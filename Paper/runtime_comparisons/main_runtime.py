#!/usr/bin/env python3
"""
Fair runtime comparison experiment for CRFE-HPC_UNAV feature selection methods.

Key fairness upgrades while keeping the same benchmark grid:
- paired datasets across sample-size conditions (same base data per class/feature setting),
- configurable fixed LinearSVC optimization regime to avoid solver-mode flips,
- convergence-warning capture per timed run,
- robust runtime summary statistics (mean/std + median/MAD),
- optional fairness diagnostics report generation.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from _crfe import CRFE, Stepwise_SMFS
from rfe_module import Stepwise_BORUTA, Stepwise_ELASTICNET, Stepwise_LASSO, Stepwise_RFE

AVAILABLE_METHODS = ("crfe", "rfe", "smfs", "lasso", "elasticnet", "boruta")
METHOD_LABELS = {
    "crfe": "CRFE",
    "rfe": "RFE",
    "smfs": "SMFS",
    "lasso": "LASSO",
    "elasticnet": "Elastic Net",
    "boruta": "BORUTA",
}


@dataclass
class ExperimentConfig:
    methods: list[str]
    n_classes_grid: list[int]
    n_samples_grid: list[int]
    n_features_grid: list[int]
    repeats: int
    n_informative: int
    target_features: int
    seed: int
    threads: int
    test_size: float
    cal_size: float
    boruta_max_iter: int
    warmup: bool
    silence_method_logs: bool
    shuffle_method_order: bool
    skip_plots: bool
    output_dir: Path
    svc_dual: bool | str
    svc_tol: float
    svc_max_iter: int
    paired_sampling: bool
    fairness_report: bool


@dataclass
class TimingResult:
    elapsed_s: float
    warning_count: int
    convergence_warning_count: int


def parse_methods(raw: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    for item in raw:
        tokens.extend([part.strip().lower() for part in item.split(",") if part.strip()])

    if not tokens:
        raise ValueError("At least one method must be provided with --methods")

    if len(tokens) == 1 and tokens[0] == "all":
        return list(AVAILABLE_METHODS)

    unique_tokens: list[str] = []
    for token in tokens:
        if token not in AVAILABLE_METHODS:
            valid = ", ".join(AVAILABLE_METHODS)
            raise ValueError(f"Unknown method '{token}'. Valid methods: {valid}")
        if token not in unique_tokens:
            unique_tokens.append(token)

    return unique_tokens


def parse_int_grid(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("Grid arguments cannot be empty")
    return values


def parse_svc_dual(raw: str) -> bool | str:
    token = raw.strip().lower()
    if token == "auto":
        return "auto"
    if token == "true":
        return True
    if token == "false":
        return False
    raise ValueError("--svc-dual must be one of: auto, true, false")


def build_config() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Fair runtime comparison for CRFE-HPC_UNAV feature selectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help=(
            "Methods to compare (space and/or comma separated). "
            "Available: crfe, rfe, smfs, lasso, elasticnet, boruta, all"
        ),
    )
    
    parser.add_argument("--n-classes-grid", type=str, default="3,5")
    parser.add_argument("--n-samples-grid", type=str, default="500,1500")
    parser.add_argument("--n-features-grid", type=str, default="50,200,1000")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-informative", type=int, default=10)
    parser.add_argument("--target-features", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--cal-size", type=float, default=0.5)
    parser.add_argument("--boruta-max-iter", type=int, default=50)

    parser.add_argument(
        "--svc-dual",
        type=str,
        default="false",
        help=(
            "LinearSVC dual regime. 'false' enforces a single optimization regime across "
            "all scenarios for more reliable complexity comparisons"
        ),
    )
    parser.add_argument("--svc-tol", type=float, default=1e-4)
    parser.add_argument("--svc-max-iter", type=int, default=14_000)

    parser.add_argument("--paired-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fairness-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--silence-method-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Redirect method internal stdout/stderr to reduce timing contamination by console I/O",
    )
    parser.add_argument(
        "--shuffle-method-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle method execution order in each repeat to reduce order bias",
    )
    parser.add_argument("--skip-plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "RESULTS" / "runtimes"),
    )

    args = parser.parse_args()

    methods = parse_methods(args.methods)
    n_classes_grid = parse_int_grid(args.n_classes_grid)
    n_samples_grid = parse_int_grid(args.n_samples_grid)
    n_features_grid = parse_int_grid(args.n_features_grid)
    svc_dual = parse_svc_dual(args.svc_dual)

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.n_informative <= 0:
        raise ValueError("--n-informative must be positive")
    if args.target_features <= 0:
        raise ValueError("--target-features must be positive")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if not (0 < args.test_size < 1):
        raise ValueError("--test-size must be in (0, 1)")
    if not (0 < args.cal_size < 1):
        raise ValueError("--cal-size must be in (0, 1)")
    if args.boruta_max_iter <= 0:
        raise ValueError("--boruta-max-iter must be positive")
    if args.svc_tol <= 0:
        raise ValueError("--svc-tol must be positive")
    if args.svc_max_iter <= 0:
        raise ValueError("--svc-max-iter must be positive")

    max_classes = max(n_classes_grid)
    min_required_informative = int(np.ceil(np.log2(max_classes)))
    if args.n_informative < min_required_informative:
        raise ValueError(
            "--n-informative is too small for the selected number of classes. "
            f"For max classes={max_classes}, use n_informative >= {min_required_informative}."
        )

    minimum_required_features = max(args.n_informative, args.target_features)
    filtered_features = [f for f in n_features_grid if f >= minimum_required_features]
    if not filtered_features:
        raise ValueError(
            "No valid --n-features-grid values remain after enforcing "
            f"n_features >= max(n_informative, target_features) = {minimum_required_features}"
        )
    if len(filtered_features) < len(n_features_grid):
        removed = sorted(set(n_features_grid) - set(filtered_features))
        print(
            "Warning: dropping feature counts that are too small for this setup: "
            f"{removed}"
        )

    if args.paired_sampling and max(n_samples_grid) <= 0:
        raise ValueError("--n-samples-grid must contain positive values")

    return ExperimentConfig(
        methods=methods,
        n_classes_grid=n_classes_grid,
        n_samples_grid=n_samples_grid,
        n_features_grid=filtered_features,
        repeats=args.repeats,
        n_informative=args.n_informative,
        target_features=args.target_features,
        seed=args.seed,
        threads=args.threads,
        test_size=args.test_size,
        cal_size=args.cal_size,
        boruta_max_iter=args.boruta_max_iter,
        warmup=args.warmup,
        silence_method_logs=args.silence_method_logs,
        shuffle_method_order=args.shuffle_method_order,
        skip_plots=args.skip_plots,
        output_dir=Path(args.output_dir),
        svc_dual=svc_dual,
        svc_tol=args.svc_tol,
        svc_max_iter=args.svc_max_iter,
        paired_sampling=args.paired_sampling,
        fairness_report=args.fairness_report,
    )


def create_estimator(config: ExperimentConfig, random_state: int) -> LinearSVC:
    return LinearSVC(
        tol=config.svc_tol,
        loss="squared_hinge",
        max_iter=config.svc_max_iter,
        dual=config.svc_dual,
        random_state=random_state,
    )


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    split_seed: int,
    test_size: float,
    cal_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        stratify=y,
        random_state=split_seed,
    )

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp,
        y_temp,
        test_size=cal_size,
        shuffle=True,
        stratify=y_temp,
        random_state=split_seed,
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_cal = scaler.transform(X_cal)
    X_test = scaler.transform(X_test)

    return X_train, X_cal, X_test, y_train, y_cal, y_test


def build_selector(method: str, config: ExperimentConfig, random_state: int):
    base_kwargs = {
        "estimator": create_estimator(config=config, random_state=random_state),
        "features_to_select": config.target_features,
        "stopping_activated": False,
    }

    if method == "crfe":
        return CRFE(**base_kwargs)
    if method == "rfe":
        return Stepwise_RFE(**base_kwargs)
    if method == "smfs":
        return Stepwise_SMFS(**base_kwargs)
    if method == "lasso":
        return Stepwise_LASSO(
            **base_kwargs,
            random_state=random_state,
            n_jobs=config.threads,
        )
    if method == "elasticnet":
        return Stepwise_ELASTICNET(
            **base_kwargs,
            random_state=random_state,
            n_jobs=config.threads,
        )
    if method == "boruta":
        return Stepwise_BORUTA(
            **base_kwargs,
            random_state=random_state,
            max_iter=config.boruta_max_iter,
            n_jobs=config.threads,
            recompute_each_step=False,
        )

    raise ValueError(f"Unknown method: {method}")


@contextmanager
def maybe_silence(enabled: bool):
    if not enabled:
        yield
        return

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def _is_convergence_warning(caught_warning: warnings.WarningMessage) -> bool:
    if issubclass(caught_warning.category, ConvergenceWarning):
        return True

    msg = str(caught_warning.message).lower()
    return "converge" in msg and ("failed" in msg or "did not" in msg)


def time_single_fit(
    method: str,
    config: ExperimentConfig,
    data_split: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    random_state: int,
) -> TimingResult:
    X_train, X_cal, X_test, y_train, y_cal, y_test = data_split

    gc.collect()
    with threadpool_limits(limits=config.threads):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with maybe_silence(config.silence_method_logs):
                t0 = time.perf_counter()
                selector = build_selector(method, config, random_state=random_state)
                selector.fit(
                    X_train.copy(),
                    y_train.copy(),
                    X_cal.copy(),
                    y_cal.copy(),
                    X_test.copy(),
                    y_test.copy(),
                )
                t1 = time.perf_counter()

    convergence_warnings = sum(1 for w in caught if _is_convergence_warning(w))
    return TimingResult(
        elapsed_s=float(t1 - t0),
        warning_count=len(caught),
        convergence_warning_count=int(convergence_warnings),
    )


def _make_base_dataset(
    n_samples: int,
    n_features: int,
    n_informative: int,
    n_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_redundant = max(0, n_features - n_informative)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.25,
        random_state=seed,
        shuffle=True,
    )
    return X, y


def _stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples == len(y):
        return X.copy(), y.copy()

    if n_samples > len(y):
        raise ValueError(
            f"Requested n_samples={n_samples} is larger than base dataset size={len(y)}"
        )

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
    idx_train, _ = next(splitter.split(X, y))
    idx_train = np.sort(idx_train)
    return X[idx_train].copy(), y[idx_train].copy()


def warmup_methods(config: ExperimentConfig) -> None:
    print("Running warm-up (not timed) to reduce first-run/JIT bias...")

    n_classes = max(config.n_classes_grid)
    n_samples = max(max(config.n_samples_grid), n_classes * 40, 240)
    n_features = max(config.n_informative + 4, config.target_features + 4, max(config.n_features_grid))

    X, y = _make_base_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(config.n_informative, n_features - 1),
        n_classes=n_classes,
        seed=config.seed + 9_999,
    )

    split = split_and_scale(
        X,
        y,
        split_seed=config.seed + 8_888,
        test_size=config.test_size,
        cal_size=config.cal_size,
    )

    for idx, method in enumerate(config.methods):
        warmup_seed = config.seed + 50_000 + idx
        try:
            _ = time_single_fit(method, config, split, warmup_seed)
            print(f"  warm-up ok: {method}")
        except Exception as exc:
            print(f"  warm-up failed: {method} ({exc})")


def _ordered_scenarios(config: ExperimentConfig) -> list[tuple[int, int, int]]:
    return [
        (n_classes, n_samples, n_features)
        for n_classes in config.n_classes_grid
        for n_samples in config.n_samples_grid
        for n_features in config.n_features_grid
    ]


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    ordered = _ordered_scenarios(config)
    total_scenarios = len(ordered)
    scenario_to_idx = {scenario: idx + 1 for idx, scenario in enumerate(ordered)}

    max_samples = max(config.n_samples_grid)
    base_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    for n_classes in config.n_classes_grid:
        for n_features in config.n_features_grid:
            base_seed = config.seed + n_classes * 100_000 + n_features * 1_000 + 17
            base_cache[(n_classes, n_features)] = _make_base_dataset(
                n_samples=max_samples,
                n_features=n_features,
                n_informative=config.n_informative,
                n_classes=n_classes,
                seed=base_seed,
            )

    # Iterate in the same scenario order as the original benchmark definition.
    for n_classes, n_samples, n_features in ordered:
        scenario_idx = scenario_to_idx[(n_classes, n_samples, n_features)]
        print(
            f"\nScenario {scenario_idx}/{total_scenarios}: "
            f"classes={n_classes}, samples={n_samples}, features={n_features}"
        )

        X_base, y_base = base_cache[(n_classes, n_features)]

        for repeat in range(config.repeats):
            if config.paired_sampling:
                subset_seed = config.seed + scenario_idx * 10_000 + repeat * 131
                X, y = _stratified_subsample(X_base, y_base, n_samples=n_samples, seed=subset_seed)
            else:
                fresh_seed = config.seed + scenario_idx * 1_000 + repeat
                X, y = _make_base_dataset(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_informative=config.n_informative,
                    n_classes=n_classes,
                    seed=fresh_seed,
                )

            split_seed = config.seed + scenario_idx * 100 + repeat
            data_split = split_and_scale(
                X,
                y,
                split_seed=split_seed,
                test_size=config.test_size,
                cal_size=config.cal_size,
            )

            if config.shuffle_method_order:
                method_order = list(np.random.default_rng(split_seed).permutation(config.methods))
            else:
                method_order = list(config.methods)

            for method in method_order:
                method_seed = split_seed * 100 + AVAILABLE_METHODS.index(method) + 1
                try:
                    timing = time_single_fit(
                        method=method,
                        config=config,
                        data_split=data_split,
                        random_state=method_seed,
                    )
                    status = "ok"
                    error = ""
                    print(
                        f"  repeat={repeat + 1}/{config.repeats} | "
                        f"{method:<10} {timing.elapsed_s:>10.4f} s "
                        f"(conv_warn={timing.convergence_warning_count})"
                    )
                    elapsed = timing.elapsed_s
                    warning_count = timing.warning_count
                    convergence_warning_count = timing.convergence_warning_count
                except Exception as exc:
                    elapsed = np.nan
                    warning_count = 0
                    convergence_warning_count = 0
                    status = "failed"
                    error = str(exc)
                    print(
                        f"  repeat={repeat + 1}/{config.repeats} | "
                        f"{method:<10} FAILED ({exc})"
                    )

                rows.append(
                    {
                        "n_classes": n_classes,
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "repeat": repeat,
                        "method": method,
                        "elapsed_s": elapsed,
                        "status": status,
                        "error": error,
                        "warning_count": warning_count,
                        "convergence_warning_count": convergence_warning_count,
                    }
                )

    return pd.DataFrame(rows)


def _median_abs_deviation(series: pd.Series) -> float:
    arr = series.to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def summarize_results(raw: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    success = raw[raw["status"] == "ok"].copy()

    agg = (
        success.groupby(["n_classes", "n_samples", "n_features", "method"], as_index=False)
        .agg(
            mean_s=("elapsed_s", "mean"),
            std_s=("elapsed_s", lambda s: float(np.std(s.to_numpy(), ddof=0))),
            median_s=("elapsed_s", "median"),
            mad_s=("elapsed_s", _median_abs_deviation),
            n_success=("elapsed_s", "count"),
            warning_count=("warning_count", "sum"),
            convergence_warning_count=("convergence_warning_count", "sum"),
        )
    )

    agg["cv_s"] = np.where(agg["mean_s"] > 0, agg["std_s"] / agg["mean_s"], np.nan)
    agg["warn_rate"] = np.where(
        agg["n_success"] > 0,
        agg["convergence_warning_count"] / agg["n_success"],
        np.nan,
    )

    full_index = pd.MultiIndex.from_product(
        [
            config.n_classes_grid,
            config.n_samples_grid,
            config.n_features_grid,
            config.methods,
        ],
        names=["n_classes", "n_samples", "n_features", "method"],
    ).to_frame(index=False)

    summary = full_index.merge(
        agg,
        on=["n_classes", "n_samples", "n_features", "method"],
        how="left",
    )
    summary["n_attempted"] = config.repeats
    summary.sort_values(["n_classes", "n_samples", "n_features", "method"], inplace=True)
    summary.reset_index(drop=True, inplace=True)

    return summary


def method_tag(methods: Sequence[str]) -> str:
    return "_".join(methods)


def save_tables(summary: pd.DataFrame, raw: pd.DataFrame, output_dir: Path, tag: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_generic = output_dir / "runtime_vs_features_results_2x2.csv"
    summary_tagged = output_dir / f"runtime_vs_features_results_2x2_{tag}.csv"
    raw_generic = output_dir / "runtime_vs_features_raw.csv"
    raw_tagged = output_dir / f"runtime_vs_features_raw_{tag}.csv"

    summary.to_csv(summary_generic, index=False)
    summary.to_csv(summary_tagged, index=False)
    raw.to_csv(raw_generic, index=False)
    raw.to_csv(raw_tagged, index=False)

    print(f"\nSaved summary CSV: {summary_generic}")
    print(f"Saved tagged summary CSV: {summary_tagged}")
    print(f"Saved raw CSV: {raw_generic}")
    print(f"Saved tagged raw CSV: {raw_tagged}")


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.facecolor": "white",
        }
    )


def get_axes_array(axes, nrows: int, ncols: int) -> np.ndarray:
    axes_arr = np.asarray(axes)
    if nrows == 1 and ncols == 1:
        return axes_arr.reshape(1, 1)
    if nrows == 1:
        return axes_arr.reshape(1, ncols)
    if ncols == 1:
        return axes_arr.reshape(nrows, 1)
    return axes_arr


def plot_runtime_grid(summary: pd.DataFrame, methods: Sequence[str], output_dir: Path, tag: str) -> None:
    valid = summary.dropna(subset=["mean_s"]).copy()
    if valid.empty:
        print("No successful runs. Skipping plot generation.")
        return

    configure_plot_style()

    method_order = list(methods)
    color_cycle = plt.get_cmap("tab10")
    color_map = {method: color_cycle(i) for i, method in enumerate(method_order)}
    marker_cycle = ["o", "s", "^", "D", "v", "P"]
    marker_map = {method: marker_cycle[i % len(marker_cycle)] for i, method in enumerate(method_order)}
    linestyle_cycle = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (4, 1, 1, 1))]
    linestyle_map = {
        method: linestyle_cycle[i % len(linestyle_cycle)] for i, method in enumerate(method_order)
    }

    class_levels = sorted(valid["n_classes"].unique())
    sample_levels = sorted(valid["n_samples"].unique())
    feature_levels = sorted(valid["n_features"].unique())

    fig, axes = plt.subplots(
        nrows=len(class_levels),
        ncols=len(sample_levels),
        figsize=(8.2, 5.8),
        sharex=True,
        sharey=True,
    )
    axes = get_axes_array(axes, len(class_levels), len(sample_levels))

    legend_handles = []
    legend_labels = []

    for row, n_classes in enumerate(class_levels):
        for col, n_samples in enumerate(sample_levels):
            ax = axes[row, col]
            block = valid[
                (valid["n_classes"] == n_classes) & (valid["n_samples"] == n_samples)
            ].copy()
            block.sort_values(["method", "n_features"], inplace=True)

            for method in method_order:
                chunk = block[block["method"] == method]
                if chunk.empty:
                    continue

                line = ax.errorbar(
                    chunk["n_features"],
                    chunk["mean_s"],
                    yerr=chunk["std_s"],
                    marker=marker_map[method],
                    markersize=4.5,
                    linewidth=1.4,
                    capsize=3,
                    color=color_map[method],
                    linestyle=linestyle_map[method],
                    label=METHOD_LABELS.get(method, method),
                )

                label = METHOD_LABELS.get(method, method)
                if label not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(label)

            ax.set_title(f"{n_classes} classes, {n_samples} samples")
            ax.set_yscale("log")
            ax.set_xlim(min(feature_levels) - 5, max(feature_levels) + 5)
            ax.set_xticks(feature_levels)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
            ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=range(1, 10)))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
            ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.4, alpha=0.35)
            ax.tick_params(axis="both", direction="out", length=4, width=0.8)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

            if col == 0:
                ax.set_ylabel("Runtime (s)")
            if row == len(class_levels) - 1:
                ax.set_xlabel("Total number of features")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=min(3, len(legend_labels)),
            frameon=False,
            columnspacing=1.6,
            handlelength=2.4,
        )

    fig.subplots_adjust(top=0.9, bottom=0.15, left=0.09, right=0.99, wspace=0.08, hspace=0.12)

    generic_pdf = output_dir / "runtime_comparison_grid.pdf"
    generic_png = output_dir / "runtime_comparison_grid.png"
    tagged_pdf = output_dir / f"runtime_comparison_grid_{tag}.pdf"
    tagged_png = output_dir / f"runtime_comparison_grid_{tag}.png"

    fig.savefig(generic_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(generic_png, dpi=300, bbox_inches="tight")
    fig.savefig(tagged_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(tagged_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved grid plot: {generic_pdf}")
    print(f"Saved tagged grid plot: {tagged_pdf}")


def plot_runtime_blocks(summary: pd.DataFrame, methods: Sequence[str], output_dir: Path, tag: str) -> None:
    valid = summary.dropna(subset=["mean_s"]).copy()
    if valid.empty:
        return

    method_order = list(methods)
    color_cycle = plt.get_cmap("tab10")
    color_map = {method: color_cycle(i) for i, method in enumerate(method_order)}
    marker_cycle = ["o", "s", "^", "D", "v", "P"]
    marker_map = {method: marker_cycle[i % len(marker_cycle)] for i, method in enumerate(method_order)}

    for n_classes in sorted(valid["n_classes"].unique()):
        for n_samples in sorted(valid["n_samples"].unique()):
            block = valid[
                (valid["n_classes"] == n_classes) & (valid["n_samples"] == n_samples)
            ].copy()
            if block.empty:
                continue

            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            any_series = False

            for method in method_order:
                chunk = block[block["method"] == method].sort_values("n_features")
                if chunk.empty:
                    continue
                any_series = True
                ax.errorbar(
                    chunk["n_features"],
                    chunk["mean_s"],
                    yerr=chunk["std_s"],
                    marker=marker_map[method],
                    markersize=4.5,
                    linewidth=1.4,
                    capsize=3,
                    color=color_map[method],
                    linestyle="-",
                    label=METHOD_LABELS.get(method, method),
                )

            if not any_series:
                plt.close(fig)
                continue

            ax.set_xlabel("Total number of features")
            ax.set_ylabel("Runtime (seconds)")
            ax.set_yscale("log")
            ax.set_title(f"Runtime vs. features ({n_classes} classes, {n_samples} samples)")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            ax.legend(frameon=False)
            fig.tight_layout()

            generic_png = output_dir / f"runtime_vs_features_{n_classes}cls_{n_samples}n.png"
            tagged_png = output_dir / f"runtime_vs_features_{n_classes}cls_{n_samples}n_{tag}.png"
            fig.savefig(generic_png, dpi=300)
            fig.savefig(tagged_png, dpi=300)
            plt.close(fig)


def write_fairness_report(
    summary: pd.DataFrame,
    raw: pd.DataFrame,
    output_dir: Path,
    tag: str,
) -> None:
    try:
        from fairness_report import generate_fairness_report

        paths = generate_fairness_report(summary=summary, raw=raw, output_dir=output_dir, tag=tag)
        print(f"Saved fairness report: {paths['json_generic']}")
        print(f"Saved fairness flags : {paths['flags_generic']}")
    except Exception as exc:
        print(f"Warning: fairness report generation failed: {exc}")


def main() -> None:
    config = build_config()

    print("Configuration")
    print(f"  methods            : {config.methods}")
    print(f"  classes grid       : {config.n_classes_grid}")
    print(f"  samples grid       : {config.n_samples_grid}")
    print(f"  features grid      : {config.n_features_grid}")
    print(f"  repeats            : {config.repeats}")
    print(f"  informative        : {config.n_informative}")
    print(f"  target features    : {config.target_features}")
    print(f"  threads            : {config.threads}")
    print(f"  svc dual           : {config.svc_dual}")
    print(f"  svc tol            : {config.svc_tol}")
    print(f"  svc max_iter       : {config.svc_max_iter}")
    print(f"  paired sampling    : {config.paired_sampling}")
    print(f"  fairness report    : {config.fairness_report}")
    print(f"  output dir         : {config.output_dir}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.warmup:
        warmup_methods(config)

    raw = run_experiment(config)
    summary = summarize_results(raw, config)

    tag = method_tag(config.methods)
    save_tables(summary, raw, config.output_dir, tag)

    failed = raw[raw["status"] == "failed"]
    if not failed.empty:
        print("\nFailed runs detected:")
        print(
            failed[["n_classes", "n_samples", "n_features", "repeat", "method", "error"]]
            .to_string(index=False)
        )

    if config.fairness_report:
        write_fairness_report(summary=summary, raw=raw, output_dir=config.output_dir, tag=tag)

    if not config.skip_plots:
        plot_runtime_grid(summary, config.methods, config.output_dir, tag)
        plot_runtime_blocks(summary, config.methods, config.output_dir, tag)

    successful = int((raw["status"] == "ok").sum())
    total = int(len(raw))
    print(f"\nCompleted runs: {successful}/{total} successful")


if __name__ == "__main__":
    main()
