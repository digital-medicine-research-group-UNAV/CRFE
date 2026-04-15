#!/usr/bin/env python3
"""
Stopping experiments for CRFE and RFE.

This script runs paired multiruns with stopping criteria activated for both
methods and generates publication-style reports under RESULTS/stopping.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _crfe import CRFE
from _crfe_utils import READER
from rfe_module import Stepwise_RFE

plt.switch_backend("Agg")
warnings.filterwarnings("ignore")

METHODS = ("crfe", "rfe")
FINAL_METRICS = (
    "coverage",
    "inefficiency",
    "certainty",
    "uncertainty",
    "mistrust",
    "S_score",
    "F_score",
    "Creditibily",
)
PLOT_COLORS = {"CRFE": "#2E86AB", "RFE": "#A23B72"}


@dataclass(frozen=True)
class RunResult:
    run_id: int
    method: str
    selected_indices: np.ndarray
    metrics: dict[str, float]


@lru_cache(maxsize=256)
def get_random_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_integer(seed: int) -> int:
    return int(get_random_generator(seed).integers(low=0, high=100000))


def create_estimator(run_id: int) -> LinearSVC:
    return LinearSVC(
        tol=1e-4,
        loss="squared_hinge",
        max_iter=14000,
        dual="auto",
        random_state=random_integer(1000 + run_id),
    )


def split_data_fairly(
    X: np.ndarray,
    Y: np.ndarray,
    run_id: int,
    test_size: float = 0.15,
    cal_size: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed = random_integer(42 + run_id)

    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X,
        Y,
        test_size=test_size,
        shuffle=True,
        stratify=Y,
        random_state=seed,
    )

    X_tr, X_cal, Y_tr, Y_cal = train_test_split(
        X_temp,
        Y_temp,
        test_size=cal_size,
        shuffle=True,
        stratify=Y_temp,
        random_state=seed,
    )

    return X_tr, X_cal, X_test, Y_tr, Y_cal, Y_test


def standardize_split_data(
    X_tr: np.ndarray,
    X_cal: np.ndarray,
    X_test: np.ndarray,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scaler is None:
        scaler = StandardScaler()

    scaler.fit(X_tr)
    X_tr_scaled = scaler.transform(X_tr)
    X_cal_scaled = scaler.transform(X_cal)
    X_test_scaled = scaler.transform(X_test)
    return X_tr_scaled, X_cal_scaled, X_test_scaled


def build_selector(method: str, estimator: LinearSVC) -> CRFE | Stepwise_RFE:
    common_args = {
        "estimator": estimator,
        "features_to_select": 1,
        "stopping_activated": True,
    }
    if method == "crfe":
        return CRFE(**common_args)
    if method == "rfe":
        return Stepwise_RFE(**common_args)
    raise ValueError(f"Unsupported method: {method}")


def run_single_experiment(
    method: str,
    run_id: int,
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> dict[str, list[Any]]:
    selector = build_selector(method, create_estimator(run_id))
    selector.fit(
        X_tr.copy(),
        Y_tr.copy(),
        X_cal.copy(),
        Y_cal.copy(),
        X_test.copy(),
        Y_test.copy(),
    )
    return selector.results_dicc


def ensure_output_dirs(dataset: str) -> tuple[Path, Path, Path]:
    stopping_root = PROJECT_ROOT / "RESULTS" / "stopping"
    dataset_root = stopping_root / f"results_{dataset}"
    run_results_dir = dataset_root / "results"

    stopping_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    run_results_dir.mkdir(parents=True, exist_ok=True)

    return stopping_root, dataset_root, run_results_dir


def result_file(run_results_dir: Path, method: str, run_id: int) -> Path:
    return run_results_dir / f"result_{method}_{run_id}.pickle"


def save_result(result_dict: dict[str, list[Any]], outpath: Path) -> None:
    with outpath.open("wb") as f:
        pickle.dump([result_dict], f, protocol=4)


def load_result(path: Path) -> dict[str, list[Any]]:
    with path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, list) or len(payload) == 0 or not isinstance(payload[0], dict):
        raise ValueError(f"Unexpected result format in file: {path}")

    return payload[0]


def extract_final_result(run_id: int, method: str, result: dict[str, list[Any]]) -> RunResult | None:
    index_steps = result.get("Index", [])
    if len(index_steps) == 0:
        return None

    final_indices = np.asarray(index_steps[-1], dtype=int)
    metrics: dict[str, float] = {}

    for metric in FINAL_METRICS:
        values = result.get(metric, [])
        metrics[metric] = float(values[-1]) if len(values) > 0 else np.nan

    return RunResult(
        run_id=run_id,
        method=method.upper(),
        selected_indices=final_indices,
        metrics=metrics,
    )


def collect_paired_results(
    run_results_dir: Path,
    run_ids: list[int],
) -> tuple[pd.DataFrame, dict[str, list[int]], list[int], dict[str, dict[int, dict[str, list[Any]]]]]:
    per_method_final: dict[str, dict[int, RunResult]] = {method: {} for method in METHODS}
    per_method_full: dict[str, dict[int, dict[str, list[Any]]]] = {method: {} for method in METHODS}

    for run_id in run_ids:
        for method in METHODS:
            path = result_file(run_results_dir, method, run_id)
            if not path.is_file():
                continue

            result = load_result(path)
            per_method_full[method][run_id] = result
            final_result = extract_final_result(run_id, method, result)
            if final_result is not None:
                per_method_final[method][run_id] = final_result

    paired_run_ids = sorted(set(per_method_final["crfe"]) & set(per_method_final["rfe"]))

    rows: list[dict[str, Any]] = []
    final_indices: dict[str, list[int]] = {"CRFE": [], "RFE": []}

    for run_id in paired_run_ids:
        for method in METHODS:
            rr = per_method_final[method][run_id]
            rows.append(
                {
                    "run": rr.run_id,
                    "method": rr.method,
                    "n_features": int(len(rr.selected_indices)),
                    "inefficiency": rr.metrics["inefficiency"],
                    "certainty": rr.metrics["certainty"],
                    "coverage": rr.metrics["coverage"],
                }
            )
            final_indices[rr.method].extend(rr.selected_indices.tolist())

    final_df = pd.DataFrame(rows)
    return final_df, final_indices, paired_run_ids, per_method_full


def build_cardinality_stats(
    per_method_full: dict[str, dict[int, dict[str, list[Any]]]],
    paired_run_ids: list[int],
    min_cardinality_count: int,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    metrics_to_plot = ("coverage", "inefficiency", "certainty")
    output: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for method in METHODS:
        method_label = method.upper()
        output[method_label] = {}

        metric_accumulators: dict[str, defaultdict[int, list[float]]] = {
            metric: defaultdict(list) for metric in metrics_to_plot
        }

        for run_id in paired_run_ids:
            result = per_method_full[method].get(run_id)
            if result is None:
                continue

            index_steps = result.get("Index", [])
            if len(index_steps) == 0:
                continue

            for step_idx, idx in enumerate(index_steps):
                cardinality = int(len(idx))
                for metric in metrics_to_plot:
                    values = result.get(metric, [])
                    if step_idx >= len(values):
                        continue
                    value = float(values[step_idx])
                    if np.isfinite(value):
                        metric_accumulators[metric][cardinality].append(value)

        for metric in metrics_to_plot:
            card_to_values = metric_accumulators[metric]
            cards = sorted(card_to_values.keys(), reverse=True)
            means: list[float] = []
            stds: list[float] = []
            counts: list[int] = []

            for card in cards:
                values = np.asarray(card_to_values[card], dtype=float)
                counts.append(int(values.size))
                if values.size >= min_cardinality_count:
                    means.append(float(np.mean(values)))
                    stds.append(float(np.std(values)))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            output[method_label][metric] = {
                "cards": np.asarray(cards, dtype=int),
                "mean": np.asarray(means, dtype=float),
                "std": np.asarray(stds, dtype=float),
                "count": np.asarray(counts, dtype=int),
            }

    return output


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["figure.titlesize"] = 18


def _paired_p_value(df: pd.DataFrame, metric: str) -> float:
    crfe_values = df[df["method"] == "CRFE"][metric]
    rfe_values = df[df["method"] == "RFE"][metric]
    if len(crfe_values) == 0 or len(rfe_values) == 0:
        return np.nan
    _, p_value = mannwhitneyu(crfe_values, rfe_values, alternative="two-sided")
    return float(p_value)


def plot_final_distributions(final_df: pd.DataFrame, dataset: str, output_dir: Path) -> Path:
    if final_df.empty:
        raise ValueError("No paired runs were found to build stopping distribution plots.")

    rng = np.random.default_rng(20260306)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metric_spec = [
        ("n_features", "Set Size", "(A) Set Size Distribution at Stopping"),
        ("inefficiency", "Uncertainty Score", "(B) Uncertainty Score Distribution"),
        ("certainty", "Certainty Score", "(C) Certainty Score Distribution"),
    ]

    for ax, (metric, ylabel, title) in zip(axes, metric_spec):
        crfe_values = final_df[final_df["method"] == "CRFE"][metric]
        rfe_values = final_df[final_df["method"] == "RFE"][metric]

        box_data = [crfe_values, rfe_values]
        bp = ax.boxplot(
            box_data,
            labels=["CRFE", "RFE"],
            patch_artist=True,
            boxprops=dict(linewidth=2),
            whiskerprops=dict(linewidth=2),
            capprops=dict(linewidth=2),
            medianprops=dict(linewidth=2, color="red"),
        )

        bp["boxes"][0].set_facecolor(PLOT_COLORS["CRFE"])
        bp["boxes"][1].set_facecolor(PLOT_COLORS["RFE"])
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_alpha(0.7)

        x1_pos = rng.normal(1, 0.02, size=len(crfe_values))
        x2_pos = rng.normal(2, 0.02, size=len(rfe_values))
        ax.scatter(
            x1_pos,
            crfe_values,
            alpha=0.8,
            color=PLOT_COLORS["CRFE"],
            s=60,
            edgecolors="black",
            linewidth=0.5,
        )
        ax.scatter(
            x2_pos,
            rfe_values,
            alpha=0.8,
            color=PLOT_COLORS["RFE"],
            s=60,
            edgecolors="black",
            linewidth=0.5,
        )

        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    outpath = output_dir / f"CRFE_vs_RFE_stopping_{dataset}.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return outpath


def plot_final_feature_frequency(
    final_indices: dict[str, list[int]],
    dataset: str,
    output_dir: Path,
    top_k: int,
) -> Path:
    crfe_counter = Counter(final_indices.get("CRFE", []))
    rfe_counter = Counter(final_indices.get("RFE", []))

    if not crfe_counter and not rfe_counter:
        raise ValueError("No final selected feature indices were found for the frequency plot.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for ax, counter, method in (
        (axes[0], crfe_counter, "CRFE"),
        (axes[1], rfe_counter, "RFE"),
    ):
        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_k]
        features = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]

        ax.bar(
            range(len(features)),
            counts,
            alpha=0.7,
            color=PLOT_COLORS[method],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title(
            f"({ 'A' if method == 'CRFE' else 'B' }) {method} - Feature Selection Frequency Distribution",
            fontweight="bold",
            fontsize=16,
            pad=20,
        )
        ax.set_xlabel("Feature Index", fontweight="bold", fontsize=14)
        ax.set_ylabel("Selection Frequency", fontweight="bold", fontsize=14)
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha="right", fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    outpath = output_dir / f"CRFE_vs_RFE_distribution_{dataset}.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return outpath


def plot_trajectory_summary(
    card_stats: dict[str, dict[str, dict[str, np.ndarray]]],
    dataset: str,
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metric_spec = [
        ("coverage", "Coverage Score", "(A) Coverage by Set Size"),
        ("inefficiency", "Uncertainty Score", "(B) Uncertainty by Set Size"),
        ("certainty", "Certainty Score", "(C) Certainty by Set Size"),
    ]

    for ax, (metric_key, ylabel, title) in zip(axes, metric_spec):
        for method in ("CRFE", "RFE"):
            stats = card_stats.get(method, {}).get(metric_key)
            if stats is None:
                continue

            cards = stats["cards"]
            mean = stats["mean"]
            std = stats["std"]

            if cards.size == 0:
                continue

            order = np.argsort(cards)
            cards_sorted = cards[order]
            mean_sorted = mean[order]
            std_sorted = std[order]

            valid = np.isfinite(mean_sorted)
            if not np.any(valid):
                continue

            cards_plot = cards_sorted[valid]
            mean_plot = mean_sorted[valid]
            std_plot = np.nan_to_num(std_sorted[valid], nan=0.0)

            ax.plot(
                cards_plot,
                mean_plot,
                color=PLOT_COLORS[method],
                linewidth=2.0,
                linestyle="-" if method == "CRFE" else "--",
                label=method,
            )
            ax.fill_between(
                cards_plot,
                mean_plot - std_plot,
                mean_plot + std_plot,
                color=PLOT_COLORS[method],
                alpha=0.18,
            )

        ax.set_xlabel("Set Size", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=16)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    outpath = output_dir / f"CRFE_vs_RFE_trajectory_{dataset}.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return outpath


def validate_dataset_path(dataset: str) -> None:
    if dataset == "synthetic":
        return

    data_dir = PROJECT_ROOT / "DATA" / dataset
    if not data_dir.is_dir():
        available = sorted(path.name for path in (PROJECT_ROOT / "DATA").iterdir() if path.is_dir())
        raise FileNotFoundError(
            f"Dataset '{dataset}' not found under DATA/. Available datasets: {available}"
        )

    for filename in ("X.csv", "Y.csv"):
        if not (data_dir / filename).is_file():
            raise FileNotFoundError(f"Missing file in dataset '{dataset}': {filename}")


def save_summary(final_df: pd.DataFrame, dataset: str, output_dir: Path) -> Path:
    summary_path = output_dir / f"stopping_summary_{dataset}.csv"
    final_df.sort_values(["run", "method"]).to_csv(summary_path, index=False)
    return summary_path


def print_summary(final_df: pd.DataFrame) -> None:
    feature_p = _paired_p_value(final_df, "n_features")
    uncertainty_p = _paired_p_value(final_df, "inefficiency")
    certainty_p = _paired_p_value(final_df, "certainty")

    crfe = final_df[final_df["method"] == "CRFE"]
    rfe = final_df[final_df["method"] == "RFE"]

    print("\nSummary Statistics")
    print("=" * 50)
    print(
        f"CRFE set size: mean={crfe['n_features'].mean():.2f}, std={crfe['n_features'].std():.2f}, "
        f"median={crfe['n_features'].median():.2f}"
    )
    print(
        f"RFE  set size: mean={rfe['n_features'].mean():.2f}, std={rfe['n_features'].std():.2f}, "
        f"median={rfe['n_features'].median():.2f}, p_value={feature_p:.4f}"
    )
    print(
        f"CRFE uncertainty: mean={crfe['inefficiency'].mean():.4f}, std={crfe['inefficiency'].std():.4f}, "
        f"median={crfe['inefficiency'].median():.4f}"
    )
    print(
        f"RFE  uncertainty: mean={rfe['inefficiency'].mean():.4f}, std={rfe['inefficiency'].std():.4f}, "
        f"median={rfe['inefficiency'].median():.4f}, p_value={uncertainty_p:.4f}"
    )
    print(
        f"CRFE certainty: mean={crfe['certainty'].mean():.4f}, std={crfe['certainty'].std():.4f}, "
        f"median={crfe['certainty'].median():.4f}"
    )
    print(
        f"RFE  certainty: mean={rfe['certainty'].mean():.4f}, std={rfe['certainty'].std():.4f}, "
        f"median={rfe['certainty'].median():.4f}, p_value={certainty_p:.4f}"
    )


def run_experiments(
    data_path: str,
    run_ids: list[int],
    run_results_dir: Path,
    overwrite: bool,
) -> None:
    reader = READER()
    X, Y, y_classes = reader.get_data(data_path)
    print(f"Loaded dataset '{data_path}' with shape={X.shape} and classes={list(y_classes)}")

    for run_id in run_ids:
        print(f"\n[Run {run_id}] Preparing paired split for CRFE and RFE")
        X_tr, X_cal, X_test, Y_tr, Y_cal, Y_test = split_data_fairly(X, Y, run_id)
        X_tr, X_cal, X_test = standardize_split_data(X_tr, X_cal, X_test)

        for method in METHODS:
            outpath = result_file(run_results_dir, method, run_id)
            if outpath.exists() and not overwrite:
                print(f"[Run {run_id}] {method.upper()} result exists, skipping: {outpath}")
                continue

            print(f"[Run {run_id}] Running {method.upper()} with stopping criteria activated")
            result = run_single_experiment(
                method,
                run_id,
                X_tr,
                Y_tr,
                X_cal,
                Y_cal,
                X_test,
                Y_test,
            )
            save_result(result, outpath)
            print(f"[Run {run_id}] Saved {method.upper()} result: {outpath}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CRFE/RFE stopping-criteria experiments and generate reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Dataset folder name inside DATA/ (or 'synthetic').",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=50,
        help="Number of multiruns to execute.",
    )
    parser.add_argument(
        "--start_run_id",
        type=int,
        default=0,
        help="Initial run id.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute runs even when result pickles already exist.",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Run experiments only and skip report generation.",
    )
    parser.add_argument(
        "--top_k_features",
        type=int,
        default=30,
        help="Top-K final selected features per method in the frequency plot.",
    )
    parser.add_argument(
        "--min_cardinality_count",
        type=int,
        default=1,
        help="Minimum number of runs required to show a cardinality point in trajectory plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.n_runs <= 0:
        raise ValueError("--n_runs must be > 0")
    if args.start_run_id < 0:
        raise ValueError("--start_run_id must be >= 0")
    if args.top_k_features <= 0:
        raise ValueError("--top_k_features must be > 0")
    if args.min_cardinality_count <= 0:
        raise ValueError("--min_cardinality_count must be > 0")

    validate_dataset_path(args.data_path)
    configure_plot_style()

    run_ids = list(range(args.start_run_id, args.start_run_id + args.n_runs))
    stopping_root, dataset_root, run_results_dir = ensure_output_dirs(args.data_path)

    print(f"Stopping root: {stopping_root}")
    print(f"Dataset result root: {dataset_root}")

    run_experiments(
        data_path=args.data_path,
        run_ids=run_ids,
        run_results_dir=run_results_dir,
        overwrite=args.overwrite,
    )

    final_df, final_indices, paired_run_ids, per_method_full = collect_paired_results(
        run_results_dir,
        run_ids,
    )

    if len(paired_run_ids) == 0:
        raise RuntimeError(
            "No paired CRFE/RFE runs were found. "
            "No plots can be generated."
        )

    if len(paired_run_ids) < len(run_ids):
        print(
            f"Warning: only {len(paired_run_ids)} paired runs available out of {len(run_ids)} requested."
        )

    summary_path = save_summary(final_df, args.data_path, stopping_root)
    print(f"Saved summary table: {summary_path}")

    if args.skip_plots:
        print("Plot generation skipped by --skip_plots")
        return

    final_plot_path = plot_final_distributions(final_df, args.data_path, stopping_root)
    freq_plot_path = plot_final_feature_frequency(
        final_indices,
        args.data_path,
        stopping_root,
        top_k=args.top_k_features,
    )

    card_stats = build_cardinality_stats(
        per_method_full,
        paired_run_ids,
        min_cardinality_count=args.min_cardinality_count,
    )
    traj_plot_path = plot_trajectory_summary(card_stats, args.data_path, stopping_root)

    print_summary(final_df)

    print("\nGenerated files")
    print("=" * 50)
    print(f"Final distributions: {final_plot_path}")
    print(f"Final feature frequency: {freq_plot_path}")
    print(f"Trajectory summary: {traj_plot_path}")


if __name__ == "__main__":
    main()
