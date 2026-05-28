#!/usr/bin/env python3
"""Compute paired bootstrap confidence intervals for CRFE vs RFE.

This script uses only already-computed result pickles. It mirrors the
aggregation in create_boxplots_2.py:
- percentages are converted to selected-feature cardinalities;
- k_center-1, k_center, and k_center+1 are averaged within each seed;
- dataset-level tests use seeds as paired units;
- global tests average seeds within dataset and use datasets as paired units.
"""

from __future__ import annotations

import argparse
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_PERCENTAGES = [0.01, 0.03, 0.05, 0.07, 0.10]
DEFAULT_FOCUSED_K = [9, 10, 11]
DEFAULT_BOOTSTRAPS = 10_000
DEFAULT_RANDOM_SEED = 20260526
METHOD_A = "crfe"
METHOD_B = "rfe"


@dataclass(frozen=True)
class MetricSpec:
    output_name: str
    result_key: str
    direction: str
    scale_uncertainty: bool = False


METRICS = [
    MetricSpec("coverage", "coverage", "crfe_minus_rfe"),
    MetricSpec("average_set_size", "inefficiency", "rfe_minus_crfe"),
    MetricSpec("uncertainty_scaled", "inefficiency", "rfe_minus_crfe", True),
    MetricSpec("certainty", "certainty", "crfe_minus_rfe"),
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compute CRFE vs RFE paired bootstrap confidence intervals."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=project_root / "RESULTS",
        help="Directory containing results_<dataset> folders.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "DATA",
        help="Directory containing <dataset>/Y.csv files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_root / "results" / "statistical_ci_summary.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=project_root / "results" / "statistical_ci_summary.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--percentages",
        nargs="+",
        type=float,
        default=DEFAULT_PERCENTAGES,
        help="Relative selected-feature sizes to evaluate.",
    )
    parser.add_argument(
        "--focused-dataset",
        default="imvigor210",
        help="Dataset for focused absolute-k analysis.",
    )
    parser.add_argument(
        "--focused-k",
        nargs="+",
        type=int,
        default=DEFAULT_FOCUSED_K,
        help="Absolute selected-feature cardinalities for the focused analysis.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=DEFAULT_BOOTSTRAPS,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset names. Defaults to auto-detected CRFE/RFE datasets.",
    )
    parser.add_argument(
        "--n-classes-map",
        nargs="*",
        default=[],
        metavar="DATASET=N",
        help="Optional class-count overrides, for example synthetic=4.",
    )
    return parser.parse_args()


def detect_datasets(results_root: Path) -> list[str]:
    datasets: list[str] = []
    for entry in sorted(results_root.glob("results_*")):
        if not entry.is_dir():
            continue
        dataset = entry.name.replace("results_", "", 1)
        if dataset == "boxplots":
            continue
        has_crfe = (entry / "results").glob("result_crfe_*.pickle")
        has_rfe = (entry / "results").glob("result_rfe_*.pickle")
        if any(has_crfe) and any(has_rfe):
            datasets.append(dataset)
    return datasets


def parse_n_classes_map(items: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --n-classes-map item '{item}'. Expected DATASET=N.")
        dataset, raw_value = item.split("=", 1)
        dataset = dataset.strip()
        if not dataset:
            raise ValueError(f"Invalid --n-classes-map item '{item}'. Empty dataset name.")
        n_classes = int(raw_value)
        if n_classes < 2:
            raise ValueError(f"Invalid --n-classes-map item '{item}'. N must be >= 2.")
        overrides[dataset] = n_classes
    return overrides


def infer_n_classes_from_logs(dataset_dir: Path) -> int | None:
    logs_dir = dataset_dir / "logs"
    if not logs_dir.is_dir():
        return None
    pattern = re.compile(r"Dataset loaded:.*?,\s*(\d+)\s+classes", re.IGNORECASE)
    for log_path in sorted(logs_dir.glob("*.out")):
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = pattern.search(line)
                if match:
                    n_classes = int(match.group(1))
                    if n_classes >= 2:
                        return n_classes
    return None


def infer_n_classes(
    data_root: Path,
    dataset: str,
    dataset_dir: Path,
    n_classes_map: dict[str, int],
) -> int:
    if dataset in n_classes_map:
        return n_classes_map[dataset]

    y_path = data_root / dataset / "Y.csv"
    if y_path.is_file():
        y = pd.read_csv(y_path)
        if y.shape[1] == 0:
            raise ValueError(f"Empty class-label file: {y_path}")
        return int(np.unique(y.iloc[:, 0].to_numpy()).size)

    n_from_logs = infer_n_classes_from_logs(dataset_dir)
    if n_from_logs is not None:
        return n_from_logs

    raise FileNotFoundError(
        f"Missing class-label file: {y_path}. Provide --n-classes-map {dataset}=N."
    )


def parse_run_id(path: Path, method: str) -> int | None:
    match = re.fullmatch(rf"result_{re.escape(method)}_(\d+)\.pickle", path.name)
    if match is None:
        return None
    return int(match.group(1))


def load_run_pickles(dataset_dir: Path, method: str) -> dict[int, dict]:
    runs_dir = dataset_dir / "results"
    runs: dict[int, dict] = {}
    for path in sorted(runs_dir.glob(f"result_{method}_*.pickle")):
        run_id = parse_run_id(path, method)
        if run_id is None:
            continue
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, list) or len(loaded) != 1:
            raise ValueError(f"Unexpected pickle structure in {path}")
        if not isinstance(loaded[0], dict):
            raise ValueError(f"Unexpected run payload in {path}")
        runs[run_id] = loaded[0]
    return runs


def minmax_scale(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    denom = upper - lower
    if denom == 0:
        return np.zeros_like(values, dtype=float)
    return (values - lower) / denom


def metric_values(run: dict, metric: MetricSpec, n_classes: int) -> np.ndarray:
    if metric.result_key not in run:
        raise KeyError(f"Missing metric key '{metric.result_key}'")
    values = np.asarray(run[metric.result_key], dtype=float).reshape(-1)
    if metric.scale_uncertainty:
        values = minmax_scale(values, 1.0, float(n_classes))
    return values


def cardinality_indices(n_features: int, percentage: float) -> tuple[list[int], list[int]]:
    k_center = int(round(percentage * n_features))
    k_center = max(1, min(n_features, k_center))
    valid_ks = sorted({k for k in (k_center - 1, k_center, k_center + 1) if 1 <= k <= n_features})
    idx_cols = [n_features - k for k in valid_ks]
    return valid_ks, idx_cols


def absolute_k_index(n_features: int, k: int) -> int:
    if k < 1 or k > n_features:
        raise ValueError(f"k={k} is outside the valid range 1..{n_features}")
    return n_features - k


def paired_seed_differences_for_indices(
    runs_crfe: dict[int, dict],
    runs_rfe: dict[int, dict],
    metric: MetricSpec,
    n_classes: int,
    idx_cols_by_run: dict[int, list[int]],
) -> np.ndarray:
    diffs: list[float] = []
    for run_id in sorted(set(runs_crfe).intersection(runs_rfe)):
        crfe_values = metric_values(runs_crfe[run_id], metric, n_classes)
        rfe_values = metric_values(runs_rfe[run_id], metric, n_classes)
        if len(crfe_values) != len(rfe_values):
            raise ValueError(f"Metric length mismatch for run_id={run_id}")
        idx_cols = idx_cols_by_run[run_id]
        crfe_score = float(crfe_values[idx_cols].mean())
        rfe_score = float(rfe_values[idx_cols].mean())
        if metric.direction == "rfe_minus_crfe":
            diffs.append(rfe_score - crfe_score)
        elif metric.direction == "crfe_minus_rfe":
            diffs.append(crfe_score - rfe_score)
        else:
            raise ValueError(f"Unknown direction: {metric.direction}")
    return np.asarray(diffs, dtype=float)


def bootstrap_mean_ci(
    diffs: np.ndarray,
    n_bootstraps: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n_pairs = len(diffs)
    if n_pairs == 0:
        return math.nan, math.nan
    sample_indices = rng.integers(0, n_pairs, size=(n_bootstraps, n_pairs))
    boot_means = diffs[sample_indices].mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def paired_wilcoxon_p_value(diffs: np.ndarray) -> float:
    if len(diffs) < 2:
        return math.nan
    if np.allclose(diffs, 0.0):
        return 1.0
    try:
        return float(wilcoxon(diffs, alternative="two-sided").pvalue)
    except ValueError:
        return math.nan


def interpretation_for(metric: MetricSpec, low: float, high: float) -> str:
    if metric.output_name == "coverage":
        if low > 0:
            return "CRFE coverage is higher; interpret relative to nominal coverage."
        if high < 0:
            return "RFE coverage is higher; interpret relative to nominal coverage."
        return "CI includes zero; coverage difference should be judged by nominal proximity."
    if low > 0:
        return "Positive difference favors CRFE."
    if high < 0:
        return "Negative difference favors RFE."
    return "CI includes zero; no clear paired difference."


def format_percentage(percentage: float) -> str:
    return f"{percentage * 100:g}%"


def make_row(
    comparison_scope: str,
    dataset: str,
    metric: MetricSpec,
    subset_label: str,
    diffs: np.ndarray,
    n_bootstraps: int,
    rng: np.random.Generator,
) -> dict:
    ci_low, ci_high = bootstrap_mean_ci(diffs, n_bootstraps, rng)
    p_value = paired_wilcoxon_p_value(diffs)
    mean_diff = float(np.mean(diffs)) if len(diffs) else math.nan
    median_diff = float(np.median(diffs)) if len(diffs) else math.nan
    return {
        "comparison_scope": comparison_scope,
        "dataset": dataset,
        "metric": metric.output_name,
        "subset_percentage_or_k": subset_label,
        "n_pairs": int(len(diffs)),
        "mean_difference": mean_diff,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "wilcoxon_p_value": p_value,
        "interpretation": interpretation_for(metric, ci_low, ci_high),
        "median_difference": median_diff,
    }


def dataset_percentage_differences(
    runs_crfe: dict[int, dict],
    runs_rfe: dict[int, dict],
    metric: MetricSpec,
    n_classes: int,
    percentage: float,
) -> np.ndarray:
    idx_cols_by_run: dict[int, list[int]] = {}
    for run_id in sorted(set(runs_crfe).intersection(runs_rfe)):
        n_features = len(metric_values(runs_crfe[run_id], metric, n_classes))
        _, idx_cols_by_run[run_id] = cardinality_indices(n_features, percentage)
    return paired_seed_differences_for_indices(
        runs_crfe, runs_rfe, metric, n_classes, idx_cols_by_run
    )


def dataset_absolute_k_differences(
    runs_crfe: dict[int, dict],
    runs_rfe: dict[int, dict],
    metric: MetricSpec,
    n_classes: int,
    k: int,
) -> np.ndarray:
    idx_cols_by_run: dict[int, list[int]] = {}
    for run_id in sorted(set(runs_crfe).intersection(runs_rfe)):
        n_features = len(metric_values(runs_crfe[run_id], metric, n_classes))
        idx_cols_by_run[run_id] = [absolute_k_index(n_features, k)]
    return paired_seed_differences_for_indices(
        runs_crfe, runs_rfe, metric, n_classes, idx_cols_by_run
    )


def load_dataset_payloads(
    results_root: Path,
    data_root: Path,
    datasets: Iterable[str],
    n_classes_map: dict[str, int],
) -> dict[str, dict]:
    payloads = {}
    for dataset in datasets:
        dataset_dir = results_root / f"results_{dataset}"
        runs_crfe = load_run_pickles(dataset_dir, METHOD_A)
        runs_rfe = load_run_pickles(dataset_dir, METHOD_B)
        paired_run_ids = sorted(set(runs_crfe).intersection(runs_rfe))
        if not paired_run_ids:
            raise ValueError(f"No paired CRFE/RFE runs found for dataset '{dataset}'")
        payloads[dataset] = {
            "n_classes": infer_n_classes(data_root, dataset, dataset_dir, n_classes_map),
            "runs_crfe": runs_crfe,
            "runs_rfe": runs_rfe,
            "paired_run_ids": paired_run_ids,
        }
    return payloads


def write_summary(
    path: Path,
    rows: list[dict],
    datasets: list[str],
    payloads: dict[str, dict],
    args: argparse.Namespace,
) -> None:
    df = pd.DataFrame(rows)
    run_counts = {
        dataset: len(payloads[dataset]["paired_run_ids"])
        for dataset in datasets
    }
    global_subset = df[df["comparison_scope"] == "global"]
    notable = global_subset[
        (global_subset["metric"].isin(["average_set_size", "uncertainty_scaled", "certainty"]))
        & ~((global_subset["ci95_low"] <= 0.0) & (global_subset["ci95_high"] >= 0.0))
    ]

    lines = [
        "# Statistical CI Summary",
        "",
        "## Feasibility",
        "",
        "Confidence intervals are feasible from the available raw paired observations.",
        "The repository contains per-run result pickles under `RESULTS/results_<dataset>/results/result_<method>_<run_id>.pickle` and merged pickles under `RESULTS/results_<dataset>/merged_<method>_results.pickle`.",
        "The per-run pickles contain metric trajectories with keys including `coverage`, `inefficiency`, `certainty`, `uncertainty`, and feature `Index` values.",
        "The CI computation uses the per-run pickles directly, paired by `run_id`, rather than manuscript tables.",
        "",
        "Available paired CRFE/RFE runs in `RESULTS`:",
        "",
    ]
    for dataset, count in run_counts.items():
        lines.append(f"- `{dataset}`: {count} paired run(s)")

    lines.extend(
        [
            "",
            "Caveat: `musk_v1` and `parkinson` have fewer than 20 paired runs in the available `RESULTS` directory, so their dataset-level CIs use the available paired runs only.",
            "",
            "## Methods",
            "",
            f"- Bootstrap resamples: {args.n_bootstraps}",
            f"- Bootstrap random seed: {args.random_seed}",
            "- Global comparison: paired units are datasets, matching the global Wilcoxon structure.",
            "- Dataset-level comparison: paired units are random seeds, matching the heatmap Wilcoxon structure.",
            "- Percentage rows use the same selected-cardinality window as `create_boxplots_2.py`: `{k_center - 1, k_center, k_center + 1}` after bounds checking.",
            "- `average_set_size` is the raw `inefficiency` metric. `uncertainty_scaled` is the same metric min-max scaled from 1 to the dataset's number of classes, matching the existing plotting/statistical scripts.",
            "- Difference signs: `RFE - CRFE` for average set size and scaled uncertainty; `CRFE - RFE` for certainty and coverage.",
            "",
            "## Outputs",
            "",
            f"- CSV table: `{args.output_csv}`",
            f"- Markdown summary: `{args.output_md}`",
            "",
            "## Main Global Pattern",
            "",
        ]
    )

    if notable.empty:
        lines.append("No global non-coverage CI among average set size, scaled uncertainty, and certainty excluded zero.")
    else:
        for _, row in notable.iterrows():
            lines.append(
                "- {metric} at {subset}: mean difference {mean:+.6f}, 95% CI [{low:+.6f}, {high:+.6f}], p={p:.4g}".format(
                    metric=row["metric"],
                    subset=row["subset_percentage_or_k"],
                    mean=row["mean_difference"],
                    low=row["ci95_low"],
                    high=row["ci95_high"],
                    p=row["wilcoxon_p_value"],
                )
            )

    lines.extend(
        [
            "",
            "## Suggested Manuscript Addition",
            "",
            "We additionally report 95% bootstrap confidence intervals for the paired score differences, computed using the same paired units as the Wilcoxon tests. Because the full CI table is large, place it in the Supplementary Material and mention the main global pattern in the Results section.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.n_bootstraps < 1:
        raise ValueError("--n-bootstraps must be positive")

    datasets = args.datasets if args.datasets is not None else detect_datasets(args.results_root)
    if not datasets:
        raise ValueError(f"No paired CRFE/RFE datasets found under {args.results_root}")

    n_classes_map = parse_n_classes_map(args.n_classes_map)
    payloads = load_dataset_payloads(args.results_root, args.data_root, datasets, n_classes_map)
    rng = np.random.default_rng(args.random_seed)

    rows: list[dict] = []
    dataset_diffs: dict[tuple[str, float, str], np.ndarray] = {}

    for dataset in datasets:
        payload = payloads[dataset]
        for percentage in args.percentages:
            subset_label = format_percentage(percentage)
            for metric in METRICS:
                diffs = dataset_percentage_differences(
                    payload["runs_crfe"],
                    payload["runs_rfe"],
                    metric,
                    payload["n_classes"],
                    percentage,
                )
                rows.append(
                    make_row(
                        "dataset",
                        dataset,
                        metric,
                        subset_label,
                        diffs,
                        args.n_bootstraps,
                        rng,
                    )
                )
                dataset_diffs[(dataset, percentage, metric.output_name)] = diffs

    for percentage in args.percentages:
        subset_label = format_percentage(percentage)
        for metric in METRICS:
            global_diffs = []
            for dataset in datasets:
                diffs = dataset_diffs[(dataset, percentage, metric.output_name)]
                global_diffs.append(float(np.mean(diffs)))
            rows.append(
                make_row(
                    "global",
                    "global",
                    metric,
                    subset_label,
                    np.asarray(global_diffs, dtype=float),
                    args.n_bootstraps,
                    rng,
                )
            )

    if args.focused_dataset in payloads:
        payload = payloads[args.focused_dataset]
        for k in args.focused_k:
            for metric in METRICS:
                diffs = dataset_absolute_k_differences(
                    payload["runs_crfe"],
                    payload["runs_rfe"],
                    metric,
                    payload["n_classes"],
                    k,
                )
                rows.append(
                    make_row(
                        "focused_imvigor210",
                        args.focused_dataset,
                        metric,
                        f"k={k}",
                        diffs,
                        args.n_bootstraps,
                        rng,
                    )
                )

    out_df = pd.DataFrame(rows)
    requested_cols = [
        "comparison_scope",
        "dataset",
        "metric",
        "subset_percentage_or_k",
        "n_pairs",
        "mean_difference",
        "ci95_low",
        "ci95_high",
        "wilcoxon_p_value",
        "interpretation",
    ]
    extra_cols = ["median_difference"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df[requested_cols + extra_cols].to_csv(args.output_csv, index=False)
    write_summary(args.output_md, rows, list(datasets), payloads, args)

    print(f"Wrote {len(out_df)} CI rows to {args.output_csv}")
    print(f"Wrote summary to {args.output_md}")


if __name__ == "__main__":
    main()
