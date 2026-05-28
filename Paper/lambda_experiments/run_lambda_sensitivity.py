"""Run OVA lambda sensitivity checks on the miRNA multiclass datasets.

The experiment is intentionally self-contained in ``lambda-experiments`` so the
main library remains unchanged.  It reads ``DATA/mirna_nb_01`` and
``DATA/mirna_nb_02``, varies the OVA lambda in the CRFE multiclass score, and
reports the conformal prediction scores requested for the reviewer response:

- inefficiency: average conformal prediction set size;
- certainty: project-compatible rate of correct singleton prediction sets.

Feature-set stability is estimated with the Nogueira stability estimator across
the repeated seeds for each dataset/lambda/top-k combination.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RESULTS_DIR = BASE_DIR / "results"

DEFAULT_DATASETS = ("mirna_nb_01", "mirna_nb_02")
DEFAULT_LAMBDAS = (0.1, 0.25, 0.5, 0.75, 1.0)
DEFAULT_SEEDS = (7, 17, 29, 41, 53)
DEFAULT_TOP_K = (25, 50, 100)
DEFAULT_PRIMARY_TOP_K = 50
DEFAULT_ALPHA = 0.10
BASELINE_LAMBDA = 0.5
DATASET_DISPLAY_NAMES = {
    "mirna_nb_01": "Nb-1",
    "mirna_nb_02": "Nb-2",
}


def log(message: str, enabled: bool = True) -> None:
    if enabled:
        print(message, flush=True)


def display_dataset_name(dataset: str) -> str:
    return DATASET_DISPLAY_NAMES.get(dataset, dataset)


@dataclass(frozen=True)
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    classes: list


@dataclass(frozen=True)
class Split:
    X_train: np.ndarray
    X_cal: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_cal: np.ndarray
    y_test: np.ndarray


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ensure_dirs(results_dir: Path) -> dict[str, Path]:
    dirs = {
        "tables": results_dir / "tables",
        "plots": results_dir / "plots",
        "serialized": results_dir / "serialized",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), std


def make_estimator(seed: int) -> LinearSVC:
    return LinearSVC(
        tol=1e-4,
        loss="squared_hinge",
        max_iter=50000,
        dual=False,
        random_state=seed,
    )


def load_dataset(dataset_name: str) -> Dataset:
    data_dir = PROJECT_ROOT / "DATA" / dataset_name
    X_path = data_dir / "X.csv"
    y_path = data_dir / "Y.csv"
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing X.csv/Y.csv under {data_dir}")

    X = pd.read_csv(X_path).to_numpy(dtype=np.float64)
    raw_y = pd.read_csv(y_path).to_numpy().ravel()
    classes, y = np.unique(raw_y, return_inverse=True)
    return Dataset(name=dataset_name, X=X, y=y.astype(int), classes=classes.tolist())


def split_and_scale(dataset: Dataset, seed: int, test_size: float = 0.15, cal_size: float = 0.50) -> Split:
    X_temp, X_test, y_temp, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        stratify=dataset.y,
        shuffle=True,
        random_state=seed,
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp,
        y_temp,
        test_size=cal_size,
        stratify=y_temp,
        shuffle=True,
        random_state=seed + 1,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_cal = scaler.transform(X_cal)
    X_test = scaler.transform(X_test)
    return Split(X_train, X_cal, X_test, y_train.astype(int), y_cal.astype(int), y_test.astype(int))


def lambda_prime(lambda_value: float, n_classes: int) -> float:
    return (1.0 - lambda_value) / (n_classes - 1) if n_classes > 1 else 0.0


def compute_beta_multiclass(w: np.ndarray, y: np.ndarray, X: np.ndarray, lambda_value: float) -> np.ndarray:
    """Vectorized CRFE multiclass beta score.

    This is the same OVA lambda structure used by the CRFE code:
    beta_j -= lambda * w[y_i, j] * x_ij - lambda_p * sum_{k != y_i} w[k, j] * x_ij.
    Larger beta values are eliminated first.
    """

    lambda_p = lambda_prime(lambda_value, w.shape[0])
    w_sum = np.sum(w, axis=0)
    w_y = w[y]
    lambda_terms = lambda_value * w_y * X
    rest_terms = lambda_p * (w_sum[None, :] - w_y) * X
    return -np.sum(lambda_terms - rest_terms, axis=0)


def fit_ova_weights(estimator: LinearSVC, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = clone(estimator)
    model.fit(X, y)
    return np.asarray(model.coef_, dtype=float), np.asarray(model.intercept_, dtype=float)


def select_feature_sets(
    estimator: LinearSVC,
    X_train: np.ndarray,
    y_train: np.ndarray,
    lambda_value: float,
    top_k_values: list[int],
    ranking_mode: str,
    progress_label: str,
    verbose: bool = False,
) -> dict[int, np.ndarray]:
    """Return selected feature sets for all requested top-k values."""

    n_features = X_train.shape[1]
    top_k_set = set(top_k_values)

    if ranking_mode == "one-shot":
        w, _ = fit_ova_weights(estimator, X_train, y_train)
        beta = compute_beta_multiclass(w, y_train, X_train, lambda_value)
        best_to_worst = np.argsort(beta)
        log(
            (
                f"{progress_label}: one-shot beta summary "
                f"min={float(np.min(beta)):.6g}, mean={float(np.mean(beta)):.6g}, "
                f"max={float(np.max(beta)):.6g}"
            ),
            verbose,
        )
        return {top_k: np.sort(best_to_worst[:top_k]).astype(int) for top_k in top_k_values}

    if ranking_mode != "recursive":
        raise ValueError(f"Unknown ranking_mode={ranking_mode!r}")

    min_top_k = min(top_k_values)
    current_features = np.arange(n_features, dtype=int)
    selected: dict[int, np.ndarray] = {}
    if n_features in top_k_set:
        selected[n_features] = current_features.copy()

    while current_features.size > min_top_k:
        X_current = X_train[:, current_features]
        w, _ = fit_ova_weights(estimator, X_current, y_train)
        beta = compute_beta_multiclass(w, y_train, X_current, lambda_value)
        delete_pos = int(np.argmax(beta))
        current_features = np.delete(current_features, delete_pos)

        if current_features.size in top_k_set:
            selected[current_features.size] = current_features.copy()

        if current_features.size % 50 == 0 or current_features.size in top_k_set:
            log(f"{progress_label}: remaining_features={current_features.size}", True)

    missing = sorted(top_k_set - set(selected))
    if missing:
        raise RuntimeError(f"Did not capture requested top-k values: {missing}")
    return selected


def compute_ncm_multiclass(
    X: np.ndarray,
    y_or_candidates: np.ndarray,
    w: np.ndarray,
    bias: np.ndarray,
    lambda_value: float,
) -> np.ndarray:
    """Compute OVA multiclass nonconformity scores for true or candidate labels."""

    lambda_p = lambda_prime(lambda_value, w.shape[0])
    scores = X @ w.T + bias

    if y_or_candidates.ndim == 1 and y_or_candidates.shape[0] == X.shape[0]:
        y = y_or_candidates.astype(int)
        true_scores = scores[np.arange(X.shape[0]), y]
        return -lambda_value * true_scores + lambda_p * (np.sum(scores, axis=1) - true_scores)

    score_sum = np.sum(scores, axis=1, keepdims=True)
    return -lambda_value * scores + lambda_p * (score_sum - scores)


def conformal_scores(
    estimator: LinearSVC,
    split: Split,
    selected_features: np.ndarray,
    lambda_value: float,
    alpha: float,
    verbose: bool = False,
    progress_label: str = "",
) -> dict[str, float]:
    X_train = split.X_train[:, selected_features]
    X_cal = split.X_cal[:, selected_features]
    X_test = split.X_test[:, selected_features]

    w, bias = fit_ova_weights(estimator, X_train, split.y_train)
    n_classes = w.shape[0]
    candidate_labels = np.arange(n_classes, dtype=int)

    ncm_cal = compute_ncm_multiclass(X_cal, split.y_cal, w, bias, lambda_value)
    ncm_test = compute_ncm_multiclass(X_test, candidate_labels, w, bias, lambda_value)

    p_values = np.empty_like(ncm_test, dtype=float)
    for row_idx in range(ncm_test.shape[0]):
        for class_idx in range(ncm_test.shape[1]):
            p_values[row_idx, class_idx] = (np.sum(ncm_cal >= ncm_test[row_idx, class_idx]) + 1.0) / (
                len(ncm_cal) + 1.0
            )

    prediction_sets = p_values >= alpha
    set_sizes = np.sum(prediction_sets, axis=1)
    covered = prediction_sets[np.arange(split.y_test.shape[0]), split.y_test]
    point_predictions = np.argmax(p_values, axis=1)

    if verbose:
        set_size_counts = {
            int(size): int(count)
            for size, count in zip(*np.unique(set_sizes.astype(int), return_counts=True))
        }
        label = f"{progress_label}: " if progress_label else ""
        log(
            (
                f"{label}p-value summary "
                f"max_mean={float(np.mean(np.max(p_values, axis=1))):.4f}, "
                f"sum_mean={float(np.mean(np.sum(p_values, axis=1))):.4f}, "
                f"set_size_counts={set_size_counts}"
            ),
            True,
        )

    certainty = np.mean((set_sizes == 1) & covered)
    uncertainty = np.mean((set_sizes == n_classes) & covered)
    mistrust = np.mean(set_sizes == 0)

    return {
        "coverage": float(np.mean(covered)),
        "inefficiency": float(np.mean(set_sizes)),
        "certainty": float(certainty),
        "singleton_rate": float(np.mean(set_sizes == 1)),
        "uncertainty": float(uncertainty),
        "mistrust": float(mistrust),
        "accuracy_from_pvalues": float(accuracy_score(split.y_test, point_predictions)),
        "balanced_accuracy_from_pvalues": float(balanced_accuracy_score(split.y_test, point_predictions)),
        "macro_f1_from_pvalues": float(f1_score(split.y_test, point_predictions, average="macro")),
        "mean_max_p_value": float(np.mean(np.max(p_values, axis=1))),
        "mean_sum_p_values": float(np.mean(np.sum(p_values, axis=1))),
    }


def nogueira_stability(binary_matrix: np.ndarray) -> float:
    """Nogueira et al. stability estimator for equal-cardinality selections."""

    Z = np.asarray(binary_matrix, dtype=float)
    if Z.ndim != 2:
        raise ValueError("Nogueira stability expects a 2D binary matrix")
    n_runs, n_features = Z.shape
    if n_runs < 2:
        return float("nan")

    hat_p_f = np.mean(Z, axis=0)
    k_bar = np.sum(hat_p_f)
    denom = (k_bar / n_features) * (1.0 - k_bar / n_features)
    if denom == 0:
        return float("nan")

    return float(1.0 - (n_runs / (n_runs - 1.0)) * np.mean(hat_p_f * (1.0 - hat_p_f)) / denom)


def selected_to_binary(selected_features: list[np.ndarray], n_features: int) -> np.ndarray:
    matrix = np.zeros((len(selected_features), n_features), dtype=int)
    for row_idx, selected in enumerate(selected_features):
        matrix[row_idx, np.asarray(selected, dtype=int)] = 1
    return matrix


def aggregate_metrics(raw_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metric_names = [
        "coverage",
        "inefficiency",
        "certainty",
        "singleton_rate",
        "uncertainty",
        "mistrust",
        "accuracy_from_pvalues",
        "balanced_accuracy_from_pvalues",
        "macro_f1_from_pvalues",
        "mean_max_p_value",
        "mean_sum_p_values",
    ]

    for row in raw_rows:
        key = (str(row["dataset"]), float(row["lambda"]), int(row["top_k"]))
        for metric in metric_names:
            grouped[key][metric].append(float(row[metric]))

    rows: list[dict] = []
    for dataset, lambda_value, top_k in sorted(grouped):
        out = {"dataset": dataset, "lambda": lambda_value, "top_k": top_k}
        for metric in metric_names:
            mean, std = mean_std(grouped[(dataset, lambda_value, top_k)][metric])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        rows.append(out)
    return rows


def compute_stability_rows(selection_rows: list[dict], n_features_by_dataset: dict[str, int]) -> list[dict]:
    grouped: dict[tuple[str, float, int], list[np.ndarray]] = defaultdict(list)
    for row in selection_rows:
        key = (str(row["dataset"]), float(row["lambda"]), int(row["top_k"]))
        selected = np.array([int(item) for item in str(row["selected_features"]).split(";") if item != ""], dtype=int)
        grouped[key].append(selected)

    rows = []
    for dataset, lambda_value, top_k in sorted(grouped):
        selections = grouped[(dataset, lambda_value, top_k)]
        binary = selected_to_binary(selections, n_features_by_dataset[dataset])
        rows.append(
            {
                "dataset": dataset,
                "lambda": lambda_value,
                "top_k": top_k,
                "n_runs": len(selections),
                "n_features": n_features_by_dataset[dataset],
                "nogueira_stability": nogueira_stability(binary),
            }
        )
    return rows


def add_baseline_deltas(summary_rows: list[dict], stability_rows: list[dict]) -> list[dict]:
    stability_lookup = {
        (row["dataset"], float(row["lambda"]), int(row["top_k"])): float(row["nogueira_stability"])
        for row in stability_rows
    }
    baseline_lookup = {
        (row["dataset"], int(row["top_k"])): row
        for row in summary_rows
        if float(row["lambda"]) == BASELINE_LAMBDA
    }

    rows: list[dict] = []
    for row in summary_rows:
        dataset = row["dataset"]
        lambda_value = float(row["lambda"])
        top_k = int(row["top_k"])
        baseline = baseline_lookup[(dataset, top_k)]
        rows.append(
            {
                "dataset": dataset,
                "lambda": lambda_value,
                "top_k": top_k,
                "inefficiency_mean": row["inefficiency_mean"],
                "inefficiency_std": row["inefficiency_std"],
                "certainty_mean": row["certainty_mean"],
                "certainty_std": row["certainty_std"],
                "nogueira_stability": stability_lookup[(dataset, lambda_value, top_k)],
                "delta_inefficiency_vs_lambda_0_5": float(row["inefficiency_mean"])
                - float(baseline["inefficiency_mean"]),
                "delta_certainty_vs_lambda_0_5": float(row["certainty_mean"]) - float(baseline["certainty_mean"]),
            }
        )
    return rows


def make_plots(results_dir: Path, response_rows: list[dict], primary_top_k: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        (results_dir / "plots" / "plot_warning.txt").write_text(
            f"Plots were not generated because matplotlib could not be imported: {exc}\n",
            encoding="utf-8",
        )
        return

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    rows = [row for row in response_rows if int(row["top_k"]) == primary_top_k]
    datasets = sorted({row["dataset"] for row in rows})

    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(3.45, 2.05 * len(datasets)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets):
        ds_rows = sorted([row for row in rows if row["dataset"] == dataset], key=lambda row: float(row["lambda"]))
        lambdas = [float(row["lambda"]) for row in ds_rows]
        ax.errorbar(
            lambdas,
            [float(row["inefficiency_mean"]) for row in ds_rows],
            yerr=[float(row["inefficiency_std"]) for row in ds_rows],
            marker="o",
            markersize=4,
            linewidth=1.1,
            elinewidth=0.8,
            capsize=2.5,
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            label="Inefficiency",
        )
        ax.errorbar(
            lambdas,
            [float(row["certainty_mean"]) for row in ds_rows],
            yerr=[float(row["certainty_std"]) for row in ds_rows],
            marker="s",
            markersize=4,
            linewidth=1.1,
            elinewidth=0.8,
            capsize=2.5,
            color="0.45",
            markerfacecolor="0.45",
            markeredgecolor="0.45",
            label="Certainty",
        )
        ax.axvline(BASELINE_LAMBDA, color="0.2", linestyle="--", linewidth=0.8)
        ax.set_title(display_dataset_name(dataset), loc="left", fontweight="bold", pad=2)
        ax.set_ylabel("Score")
        ax.grid(axis="y", color="0.88", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, handlelength=1.6, borderpad=0.2, labelspacing=0.3)
    axes[-1].set_xlabel(r"$\lambda$")
    fig.savefig(results_dir / "plots" / "conformal_scores_by_lambda.png", dpi=300)
    fig.savefig(results_dir / "plots" / "conformal_scores_by_lambda.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.45, 2.35), constrained_layout=True)
    colors = ["black", "0.45"]
    markers = ["o", "s"]
    for dataset in datasets:
        ds_rows = sorted([row for row in rows if row["dataset"] == dataset], key=lambda row: float(row["lambda"]))
        color = colors[datasets.index(dataset) % len(colors)]
        ax.plot(
            [float(row["lambda"]) for row in ds_rows],
            [float(row["nogueira_stability"]) for row in ds_rows],
            marker=markers[datasets.index(dataset) % len(markers)],
            markersize=4,
            linewidth=1.1,
            color=color,
            markerfacecolor="white" if color == "black" else color,
            markeredgecolor=color,
            label=display_dataset_name(dataset),
        )
    ax.axvline(BASELINE_LAMBDA, color="0.2", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Nogueira stability")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, handlelength=1.6, borderpad=0.2, labelspacing=0.3)
    fig.savefig(results_dir / "plots" / "nogueira_stability_by_lambda.png", dpi=300)
    fig.savefig(results_dir / "plots" / "nogueira_stability_by_lambda.pdf")
    plt.close(fig)


def write_summary(
    path: Path,
    config: dict,
    response_rows: list[dict],
    primary_top_k: int,
    elapsed_seconds: float,
) -> None:
    rows = [row for row in response_rows if int(row["top_k"]) == primary_top_k]
    lines = [
        "Lambda sensitivity experiment for the OVA multiclass CRFE score",
        "",
        f"Datasets: {', '.join(config['datasets'])}.",
        f"Lambda grid: {', '.join(str(v) for v in config['lambda_grid'])}.",
        f"Seeds: {', '.join(str(v) for v in config['seeds'])}.",
        f"Top-k feature-set sizes: {', '.join(str(v) for v in config['top_k'])}.",
        f"Primary top-k for this summary: {primary_top_k}.",
        f"Ranking mode: {config['ranking_mode']}.",
        f"Verbose diagnostics enabled: {config.get('verbose', False)}.",
        f"Alpha for split conformal prediction: {config['alpha']}.",
        "",
        "Metric definitions:",
        "Inefficiency is the average conformal prediction set size.",
        "Certainty follows the project convention: fraction of test samples with a singleton prediction set that contains the true label.",
        "Nogueira stability is computed from the binary selected-feature matrix across the five seeds for each dataset/lambda/top-k.",
        "",
        "Primary results:",
    ]

    for row in rows:
        lines.append(
            (
                f"{row['dataset']} lambda={float(row['lambda']):.2f}: "
                f"inefficiency={float(row['inefficiency_mean']):.4f}+/-{float(row['inefficiency_std']):.4f}, "
                f"certainty={float(row['certainty_mean']):.4f}+/-{float(row['certainty_std']):.4f}, "
                f"Nogueira={float(row['nogueira_stability']):.4f}, "
                f"delta ineff. vs 0.5={float(row['delta_inefficiency_vs_lambda_0_5']):+.4f}, "
                f"delta cert. vs 0.5={float(row['delta_certainty_vs_lambda_0_5']):+.4f}"
            )
        )

    lines.extend(
        [
            "",
            f"Elapsed wall time: {elapsed_seconds:.1f} seconds.",
            "",
            "Generated outputs:",
            "results/tables/raw_conformal_metrics.csv",
            "results/tables/summary_by_dataset_lambda_k.csv",
            "results/tables/nogueira_stability_by_dataset_lambda_k.csv",
            "results/tables/reviewer_response_table.csv",
            "results/tables/reviewer_response_primary_table.csv",
            "results/tables/selected_features_by_seed_lambda_k.csv",
            "results/plots/conformal_scores_by_lambda.png",
            "results/plots/nogueira_stability_by_lambda.png",
            "results/serialized/lambda_sensitivity_results.pkl",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if BASELINE_LAMBDA not in args.lambdas:
        raise ValueError("The lambda grid must include the baseline lambda=0.5")
    if args.primary_top_k not in args.top_k:
        raise ValueError("primary_top_k must be included in top_k")
    if any(top_k <= 0 for top_k in args.top_k):
        raise ValueError("top-k values must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    start_time = time.time()
    dirs = ensure_dirs(RESULTS_DIR)

    datasets = [load_dataset(name) for name in args.datasets]
    n_features_by_dataset = {dataset.name: dataset.X.shape[1] for dataset in datasets}

    for dataset in datasets:
        too_large = [top_k for top_k in args.top_k if top_k > dataset.X.shape[1]]
        if too_large:
            raise ValueError(f"top-k values {too_large} exceed n_features={dataset.X.shape[1]} for {dataset.name}")

    raw_rows: list[dict] = []
    selection_rows: list[dict] = []

    total_jobs = len(datasets) * len(args.seeds) * len(args.lambdas)
    job_idx = 0

    for dataset in datasets:
        log(f"Loaded {dataset.name}: X={dataset.X.shape}, classes={dataset.classes}")
        for seed in args.seeds:
            split = split_and_scale(dataset, seed)
            if args.verbose:
                class_counts = {
                    "train": np.bincount(split.y_train).astype(int).tolist(),
                    "cal": np.bincount(split.y_cal).astype(int).tolist(),
                    "test": np.bincount(split.y_test).astype(int).tolist(),
                }
                log(
                    (
                        f"{dataset.name} seed={seed}: split shapes "
                        f"train={split.X_train.shape}, cal={split.X_cal.shape}, test={split.X_test.shape}, "
                        f"class_counts={class_counts}"
                    )
                )
            for lambda_value in args.lambdas:
                job_idx += 1
                label = f"[{job_idx}/{total_jobs}] {dataset.name} seed={seed} lambda={lambda_value}"
                log(f"{label}: selecting features")

                estimator = make_estimator(seed)
                selected_by_k = select_feature_sets(
                    estimator,
                    split.X_train,
                    split.y_train,
                    lambda_value,
                    sorted(args.top_k),
                    args.ranking_mode,
                    label,
                    args.verbose,
                )

                for top_k in sorted(args.top_k):
                    selected = np.asarray(selected_by_k[top_k], dtype=int)
                    metrics = conformal_scores(
                        estimator,
                        split,
                        selected,
                        lambda_value,
                        args.alpha,
                        verbose=args.verbose,
                        progress_label=f"{label} top_k={top_k}",
                    )
                    selected_text = ";".join(str(idx) for idx in sorted(selected.tolist()))
                    if args.verbose:
                        log(
                            (
                                f"{label}: top_k={top_k} selected_features_head="
                                f"{sorted(selected.tolist())[:10]}"
                            )
                        )
                    row = {
                        "dataset": dataset.name,
                        "seed": seed,
                        "lambda": lambda_value,
                        "lambda_p": lambda_prime(lambda_value, len(dataset.classes)),
                        "top_k": top_k,
                        "selected_features": selected_text,
                        **metrics,
                    }
                    raw_rows.append(row)
                    selection_rows.append(
                        {
                            "dataset": dataset.name,
                            "seed": seed,
                            "lambda": lambda_value,
                            "top_k": top_k,
                            "selected_features": selected_text,
                        }
                    )
                    log(
                        (
                            f"{label}: top_k={top_k} "
                            f"inefficiency={metrics['inefficiency']:.4f} "
                            f"certainty={metrics['certainty']:.4f}"
                        )
                    )

    summary_rows = aggregate_metrics(raw_rows)
    stability_rows = compute_stability_rows(selection_rows, n_features_by_dataset)
    response_rows = add_baseline_deltas(summary_rows, stability_rows)
    primary_response_rows = [row for row in response_rows if int(row["top_k"]) == args.primary_top_k]

    raw_fields = [
        "dataset",
        "seed",
        "lambda",
        "lambda_p",
        "top_k",
        "selected_features",
        "coverage",
        "inefficiency",
        "certainty",
        "singleton_rate",
        "uncertainty",
        "mistrust",
        "accuracy_from_pvalues",
        "balanced_accuracy_from_pvalues",
        "macro_f1_from_pvalues",
        "mean_max_p_value",
        "mean_sum_p_values",
    ]
    summary_fields = ["dataset", "lambda", "top_k"] + [
        f"{metric}_{suffix}"
        for metric in [
            "coverage",
            "inefficiency",
            "certainty",
            "singleton_rate",
            "uncertainty",
            "mistrust",
            "accuracy_from_pvalues",
            "balanced_accuracy_from_pvalues",
            "macro_f1_from_pvalues",
            "mean_max_p_value",
            "mean_sum_p_values",
        ]
        for suffix in ["mean", "std"]
    ]
    stability_fields = ["dataset", "lambda", "top_k", "n_runs", "n_features", "nogueira_stability"]
    response_fields = [
        "dataset",
        "lambda",
        "top_k",
        "inefficiency_mean",
        "inefficiency_std",
        "certainty_mean",
        "certainty_std",
        "nogueira_stability",
        "delta_inefficiency_vs_lambda_0_5",
        "delta_certainty_vs_lambda_0_5",
    ]
    selection_fields = ["dataset", "seed", "lambda", "top_k", "selected_features"]

    write_csv(dirs["tables"] / "raw_conformal_metrics.csv", raw_rows, raw_fields)
    write_csv(dirs["tables"] / "summary_by_dataset_lambda_k.csv", summary_rows, summary_fields)
    write_csv(dirs["tables"] / "nogueira_stability_by_dataset_lambda_k.csv", stability_rows, stability_fields)
    write_csv(dirs["tables"] / "reviewer_response_table.csv", response_rows, response_fields)
    write_csv(dirs["tables"] / "reviewer_response_primary_table.csv", primary_response_rows, response_fields)
    write_csv(dirs["tables"] / "selected_features_by_seed_lambda_k.csv", selection_rows, selection_fields)

    config = {
        "datasets": args.datasets,
        "lambda_grid": args.lambdas,
        "baseline_lambda": BASELINE_LAMBDA,
        "seeds": args.seeds,
        "top_k": args.top_k,
        "primary_top_k": args.primary_top_k,
        "alpha": args.alpha,
        "ranking_mode": args.ranking_mode,
        "verbose": args.verbose,
        "test_size": 0.15,
        "calibration_size_after_test_split": 0.50,
        "estimator": "LinearSVC(tol=1e-4, loss='squared_hinge', max_iter=50000, dual=False)",
        "metric_notes": {
            "inefficiency": "Average conformal prediction set size.",
            "certainty": "Project-compatible correct singleton prediction-set rate.",
            "singleton_rate": "All singleton prediction sets, regardless of correctness.",
            "stability": "Nogueira estimator across seeds for each dataset/lambda/top_k.",
        },
    }
    (RESULTS_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    with (dirs["serialized"] / "lambda_sensitivity_results.pkl").open("wb") as handle:
        pickle.dump(
            {
                "config": config,
                "raw_metrics": raw_rows,
                "summary": summary_rows,
                "stability": stability_rows,
                "response_table": response_rows,
                "primary_response_table": primary_response_rows,
                "selected_features": selection_rows,
            },
            handle,
            protocol=4,
        )

    np.savez_compressed(
        dirs["serialized"] / "lambda_sensitivity_results.npz",
        raw_metrics=np.array(raw_rows, dtype=object),
        summary=np.array(summary_rows, dtype=object),
        stability=np.array(stability_rows, dtype=object),
        response_table=np.array(response_rows, dtype=object),
        primary_response_table=np.array(primary_response_rows, dtype=object),
        selected_features=np.array(selection_rows, dtype=object),
    )

    make_plots(RESULTS_DIR, response_rows, args.primary_top_k)
    elapsed = time.time() - start_time
    write_summary(RESULTS_DIR / "summary.txt", config, response_rows, args.primary_top_k, elapsed)
    log(f"Finished lambda sensitivity experiment. Results written to: {RESULTS_DIR}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=parse_str_list,
        default=list(DEFAULT_DATASETS),
        help="Comma-separated dataset folders under DATA.",
    )
    parser.add_argument(
        "--lambdas",
        type=parse_float_list,
        default=list(DEFAULT_LAMBDAS),
        help="Comma-separated lambda grid. Must include 0.5.",
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=list(DEFAULT_SEEDS),
        help="Comma-separated random seeds; defaults to five multiruns.",
    )
    parser.add_argument(
        "--top-k",
        type=parse_int_list,
        default=list(DEFAULT_TOP_K),
        help="Comma-separated selected feature-set sizes to evaluate.",
    )
    parser.add_argument(
        "--primary-top-k",
        type=int,
        default=DEFAULT_PRIMARY_TOP_K,
        help="Top-k subset used for summary text and plots.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Split conformal significance level.",
    )
    parser.add_argument(
        "--ranking-mode",
        choices=["one-shot", "recursive"],
        default="one-shot",
        help=(
            "one-shot computes the lambda-weighted OVA feature score once per run; "
            "recursive repeats CRFE elimination until the smallest top-k is reached."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print split sizes, class counts, beta summaries, p-value summaries, and selected feature previews.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
