#!/usr/bin/env python3
"""
Generate only CRFE vs RFE distribution plots from existing stopping results.

This utility does not run experiments. It reads previously saved paired
`result_crfe_*.pickle` / `result_rfe_*.pickle` files and regenerates:
    CRFE_vs_RFE_distribution_<dataset>.pdf
under RESULTS/stopping.
"""

from __future__ import annotations

import argparse
import pickle
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_FILE_RE = re.compile(r"^result_(?:crfe|rfe)_(\d+)\.pickle$")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOT_COLORS = {"CRFE": "#2E86AB", "RFE": "#A23B72"}
MIRNA_DATASETS = {"mirna_nb_01", "mirna_nb_02"}
MIRNA_TOP_K = 25
SYNTHETIC_TOP_K = 35
REFERENCE_INDICES = [
    9,
    39,
    43,
    58,
    68,
    72,
    79,
    81,
    91,
    155,
    158,
    169,
    176,
    187,
    197,
    224,
    264,
    267,
    268,
    291,
    292,
    295,
    299,
    310,
    321,
    335,
    340,
    366,
    371,
    381,
    383,
    393,
    407,
    411,
    412,
    431,
    473,
    483,
    492,
    499,
]


def discover_datasets(stopping_root: Path) -> list[str]:
    datasets: list[str] = []
    if not stopping_root.is_dir():
        return datasets

    for path in sorted(stopping_root.iterdir()):
        if not path.is_dir():
            continue
        if not path.name.startswith("results_"):
            continue
        datasets.append(path.name.removeprefix("results_"))
    return datasets


def discover_run_ids(run_results_dir: Path) -> list[int]:
    run_ids: set[int] = set()
    if not run_results_dir.is_dir():
        return []

    for path in run_results_dir.iterdir():
        if not path.is_file():
            continue
        match = RESULT_FILE_RE.match(path.name)
        if match:
            run_ids.add(int(match.group(1)))
    return sorted(run_ids)


def load_result(path: Path) -> dict[str, list[object]]:
    with path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise ValueError(f"Unexpected result format in {path}")
    return payload[0]


def collect_final_indices(run_results_dir: Path, run_ids: list[int]) -> dict[str, list[int]]:
    final_indices: dict[str, list[int]] = {"CRFE": [], "RFE": []}

    for run_id in run_ids:
        crfe_path = run_results_dir / f"result_crfe_{run_id}.pickle"
        rfe_path = run_results_dir / f"result_rfe_{run_id}.pickle"
        if not (crfe_path.is_file() and rfe_path.is_file()):
            continue

        crfe_result = load_result(crfe_path)
        rfe_result = load_result(rfe_path)
        crfe_steps = crfe_result.get("Index", [])
        rfe_steps = rfe_result.get("Index", [])

        if not crfe_steps or not rfe_steps:
            continue

        final_indices["CRFE"].extend(np.asarray(crfe_steps[-1], dtype=int).tolist())
        final_indices["RFE"].extend(np.asarray(rfe_steps[-1], dtype=int).tolist())

    return final_indices


def plot_final_feature_frequency(
    final_indices: dict[str, list[int]],
    dataset: str,
    output_dir: Path,
    top_k: int,
) -> tuple[Path, dict[str, list[int]]]:
    crfe_counter = Counter(final_indices.get("CRFE", []))
    rfe_counter = Counter(final_indices.get("RFE", []))
    if not crfe_counter and not rfe_counter:
        raise ValueError("No final selected feature indices were found for plotting.")

    # Keep synthetic size aligned with the original paper-style figure so LaTeX
    # inclusion does not shrink text excessively.
    if dataset == "synthetic":
        fig_width = 10.0
    else:
        fig_width = max(10.0, min(22.0, 0.4 * top_k))
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 8))
    top_features_by_method: dict[str, list[int]] = {}
    # Match original typography from main_stopping.py
    x_label_size = 12

    for ax, counter, method in (
        (axes[0], crfe_counter, "CRFE"),
        (axes[1], rfe_counter, "RFE"),
    ):
        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_k]
        features = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]
        top_features_by_method[method] = features
        padded_features = features + [""] * (top_k - len(features))
        padded_counts = counts + [0] * (top_k - len(counts))
        x = np.arange(top_k)

        ax.bar(
            x,
            padded_counts,
            alpha=0.7,
            color=PLOT_COLORS[method],
            edgecolor="black",
            linewidth=0.5,
            width=0.8,
        )
        ax.set_title(
            f"({ 'A' if method == 'CRFE' else 'B' }) {method} - Feature Selection Frequency Distribution",
            fontweight="bold",
            fontsize=17,
            pad=20,
        )
        ax.set_xlabel("Feature Index", fontweight="bold", fontsize=15)
        ax.set_ylabel("Selection Frequency", fontweight="bold", fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(padded_features, rotation=45, ha="right", va="top", fontsize=x_label_size)
        ax.tick_params(axis="x", pad=2)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        if dataset == "synthetic":
            ax.set_ylim(0, 50)

    fig.tight_layout()
    outpath = output_dir / f"CRFE_vs_RFE_distribution_{dataset}.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath, top_features_by_method


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["figure.titlesize"] = 18


def print_reference_overlap(
    dataset: str,
    top_features_by_method: dict[str, list[int]],
    top_k_used: int,
) -> None:
    reference_set = set(REFERENCE_INDICES)
    print(f"[info] {dataset} reference indices ({len(reference_set)} total): {REFERENCE_INDICES}")
    for method in ("CRFE", "RFE"):
        selected = top_features_by_method.get(method, [])
        selected_set = set(selected)
        overlap = sorted(selected_set & reference_set)
        hits_top_k = len(overlap)
        top_k_pct = (100.0 * hits_top_k / top_k_used) if top_k_used > 0 else 0.0
        recall_pct = (100.0 * len(overlap) / len(reference_set)) if reference_set else 0.0
        precision_pct = (100.0 * len(overlap) / len(selected_set)) if selected_set else 0.0
        print(
            f"[info] {dataset} {method}: "
            f"hits_top{top_k_used}={hits_top_k}/{top_k_used} "
            f"({top_k_pct:.2f}%) "
            f"match={len(overlap)}/{len(reference_set)} "
            f"recall={recall_pct:.2f}% "
            f"precision={precision_pct:.2f}%"
        )
        print(f"[info] {dataset} {method} matched indices: {overlap}")


def generate_distribution_for_dataset(dataset: str, top_k_features: int) -> Path:
    stopping_root = PROJECT_ROOT / "RESULTS" / "stopping"
    run_results_dir = stopping_root / f"results_{dataset}" / "results"
    if not run_results_dir.is_dir():
        raise FileNotFoundError(f"Result directory not found for dataset '{dataset}': {run_results_dir}")

    run_ids = discover_run_ids(run_results_dir)
    if not run_ids:
        raise FileNotFoundError(
            f"No result pickles found for dataset '{dataset}' in {run_results_dir}"
        )

    final_indices = collect_final_indices(run_results_dir, run_ids)
    if not final_indices["CRFE"] or not final_indices["RFE"]:
        raise RuntimeError(
            f"No paired CRFE/RFE runs found for dataset '{dataset}' in {run_results_dir}"
        )

    if dataset in MIRNA_DATASETS:
        dataset_top_k = MIRNA_TOP_K
    elif dataset == "synthetic":
        dataset_top_k = SYNTHETIC_TOP_K
    else:
        dataset_top_k = top_k_features
    outpath, top_features_by_method = plot_final_feature_frequency(
        final_indices=final_indices,
        dataset=dataset,
        output_dir=stopping_root,
        top_k=dataset_top_k,
    )
    if dataset in MIRNA_DATASETS:
        print(f"[info] {dataset}: using fixed top_k={dataset_top_k}.")
        print_reference_overlap(dataset, top_features_by_method, dataset_top_k)
    return outpath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CRFE_vs_RFE_distribution_<dataset>.pdf from existing result pickles."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Dataset names (e.g. mirna_nb_01 mirna_nb_02 synthetic). "
            "If omitted, all datasets under RESULTS/stopping/results_* are used."
        ),
    )
    parser.add_argument(
        "--top_k_features",
        type=int,
        default=30,
        help="Top-K selected features per method in the distribution plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k_features <= 0:
        raise ValueError("--top_k_features must be > 0")

    stopping_root = PROJECT_ROOT / "RESULTS" / "stopping"
    datasets = args.datasets if args.datasets else discover_datasets(stopping_root)
    if not datasets:
        raise RuntimeError(
            f"No datasets found. Expected folders like results_<dataset> under {stopping_root}"
        )

    configure_plot_style()
    print(f"Stopping root: {stopping_root}")
    print(f"Datasets: {datasets}")

    generated: list[Path] = []
    for dataset in datasets:
        outpath = generate_distribution_for_dataset(dataset, args.top_k_features)
        generated.append(outpath)
        print(f"[ok] {dataset}: {outpath}")

    print(f"\nGenerated {len(generated)} distribution plot(s).")


if __name__ == "__main__":
    main()
