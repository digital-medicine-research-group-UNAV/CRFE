#!/usr/bin/env python3
"""Re-plot runtime comparison figures from a saved CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

METHOD_LABELS = {
    "crfe": "CRFE",
    "rfe": "RFE",
    "smfs": "SMFS",
    "lasso": "LASSO",
    "elasticnet": "Elastic Net",
    "boruta": "BORUTA",
}


def parse_methods(raw: str | None, available: Sequence[str]) -> list[str]:
    if raw is None or raw.strip() == "":
        return list(available)

    methods: list[str] = []
    for chunk in raw.split(","):
        token = chunk.strip().lower()
        if not token:
            continue
        if token not in available:
            raise ValueError(
                f"Method '{token}' not present in CSV. Available: {', '.join(available)}"
            )
        if token not in methods:
            methods.append(token)

    if not methods:
        raise ValueError("--methods resolved to an empty selection")
    return methods


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


def method_tag(methods: Sequence[str]) -> str:
    return "_".join(methods)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate runtime comparison plots from CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="RESULTS/runtimes/runtime_vs_features_results_2x2.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="RESULTS/runtimes",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated subset of methods to display",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "mean_s" not in df.columns:
        raise ValueError("Input CSV must include at least columns: mean_s, std_s")

    valid = df.dropna(subset=["mean_s"]).copy()
    if valid.empty:
        raise ValueError("Input CSV contains no successful runs to plot")

    available_methods = sorted(valid["method"].unique().tolist())
    methods = parse_methods(args.methods, available_methods)
    valid = valid[valid["method"].isin(methods)].copy()

    configure_plot_style()

    color_cycle = plt.get_cmap("tab10")
    color_map = {method: color_cycle(i) for i, method in enumerate(methods)}
    marker_cycle = ["o", "s", "^", "D", "v", "P"]
    marker_map = {method: marker_cycle[i % len(marker_cycle)] for i, method in enumerate(methods)}
    linestyle_cycle = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (4, 1, 1, 1))]
    linestyle_map = {
        method: linestyle_cycle[i % len(linestyle_cycle)]
        for i, method in enumerate(methods)
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
                (valid["n_classes"] == n_classes)
                & (valid["n_samples"] == n_samples)
            ].copy()

            for method in methods:
                chunk = block[block["method"] == method].sort_values("n_features")
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

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = method_tag(methods)

    output_pdf = output_dir / "runtime_comparison_grid.pdf"
    output_png = output_dir / "runtime_comparison_grid.png"
    tagged_pdf = output_dir / f"runtime_comparison_grid_{tag}.pdf"
    tagged_png = output_dir / f"runtime_comparison_grid_{tag}.png"

    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(tagged_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(tagged_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {output_pdf.resolve()}")
    print(f"Saved tagged figure to {tagged_pdf.resolve()}")


if __name__ == "__main__":
    main()
