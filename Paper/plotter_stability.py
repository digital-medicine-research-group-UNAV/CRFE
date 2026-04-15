"""
Stability comparison plotting for local Linux pipeline outputs.
"""

from pathlib import Path
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")
RESULTS_PARENT = os.environ.get("RESULTS_PARENT", "RESULTS")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["crfe", "rfe", "lasso", "elasticnet", "boruta", "smfs"],
        help="Methods to include in the stability comparison plot",
    )
    return parser.parse_args()


def _detect_columns(df: pd.DataFrame) -> tuple[str, str | None, str | None]:
    col_map = {c.lower(): c for c in df.columns}
    stability_col = col_map.get("nogue") or col_map.get("stability") or col_map.get("stability_score")
    low_col = col_map.get("low_int") or col_map.get("ci_lower") or col_map.get("lower")
    high_col = col_map.get("high_int") or col_map.get("ci_upper") or col_map.get("upper")

    if stability_col is None:
        stability_col = df.columns[0]
    return stability_col, low_col, high_col


def main():
    args = parse_arguments()
    dataset = args.data_path
    methods = args.methods

    project_root = Path(__file__).resolve().parent
    results_root = project_root / RESULTS_PARENT / f"results_{dataset}"
    plots_dir = results_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    style_map = {
        "crfe": {"color": "black", "linestyle": "-", "alpha": 0.23},
        "rfe": {"color": "black", "linestyle": ":", "alpha": 0.16},
        "lasso": {"color": "tab:blue", "linestyle": "--", "alpha": 0.12},
        "elasticnet": {"color": "tab:orange", "linestyle": "-.", "alpha": 0.12},
        "boruta": {"color": "tab:green", "linestyle": (0, (3, 1, 1, 1)), "alpha": 0.12},
        "smfs": {"color": "tab:red", "linestyle": (0, (5, 1)), "alpha": 0.12},
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = 0

    for method in methods:
        csv_path = results_root / f"stability_{method}.csv"
        if not csv_path.is_file():
            print(f"Warning: stability file not found, skipping {method}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Warning: stability file is empty, skipping {method}: {csv_path}")
            continue

        stability_col, low_col, high_col = _detect_columns(df)
        x = np.arange(len(df), 0, -1)

        style = style_map.get(
            method,
            {"color": "black", "linestyle": "-", "alpha": 0.12},
        )

        label = f"{method.upper()} Stability"
        ax.plot(
            x,
            df[stability_col],
            linestyle=style["linestyle"],
            color=style["color"],
            markersize=5,
            linewidth=1.5,
            label=label,
        )

        if low_col in df.columns and high_col in df.columns:
            ax.fill_between(
                x,
                df[low_col],
                df[high_col],
                color=style["color"],
                alpha=style["alpha"],
            )
        plotted += 1

    if plotted == 0:
        raise FileNotFoundError(
            f"No stability CSV files found in {results_root}. "
            "Run submit_jobs.py successfully before plotting stability."
        )

    ax.set_xlabel("Num. of Selected Features", fontsize=18, fontweight="bold")
    ax.set_ylabel("Stability Score", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.legend(loc="best", fontsize=13, frameon=True, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.65)
    plt.tight_layout()

    plot_path = plots_dir / "stability_comparison.svg"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Stability comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
