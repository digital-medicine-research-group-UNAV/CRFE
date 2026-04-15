#!/usr/bin/env python3
"""
Sensitivity analysis for stopping criteria parameters eta and xi.

For each (eta, xi) pair in the grid, this script runs CRFE and RFE with
stopping activated using paired seeds/splits for fair comparison.
It reports mean final set size, mean uncertainty and mean certainty, and
renders a 6-panel heatmap figure (3 metrics x 2 methods).

In this sensitivity module, eta controls the absolute drawdown stop threshold,
while xi controls proximity quantiles inside the stopping proximity matrix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _crfe import CRFE
from _crfe_utils import READER
from rfe_module import Stepwise_RFE
from stopping_module_sensitivity import ParamParada, StoppingCriteria

plt.switch_backend("Agg")

METHODS = ("crfe", "rfe")
DEFAULT_ETA_VALUES = (0.0, 0.01, 0.05, 0.1, 0.15)
DEFAULT_XI_VALUES = (0.0, 0.05, 0.1, 0.5)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stopping criteria sensitivity analysis for eta and xi.",
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
        help="Number of paired runs per (eta, xi, method).",
    )
    parser.add_argument(
        "--start_run_id",
        type=int,
        default=0,
        help="First run id used to generate seeds/partitions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute all combinations even if raw CSV already contains them.",
    )
    parser.add_argument(
        "--eta_values",
        type=str,
        default=",".join(str(v) for v in DEFAULT_ETA_VALUES),
        help="Comma-separated eta values.",
    )
    parser.add_argument(
        "--xi_values",
        type=str,
        default=",".join(str(v) for v in DEFAULT_XI_VALUES),
        help="Comma-separated xi values (used as inter-class proximity quantile).",
    )
    return parser.parse_args()


def parse_float_grid(values: str) -> list[float]:
    grid = [float(v.strip()) for v in values.split(",") if v.strip()]
    if not grid:
        raise ValueError("Grid cannot be empty")
    return grid


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.25)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10


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


def split_seed(run_id: int) -> int:
    return int(np.random.default_rng(42 + run_id).integers(low=0, high=100000))


def estimator_seed(run_id: int) -> int:
    return int(np.random.default_rng(1000 + run_id).integers(low=0, high=100000))


def split_data_fairly(
    X: np.ndarray,
    Y: np.ndarray,
    run_id: int,
    test_size: float = 0.15,
    cal_size: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed = split_seed(run_id)

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    scaler.fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_cal), scaler.transform(X_test)


def create_estimator(run_id: int) -> LinearSVC:
    return LinearSVC(
        tol=1e-4,
        loss="squared_hinge",
        max_iter=14000,
        dual="auto",
        random_state=estimator_seed(run_id),
    )


def build_selector(method: str, run_id: int, stop_params: ParamParada) -> CRFE | Stepwise_RFE:
    common_args = {
        "estimator": create_estimator(run_id),
        "features_to_select": 1,
        "stopping_activated": True,
        "stopping_params": stop_params,
    }

    if method == "crfe":
        selector = CRFE(**common_args)
    elif method == "rfe":
        selector = Stepwise_RFE(**common_args)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Force local sensitivity-aware stopping implementation.
    selector.stopping_params = stop_params
    selector.new_stopping_criteria = StoppingCriteria(stop_params)
    return selector


def run_single_experiment(
    method: str,
    run_id: int,
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    stop_params: ParamParada,
) -> dict[str, list[Any]]:
    selector = build_selector(method, run_id, stop_params)
    selector.fit(
        X_tr.copy(),
        Y_tr.copy(),
        X_cal.copy(),
        Y_cal.copy(),
        X_test.copy(),
        Y_test.copy(),
    )
    return selector.results_dicc


def extract_final_metrics(result: dict[str, list[Any]]) -> dict[str, float] | None:
    index_steps = result.get("Index", [])
    if len(index_steps) == 0:
        return None

    final_indices = np.asarray(index_steps[-1], dtype=int)

    def _last(name: str) -> float:
        values = result.get(name, [])
        return float(values[-1]) if len(values) > 0 else np.nan

    return {
        "n_features": float(len(final_indices)),
        "inefficiency": _last("inefficiency"),
        "certainty": _last("certainty"),
    }


def ensure_output_dirs(dataset: str) -> tuple[Path, Path]:
    root = PROJECT_ROOT / "RESULTS" / "stopping_sensitivity"
    dataset_root = root / f"results_{dataset}"
    root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    return root, dataset_root


def _combo_key(method: str, run_id: int, eta: float, xi: float) -> tuple[str, int, str, str]:
    return (method, run_id, f"{eta:.6f}", f"{xi:.6f}")


def run_grid(
    X: np.ndarray,
    Y: np.ndarray,
    run_ids: list[int],
    eta_values: list[float],
    xi_values: list[float],
    raw_csv_path: Path,
    overwrite: bool,
) -> pd.DataFrame:
    if raw_csv_path.is_file() and not overwrite:
        raw_df = pd.read_csv(raw_csv_path)
        existing_keys = {
            _combo_key(str(row.method), int(row.run_id), float(row.eta), float(row.xi))
            for row in raw_df.itertuples(index=False)
        }
        rows = raw_df.to_dict(orient="records")
        print(f"Loaded existing rows from {raw_csv_path}: {len(rows)}")
    else:
        existing_keys = set()
        rows: list[dict[str, Any]] = []

    for run_id in run_ids:
        print(f"\n[Run {run_id}] Creating paired split once for all (eta, xi) combinations")
        X_tr, X_cal, X_test, Y_tr, Y_cal, Y_test = split_data_fairly(X, Y, run_id)
        X_tr, X_cal, X_test = standardize_split_data(X_tr, X_cal, X_test)

        for xi in xi_values:
            for eta in eta_values:
                params = ParamParada(eps=xi, eta=eta, verbose=False)

                for method in METHODS:
                    key = _combo_key(method, run_id, eta, xi)
                    if key in existing_keys:
                        continue

                    print(
                        f"[Run {run_id}] {method.upper()} eta={eta:.3f} xi={xi:.3f}"
                    )
                    result = run_single_experiment(
                        method,
                        run_id,
                        X_tr,
                        Y_tr,
                        X_cal,
                        Y_cal,
                        X_test,
                        Y_test,
                        params,
                    )
                    final_metrics = extract_final_metrics(result)
                    if final_metrics is None:
                        continue

                    row = {
                        "dataset": str(raw_csv_path.parent.name).replace("results_", "", 1),
                        "run_id": run_id,
                        "method": method,
                        "eta": eta,
                        "xi": xi,
                        "n_features": final_metrics["n_features"],
                        "inefficiency": final_metrics["inefficiency"],
                        "certainty": final_metrics["certainty"],
                    }
                    rows.append(row)
                    existing_keys.add(key)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows collected during sensitivity analysis.")

    df = df.drop_duplicates(subset=["run_id", "method", "eta", "xi"], keep="last")
    df = df.sort_values(["method", "xi", "eta", "run_id"]).reset_index(drop=True)
    return df


def summarize_grid(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw_df.groupby(["method", "eta", "xi"], as_index=False)
        .agg(
            mean_set_size=("n_features", "mean"),
            mean_uncertainty=("inefficiency", "mean"),
            mean_certainty=("certainty", "mean"),
            std_set_size=("n_features", "std"),
            std_uncertainty=("inefficiency", "std"),
            std_certainty=("certainty", "std"),
            n_runs=("run_id", "nunique"),
        )
        .sort_values(["method", "xi", "eta"])
        .reset_index(drop=True)
    )
    return summary


def _make_heatmap_matrix(
    summary_df: pd.DataFrame,
    method: str,
    metric_col: str,
    xi_values: list[float],
    eta_values: list[float],
) -> pd.DataFrame:
    subset = summary_df[summary_df["method"] == method]
    pivot = subset.pivot(index="xi", columns="eta", values=metric_col)
    pivot = pivot.reindex(index=xi_values, columns=eta_values)
    return pivot


def plot_heatmaps(
    summary_df: pd.DataFrame,
    dataset: str,
    eta_values: list[float],
    xi_values: list[float],
    n_runs: int,
    outpath: Path,
) -> None:
    metric_specs = [
        ("mean_set_size", "Average Set Size", "YlGnBu", ".2f"),
        ("mean_uncertainty", "Uncertainty Average", "magma", ".3f"),
        ("mean_certainty", "Certainty Average", "viridis", ".3f"),
    ]
    method_specs = [("crfe", "CRFE"), ("rfe", "RFE")]

    fig, axes = plt.subplots(3, 2, figsize=(15, 16), constrained_layout=False)

    for row_idx, (metric_col, metric_title, cmap, fmt) in enumerate(metric_specs):
        row_matrices: dict[str, pd.DataFrame] = {}
        row_values: list[np.ndarray] = []

        for method_key, _ in method_specs:
            matrix = _make_heatmap_matrix(
                summary_df,
                method_key,
                metric_col,
                xi_values=xi_values,
                eta_values=eta_values,
            )
            row_matrices[method_key] = matrix
            vals = matrix.to_numpy(dtype=float).ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                row_values.append(vals)

        if row_values:
            all_vals = np.concatenate(row_values)
            row_vmin = float(np.min(all_vals))
            row_vmax = float(np.max(all_vals))
            if np.isclose(row_vmin, row_vmax):
                delta = 1e-12 if row_vmin == 0 else abs(row_vmin) * 1e-12
                row_vmin -= delta
                row_vmax += delta
        else:
            row_vmin, row_vmax = 0.0, 1.0

        for col_idx, (method_key, method_label) in enumerate(method_specs):
            ax = axes[row_idx, col_idx]
            matrix = row_matrices[method_key]

            sns.heatmap(
                matrix,
                ax=ax,
                cmap=cmap,
                vmin=row_vmin,
                vmax=row_vmax,
                annot=True,
                fmt=fmt,
                linewidths=0.6,
                linecolor="white",
                cbar=True,
                square=True,
                annot_kws={"fontsize": 9},
            )

            ax.set_title(f"{method_label} - {metric_title}", fontweight="bold", pad=10)
            ax.set_xlabel(r"$\eta$", fontweight="bold")
            ax.set_ylabel(r"$\xi$", fontweight="bold")
            ax.set_xticklabels([f"{v:.2f}" for v in eta_values], rotation=0)
            ax.set_yticklabels([f"{v:.2f}" for v in xi_values], rotation=0)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_arguments()

    if args.n_runs <= 0:
        raise ValueError("--n_runs must be > 0")
    if args.start_run_id < 0:
        raise ValueError("--start_run_id must be >= 0")

    eta_values = parse_float_grid(args.eta_values)
    xi_values = parse_float_grid(args.xi_values)

    validate_dataset_path(args.data_path)
    configure_plot_style()

    output_root, dataset_root = ensure_output_dirs(args.data_path)
    raw_csv_path = dataset_root / f"sensitivity_raw_{args.data_path}.csv"
    summary_csv_path = dataset_root / f"sensitivity_summary_{args.data_path}.csv"
    heatmap_path = output_root / f"stopping_sensitivity_heatmaps_{args.data_path}.svg"

    run_ids = list(range(args.start_run_id, args.start_run_id + args.n_runs))

    reader = READER()
    X, Y, y_classes = reader.get_data(args.data_path)
    print(
        f"Dataset '{args.data_path}' loaded with shape={X.shape}, "
        f"classes={list(y_classes)}"
    )
    print(f"Eta grid: {eta_values}")
    print(f"Xi grid: {xi_values}")

    raw_df = run_grid(
        X=X,
        Y=Y,
        run_ids=run_ids,
        eta_values=eta_values,
        xi_values=xi_values,
        raw_csv_path=raw_csv_path,
        overwrite=args.overwrite,
    )
    raw_df.to_csv(raw_csv_path, index=False)

    summary_df = summarize_grid(raw_df)
    summary_df.to_csv(summary_csv_path, index=False)

    plot_heatmaps(
        summary_df=summary_df,
        dataset=args.data_path,
        eta_values=eta_values,
        xi_values=xi_values,
        n_runs=args.n_runs,
        outpath=heatmap_path,
    )

    print("\nSensitivity analysis completed")
    print(f"Raw rows: {len(raw_df)}")
    print(f"Summary rows: {len(summary_df)}")
    print(f"Saved raw CSV: {raw_csv_path}")
    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Saved heatmap SVG: {heatmap_path}")


if __name__ == "__main__":
    main()
