#!/usr/bin/env python3
"""Generate fairness diagnostics for runtime comparison outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _feature_monotonicity_violations(summary: pd.DataFrame) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    for (method, n_classes, n_samples), block in summary.groupby(
        ["method", "n_classes", "n_samples"]
    ):
        chunk = block.sort_values("n_features")
        means = chunk["mean_s"].to_numpy(dtype=float)
        features = chunk["n_features"].to_numpy(dtype=int)

        if means.size <= 1:
            continue

        if np.any(np.diff(means) < 0):
            violations.append(
                {
                    "check": "feature_monotonicity",
                    "method": method,
                    "n_classes": int(n_classes),
                    "n_samples": int(n_samples),
                    "n_features": -1,
                    "detail": "; ".join(
                        f"{f}:{m:.6f}" for f, m in zip(features.tolist(), means.tolist())
                    ),
                }
            )
    return violations


def _sample_scaling_violations(
    summary: pd.DataFrame,
    tolerance_ratio: float,
) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    for (method, n_classes, n_features), block in summary.groupby(
        ["method", "n_classes", "n_features"]
    ):
        chunk = block.sort_values("n_samples")
        means = chunk["mean_s"].to_numpy(dtype=float)
        samples = chunk["n_samples"].to_numpy(dtype=int)

        if means.size <= 1:
            continue

        for idx in range(means.size - 1):
            current = means[idx]
            nxt = means[idx + 1]
            if current <= 0:
                continue

            # Flag if runtime decreases more than tolerance when samples increase.
            if nxt < current * (1.0 - tolerance_ratio):
                violations.append(
                    {
                        "check": "sample_scaling_drop",
                        "method": method,
                        "n_classes": int(n_classes),
                        "n_samples": int(samples[idx + 1]),
                        "n_features": int(n_features),
                        "detail": (
                            f"{samples[idx]}->{samples[idx + 1]}: "
                            f"{current:.6f}s->{nxt:.6f}s"
                        ),
                    }
                )
    return violations


def _high_cv_flags(summary: pd.DataFrame, cv_threshold: float) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    subset = summary[summary["cv_s"].fillna(0.0) > cv_threshold]
    for _, row in subset.iterrows():
        out.append(
            {
                "check": "high_cv",
                "method": row["method"],
                "n_classes": int(row["n_classes"]),
                "n_samples": int(row["n_samples"]),
                "n_features": int(row["n_features"]),
                "detail": f"cv_s={float(row['cv_s']):.4f}",
            }
        )
    return out


def _convergence_flags(summary: pd.DataFrame) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    subset = summary[summary["convergence_warning_count"].fillna(0.0) > 0]
    for _, row in subset.iterrows():
        out.append(
            {
                "check": "convergence_warning",
                "method": row["method"],
                "n_classes": int(row["n_classes"]),
                "n_samples": int(row["n_samples"]),
                "n_features": int(row["n_features"]),
                "detail": (
                    f"count={int(row['convergence_warning_count'])}, "
                    f"warn_rate={float(row.get('warn_rate', np.nan)):.4f}"
                ),
            }
        )
    return out


def generate_fairness_report(
    summary: pd.DataFrame,
    raw: pd.DataFrame,
    output_dir: Path,
    tag: str,
    sample_drop_tolerance: float = 0.15,
    cv_threshold: float = 0.20,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    valid = summary.dropna(subset=["mean_s"]).copy()

    flags: list[dict[str, object]] = []
    flags.extend(_feature_monotonicity_violations(valid))
    flags.extend(_sample_scaling_violations(valid, tolerance_ratio=sample_drop_tolerance))
    flags.extend(_high_cv_flags(valid, cv_threshold=cv_threshold))
    flags.extend(_convergence_flags(valid))

    flags_df = pd.DataFrame(
        flags,
        columns=["check", "method", "n_classes", "n_samples", "n_features", "detail"],
    )

    failed_runs = int((raw["status"] == "failed").sum())
    successful_runs = int((raw["status"] == "ok").sum())
    total_runs = int(len(raw))

    report = {
        "total_runs": total_runs,
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "rows_in_summary": int(len(valid)),
        "checks": {
            "feature_monotonicity_violations": int(
                (flags_df["check"] == "feature_monotonicity").sum() if not flags_df.empty else 0
            ),
            "sample_scaling_drop_violations": int(
                (flags_df["check"] == "sample_scaling_drop").sum() if not flags_df.empty else 0
            ),
            "high_cv_flags": int((flags_df["check"] == "high_cv").sum() if not flags_df.empty else 0),
            "convergence_warning_flags": int(
                (flags_df["check"] == "convergence_warning").sum() if not flags_df.empty else 0
            ),
        },
        "thresholds": {
            "sample_drop_tolerance": sample_drop_tolerance,
            "cv_threshold": cv_threshold,
        },
    }

    json_generic = output_dir / "runtime_fairness_report.json"
    json_tagged = output_dir / f"runtime_fairness_report_{tag}.json"
    flags_generic = output_dir / "runtime_fairness_flags.csv"
    flags_tagged = output_dir / f"runtime_fairness_flags_{tag}.csv"

    json_generic.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    json_tagged.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if flags_df.empty:
        flags_df = pd.DataFrame(
            [{"check": "none", "method": "", "n_classes": "", "n_samples": "", "n_features": "", "detail": ""}]
        )
    flags_df.to_csv(flags_generic, index=False)
    flags_df.to_csv(flags_tagged, index=False)

    return {
        "json_generic": str(json_generic),
        "json_tagged": str(json_tagged),
        "flags_generic": str(flags_generic),
        "flags_tagged": str(flags_tagged),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fairness diagnostics from runtime benchmark CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="RESULTS/runtimes/runtime_vs_features_results_2x2.csv",
    )
    parser.add_argument(
        "--raw-csv",
        type=str,
        default="RESULTS/runtimes/runtime_vs_features_raw.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="RESULTS/runtimes",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="manual",
    )
    parser.add_argument(
        "--sample-drop-tolerance",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--cv-threshold",
        type=float,
        default=0.20,
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    raw_path = Path(args.raw_csv)
    output_dir = Path(args.output_dir)

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_path}")

    summary = pd.read_csv(summary_path)
    raw = pd.read_csv(raw_path)

    paths = generate_fairness_report(
        summary=summary,
        raw=raw,
        output_dir=output_dir,
        tag=args.tag,
        sample_drop_tolerance=args.sample_drop_tolerance,
        cv_threshold=args.cv_threshold,
    )

    print(f"Saved fairness report: {paths['json_generic']}")
    print(f"Saved fairness flags : {paths['flags_generic']}")


if __name__ == "__main__":
    main()
