#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare frozen inner-validation SHAP group structure between the original
catalog and a higher-magnitude-threshold feature catalog.

Expected inputs are the output directories from
05_inner_validation_shap_discovery.py.  The comparison is performed
on group_consensus_inner_validation.csv and is restricted to the 11
representation-specific catalog-information groups.

Outputs
-------
- all_group_rank_comparison.csv
- horizon_rank_and_top3_summary.csv
- key_group_stability.csv
- latest100_pattern.csv
- feature_threshold_shap_sensitivity_report.md
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def expand_path(path):
    return str(Path(path).expanduser().resolve())


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def parse_int_list(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_str_list(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def load_consensus(directory, label, expected_groups, expected_models, expected_folds):
    path = os.path.join(directory, "group_consensus_inner_validation.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {
        "horizon_month",
        "group_name",
        "mean_group_importance",
        "rank",
        "n_models",
        "n_folds",
        "n_runs",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{label} consensus is missing columns: {sorted(missing)}")

    df = df.copy()
    df["horizon_month"] = df["horizon_month"].astype(int)
    df["rank"] = df["rank"].astype(int)
    df["group_name"] = df["group_name"].astype(str)
    df["catalog_version"] = label

    audit_rows = []
    for horizon, g in df.groupby("horizon_month"):
        audit_rows.append(
            {
                "catalog_version": label,
                "horizon_month": int(horizon),
                "n_groups": int(g["group_name"].nunique()),
                "n_models_min": int(g["n_models"].min()),
                "n_folds_min": int(g["n_folds"].min()),
                "n_runs_min": int(g["n_runs"].min()),
                "group_count_ok": int(g["group_name"].nunique()) == expected_groups,
                "model_coverage_ok": int(g["n_models"].min()) == expected_models,
                "fold_coverage_ok": int(g["n_folds"].min()) == expected_folds,
                "run_coverage_ok": int(g["n_runs"].min()) == expected_models * expected_folds,
            }
        )

    audit = pd.DataFrame(audit_rows)
    if not bool(
        audit[
            [
                "group_count_ok",
                "model_coverage_ok",
                "fold_coverage_ok",
                "run_coverage_ok",
            ]
        ].all(axis=None)
    ):
        raise ValueError(
            f"Incomplete consensus coverage in {label}:\n{audit.to_string(index=False)}"
        )

    return df, audit, path


def ordered_top_groups(df, top_k):
    return (
        df.sort_values(["rank", "group_name"])
        .head(top_k)["group_name"]
        .astype(str)
        .tolist()
    )


def safe_rank_correlation(rank_a, rank_b):
    a = pd.Series(rank_a, dtype=float)
    b = pd.Series(rank_b, dtype=float)
    if len(a) < 2 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(a.corr(b, method="pearson"))


def format_markdown(df, columns=None):
    show = df if columns is None else df[columns]
    try:
        return show.to_markdown(index=False)
    except Exception:
        return "```\n" + show.to_string(index=False) + "\n```"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--sensitivity-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--expected-groups", type=int, default=11)
    parser.add_argument("--expected-models", type=int, default=3)
    parser.add_argument("--expected-folds", type=int, default=3)
    parser.add_argument(
        "--key-groups",
        default=(
            "monthly_lag/magnitude_extreme,"
            "annual/magnitude_extreme,"
            "last100/magnitude_distribution,"
            "last100/temporal_clustering"
        ),
    )
    parser.add_argument("--monthly-extreme-group", default="monthly_lag/magnitude_extreme")
    parser.add_argument("--annual-extreme-group", default="annual/magnitude_extreme")
    parser.add_argument("--latest100-prefix", default="last100/")
    parser.add_argument(
        "--latest100-magnitude-group",
        default="last100/magnitude_distribution",
    )
    parser.add_argument(
        "--latest100-sequence-group",
        default="last100/temporal_clustering",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    reference_dir = expand_path(args.reference_dir)
    sensitivity_dir = expand_path(args.sensitivity_dir)
    out_dir = ensure_dir(expand_path(args.out_dir))
    horizons = parse_int_list(args.horizons)
    key_groups = parse_str_list(args.key_groups)

    reference, audit_ref, reference_path = load_consensus(
        reference_dir,
        label="reference",
        expected_groups=args.expected_groups,
        expected_models=args.expected_models,
        expected_folds=args.expected_folds,
    )
    sensitivity, audit_sens, sensitivity_path = load_consensus(
        sensitivity_dir,
        label="high_threshold",
        expected_groups=args.expected_groups,
        expected_models=args.expected_models,
        expected_folds=args.expected_folds,
    )

    reference = reference[reference["horizon_month"].isin(horizons)].copy()
    sensitivity = sensitivity[sensitivity["horizon_month"].isin(horizons)].copy()

    ref_groups = set(reference["group_name"])
    sens_groups = set(sensitivity["group_name"])
    if ref_groups != sens_groups:
        only_ref = sorted(ref_groups.difference(sens_groups))
        only_sens = sorted(sens_groups.difference(ref_groups))
        raise ValueError(
            "Group definitions differ between catalogs. "
            f"Only reference={only_ref}; only high-threshold={only_sens}"
        )

    merged = reference.merge(
        sensitivity,
        on=["horizon_month", "group_name"],
        how="inner",
        suffixes=("_reference", "_high_threshold"),
        validate="one_to_one",
    )
    merged["rank_change_high_minus_reference"] = (
        merged["rank_high_threshold"] - merged["rank_reference"]
    )
    merged["importance_change_high_minus_reference"] = (
        merged["mean_group_importance_high_threshold"]
        - merged["mean_group_importance_reference"]
    )
    merged["in_top3_reference"] = merged["rank_reference"] <= args.top_k
    merged["in_top3_high_threshold"] = merged["rank_high_threshold"] <= args.top_k

    all_group_path = os.path.join(out_dir, "all_group_rank_comparison.csv")
    merged.sort_values(["horizon_month", "rank_reference", "group_name"]).to_csv(
        all_group_path, index=False
    )

    summary_rows = []
    for horizon in horizons:
        ref_h = reference[reference["horizon_month"].eq(horizon)].copy()
        sens_h = sensitivity[sensitivity["horizon_month"].eq(horizon)].copy()
        pair_h = merged[merged["horizon_month"].eq(horizon)].copy()
        if len(pair_h) != args.expected_groups:
            raise ValueError(
                f"H={horizon}: matched groups={len(pair_h)}, expected={args.expected_groups}"
            )

        top_ref = ordered_top_groups(ref_h, args.top_k)
        top_sens = ordered_top_groups(sens_h, args.top_k)
        set_ref = set(top_ref)
        set_sens = set(top_sens)
        overlap = len(set_ref.intersection(set_sens))
        union = len(set_ref.union(set_sens))
        same_rank = sum(a == b for a, b in zip(top_ref, top_sens))

        summary_rows.append(
            {
                "horizon_month": horizon,
                "n_groups": int(len(pair_h)),
                "spearman_rank_correlation": safe_rank_correlation(
                    pair_h["rank_reference"], pair_h["rank_high_threshold"]
                ),
                "mean_absolute_rank_change": float(
                    pair_h["rank_change_high_minus_reference"].abs().mean()
                ),
                "max_absolute_rank_change": int(
                    pair_h["rank_change_high_minus_reference"].abs().max()
                ),
                "top3_overlap_count": overlap,
                "top3_jaccard": float(overlap / union) if union else np.nan,
                "top3_exact_set_match": set_ref == set_sens,
                "top3_same_rank_count": same_rank,
                "top3_reference": " | ".join(top_ref),
                "top3_high_threshold": " | ".join(top_sens),
            }
        )

    horizon_summary = pd.DataFrame(summary_rows)
    horizon_summary_path = os.path.join(out_dir, "horizon_rank_and_top3_summary.csv")
    horizon_summary.to_csv(horizon_summary_path, index=False)

    key_rows = []
    for horizon in horizons:
        for group_name in key_groups:
            hit = merged[
                merged["horizon_month"].eq(horizon)
                & merged["group_name"].eq(group_name)
            ]
            if hit.empty:
                raise ValueError(f"Missing key group H={horizon}: {group_name}")
            row = hit.iloc[0]
            key_rows.append(
                {
                    "horizon_month": horizon,
                    "group_name": group_name,
                    "rank_reference": int(row["rank_reference"]),
                    "rank_high_threshold": int(row["rank_high_threshold"]),
                    "rank_change_high_minus_reference": int(
                        row["rank_change_high_minus_reference"]
                    ),
                    "importance_reference": float(
                        row["mean_group_importance_reference"]
                    ),
                    "importance_high_threshold": float(
                        row["mean_group_importance_high_threshold"]
                    ),
                    "importance_change_high_minus_reference": float(
                        row["importance_change_high_minus_reference"]
                    ),
                    "top3_in_both": bool(
                        row["in_top3_reference"] and row["in_top3_high_threshold"]
                    ),
                }
            )

    key_stability = pd.DataFrame(key_rows)
    key_stability_path = os.path.join(out_dir, "key_group_stability.csv")
    key_stability.to_csv(key_stability_path, index=False)

    latest_rows = []
    for horizon in horizons:
        ref_h = reference[
            reference["horizon_month"].eq(horizon)
            & reference["group_name"].str.startswith(args.latest100_prefix)
        ].sort_values(["rank", "group_name"])
        sens_h = sensitivity[
            sensitivity["horizon_month"].eq(horizon)
            & sensitivity["group_name"].str.startswith(args.latest100_prefix)
        ].sort_values(["rank", "group_name"])
        if ref_h.empty or sens_h.empty:
            raise ValueError(f"No Latest-100 groups found for H={horizon}")

        ref_top = ref_h.iloc[0]
        sens_top = sens_h.iloc[0]
        latest_rows.append(
            {
                "horizon_month": horizon,
                "top_latest100_reference": str(ref_top["group_name"]),
                "top_latest100_rank_reference": int(ref_top["rank"]),
                "top_latest100_importance_reference": float(
                    ref_top["mean_group_importance"]
                ),
                "top_latest100_high_threshold": str(sens_top["group_name"]),
                "top_latest100_rank_high_threshold": int(sens_top["rank"]),
                "top_latest100_importance_high_threshold": float(
                    sens_top["mean_group_importance"]
                ),
                "latest100_top_group_match": str(ref_top["group_name"])
                == str(sens_top["group_name"]),
            }
        )

    latest100 = pd.DataFrame(latest_rows)
    latest100["expected_reference_pattern"] = np.where(
        latest100["horizon_month"].eq(1),
        latest100["top_latest100_reference"].eq(args.latest100_magnitude_group),
        latest100["top_latest100_reference"].eq(args.latest100_sequence_group),
    )
    latest100["expected_high_threshold_pattern"] = np.where(
        latest100["horizon_month"].eq(1),
        latest100["top_latest100_high_threshold"].eq(
            args.latest100_magnitude_group
        ),
        latest100["top_latest100_high_threshold"].eq(
            args.latest100_sequence_group
        ),
    )
    latest100_path = os.path.join(out_dir, "latest100_pattern.csv")
    latest100.to_csv(latest100_path, index=False)

    monthly = key_stability[
        key_stability["group_name"].eq(args.monthly_extreme_group)
    ]
    annual = key_stability[
        key_stability["group_name"].eq(args.annual_extreme_group)
    ]
    backbone_stable = bool(
        len(monthly) == len(horizons)
        and len(annual) == len(horizons)
        and monthly["rank_high_threshold"].le(2).all()
        and annual["rank_high_threshold"].le(2).all()
    )
    latest_pattern_preserved = bool(
        latest100["expected_high_threshold_pattern"].all()
    )

    audit = pd.concat([audit_ref, audit_sens], ignore_index=True)
    audit_path = os.path.join(out_dir, "consensus_coverage_audit.csv")
    audit.to_csv(audit_path, index=False)

    manifest = {
        "reference_dir": reference_dir,
        "sensitivity_dir": sensitivity_dir,
        "reference_consensus": reference_path,
        "sensitivity_consensus": sensitivity_path,
        "out_dir": out_dir,
        "horizons": horizons,
        "top_k": args.top_k,
        "expected_groups": args.expected_groups,
        "expected_models": args.expected_models,
        "expected_folds": args.expected_folds,
        "monthly_annual_backbone_stable_at_ranks_1_to_2": backbone_stable,
        "latest100_expected_horizon_pattern_preserved": latest_pattern_preserved,
    }
    manifest_path = os.path.join(out_dir, "comparison_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    report_path = os.path.join(
        out_dir, "feature_threshold_shap_sensitivity_report.md"
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Higher-magnitude-threshold SHAP sensitivity\n\n")
        handle.write(
            "This report compares the 11-group inner-validation SHAP consensus "
            "from the original catalog with the same workflow applied after "
            "raising the magnitude threshold used for feature construction.\n\n"
        )
        handle.write("## Coverage audit\n\n")
        handle.write(format_markdown(audit))
        handle.write("\n\n## Rank correlation and Top-3 overlap\n\n")
        handle.write(
            format_markdown(
                horizon_summary,
                [
                    "horizon_month",
                    "spearman_rank_correlation",
                    "mean_absolute_rank_change",
                    "top3_overlap_count",
                    "top3_jaccard",
                    "top3_same_rank_count",
                    "top3_reference",
                    "top3_high_threshold",
                ],
            )
        )
        handle.write("\n\n## Key-group stability\n\n")
        handle.write(
            format_markdown(
                key_stability,
                [
                    "horizon_month",
                    "group_name",
                    "rank_reference",
                    "rank_high_threshold",
                    "rank_change_high_minus_reference",
                    "top3_in_both",
                ],
            )
        )
        handle.write("\n\n## Latest-100 horizon pattern\n\n")
        handle.write(format_markdown(latest100))
        handle.write("\n\n## Prespecified checks\n\n")
        handle.write(
            f"- Monthly and Annual magnitude-extreme groups remain within the "
            f"leading two groups at every horizon: **{backbone_stable}**\n"
        )
        handle.write(
            f"- H1 selects Latest-100 magnitude distribution and H3-H12 select "
            f"Latest-100 sequence timing in the high-threshold analysis: "
            f"**{latest_pattern_preserved}**\n"
        )
        handle.write(
            "\nThese checks assess structural sensitivity of the discovery-stage "
            "SHAP hierarchy. They do not constitute a completeness-corrected "
            "forecast evaluation.\n"
        )

    print("=" * 100, flush=True)
    print("FEATURE-THRESHOLD SHAP SENSITIVITY FINISHED", flush=True)
    print(horizon_summary.to_string(index=False), flush=True)
    print("\nBackbone stable:", backbone_stable, flush=True)
    print("Latest-100 expected pattern preserved:", latest_pattern_preserved, flush=True)
    print("Saved:", all_group_path, flush=True)
    print("Saved:", horizon_summary_path, flush=True)
    print("Saved:", key_stability_path, flush=True)
    print("Saved:", latest100_path, flush=True)
    print("Saved:", report_path, flush=True)


if __name__ == "__main__":
    main()
