#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit and report the shallow-depth inner-validation SHAP sensitivity run.

The companion SLURM workflow first reconstructs the unchanged 282-column
representation using only feature-catalog events with known focal depths
0 <= d < 70 km, then repeats only the three discovery models' inner-validation
SHAP ranking.  This script performs the final invariant and ranking audits.

No final-test predictions, labels, or metrics are used for SHAP ranking.

Outputs
-------
- dataset_invariance_audit.csv
- discovery_configuration_audit.csv
- consensus_coverage_audit.csv
- all_group_rank_comparison.csv
- horizon_rank_and_top3_summary.csv
- key_group_stability.csv
- latest100_horizon_transition.csv
- depth_filtered_inner_shap_sensitivity_report.md
- depth_filtered_inner_shap_sensitivity_manifest.json
"""

import argparse
import gc
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_DISCOVERY_MODELS = {"ExtraTrees", "CatBoost", "XGBoost"}


def expand_path(path):
    return str(Path(path).expanduser().resolve())


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def parse_int_list(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def format_markdown(df, columns=None):
    show = df if columns is None else df[columns]
    try:
        return show.to_markdown(index=False)
    except Exception:
        return "```\n" + show.to_string(index=False) + "\n```"


def boolean_series(values):
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(True).astype(bool)
    normalized = values.fillna("true").astype(str).str.strip().str.lower()
    unknown = ~normalized.isin({"true", "false", "1", "0", "yes", "no"})
    if unknown.any():
        raise ValueError(
            "Cannot parse boolean manifest values: "
            f"{sorted(normalized[unknown].unique())}"
        )
    return normalized.isin({"true", "1", "yes"})


def dataset_path(directory, horizon):
    return os.path.join(directory, f"china_raw282_dataset_horizon_{horizon}m.pkl")


def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def comparable_meta(meta):
    required = [
        "t0",
        "split",
        "horizon_month",
        "patch_order",
        "region",
        "future_start_exclusive",
        "future_end_inclusive",
        "future_max_mag",
        "y_class",
        "y_m5",
        "fold1_role",
        "fold2_role",
        "fold3_role",
    ]
    missing = [column for column in required if column not in meta.columns]
    if missing:
        raise ValueError(f"Dataset metadata is missing invariant columns: {missing}")
    out = meta[required].copy().reset_index(drop=True)
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].astype(str)
    return out


def load_feature_metadata(directory):
    path = os.path.join(directory, "feature_metadata_282_relative_lag.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    meta = pd.read_csv(path).sort_values("feature_idx").reset_index(drop=True)
    if len(meta) != 282:
        raise ValueError(f"Expected 282 feature metadata rows, found {len(meta)}")
    return meta, path


def audit_datasets(reference_dir, shallow_dir, horizons, depth_min, depth_max):
    feature_meta, feature_meta_path = load_feature_metadata(shallow_dir)
    deep_indices = feature_meta.loc[
        feature_meta["depth_bin"].astype(str).eq("depth_ge70"), "feature_idx"
    ].astype(int).to_numpy()
    if len(deep_indices) != 12:
        raise ValueError(
            "Expected 12 structurally zero depth_ge70 monthly features, "
            f"found {len(deep_indices)}"
        )

    rows = []
    for horizon in horizons:
        reference_path = dataset_path(reference_dir, horizon)
        shallow_path = dataset_path(shallow_dir, horizon)
        reference = load_pickle(reference_path)
        shallow = load_pickle(shallow_path)

        X_ref = np.asarray(reference["X"])
        X_shallow = np.asarray(shallow["X"])
        y_ref = np.asarray(reference.get("y_m5", np.asarray(reference["y_class"]) > 0))
        y_shallow = np.asarray(shallow.get("y_m5", np.asarray(shallow["y_class"]) > 0))
        meta_ref = comparable_meta(reference["meta"])
        meta_shallow = comparable_meta(shallow["meta"])

        shape_match = X_ref.shape == X_shallow.shape
        dimension_ok = (
            X_shallow.ndim == 2
            and X_shallow.shape[1] == 282
            and int(shallow.get("feature_dim", X_shallow.shape[1])) == 282
        )
        labels_identical = y_ref.shape == y_shallow.shape and np.array_equal(
            y_ref, y_shallow
        )
        metadata_identical = meta_ref.equals(meta_shallow)
        finite_features = bool(np.isfinite(X_shallow).all())
        deep_bin_zero = bool(
            deep_indices.size
            and np.all(X_shallow[:, deep_indices] == 0.0)
        )
        if shape_match:
            changed_row_fraction = float(
                np.mean(np.any(X_ref != X_shallow, axis=1))
            )
        else:
            changed_row_fraction = np.nan

        payload_depth_min = shallow.get("feature_depth_min_km")
        payload_depth_max = shallow.get("feature_depth_max_km_exclusive")
        depth_filter_recorded = (
            payload_depth_min is not None
            and payload_depth_max is not None
            and np.isclose(float(payload_depth_min), float(depth_min))
            and np.isclose(float(payload_depth_max), float(depth_max))
        )

        row = {
            "horizon_month": int(horizon),
            "n_rows_reference": int(X_ref.shape[0]),
            "n_rows_shallow": int(X_shallow.shape[0]),
            "n_features_shallow": int(X_shallow.shape[1]),
            "shape_match": bool(shape_match),
            "dimension_282_ok": bool(dimension_ok),
            "labels_identical": bool(labels_identical),
            "sample_order_cells_folds_metadata_identical": bool(metadata_identical),
            "all_shallow_features_finite": finite_features,
            "depth_ge70_feature_columns_structurally_zero": deep_bin_zero,
            "depth_filter_recorded_in_payload": bool(depth_filter_recorded),
            "fraction_rows_with_any_feature_change": changed_row_fraction,
        }
        rows.append(row)

        required_ok = [
            shape_match,
            dimension_ok,
            labels_identical,
            metadata_identical,
            finite_features,
            deep_bin_zero,
            depth_filter_recorded,
        ]
        if not all(required_ok):
            raise ValueError(
                f"Dataset invariant audit failed for H={horizon}: {row}"
            )

        del reference, shallow, X_ref, X_shallow, y_ref, y_shallow
        del meta_ref, meta_shallow
        gc.collect()

    return pd.DataFrame(rows), feature_meta_path


def load_shap_manifest(directory, label):
    path = os.path.join(directory, "inner_validation_shap_manifest.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {
        "horizon_month",
        "fold",
        "model_name",
        "config_name",
        "n_inner_train",
        "n_inner_valid_full",
        "n_inner_valid_shap",
        "used_final_test_rows",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{label} SHAP manifest is missing: {sorted(missing)}")
    df = df.copy()
    df["horizon_month"] = df["horizon_month"].astype(int)
    df["fold"] = df["fold"].astype(int)
    return df, path


def audit_discovery_configuration(reference_dir, shallow_dir, horizons):
    reference, reference_path = load_shap_manifest(reference_dir, "reference")
    shallow, shallow_path = load_shap_manifest(shallow_dir, "shallow_depth")
    reference = reference[reference["horizon_month"].isin(horizons)].copy()
    shallow = shallow[shallow["horizon_month"].isin(horizons)].copy()

    keys = ["horizon_month", "fold", "model_name"]
    columns = keys + [
        "config_name",
        "n_inner_train",
        "n_inner_valid_full",
        "n_inner_valid_shap",
        "used_final_test_rows",
    ]
    merged = reference[columns].merge(
        shallow[columns],
        on=keys,
        how="outer",
        suffixes=("_reference", "_shallow_depth"),
        indicator=True,
        validate="one_to_one",
    )
    merged["config_name_match"] = (
        merged["config_name_reference"]
        == merged["config_name_shallow_depth"]
    )
    merged["fold_sizes_match"] = (
        (merged["n_inner_train_reference"] == merged["n_inner_train_shallow_depth"])
        & (
            merged["n_inner_valid_full_reference"]
            == merged["n_inner_valid_full_shallow_depth"]
        )
        & (
            merged["n_inner_valid_shap_reference"]
            == merged["n_inner_valid_shap_shallow_depth"]
        )
    )
    merged["no_final_test_rows"] = ~(
        boolean_series(merged["used_final_test_rows_reference"])
        | boolean_series(merged["used_final_test_rows_shallow_depth"])
    )

    expected_keys = len(horizons) * 3 * len(EXPECTED_DISCOVERY_MODELS)
    models_ok = set(merged["model_name"].dropna().astype(str)) == EXPECTED_DISCOVERY_MODELS
    audit_ok = bool(
        len(merged) == expected_keys
        and models_ok
        and merged["_merge"].eq("both").all()
        and merged["config_name_match"].all()
        and merged["fold_sizes_match"].all()
        and merged["no_final_test_rows"].all()
    )
    if not audit_ok:
        raise ValueError(
            "Discovery-model configuration or fold audit failed:\n"
            + merged.to_string(index=False)
        )
    return merged, reference_path, shallow_path


def load_consensus(directory, label, horizons, expected_groups=11):
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
        raise ValueError(f"{label} consensus is missing: {sorted(missing)}")
    df = df.copy()
    df["horizon_month"] = df["horizon_month"].astype(int)
    df["rank"] = df["rank"].astype(int)
    df["group_name"] = df["group_name"].astype(str)
    df = df[df["horizon_month"].isin(horizons)].copy()

    rows = []
    for horizon in horizons:
        group = df[df["horizon_month"].eq(horizon)]
        row = {
            "catalog_version": label,
            "horizon_month": int(horizon),
            "n_groups": int(group["group_name"].nunique()),
            "n_models_min": int(group["n_models"].min()) if len(group) else 0,
            "n_folds_min": int(group["n_folds"].min()) if len(group) else 0,
            "n_runs_min": int(group["n_runs"].min()) if len(group) else 0,
        }
        row["complete"] = bool(
            row["n_groups"] == expected_groups
            and row["n_models_min"] == 3
            and row["n_folds_min"] == 3
            and row["n_runs_min"] == 9
        )
        rows.append(row)
    audit = pd.DataFrame(rows)
    if not audit["complete"].all():
        raise ValueError(
            f"Incomplete {label} consensus coverage:\n{audit.to_string(index=False)}"
        )
    return df, audit, path


def ordered_top_groups(df, top_k):
    return (
        df.sort_values(["rank", "group_name"])
        .head(top_k)["group_name"]
        .astype(str)
        .tolist()
    )


def spearman_from_complete_ranks(rank_a, rank_b):
    """Spearman rho; complete untied ranks make this Pearson on rank values."""
    a = np.asarray(rank_a, dtype=float)
    b = np.asarray(rank_b, dtype=float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def compare_rankings(reference, shallow, horizons, top_k):
    reference_groups = set(reference["group_name"])
    shallow_groups = set(shallow["group_name"])
    if reference_groups != shallow_groups:
        raise ValueError(
            "The 11 group definitions differ: "
            f"reference_only={sorted(reference_groups - shallow_groups)}, "
            f"shallow_only={sorted(shallow_groups - reference_groups)}"
        )

    merged = reference.merge(
        shallow,
        on=["horizon_month", "group_name"],
        how="inner",
        suffixes=("_reference", "_shallow_depth"),
        validate="one_to_one",
    )
    merged["rank_change_shallow_minus_reference"] = (
        merged["rank_shallow_depth"] - merged["rank_reference"]
    )
    merged["importance_change_shallow_minus_reference"] = (
        merged["mean_group_importance_shallow_depth"]
        - merged["mean_group_importance_reference"]
    )
    merged["in_top3_reference"] = merged["rank_reference"] <= top_k
    merged["in_top3_shallow_depth"] = merged["rank_shallow_depth"] <= top_k

    summary_rows = []
    for horizon in horizons:
        ref_h = reference[reference["horizon_month"].eq(horizon)]
        shallow_h = shallow[shallow["horizon_month"].eq(horizon)]
        pair_h = merged[merged["horizon_month"].eq(horizon)]
        top_ref = ordered_top_groups(ref_h, top_k)
        top_shallow = ordered_top_groups(shallow_h, top_k)
        set_ref, set_shallow = set(top_ref), set(top_shallow)
        overlap = len(set_ref & set_shallow)
        union = len(set_ref | set_shallow)
        summary_rows.append(
            {
                "horizon_month": int(horizon),
                "n_information_groups": int(len(pair_h)),
                "spearman_rank_correlation": spearman_from_complete_ranks(
                    pair_h["rank_reference"], pair_h["rank_shallow_depth"]
                ),
                "mean_absolute_rank_change": float(
                    pair_h["rank_change_shallow_minus_reference"].abs().mean()
                ),
                "max_absolute_rank_change": int(
                    pair_h["rank_change_shallow_minus_reference"].abs().max()
                ),
                "top3_overlap_count": int(overlap),
                "top3_overlap_fraction": float(overlap / top_k),
                "top3_jaccard": float(overlap / union),
                "top3_exact_set_match": bool(set_ref == set_shallow),
                "top3_same_rank_count": int(
                    sum(a == b for a, b in zip(top_ref, top_shallow))
                ),
                "top3_reference": " | ".join(top_ref),
                "top3_shallow_depth": " | ".join(top_shallow),
            }
        )
    return merged, pd.DataFrame(summary_rows)


def key_group_table(merged, horizons, group_names):
    rows = []
    for horizon in horizons:
        for group_name in group_names:
            hit = merged[
                merged["horizon_month"].eq(horizon)
                & merged["group_name"].eq(group_name)
            ]
            if len(hit) != 1:
                raise ValueError(f"Missing key group H={horizon}: {group_name}")
            row = hit.iloc[0]
            rows.append(
                {
                    "horizon_month": int(horizon),
                    "group_name": group_name,
                    "rank_reference": int(row["rank_reference"]),
                    "rank_shallow_depth": int(row["rank_shallow_depth"]),
                    "rank_change_shallow_minus_reference": int(
                        row["rank_change_shallow_minus_reference"]
                    ),
                    "importance_reference": float(
                        row["mean_group_importance_reference"]
                    ),
                    "importance_shallow_depth": float(
                        row["mean_group_importance_shallow_depth"]
                    ),
                    "in_top3_reference": bool(row["in_top3_reference"]),
                    "in_top3_shallow_depth": bool(row["in_top3_shallow_depth"]),
                    "top3_membership_preserved": bool(
                        row["in_top3_reference"] == row["in_top3_shallow_depth"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def latest100_transition_table(
    reference,
    shallow,
    horizons,
    latest_prefix,
    magnitude_group,
    temporal_group,
):
    rows = []
    for horizon in horizons:
        ref = reference[
            reference["horizon_month"].eq(horizon)
            & reference["group_name"].str.startswith(latest_prefix)
        ].sort_values(["rank", "group_name"])
        dep = shallow[
            shallow["horizon_month"].eq(horizon)
            & shallow["group_name"].str.startswith(latest_prefix)
        ].sort_values(["rank", "group_name"])
        if ref.empty or dep.empty:
            raise ValueError(f"No Latest-100 groups found for H={horizon}")
        ref_top, dep_top = ref.iloc[0], dep.iloc[0]
        expected = magnitude_group if horizon == 1 else temporal_group
        rows.append(
            {
                "horizon_month": int(horizon),
                "expected_transition_group": expected,
                "top_latest100_reference": str(ref_top["group_name"]),
                "top_latest100_rank_reference": int(ref_top["rank"]),
                "top_latest100_shallow_depth": str(dep_top["group_name"]),
                "top_latest100_rank_shallow_depth": int(dep_top["rank"]),
                "reference_matches_transition": bool(ref_top["group_name"] == expected),
                "shallow_depth_matches_transition": bool(dep_top["group_name"] == expected),
                "top_latest100_group_preserved": bool(
                    ref_top["group_name"] == dep_top["group_name"]
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-data-dir", required=True)
    parser.add_argument("--shallow-data-dir", required=True)
    parser.add_argument("--reference-shap-dir", required=True)
    parser.add_argument("--shallow-shap-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--depth-min-km", type=float, default=0.0)
    parser.add_argument("--depth-max-km", type=float, default=70.0)
    parser.add_argument(
        "--monthly-extreme-group", default="monthly_lag/magnitude_extreme"
    )
    parser.add_argument(
        "--annual-extreme-group", default="annual/magnitude_extreme"
    )
    parser.add_argument("--latest100-prefix", default="last100/")
    parser.add_argument(
        "--latest100-magnitude-group",
        default="last100/magnitude_distribution",
    )
    parser.add_argument(
        "--latest100-temporal-group",
        default="last100/temporal_clustering",
    )
    args = parser.parse_args()

    if args.top_k != 3:
        raise ValueError("This prespecified report requires --top-k 3")
    if args.depth_min_km >= args.depth_max_km:
        raise ValueError("--depth-min-km must be smaller than --depth-max-km")

    horizons = parse_int_list(args.horizons)
    if horizons != [1, 3, 6, 12]:
        raise ValueError(
            "To keep the original horizons unchanged, --horizons must be 1,3,6,12"
        )
    reference_data_dir = expand_path(args.reference_data_dir)
    shallow_data_dir = expand_path(args.shallow_data_dir)
    reference_shap_dir = expand_path(args.reference_shap_dir)
    shallow_shap_dir = expand_path(args.shallow_shap_dir)
    out_dir = ensure_dir(expand_path(args.out_dir))

    dataset_audit, feature_meta_path = audit_datasets(
        reference_data_dir,
        shallow_data_dir,
        horizons,
        args.depth_min_km,
        args.depth_max_km,
    )
    config_audit, reference_shap_manifest, shallow_shap_manifest = (
        audit_discovery_configuration(
            reference_shap_dir,
            shallow_shap_dir,
            horizons,
        )
    )
    reference, audit_ref, reference_consensus_path = load_consensus(
        reference_shap_dir, "reference", horizons
    )
    shallow, audit_shallow, shallow_consensus_path = load_consensus(
        shallow_shap_dir, "shallow_depth_0_to_70km", horizons
    )
    coverage_audit = pd.concat([audit_ref, audit_shallow], ignore_index=True)

    merged, horizon_summary = compare_rankings(
        reference, shallow, horizons, args.top_k
    )
    key_groups = [
        args.monthly_extreme_group,
        args.annual_extreme_group,
        args.latest100_magnitude_group,
        args.latest100_temporal_group,
    ]
    key_stability = key_group_table(merged, horizons, key_groups)
    latest100 = latest100_transition_table(
        reference,
        shallow,
        horizons,
        args.latest100_prefix,
        args.latest100_magnitude_group,
        args.latest100_temporal_group,
    )

    monthly = key_stability[
        key_stability["group_name"].eq(args.monthly_extreme_group)
    ]
    annual = key_stability[
        key_stability["group_name"].eq(args.annual_extreme_group)
    ]
    monthly_stable = bool(
        len(monthly) == len(horizons)
        and monthly["rank_shallow_depth"].le(2).all()
    )
    annual_stable = bool(
        len(annual) == len(horizons)
        and annual["rank_shallow_depth"].le(2).all()
    )
    latest_transition_exists = bool(
        len(latest100) == len(horizons)
        and latest100["shallow_depth_matches_transition"].all()
    )

    paths = {
        "dataset_audit": os.path.join(out_dir, "dataset_invariance_audit.csv"),
        "configuration_audit": os.path.join(
            out_dir, "discovery_configuration_audit.csv"
        ),
        "coverage_audit": os.path.join(out_dir, "consensus_coverage_audit.csv"),
        "all_groups": os.path.join(out_dir, "all_group_rank_comparison.csv"),
        "horizon_summary": os.path.join(
            out_dir, "horizon_rank_and_top3_summary.csv"
        ),
        "key_stability": os.path.join(out_dir, "key_group_stability.csv"),
        "latest100": os.path.join(out_dir, "latest100_horizon_transition.csv"),
        "report": os.path.join(
            out_dir, "depth_filtered_inner_shap_sensitivity_report.md"
        ),
        "manifest": os.path.join(
            out_dir, "depth_filtered_inner_shap_sensitivity_manifest.json"
        ),
    }
    dataset_audit.to_csv(paths["dataset_audit"], index=False)
    config_audit.to_csv(paths["configuration_audit"], index=False)
    coverage_audit.to_csv(paths["coverage_audit"], index=False)
    merged.sort_values(["horizon_month", "rank_reference", "group_name"]).to_csv(
        paths["all_groups"], index=False
    )
    horizon_summary.to_csv(paths["horizon_summary"], index=False)
    key_stability.to_csv(paths["key_stability"], index=False)
    latest100.to_csv(paths["latest100"], index=False)

    manifest = {
        "analysis": "known shallow focal-depth feature-catalog sensitivity",
        "feature_event_depth_filter": {
            "minimum_km_inclusive": args.depth_min_km,
            "maximum_km_exclusive": args.depth_max_km,
        },
        "target": "unchanged binary future ML >= 5",
        "horizons_month": horizons,
        "cells_samples_folds_labels_identical": True,
        "models": sorted(EXPECTED_DISCOVERY_MODELS),
        "models_and_configurations_identical": True,
        "selection_stage": "inner_validation_only",
        "n_information_groups": 11,
        "monthly_magnitude_extremes_stable_top2_all_horizons": monthly_stable,
        "annual_magnitude_extremes_stable_top2_all_horizons": annual_stable,
        "latest100_h1_magnitude_to_h3_h12_temporal_transition_exists": (
            latest_transition_exists
        ),
        "inputs": {
            "reference_data_dir": reference_data_dir,
            "shallow_data_dir": shallow_data_dir,
            "reference_shap_dir": reference_shap_dir,
            "shallow_shap_dir": shallow_shap_dir,
            "feature_metadata": feature_meta_path,
            "reference_shap_manifest": reference_shap_manifest,
            "shallow_shap_manifest": shallow_shap_manifest,
            "reference_consensus": reference_consensus_path,
            "shallow_consensus": shallow_consensus_path,
        },
        "outputs": paths,
    }
    with open(paths["manifest"], "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    with open(paths["report"], "w", encoding="utf-8") as handle:
        handle.write("# Shallow focal-depth inner-validation SHAP sensitivity\n\n")
        handle.write(
            "The 282-feature representation was reconstructed from feature "
            "events with known focal depths `0 <= d < 70 km`. Cells, samples, "
            "the binary future `ML >= 5` target, rolling folds, discovery "
            "models, and model configurations were audited against the "
            "reference run. Only inner-validation SHAP rankings were repeated; "
            "the final test set was not used.\n\n"
        )
        handle.write("## Invariant audit\n\n")
        handle.write(format_markdown(dataset_audit))
        handle.write("\n\n## 11-group Spearman correlation and Top-3 overlap\n\n")
        handle.write(format_markdown(horizon_summary))
        handle.write("\n\n## Monthly and Annual magnitude extremes\n\n")
        handle.write(
            format_markdown(
                key_stability[
                    key_stability["group_name"].isin(
                        [args.monthly_extreme_group, args.annual_extreme_group]
                    )
                ],
                [
                    "horizon_month",
                    "group_name",
                    "rank_reference",
                    "rank_shallow_depth",
                    "rank_change_shallow_minus_reference",
                    "in_top3_shallow_depth",
                ],
            )
        )
        handle.write("\n\n## Latest-100 horizon transition\n\n")
        handle.write(format_markdown(latest100))
        handle.write("\n\n## Prespecified conclusions\n\n")
        handle.write(
            "Stability means that the group remains within the leading two "
            "information groups at every horizon. The Latest-100 transition "
            "means magnitude distribution leads at H1, while temporal "
            "clustering leads at H3, H6, and H12.\n\n"
        )
        handle.write(
            f"- Monthly magnitude extremes remain stable: **{monthly_stable}**\n"
        )
        handle.write(
            f"- Annual magnitude extremes remain stable: **{annual_stable}**\n"
        )
        handle.write(
            f"- Latest-100 horizon transition still exists: "
            f"**{latest_transition_exists}**\n"
        )

    print("=" * 100, flush=True)
    print("SHALLOW-DEPTH INNER-VALIDATION SHAP SENSITIVITY FINISHED", flush=True)
    print(horizon_summary.to_string(index=False), flush=True)
    print("Monthly magnitude extremes stable:", monthly_stable, flush=True)
    print("Annual magnitude extremes stable:", annual_stable, flush=True)
    print("Latest-100 transition exists:", latest_transition_exists, flush=True)
    print("Report:", paths["report"], flush=True)


if __name__ == "__main__":
    main()
