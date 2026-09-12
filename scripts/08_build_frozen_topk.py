#!/usr/bin/env python3
"""Build leakage-controlled Top-K datasets from frozen inner-validation SHAP.

The input table must contain one row per feature for each
(model_name, horizon_month, fold) inner-validation task. A single consensus
ranking is formed across all requested discovery models, horizons, and folds,
then the same frozen Top-K feature sets are applied to every forecast horizon.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def expand_path(path):
    return str(Path(path).expanduser().resolve())


def parse_str_list(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_list(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def load_feature_metadata(base_data_dir):
    candidates = [
        os.path.join(base_data_dir, "feature_metadata_282_relative_lag.csv"),
        os.path.join(base_data_dir, "feature_metadata.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            meta = pd.read_csv(path)
            if "feature_idx" not in meta.columns:
                meta["feature_idx"] = np.arange(len(meta), dtype=int)
            if "feature_name" not in meta.columns:
                meta["feature_name"] = [f"feature_{i}" for i in range(len(meta))]
            meta["feature_idx"] = meta["feature_idx"].astype(int)
            return meta.sort_values("feature_idx").reset_index(drop=True), path
    raise FileNotFoundError("Cannot find feature metadata in base data directory.")


def validate_and_rank(importance_csv, models, horizons, folds, expected_n_features):
    df = pd.read_csv(importance_csv)
    required = {
        "feature_idx",
        "feature_name",
        "model_name",
        "horizon_month",
        "fold",
        "source_stage",
        "mean_abs_shap",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {importance_csv}: {sorted(missing)}")

    df = df.copy()
    df["feature_idx"] = df["feature_idx"].astype(int)
    df["horizon_month"] = df["horizon_month"].astype(int)
    df["fold"] = df["fold"].astype(int)
    df["model_name"] = df["model_name"].astype(str)
    df["source_stage"] = df["source_stage"].astype(str).str.lower().str.strip()
    df["mean_abs_shap"] = pd.to_numeric(df["mean_abs_shap"], errors="coerce")

    bad_stage = sorted(set(df["source_stage"].dropna()) - {"inner_validation"})
    if bad_stage:
        raise ValueError(
            "Input contains non-inner-validation rows: " + ", ".join(bad_stage)
        )

    selected = df[
        df["model_name"].isin(models)
        & df["horizon_month"].isin(horizons)
        & df["fold"].isin(folds)
    ].copy()
    if selected.empty:
        raise ValueError("No rows remain after model/horizon/fold filtering.")

    available_models = set(selected["model_name"].unique())
    missing_models = [m for m in models if m not in available_models]
    if missing_models:
        raise ValueError(f"Requested models absent from input: {missing_models}")

    task_cols = ["model_name", "horizon_month", "fold"]
    expected_tasks = pd.MultiIndex.from_product(
        [models, horizons, folds], names=task_cols
    )
    observed_tasks = pd.MultiIndex.from_frame(selected[task_cols].drop_duplicates())
    missing_tasks = expected_tasks.difference(observed_tasks)
    if len(missing_tasks):
        raise ValueError(f"Missing requested inner-validation tasks: {list(missing_tasks)}")

    duplicate_count = selected.duplicated(task_cols + ["feature_idx"]).sum()
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate task-feature rows; expected one row per feature per task."
        )

    task_audit_rows = []
    normalized_parts = []
    for task_key, g in selected.groupby(task_cols, sort=True):
        g = g.copy()
        n_features = g["feature_idx"].nunique()
        if n_features != expected_n_features:
            raise ValueError(
                f"Task {task_key} contains {n_features} unique features; "
                f"expected {expected_n_features}."
            )
        if g["mean_abs_shap"].isna().any() or (g["mean_abs_shap"] < 0).any():
            raise ValueError(f"Invalid mean_abs_shap values in task {task_key}.")

        total = float(g["mean_abs_shap"].sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"Non-positive SHAP total in task {task_key}.")

        # Re-normalize within every model-horizon-fold task so each task has equal weight.
        g["task_normalized_importance"] = g["mean_abs_shap"] / total
        g["task_rank"] = (
            g["mean_abs_shap"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        normalized_parts.append(g)
        task_audit_rows.append(
            {
                "model_name": task_key[0],
                "horizon_month": int(task_key[1]),
                "fold": int(task_key[2]),
                "source_stage": "inner_validation",
                "n_features": int(n_features),
                "mean_abs_shap_sum": total,
                "normalized_importance_sum": float(
                    g["task_normalized_importance"].sum()
                ),
            }
        )

    norm = pd.concat(normalized_parts, ignore_index=True)
    meta_cols = [
        c
        for c in [
            "feature_idx",
            "feature_name",
            "time_group",
            "signal_group",
            "month_lag",
            "stat_type",
            "mag_bin",
            "depth_bin",
            "feature_group",
        ]
        if c in norm.columns
    ]

    agg = (
        norm.groupby(meta_cols, dropna=False)
        .agg(
            mean_inner_normalized_importance=("task_normalized_importance", "mean"),
            median_inner_normalized_importance=("task_normalized_importance", "median"),
            max_inner_normalized_importance=("task_normalized_importance", "max"),
            min_task_rank=("task_rank", "min"),
            mean_task_rank=("task_rank", "mean"),
            max_task_rank=("task_rank", "max"),
            top20_count=("task_rank", lambda x: int((x <= 20).sum())),
            top50_count=("task_rank", lambda x: int((x <= 50).sum())),
            top100_count=("task_rank", lambda x: int((x <= 100).sum())),
            task_count=("task_rank", "count"),
            model_count=("model_name", "nunique"),
            horizon_count=("horizon_month", "nunique"),
            fold_count=("fold", "nunique"),
        )
        .reset_index()
    )

    expected_task_count = len(models) * len(horizons) * len(folds)
    if len(agg) != expected_n_features:
        raise ValueError(
            f"Consensus table has {len(agg)} features; expected {expected_n_features}."
        )
    if not (agg["task_count"] == expected_task_count).all():
        bad = agg.loc[agg["task_count"] != expected_task_count, ["feature_idx", "task_count"]]
        raise ValueError(
            "Some features are absent from requested tasks:\n" + bad.head(20).to_string(index=False)
        )

    # Primary criterion is the mean within-task normalized importance.
    # Stability counts and mean rank are deterministic tie-breakers only.
    agg = agg.sort_values(
        [
            "mean_inner_normalized_importance",
            "top20_count",
            "top50_count",
            "mean_task_rank",
            "feature_idx",
        ],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    agg["global_stability_rank"] = np.arange(1, len(agg) + 1, dtype=int)
    agg["ranking_source"] = "frozen_inner_validation_consensus"
    agg["discovery_models"] = ",".join(models)
    agg["discovery_horizons"] = ",".join(map(str, horizons))
    agg["discovery_folds"] = ",".join(map(str, folds))

    return agg, pd.DataFrame(task_audit_rows), norm


def reduce_dataset(src_path, dst_path, keep_indices, reduced_meta):
    with open(src_path, "rb") as f:
        data = pickle.load(f)

    X = np.asarray(data["X"])
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D X in {src_path}, got {X.shape}")
    if X.shape[1] <= max(keep_indices):
        raise ValueError(
            f"Feature index out of range for {src_path}: max={max(keep_indices)}, "
            f"n_features={X.shape[1]}"
        )

    X_new = X[:, keep_indices].astype(np.float32, copy=False)
    out = dict(data)
    out["X"] = X_new
    out["feature_names"] = reduced_meta["feature_name"].astype(str).tolist()
    out["feature_idx_original"] = [int(i) for i in keep_indices]
    out["feature_metadata"] = reduced_meta.copy()

    with open(dst_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    return [int(X.shape[0]), int(X.shape[1])], [int(X_new.shape[0]), int(X_new.shape[1])]


def build_topk_datasets(base_data_dir, out_root, ranking, metadata, topk_list, horizons, sources):
    manifests = []
    for k in topk_list:
        if k <= 0 or k > len(ranking):
            raise ValueError(f"Invalid K={k}; ranking contains {len(ranking)} features.")

        ranked = ranking.nsmallest(k, "global_stability_rank").copy()
        keep_indices = sorted(ranked["feature_idx"].astype(int).tolist())

        out_dir = os.path.join(out_root, f"top{k}_innerfrozen_consensus")
        os.makedirs(out_dir, exist_ok=True)

        selected = metadata[metadata["feature_idx"].isin(keep_indices)].copy()
        selected["original_feature_idx"] = selected["feature_idx"].astype(int)
        selected["new_feature_idx"] = np.arange(len(selected), dtype=int)
        selected = selected.sort_values("new_feature_idx").reset_index(drop=True)

        # Attach consensus rank and summary metrics for auditability.
        selected = selected.merge(
            ranking[
                [
                    "feature_idx",
                    "global_stability_rank",
                    "mean_inner_normalized_importance",
                    "mean_task_rank",
                    "top20_count",
                    "top50_count",
                    "top100_count",
                ]
            ].rename(columns={"feature_idx": "original_feature_idx"}),
            on="original_feature_idx",
            how="left",
            validate="one_to_one",
        )

        reduced_meta = selected.copy()
        reduced_meta["feature_idx"] = reduced_meta["new_feature_idx"].astype(int)
        removed = metadata[~metadata["feature_idx"].isin(keep_indices)].copy()
        removed["original_feature_idx"] = removed["feature_idx"].astype(int)

        selected.to_csv(os.path.join(out_dir, "selected_features.csv"), index=False)
        removed.to_csv(os.path.join(out_dir, "removed_features.csv"), index=False)
        reduced_meta.to_csv(
            os.path.join(out_dir, "feature_metadata_282_relative_lag.csv"), index=False
        )
        reduced_meta.to_csv(
            os.path.join(out_dir, "feature_metadata_reduced.csv"), index=False
        )

        file_rows = []
        for horizon in horizons:
            src = os.path.join(
                base_data_dir, f"china_raw282_dataset_horizon_{horizon}m.pkl"
            )
            dst = os.path.join(
                out_dir, f"china_raw282_dataset_horizon_{horizon}m.pkl"
            )
            if not os.path.exists(src):
                raise FileNotFoundError(src)
            old_shape, new_shape = reduce_dataset(
                src, dst, keep_indices, reduced_meta
            )
            file_rows.append(
                {
                    "horizon_month": int(horizon),
                    "src": src,
                    "dst": dst,
                    "old_shape": old_shape,
                    "new_shape": new_shape,
                }
            )
            print(f"Top{k} H{horizon}: {old_shape} -> {new_shape}", flush=True)

        manifest = {
            "variant": f"top{k}_innerfrozen_consensus",
            "k": int(k),
            "selection_protocol": (
                "A single consensus ranking was frozen from task-normalized SHAP "
                "importance on inner-validation samples across the requested discovery "
                "models, horizons, and folds; the same Top-K set was used for all horizons."
            ),
            "importance_csv": sources["importance_csv"],
            "feature_metadata_source": sources["feature_metadata_source"],
            "ranking_csv": sources["ranking_csv"],
            "base_data_dir": base_data_dir,
            "out_dir": out_dir,
            "n_selected_features": int(len(selected)),
            "selected_feature_original_indices": [int(i) for i in keep_indices],
            "horizons": [int(h) for h in horizons],
            "files": file_rows,
        }
        with open(os.path.join(out_dir, "feature_selection_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        manifests.append(manifest)

    return manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--importance-csv", required=True)
    parser.add_argument(
        "--base-data-dir",
        default="data/raw282",
    )
    parser.add_argument(
        "--out-root",
        default="data/topk",
    )
    parser.add_argument(
        "--audit-out-dir",
        default="outputs/topk_ranking",
    )
    parser.add_argument("--models", default="CatBoost,ExtraTrees,XGBoost")
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--folds", default="1,2,3")
    parser.add_argument("--topk-list", default="50,75,100")
    parser.add_argument("--expected-n-features", type=int, default=282)
    args = parser.parse_args()

    importance_csv = expand_path(args.importance_csv)
    base_data_dir = expand_path(args.base_data_dir)
    out_root = expand_path(args.out_root)
    audit_out_dir = expand_path(args.audit_out_dir)
    models = parse_str_list(args.models)
    horizons = parse_int_list(args.horizons)
    folds = parse_int_list(args.folds)
    topk_list = parse_int_list(args.topk_list)

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(audit_out_dir, exist_ok=True)

    ranking, task_audit, normalized_long = validate_and_rank(
        importance_csv=importance_csv,
        models=models,
        horizons=horizons,
        folds=folds,
        expected_n_features=args.expected_n_features,
    )

    ranking_csv = os.path.join(
        audit_out_dir, "frozen_inner_consensus_feature_ranking.csv"
    )
    task_audit_csv = os.path.join(
        audit_out_dir, "frozen_inner_consensus_task_audit.csv"
    )
    normalized_csv = os.path.join(
        audit_out_dir, "frozen_inner_consensus_importance_long.csv.gz"
    )
    ranking.to_csv(ranking_csv, index=False)
    task_audit.to_csv(task_audit_csv, index=False)
    normalized_long.to_csv(normalized_csv, index=False, compression="gzip")

    metadata, metadata_path = load_feature_metadata(base_data_dir)
    if len(metadata) != args.expected_n_features:
        raise ValueError(
            f"Feature metadata contains {len(metadata)} rows; expected {args.expected_n_features}."
        )

    build_topk_datasets(
        base_data_dir=base_data_dir,
        out_root=out_root,
        ranking=ranking,
        metadata=metadata,
        topk_list=topk_list,
        horizons=horizons,
        sources={
            "importance_csv": importance_csv,
            "feature_metadata_source": metadata_path,
            "ranking_csv": ranking_csv,
        },
    )

    run_manifest = {
        "importance_csv": importance_csv,
        "base_data_dir": base_data_dir,
        "out_root": out_root,
        "audit_out_dir": audit_out_dir,
        "models": models,
        "horizons": horizons,
        "folds": folds,
        "topk_list": topk_list,
        "expected_n_features": int(args.expected_n_features),
        "n_tasks": int(len(models) * len(horizons) * len(folds)),
        "ranking_csv": ranking_csv,
        "task_audit_csv": task_audit_csv,
        "normalized_long_csv": normalized_csv,
    }
    with open(os.path.join(audit_out_dir, "run_manifest.json"), "w") as f:
        json.dump(run_manifest, f, indent=2)

    print("=" * 100, flush=True)
    print("Frozen inner-validation consensus Top-K datasets generated.", flush=True)
    print("Ranking:", ranking_csv, flush=True)
    print("Top 15 features:", flush=True)
    print(
        ranking[
            [
                "global_stability_rank",
                "feature_idx",
                "feature_name",
                "mean_inner_normalized_importance",
                "mean_task_rank",
                "top20_count",
            ]
        ].head(15).to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
