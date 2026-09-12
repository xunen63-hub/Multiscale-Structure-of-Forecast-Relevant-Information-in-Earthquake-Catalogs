#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Annual Top-3 versus Top-5 distribution audit and matched retraining.

This script extends the project's frozen inner-validation design without using
the final test period for feature selection or threshold selection.

Primary comparison
------------------
For each forecast horizon, read the Top-3 information groups frozen from
inner-validation SHAP.  Remove the complete ``annual/magnitude_extreme`` group
from that compact core, then compare two matched augmentations:

    core_plus_annual_top3 = common compact core + annual M1..M3
    core_plus_annual_top5 = common compact core + annual M1..M5

The two conditions therefore differ only by annual M4 and M5.  They use the
same samples, temporal folds, target, model configuration and paired seeds.
Decision thresholds are selected separately from pooled inner-validation
predictions, exactly as in the existing frozen-group ablation pipeline.

Distribution audit
------------------
The script extracts annual M1..M5 directly from the already-built 282-feature
datasets, verifies that shared cell-origin samples agree across horizons, and
reports order-statistic, adjacent-gap, Top-3 range and Top-5 range summaries.
Statistics are emitted both for all eligible samples (including the project's
deterministic zero encoding) and for samples with a genuinely present M5.

No final-test labels or metrics are used to define either feature condition.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        fbeta_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    HAS_SKLEARN = True
except Exception as exc:  # distribution-only mode remains usable
    ExtraTreesClassifier = None
    HAS_SKLEARN = False
    SKLEARN_IMPORT_ERROR = repr(exc)
else:
    SKLEARN_IMPORT_ERROR = ""

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except Exception as exc:  # pragma: no cover - depends on cluster environment
    CatBoostClassifier = None
    HAS_CATBOOST = False
    CATBOOST_IMPORT_ERROR = repr(exc)
else:
    CATBOOST_IMPORT_ERROR = ""

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception as exc:  # pragma: no cover - depends on cluster environment
    XGBClassifier = None
    HAS_XGBOOST = False
    XGBOOST_IMPORT_ERROR = repr(exc)
else:
    XGBOOST_IMPORT_ERROR = ""

warnings.filterwarnings("ignore", category=UserWarning)


MODEL_ALIAS = {
    "extratrees": "ExtraTrees",
    "extra_trees": "ExtraTrees",
    "et": "ExtraTrees",
    "catboost": "CatBoost",
    "cat": "CatBoost",
    "xgboost": "XGBoost",
    "xgb": "XGBoost",
}

# These are copied from 06_feature_group_ablation.py.
# They are only available behind --allow-default-configs.  The normal and
# recommended path is to load selected_config_by_model.json from the original
# comparison run, so the manuscript model configurations are reused exactly.
DEFAULT_CONFIGS = {
    "ExtraTrees": {
        "config_name": "et_cfg0_depth12_leaf5",
        "params": {
            "n_estimators": 500,
            "max_depth": 12,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "bootstrap": False,
        },
    },
    "CatBoost": {
        "config_name": "cat_cfg1_depth5_lr002",
        "params": {
            "iterations": 800,
            "depth": 5,
            "learning_rate": 0.020,
            "l2_leaf_reg": 10.0,
            "random_strength": 1.0,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.85,
        },
    },
    "XGBoost": {
        "config_name": "xgb_cfg4_strong_reg",
        "params": {
            "n_estimators": 500,
            "max_depth": 3,
            "learning_rate": 0.030,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "reg_lambda": 8.0,
            "reg_alpha": 0.8,
            "min_child_weight": 5.0,
            "gamma": 0.20,
        },
    },
}

ANNUAL_MAGNITUDE_NAMES = [f"annual_top{i}_magnitude" for i in range(1, 6)]
HIGHER_IS_BETTER = ["auc", "pr_auc", "f1"]
LOWER_IS_BETTER = ["brier"]


def expand_path(path):
    return str(Path(path).expanduser().resolve())


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def normalize_model_name(value):
    key = str(value).strip().lower().replace(" ", "_")
    if key not in MODEL_ALIAS:
        raise ValueError(f"Unknown model name: {value}")
    return MODEL_ALIAS[key]


def safe_name(value):
    return (
        str(value)
        .lower()
        .replace(" ", "_")
        .replace("/", "__")
        .replace("+", "plus")
    )


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dataset_path(data_dir, horizon):
    return os.path.join(data_dir, f"china_raw282_dataset_horizon_{horizon}m.pkl")


def load_dataset(data_dir, horizon):
    path = dataset_path(data_dir, horizon)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    X = np.asarray(payload["X"], dtype=np.float32)
    if "y_m5" in payload:
        y = np.asarray(payload["y_m5"], dtype=np.int64)
    else:
        y = (np.asarray(payload["y_class"], dtype=np.int64) > 0).astype(np.int64)
    meta = payload["meta"].copy().reset_index(drop=True)
    if len(X) != len(y) or len(X) != len(meta):
        raise ValueError(
            f"H={horizon} row mismatch: X={len(X)}, y={len(y)}, meta={len(meta)}"
        )
    return X, y, meta, payload


def load_feature_metadata(data_dir, n_features):
    path = os.path.join(data_dir, "feature_metadata_282_relative_lag.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature metadata is required for leakage-safe feature construction: {path}"
        )
    metadata = pd.read_csv(path)
    if "feature_idx" not in metadata.columns:
        metadata.insert(0, "feature_idx", np.arange(len(metadata), dtype=int))
    metadata = metadata.sort_values("feature_idx").reset_index(drop=True)
    if len(metadata) != n_features:
        raise ValueError(
            f"feature metadata length {len(metadata)} != n_features {n_features}"
        )
    required = {"feature_idx", "feature_name", "time_group", "signal_group"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Feature metadata is missing columns: {missing}")
    if "feature_group" not in metadata.columns:
        metadata["feature_group"] = (
            metadata["time_group"].fillna("unknown").astype(str)
            + "/"
            + metadata["signal_group"].fillna("unknown").astype(str)
        )
    return metadata


def _to_bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", "", "nan", "none"}
    unknown = sorted(set(normalized) - true_values - false_values)
    if unknown:
        raise ValueError(f"Cannot parse Boolean values: {unknown}")
    return normalized.isin(true_values)


def load_frozen_groups(frozen_csv, horizons, group_col, frozen_top_k):
    if not os.path.exists(frozen_csv):
        raise FileNotFoundError(frozen_csv)
    table = pd.read_csv(frozen_csv).copy()
    required = {"horizon_month", "rank", "group_name"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Frozen-group CSV is missing columns: {missing}")

    table["horizon_month"] = pd.to_numeric(
        table["horizon_month"], errors="raise"
    ).astype(int)
    table["rank"] = pd.to_numeric(table["rank"], errors="raise").astype(int)
    table["group_name"] = table["group_name"].astype(str).str.strip()

    if table.duplicated(["horizon_month", "rank"]).any():
        raise ValueError("Frozen-group CSV has duplicate horizon/rank rows")
    if "source_stage" in table.columns:
        valid = table["source_stage"].astype(str).str.strip().str.lower()
        if not valid.eq("inner_validation").all():
            raise RuntimeError("Frozen groups were not exclusively selected in inner validation")
    if "used_final_test_for_selection" in table.columns:
        if bool(_to_bool_series(table["used_final_test_for_selection"]).any()):
            raise RuntimeError("Leakage guard: frozen groups used final-test information")
    if "complete_model_fold_coverage" in table.columns:
        if not bool(_to_bool_series(table["complete_model_fold_coverage"]).all()):
            raise ValueError("Frozen groups lack complete model-fold coverage")
    if "group_col" in table.columns:
        observed = set(table["group_col"].dropna().astype(str).str.strip())
        if observed and observed != {group_col}:
            raise ValueError(
                f"Frozen group_col {sorted(observed)} does not match requested {group_col}"
            )

    selected = []
    for horizon in horizons:
        subset = table[table["horizon_month"] == int(horizon)].sort_values("rank")
        actual = subset.loc[subset["rank"] <= frozen_top_k, "rank"].tolist()
        expected = list(range(1, frozen_top_k + 1))
        if actual != expected:
            raise ValueError(
                f"H={horizon} frozen ranks are {actual}; expected {expected}"
            )
        selected.append(subset[subset["rank"] <= frozen_top_k])
    result = pd.concat(selected, ignore_index=True, sort=False)
    result["selection_source"] = "inner_validation_frozen_csv"
    result["used_final_test_for_selection"] = False
    result["frozen_groups_csv"] = str(Path(frozen_csv).resolve())
    result["frozen_groups_sha256"] = file_sha256(frozen_csv)
    return result.sort_values(["horizon_month", "rank"]).reset_index(drop=True)


def load_selected_configs(comparison_dir):
    path = os.path.join(comparison_dir, "selected_config_by_model.json")
    if not os.path.exists(path):
        return {}, path
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = {}
    for item in payload.get("selected_configs", []):
        if "model_name" in item and item.get("params"):
            selected[str(item["model_name"])] = {
                "config_name": item.get("config_name", "selected"),
                "params": item["params"],
            }
    return selected, path


def get_model_config(model_name, selected_configs, allow_defaults):
    if model_name in selected_configs and selected_configs[model_name].get("params"):
        return deepcopy(selected_configs[model_name])
    if not allow_defaults:
        raise FileNotFoundError(
            f"No selected configuration for {model_name}. Refusing to change the "
            "manuscript model configuration. Check --comparison-dir, or use "
            "--allow-default-configs only if intentional."
        )
    return deepcopy(DEFAULT_CONFIGS[model_name])


def make_model(model_name, params, seed, n_jobs):
    params = deepcopy(dict(params))
    if model_name == "ExtraTrees":
        if not HAS_SKLEARN:
            raise ImportError(f"scikit-learn is not installed: {SKLEARN_IMPORT_ERROR}")
        params["random_state"] = seed
        params["n_jobs"] = n_jobs
        return ExtraTreesClassifier(**params)
    if model_name == "CatBoost":
        if not HAS_CATBOOST:
            raise ImportError(f"catboost is not installed: {CATBOOST_IMPORT_ERROR}")
        params.setdefault("loss_function", "Logloss")
        params.setdefault("eval_metric", "Logloss")
        params["random_seed"] = seed
        params["thread_count"] = n_jobs
        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
        return CatBoostClassifier(**params)
    if model_name == "XGBoost":
        if not HAS_XGBOOST:
            raise ImportError(f"xgboost is not installed: {XGBOOST_IMPORT_ERROR}")
        params.setdefault("objective", "binary:logistic")
        params.setdefault("eval_metric", "logloss")
        params.setdefault("tree_method", "hist")
        params["random_state"] = seed
        params["n_jobs"] = n_jobs
        return XGBClassifier(**params)
    raise ValueError(model_name)


def get_score(model, X):
    if hasattr(model, "predict_proba"):
        probability = np.asarray(model.predict_proba(X))
        if probability.ndim == 2 and probability.shape[1] >= 2:
            return probability[:, 1].astype(float)
        return probability.ravel().astype(float)
    return np.asarray(model.predict(X), dtype=float).ravel()


def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, score))


def safe_ap(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, score))


def evaluate_binary(y_true, score, threshold):
    if not HAS_SKLEARN:
        raise ImportError(f"scikit-learn is not installed: {SKLEARN_IMPORT_ERROR}")
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    prediction = (score >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    metrics = {
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true == 1)),
        "threshold": float(threshold),
        "pred_positive_rate": float(np.mean(prediction == 1)),
        "acc": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "f05": float(fbeta_score(y_true, prediction, beta=0.5, zero_division=0)),
        "auc": safe_auc(y_true, score),
        "pr_auc": safe_ap(y_true, score),
        "brier": float(brier_score_loss(y_true, score)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
    return metrics, prediction


def choose_threshold(y_true, score, thresholds, target="f1"):
    rows = [evaluate_binary(y_true, score, threshold)[0] for threshold in thresholds]
    curve = pd.DataFrame(rows)
    if target not in curve.columns:
        raise ValueError(f"Unknown threshold target: {target}")
    curve = curve.sort_values(
        [target, "threshold"], ascending=[False, True]
    ).reset_index(drop=True)
    return float(curve.loc[0, "threshold"]), curve


def append_row_csv(path, row):
    new = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        columns = list(old.columns)
        columns.extend(column for column in new.columns if column not in columns)
        pd.concat(
            [old.reindex(columns=columns), new.reindex(columns=columns)],
            ignore_index=True,
        ).to_csv(path, index=False)
    else:
        new.to_csv(path, index=False)


def remove_matching_rows(path, conditions):
    if not os.path.exists(path):
        return
    table = pd.read_csv(path)
    match = np.ones(len(table), dtype=bool)
    for column, value in conditions.items():
        if column in table.columns:
            match &= table[column].astype(str).eq(str(value)).to_numpy()
        else:
            match &= False
    table.loc[~match].to_csv(path, index=False)


def build_matched_feature_sets(
    feature_meta,
    frozen_horizon_groups,
    group_col,
    annual_group,
):
    ranked_groups = (
        frozen_horizon_groups.sort_values("rank")["group_name"].astype(str).tolist()
    )
    if annual_group not in ranked_groups:
        raise ValueError(
            f"Frozen compact core does not contain {annual_group}: {ranked_groups}"
        )
    if feature_meta["feature_name"].duplicated().any():
        raise ValueError("Feature metadata contains duplicate feature names")

    name_to_index = dict(
        zip(feature_meta["feature_name"], feature_meta["feature_idx"].astype(int))
    )
    missing_annual = [name for name in ANNUAL_MAGNITUDE_NAMES if name not in name_to_index]
    if missing_annual:
        raise ValueError(f"Missing annual magnitude features: {missing_annual}")

    base_groups = [group for group in ranked_groups if group != annual_group]
    base_mask = feature_meta[group_col].astype(str).isin(base_groups)
    base_features = feature_meta.loc[base_mask, "feature_idx"].astype(int).to_numpy()
    if len(base_features) == 0:
        raise ValueError("Matched compact-core base is empty")

    top3 = np.asarray([name_to_index[name] for name in ANNUAL_MAGNITUDE_NAMES[:3]])
    top5 = np.asarray([name_to_index[name] for name in ANNUAL_MAGNITUDE_NAMES])
    masks = {
        "core_plus_annual_top3": np.unique(np.concatenate([base_features, top3])),
        "core_plus_annual_top5": np.unique(np.concatenate([base_features, top5])),
    }
    if set(masks["core_plus_annual_top5"]) - set(masks["core_plus_annual_top3"]) != set(
        top5[-2:]
    ):
        raise RuntimeError("Matched comparison differs by features other than M4 and M5")

    rows = []
    for condition, indices in masks.items():
        selected_names = feature_meta.set_index("feature_idx").loc[
            indices, "feature_name"
        ].tolist()
        rows.append(
            {
                "feature_set_name": condition,
                "feature_set_type": "matched_compact_augmentation",
                "base_groups": "; ".join(base_groups),
                "annual_features": "; ".join(
                    ANNUAL_MAGNITUDE_NAMES[:3]
                    if condition.endswith("top3")
                    else ANNUAL_MAGNITUDE_NAMES
                ),
                "n_features": int(len(indices)),
                "feature_indices": ";".join(map(str, indices.tolist())),
                "feature_names": "; ".join(selected_names),
            }
        )
    return masks, pd.DataFrame(rows), base_groups


def sample_key_table(meta, horizon):
    result = pd.DataFrame(index=np.arange(len(meta)))
    if {"patch_order", "t0"}.issubset(meta.columns):
        result["sample_key"] = (
            meta["patch_order"].astype(str) + "|" + meta["t0"].astype(str)
        )
    elif {"region", "t0"}.issubset(meta.columns):
        result["sample_key"] = meta["region"].astype(str) + "|" + meta["t0"].astype(str)
    else:
        result["sample_key"] = f"H{horizon}|" + pd.Series(
            np.arange(len(meta)), dtype=str
        )
    for column in ["t0", "split", "patch_order", "region"]:
        if column in meta.columns:
            result[column] = meta[column].to_numpy()
    return result


def extract_distribution_rows(X, meta, feature_meta, horizon):
    name_to_index = dict(
        zip(feature_meta["feature_name"], feature_meta["feature_idx"].astype(int))
    )
    indices = [name_to_index[name] for name in ANNUAL_MAGNITUDE_NAMES]
    output = sample_key_table(meta, horizon)
    output.insert(0, "horizon_month", int(horizon))
    for rank, index in enumerate(indices, start=1):
        output[f"M{rank}"] = X[:, index].astype(float)

    count_candidates = [
        "feature_annual_event_count",
        "annual_feature_event_count",
        "annual_event_count",
    ]
    count_column = next((name for name in count_candidates if name in meta.columns), None)
    if count_column:
        output["annual_event_count"] = pd.to_numeric(
            meta[count_column], errors="coerce"
        ).to_numpy()
        output["m5_present"] = output["annual_event_count"].ge(5)
        output["availability_source"] = count_column
    else:
        output["annual_event_count"] = np.nan
        output["m5_present"] = output["M5"].gt(0.0)
        output["availability_source"] = "M5>0 (deterministic-zero encoding)"

    output["n_nonzero_top5"] = output[[f"M{i}" for i in range(1, 6)]].gt(0).sum(axis=1)
    for rank in range(1, 5):
        output[f"gap_M{rank}_M{rank + 1}"] = output[f"M{rank}"] - output[
            f"M{rank + 1}"
        ]
    output["top3_range_M1_M3"] = output["M1"] - output["M3"]
    output["top5_range_M1_M5"] = output["M1"] - output["M5"]
    output["top3_mean"] = output[["M1", "M2", "M3"]].mean(axis=1)
    output["top5_mean"] = output[["M1", "M2", "M3", "M4", "M5"]].mean(axis=1)
    output["top3_range_le_0p5"] = output["top3_range_M1_M3"].le(0.5).astype(int)
    output["top3_range_le_1p0"] = output["top3_range_M1_M3"].le(1.0).astype(int)
    output["top5_range_le_0p5"] = output["top5_range_M1_M5"].le(0.5).astype(int)
    output["top5_range_le_1p0"] = output["top5_range_M1_M5"].le(1.0).astype(int)
    return output


def numeric_summary(table, scope, subset, horizon=None):
    variables = [
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "gap_M1_M2",
        "gap_M2_M3",
        "gap_M3_M4",
        "gap_M4_M5",
        "top3_range_M1_M3",
        "top5_range_M1_M5",
        "top3_mean",
        "top5_mean",
        "annual_event_count",
        "n_nonzero_top5",
        "top3_range_le_0p5",
        "top3_range_le_1p0",
        "top5_range_le_0p5",
        "top5_range_le_1p0",
    ]
    rows = []
    for variable in variables:
        values = pd.to_numeric(table[variable], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "scope": scope,
                "subset": subset,
                "horizon_month": horizon,
                "variable": variable,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "min": float(values.min()),
                "p05": float(values.quantile(0.05)),
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q75": float(values.quantile(0.75)),
                "p95": float(values.quantile(0.95)),
                "max": float(values.max()),
            }
        )
    return rows


def verify_cross_horizon_consistency(distribution_rows, tolerance=1e-6):
    duplicated = distribution_rows[
        distribution_rows.duplicated("sample_key", keep=False)
    ].copy()
    if duplicated.empty:
        return
    spread = duplicated.groupby("sample_key")[[f"M{i}" for i in range(1, 6)]].agg(
        lambda values: float(np.nanmax(values) - np.nanmin(values))
    )
    if bool((spread > tolerance).any().any()):
        bad = spread[(spread > tolerance).any(axis=1)].head()
        raise RuntimeError(
            "Annual M1-M5 disagree across horizon datasets for shared cell-origin "
            f"samples. First mismatches:\n{bad}"
        )


def write_distribution_outputs(distribution_rows, out_dir):
    verify_cross_horizon_consistency(distribution_rows)
    sample_csv = os.path.join(out_dir, "annual_top3_top5_sample_level.csv.gz")
    sample_parquet = os.path.join(out_dir, "annual_top3_top5_sample_level.parquet")
    distribution_rows.to_csv(sample_csv, index=False, compression="gzip")
    parquet_status = "written"
    try:
        distribution_rows.to_parquet(sample_parquet, index=False)
    except Exception as exc:  # optional pyarrow/fastparquet dependency
        parquet_status = f"not written ({type(exc).__name__}: {exc})"

    unique = (
        distribution_rows.sort_values(["sample_key", "horizon_month"])
        .drop_duplicates("sample_key", keep="first")
        .reset_index(drop=True)
    )
    overall_rows = []
    for subset_name, subset in [
        ("all_eligible", unique),
        ("m5_present", unique[unique["m5_present"]]),
    ]:
        overall_rows.extend(
            numeric_summary(subset, "unique_cell_origin", subset_name, horizon=None)
        )
    overall = pd.DataFrame(overall_rows)
    overall_path = os.path.join(out_dir, "annual_top3_top5_distribution_summary.csv")
    overall.to_csv(overall_path, index=False)

    horizon_rows = []
    for horizon, horizon_table in distribution_rows.groupby("horizon_month"):
        for subset_name, subset in [
            ("all_eligible", horizon_table),
            ("m5_present", horizon_table[horizon_table["m5_present"]]),
        ]:
            horizon_rows.extend(
                numeric_summary(
                    subset,
                    "horizon_dataset",
                    subset_name,
                    horizon=int(horizon),
                )
            )
    by_horizon = pd.DataFrame(horizon_rows)
    by_horizon_path = os.path.join(
        out_dir, "annual_top3_top5_distribution_by_horizon.csv"
    )
    by_horizon.to_csv(by_horizon_path, index=False)

    plot_distribution(unique, out_dir)
    return {
        "sample_csv": sample_csv,
        "sample_parquet": sample_parquet,
        "sample_parquet_status": parquet_status,
        "overall_summary": overall_path,
        "by_horizon_summary": by_horizon_path,
        "n_horizon_rows": int(len(distribution_rows)),
        "n_unique_cell_origins": int(len(unique)),
        "n_unique_m5_present": int(unique["m5_present"].sum()),
        "availability_sources": sorted(unique["availability_source"].astype(str).unique()),
    }


def plot_distribution(unique, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Skip distribution plot: {exc}", flush=True)
        return
    available = unique[unique["m5_present"]].copy()
    if available.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.35))
    magnitude_data = [available[f"M{i}"].to_numpy() for i in range(1, 6)]
    axes[0].boxplot(magnitude_data, labels=[f"M{i}" for i in range(1, 6)], showfliers=False)
    axes[0].set_ylabel("Annual magnitude order statistic")
    axes[0].set_title("(a) Annual M1-M5 (M5 present)", loc="left", fontweight="bold")

    range_data = [
        available["top3_range_M1_M3"].to_numpy(),
        available["top5_range_M1_M5"].to_numpy(),
    ]
    axes[1].boxplot(range_data, labels=["M1-M3", "M1-M5"], showfliers=False)
    axes[1].axhline(0.5, color="0.55", linestyle="--", linewidth=0.8)
    axes[1].axhline(1.0, color="0.35", linestyle=":", linewidth=0.8)
    axes[1].set_ylabel("Magnitude range")
    axes[1].set_title("(b) Top-3 and Top-5 ranges", loc="left", fontweight="bold")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    figure.tight_layout()
    figure.savefig(
        os.path.join(out_dir, "annual_top3_top5_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def build_delta_tables(final_metrics, out_dir):
    key = ["horizon_month", "model_name"]
    top3 = final_metrics[
        final_metrics["feature_set_name"] == "core_plus_annual_top3"
    ].copy()
    top5 = final_metrics[
        final_metrics["feature_set_name"] == "core_plus_annual_top5"
    ].copy()
    if top3.duplicated(key).any() or top5.duplicated(key).any():
        raise ValueError("Duplicate final metric rows prevent matched comparison")

    columns = [
        "horizon_month",
        "model_name",
        "config_name",
        "paired_final_seed",
        "n_features",
        "threshold",
        "auc",
        "pr_auc",
        "brier",
        "f1",
    ]
    matched = top3[columns].merge(
        top5[columns],
        on=key,
        suffixes=("_top3", "_top5"),
        validate="one_to_one",
    )
    for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        matched[f"delta_top5_minus_top3_{metric}"] = (
            matched[f"{metric}_top5"] - matched[f"{metric}_top3"]
        )
        direction = 1.0 if metric in HIGHER_IS_BETTER else -1.0
        matched[f"top5_improvement_{metric}"] = (
            direction * matched[f"delta_top5_minus_top3_{metric}"]
        )

    delta_path = os.path.join(out_dir, "annual_top3_top5_delta_performance.csv")
    detail_path = os.path.join(
        out_dir, "annual_top3_top5_model_horizon_detail.csv"
    )
    matched.to_csv(delta_path, index=False)
    matched.to_csv(detail_path, index=False)

    summary_rows = []
    for horizon, subset in matched.groupby("horizon_month"):
        row = {"horizon_month": int(horizon), "n_models": int(len(subset))}
        for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
            delta_column = f"delta_top5_minus_top3_{metric}"
            row[f"mean_{delta_column}"] = float(subset[delta_column].mean())
            row[f"min_{delta_column}"] = float(subset[delta_column].min())
            row[f"max_{delta_column}"] = float(subset[delta_column].max())
            row[f"n_models_top5_better_{metric}"] = int(
                (subset[f"top5_improvement_{metric}"] > 0).sum()
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(
        out_dir, "annual_top3_top5_performance_summary.csv"
    )
    summary.to_csv(summary_path, index=False)
    plot_performance(final_metrics, out_dir)
    return matched, summary, delta_path, detail_path, summary_path


def plot_performance(final_metrics, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Skip performance plot: {exc}", flush=True)
        return
    metrics = [("auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("brier", "Brier"), ("f1", "F1")]
    conditions = ["core_plus_annual_top3", "core_plus_annual_top5"]
    labels = {conditions[0]: "Top3", conditions[1]: "Top5"}
    colors = {conditions[0]: "#4C78A8", conditions[1]: "#E68643"}
    aggregated = (
        final_metrics.groupby(["horizon_month", "feature_set_name"], as_index=False)[
            [metric for metric, _ in metrics]
        ]
        .mean()
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.6, 5.8), sharex=True)
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        for condition in conditions:
            subset = aggregated[aggregated["feature_set_name"] == condition].sort_values(
                "horizon_month"
            )
            axis.plot(
                subset["horizon_month"],
                subset[metric],
                marker="o",
                linewidth=1.3,
                markersize=4,
                label=labels[condition],
                color=colors[condition],
            )
        axis.set_ylabel(label)
        axis.set_xticks(sorted(aggregated["horizon_month"].unique()))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_xlabel("Forecast horizon (months)")
    axes[1, 1].set_xlabel("Forecast horizon (months)")
    figure.suptitle("Matched compact-core retraining (mean across models)", fontsize=10)
    figure.tight_layout()
    figure.savefig(
        os.path.join(out_dir, "annual_top3_top5_performance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_report(out_dir, manifest, distribution_info, final_metrics, matched, summary):
    path = os.path.join(out_dir, "annual_top3_top5_report.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Annual Top-3 versus Top-5 validation\n\n")
        handle.write(
            "The matched conditions were defined from information groups frozen "
            "using inner-validation SHAP. The final test was not used for feature "
            "definition or threshold selection.\n\n"
        )
        handle.write("## Distribution audit\n\n")
        handle.write(
            f"- Unique cell-origin samples: {distribution_info['n_unique_cell_origins']}\n"
        )
        handle.write(
            f"- Samples with M5 present: {distribution_info['n_unique_m5_present']}\n"
        )
        handle.write(
            "- Availability definition: "
            + "; ".join(distribution_info["availability_sources"])
            + "\n"
        )
        handle.write(
            f"- Sample-level Parquet: {distribution_info['sample_parquet_status']}\n\n"
        )
        if not final_metrics.empty:
            handle.write("## Matched performance deltas\n\n")
            handle.write(
                "Raw deltas are Top5 minus Top3. Positive values favor Top5 for "
                "ROC-AUC, PR-AUC and F1; negative values favor Top5 for Brier.\n\n"
            )
            display_columns = [
                "horizon_month",
                "model_name",
                "delta_top5_minus_top3_auc",
                "delta_top5_minus_top3_pr_auc",
                "delta_top5_minus_top3_brier",
                "delta_top5_minus_top3_f1",
            ]
            handle.write("```text\n")
            handle.write(matched[display_columns].to_string(index=False))
            handle.write("\n```\n\n")
            handle.write("## Across-model horizon summary\n\n```text\n")
            handle.write(summary.to_string(index=False))
            handle.write("\n```\n\n")
        handle.write("## Interpretation\n\n")
        handle.write(
            "- Consistently better Top5 results indicate incremental information "
            "from M4-M5 beyond the shared core and M1-M3.\n"
        )
        handle.write(
            "- Near-zero deltas indicate that M4-M5 are largely redundant for "
            "forecast performance, even if M5 has high SHAP attribution.\n"
        )
        handle.write(
            "- Consistently worse Top5 results do not support a claim of predictive "
            "gain from expanding the annual upper tail from three to five events.\n\n"
        )
        handle.write("## Reproducibility manifest\n\n```json\n")
        handle.write(json.dumps(manifest, indent=2, ensure_ascii=False))
        handle.write("\n```\n")
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annual Top-3 versus Top-5 distribution and matched retraining"
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw282",
    )
    parser.add_argument(
        "--frozen-groups-csv",
        default="outputs/inner_validation_shap/selected_groups_inner_validation.csv",
    )
    parser.add_argument(
        "--comparison-dir",
        default="outputs/discovery_models",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/annual_top3_vs_top5",
    )
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--models", default="extratrees,catboost,xgboost")
    parser.add_argument("--folds", default="1,2,3")
    parser.add_argument("--frozen-top-k", type=int, default=3)
    parser.add_argument("--group-col", default="feature_group")
    parser.add_argument("--annual-group", default="annual/magnitude_extreme")
    parser.add_argument(
        "--threshold-target",
        default="f1",
        choices=["f1", "f05", "precision", "recall", "acc"],
    )
    parser.add_argument(
        "--threshold-grid",
        default=(
            "0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,"
            "0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,"
            "0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38,0.40,"
            "0.45,0.50,0.55,0.60"
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--distribution-only", action="store_true")
    parser.add_argument("--skip-distribution", action="store_true")
    parser.add_argument(
        "--allow-default-configs",
        action="store_true",
        help="Allow copied built-in defaults if selected_config_by_model.json is unavailable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_dir = expand_path(args.data_dir)
    args.frozen_groups_csv = expand_path(args.frozen_groups_csv)
    args.comparison_dir = expand_path(args.comparison_dir)
    args.out_dir = expand_path(args.out_dir)
    args.horizons = parse_int_list(args.horizons)
    args.models = [normalize_model_name(item) for item in parse_str_list(args.models)]
    args.folds = parse_int_list(args.folds)
    args.threshold_grid = parse_float_list(args.threshold_grid)

    if not args.horizons or not args.folds:
        raise ValueError("At least one horizon and one inner fold are required")
    if args.frozen_top_k < 1:
        raise ValueError("--frozen-top-k must be positive")
    if args.distribution_only and args.skip_distribution:
        raise ValueError("--distribution-only and --skip-distribution are incompatible")

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "predictions"))
    ensure_dir(os.path.join(args.out_dir, "models"))
    ensure_dir(os.path.join(args.out_dir, "threshold_curves"))

    frozen_groups = load_frozen_groups(
        args.frozen_groups_csv,
        args.horizons,
        args.group_col,
        args.frozen_top_k,
    )
    frozen_hash = file_sha256(args.frozen_groups_csv)
    selected_configs, selected_config_path = load_selected_configs(args.comparison_dir)

    manifest_path = os.path.join(args.out_dir, "annual_top3_top5_manifest.json")
    manifest = {
        "experiment": "annual_top3_vs_top5_matched_compact_augmentation",
        "data_dir": args.data_dir,
        "frozen_groups_csv": args.frozen_groups_csv,
        "frozen_groups_sha256": frozen_hash,
        "comparison_dir": args.comparison_dir,
        "selected_config_json": selected_config_path,
        "horizons": args.horizons,
        "models": args.models,
        "folds": args.folds,
        "frozen_top_k": args.frozen_top_k,
        "group_col": args.group_col,
        "annual_group": args.annual_group,
        "threshold_target": args.threshold_target,
        "threshold_grid": args.threshold_grid,
        "seed": args.seed,
        "paired_seed_within_model_horizon": True,
        "selection_source": "inner_validation_frozen_csv",
        "used_final_test_for_feature_selection": False,
        "used_final_test_for_threshold_selection": False,
        "target": "y_m5 (fallback: y_class > 0)",
        "top3_condition": ANNUAL_MAGNITUDE_NAMES[:3],
        "top5_condition": ANNUAL_MAGNITUDE_NAMES,
    }
    if os.path.exists(manifest_path) and args.resume:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        stable_keys = [
            "frozen_groups_sha256",
            "horizons",
            "models",
            "folds",
            "frozen_top_k",
            "annual_group",
            "threshold_target",
            "threshold_grid",
            "seed",
        ]
        mismatch = {
            key: [previous.get(key), manifest.get(key)]
            for key in stable_keys
            if previous.get(key) != manifest.get(key)
        }
        if mismatch:
            raise RuntimeError(
                "Refusing to resume because the manifest changed: "
                + json.dumps(mismatch, ensure_ascii=False)
            )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    frozen_used_path = os.path.join(args.out_dir, "frozen_selected_groups_used.csv")
    frozen_groups.to_csv(frozen_used_path, index=False)
    distribution_parts = []
    feature_set_parts = []

    inner_path = os.path.join(args.out_dir, "annual_top3_top5_inner_valid_metrics.csv")
    threshold_path = os.path.join(args.out_dir, "annual_top3_top5_thresholds.csv")
    final_path = os.path.join(args.out_dir, "annual_top3_top5_performance.csv")

    for horizon in args.horizons:
        print(f"\nLoad H={horizon}m", flush=True)
        X, y, meta, payload = load_dataset(args.data_dir, horizon)
        feature_meta = load_feature_metadata(args.data_dir, X.shape[1])
        if args.group_col not in feature_meta.columns:
            raise ValueError(f"Feature metadata has no {args.group_col} column")
        if "split" not in meta.columns:
            raise ValueError("Dataset metadata lacks required split column")

        if not args.skip_distribution:
            distribution_parts.append(
                extract_distribution_rows(X, meta, feature_meta, horizon)
            )

        frozen_horizon = frozen_groups[
            frozen_groups["horizon_month"] == int(horizon)
        ].copy()
        feature_masks, feature_sets, base_groups = build_matched_feature_sets(
            feature_meta,
            frozen_horizon,
            args.group_col,
            args.annual_group,
        )
        feature_sets["horizon_month"] = int(horizon)
        feature_sets["selection_source"] = "inner_validation_frozen_csv"
        feature_sets["frozen_groups_sha256"] = frozen_hash
        feature_set_parts.append(feature_sets)
        print(
            f"H={horizon} common base groups: {'; '.join(base_groups)}; "
            + ", ".join(
                f"{name} n={len(indices)}" for name, indices in feature_masks.items()
            ),
            flush=True,
        )

        if args.distribution_only:
            del X, y, meta, payload
            gc.collect()
            continue

        test_mask = meta["split"].astype(str).eq("test").to_numpy()
        train_pool_mask = meta["split"].astype(str).eq("train_pool").to_numpy()
        if not test_mask.any() or not train_pool_mask.any():
            raise ValueError(
                f"H={horizon} has empty train_pool/test: "
                f"{int(train_pool_mask.sum())}/{int(test_mask.sum())}"
            )
        X_test = X[test_mask]
        y_test = y[test_mask]
        test_meta = meta.loc[test_mask].reset_index(drop=True).copy()
        X_train_pool = X[train_pool_mask]
        y_train_pool = y[train_pool_mask]

        for model_index, model_name in enumerate(args.models):
            config = get_model_config(
                model_name, selected_configs, args.allow_default_configs
            )
            config_name = config["config_name"]
            params = config["params"]
            paired_final_seed = int(args.seed + 10000 * int(horizon) + 1000 * model_index)

            for feature_set_name, feature_indices in feature_masks.items():
                feature_indices = np.asarray(feature_indices, dtype=int)
                completed = False
                if args.resume and os.path.exists(final_path):
                    old = pd.read_csv(final_path)
                    completed = bool(
                        (
                            old["horizon_month"].astype(int).eq(horizon)
                            & old["model_name"].astype(str).eq(model_name)
                            & old["feature_set_name"].astype(str).eq(feature_set_name)
                        ).any()
                    )
                if completed:
                    print(
                        f"[RESUME] H={horizon} {model_name} {feature_set_name}",
                        flush=True,
                    )
                    continue

                conditions = {
                    "horizon_month": horizon,
                    "model_name": model_name,
                    "feature_set_name": feature_set_name,
                }
                remove_matching_rows(inner_path, conditions)
                remove_matching_rows(threshold_path, conditions)

                valid_y_parts = []
                valid_score_parts = []
                for fold in args.folds:
                    role_column = f"fold{fold}_role"
                    if role_column not in meta.columns:
                        raise ValueError(f"Missing fold role column: {role_column}")
                    train_mask = meta[role_column].astype(str).eq("inner_train").to_numpy()
                    valid_mask = meta[role_column].astype(str).eq("inner_valid").to_numpy()
                    if not train_mask.any() or not valid_mask.any():
                        raise ValueError(
                            f"H={horizon} fold={fold} has empty inner train/valid"
                        )
                    if bool(meta.loc[valid_mask, "split"].astype(str).eq("test").any()):
                        raise RuntimeError(
                            f"Leakage guard: H={horizon} fold={fold} validation contains test rows"
                        )
                    fold_seed = paired_final_seed + int(fold)
                    model = make_model(model_name, params, fold_seed, args.n_jobs)
                    model.fit(X[train_mask][:, feature_indices], y[train_mask])
                    valid_score = get_score(model, X[valid_mask][:, feature_indices])
                    valid_y = y[valid_mask]
                    valid_y_parts.append(valid_y)
                    valid_score_parts.append(valid_score)
                    metrics05, _ = evaluate_binary(valid_y, valid_score, 0.5)
                    row = {
                        "horizon_month": horizon,
                        "model_name": model_name,
                        "config_name": config_name,
                        "feature_set_name": feature_set_name,
                        "fold": fold,
                        "paired_final_seed": paired_final_seed,
                        "fold_seed": fold_seed,
                        "n_features": int(len(feature_indices)),
                        "selection_source": "inner_validation_frozen_csv",
                        "frozen_groups_sha256": frozen_hash,
                    }
                    row.update(
                        {f"valid05_{key}": value for key, value in metrics05.items()}
                    )
                    append_row_csv(inner_path, row)
                    del model, valid_score, valid_y
                    gc.collect()

                pooled_y = np.concatenate(valid_y_parts)
                pooled_score = np.concatenate(valid_score_parts)
                threshold, curve = choose_threshold(
                    pooled_y,
                    pooled_score,
                    args.threshold_grid,
                    args.threshold_target,
                )
                curve_path = os.path.join(
                    args.out_dir,
                    "threshold_curves",
                    f"threshold_{safe_name(model_name)}_H{horizon}m_{feature_set_name}.csv",
                )
                curve.to_csv(curve_path, index=False)
                pooled_metrics, _ = evaluate_binary(pooled_y, pooled_score, threshold)
                threshold_row = {
                    "horizon_month": horizon,
                    "model_name": model_name,
                    "config_name": config_name,
                    "feature_set_name": feature_set_name,
                    "n_features": int(len(feature_indices)),
                    "paired_final_seed": paired_final_seed,
                    "threshold_target": args.threshold_target,
                    "selected_threshold": threshold,
                    "threshold_curve_csv": curve_path,
                    "selection_source": "inner_validation_frozen_csv",
                    "frozen_groups_sha256": frozen_hash,
                }
                threshold_row.update(
                    {f"pooled_valid_{key}": value for key, value in pooled_metrics.items()}
                )
                append_row_csv(threshold_path, threshold_row)

                final_model = make_model(
                    model_name, params, paired_final_seed, args.n_jobs
                )
                final_model.fit(X_train_pool[:, feature_indices], y_train_pool)
                test_score = get_score(final_model, X_test[:, feature_indices])
                test_metrics, test_prediction = evaluate_binary(
                    y_test, test_score, threshold
                )
                prediction_path = os.path.join(
                    args.out_dir,
                    "predictions",
                    f"test_{safe_name(model_name)}_H{horizon}m_{feature_set_name}.csv.gz",
                )
                prediction_table = test_meta.copy()
                prediction_table.insert(
                    0, "dataset_test_row", np.arange(len(test_meta), dtype=int)
                )
                prediction_table["y_true"] = y_test
                prediction_table["score_m5"] = test_score
                prediction_table["y_pred"] = test_prediction
                prediction_table["threshold"] = threshold
                prediction_table.to_csv(
                    prediction_path, index=False, compression="gzip"
                )

                model_path = ""
                if args.save_models:
                    import joblib

                    model_path = os.path.join(
                        args.out_dir,
                        "models",
                        f"{safe_name(model_name)}_H{horizon}m_{feature_set_name}.joblib",
                    )
                    joblib.dump(final_model, model_path)

                set_row = feature_sets[
                    feature_sets["feature_set_name"] == feature_set_name
                ].iloc[0]
                final_row = {
                    "horizon_month": horizon,
                    "model_name": model_name,
                    "config_name": config_name,
                    "feature_set_name": feature_set_name,
                    "feature_set_type": set_row["feature_set_type"],
                    "base_groups": set_row["base_groups"],
                    "annual_features": set_row["annual_features"],
                    "n_features": int(len(feature_indices)),
                    "paired_final_seed": paired_final_seed,
                    "threshold_target": args.threshold_target,
                    "threshold": threshold,
                    "selection_source": "inner_validation_frozen_csv",
                    "frozen_groups_sha256": frozen_hash,
                    "used_final_test_for_selection": False,
                    "prediction_csv": prediction_path,
                    "model_file": model_path,
                }
                final_row.update(test_metrics)
                append_row_csv(final_path, final_row)
                print(
                    f"FINAL H={horizon} {model_name} {feature_set_name}: "
                    f"AUC={test_metrics['auc']:.6f} PR-AUC={test_metrics['pr_auc']:.6f} "
                    f"Brier={test_metrics['brier']:.6f} F1={test_metrics['f1']:.6f}",
                    flush=True,
                )
                del final_model, test_score, test_prediction
                gc.collect()

        del X, y, meta, payload, X_test, y_test, X_train_pool, y_train_pool, test_meta
        gc.collect()

    feature_sets_all = pd.concat(feature_set_parts, ignore_index=True, sort=False)
    feature_sets_path = os.path.join(args.out_dir, "annual_top3_top5_feature_sets.csv")
    feature_sets_all.to_csv(feature_sets_path, index=False)

    distribution_info = {
        "n_horizon_rows": 0,
        "n_unique_cell_origins": 0,
        "n_unique_m5_present": 0,
        "availability_sources": [],
        "sample_parquet_status": "skipped",
    }
    if distribution_parts:
        distribution_info = write_distribution_outputs(
            pd.concat(distribution_parts, ignore_index=True, sort=False),
            args.out_dir,
        )
    manifest["distribution_outputs"] = distribution_info
    manifest["feature_sets_csv"] = feature_sets_path
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    if args.distribution_only:
        print("Distribution-only run finished", flush=True)
        return

    final_metrics = pd.read_csv(final_path)
    matched, summary, delta_path, detail_path, summary_path = build_delta_tables(
        final_metrics, args.out_dir
    )
    report_path = write_report(
        args.out_dir,
        manifest,
        distribution_info,
        final_metrics,
        matched,
        summary,
    )
    print("\nAnnual Top-3 versus Top-5 experiment finished", flush=True)
    for path in [
        final_path,
        delta_path,
        detail_path,
        summary_path,
        report_path,
        manifest_path,
    ]:
        print("Saved:", path, flush=True)


if __name__ == "__main__":
    main()
