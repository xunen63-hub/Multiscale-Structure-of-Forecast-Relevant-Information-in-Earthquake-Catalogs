#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_feature_group_ablation.py

Purpose
-------
Confirmatory feature-group retention/removal experiment for binary M>=5
regional earthquake forecasting.

The feature groups are NOT selected inside this script. They are read from the
frozen CSV produced exclusively from inner rolling-validation SHAP consensus:

    selected_groups_inner_validation.csv

This prevents the final test period from participating in group definition.
For every model-horizon pair, the full and restricted feature sets use the same
training samples, hyperparameters, and paired random seed.

Default design
--------------
Models:   ExtraTrees + CatBoost + XGBoost
Horizons: 1, 3, 6, and 12 months
Feature sets:
    full_features
    only_top1_groups / only_top2_groups / only_top3_groups
    remove_top1_groups / remove_top2_groups / remove_top3_groups

Outputs
-------
- frozen_selected_groups_used.csv
- ablation_feature_sets.csv
- ablation_inner_valid_metrics.csv
- ablation_thresholds.csv
- ablation_final_test_metrics.csv
- ablation_deltas_vs_full.csv
- ablation_manifest.json
- ablation_feature_group_report.md
- predictions/*.csv
- models/*.joblib (optional)
"""

import os
import gc
import json
import math
import pickle
import argparse
import warnings
import hashlib
from pathlib import Path
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception as e:
    CatBoostClassifier = None
    HAS_CATBOOST = False
    CATBOOST_IMPORT_ERROR = repr(e)
else:
    CATBOOST_IMPORT_ERROR = ""

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception as e:
    XGBClassifier = None
    HAS_XGBOOST = False
    XGBOOST_IMPORT_ERROR = repr(e)
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


def expand_path(p):
    return str(Path(p).expanduser().resolve())


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def parse_int_list(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_float_list(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_str_list(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def normalize_model_name(x):
    key = str(x).strip().lower().replace(" ", "_")
    if key not in MODEL_ALIAS:
        raise ValueError(f"Unknown model name: {x}")
    return MODEL_ALIAS[key]


def safe_name(x):
    return str(x).lower().replace(" ", "_").replace("/", "__").replace("+", "plus")


def dataset_path(data_dir, horizon):
    return os.path.join(data_dir, f"china_raw282_dataset_horizon_{horizon}m.pkl")


def load_dataset(data_dir, horizon):
    p = dataset_path(data_dir, horizon)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with open(p, "rb") as f:
        d = pickle.load(f)
    X = np.asarray(d["X"], dtype=np.float32)
    if "y_m5" in d:
        y = np.asarray(d["y_m5"], dtype=np.int64)
    else:
        y = (np.asarray(d["y_class"], dtype=np.int64) > 0).astype(np.int64)
    meta = d["meta"].copy()
    return X, y, meta, d


def load_feature_metadata(data_dir, n_features):
    p = os.path.join(data_dir, "feature_metadata_282_relative_lag.csv")
    if not os.path.exists(p):
        print(f"[WARN] Missing feature metadata: {p}. Use generic feature groups.", flush=True)
        return pd.DataFrame({
            "feature_idx": np.arange(n_features, dtype=int),
            "feature_name": [f"feature_{i:03d}" for i in range(n_features)],
            "time_group": "unknown",
            "signal_group": "unknown",
            "feature_group": "unknown/unknown",
        })
    m = pd.read_csv(p)
    if "feature_idx" not in m.columns:
        m.insert(0, "feature_idx", np.arange(len(m), dtype=int))
    m = m.sort_values("feature_idx").reset_index(drop=True)
    if len(m) != n_features:
        raise ValueError(f"feature metadata length {len(m)} != n_features {n_features}")
    if "feature_name" not in m.columns:
        m["feature_name"] = [f"feature_{i:03d}" for i in range(n_features)]
    if "time_group" not in m.columns:
        m["time_group"] = "unknown"
    if "signal_group" not in m.columns:
        m["signal_group"] = "unknown"
    if "feature_group" not in m.columns:
        m["feature_group"] = m["time_group"].fillna("unknown").astype(str) + "/" + m["signal_group"].fillna("unknown").astype(str)
    return m



def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def load_frozen_groups(frozen_csv, horizons, group_col, top_k_max):
    if not os.path.exists(frozen_csv):
        raise FileNotFoundError(frozen_csv)

    df = pd.read_csv(frozen_csv)
    required = {"horizon_month", "rank", "group_name"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Frozen-group CSV is missing required columns: {missing}")

    df = df.copy()
    df["horizon_month"] = pd.to_numeric(df["horizon_month"], errors="raise").astype(int)
    df["rank"] = pd.to_numeric(df["rank"], errors="raise").astype(int)
    df["group_name"] = df["group_name"].astype(str).str.strip()

    if (df["group_name"] == "").any():
        raise ValueError("Frozen-group CSV contains an empty group_name")
    if df.duplicated(["horizon_month", "rank"]).any():
        dup = df[df.duplicated(["horizon_month", "rank"], keep=False)]
        raise ValueError(
            "Frozen-group CSV contains duplicate horizon/rank rows:\n"
            + dup[["horizon_month", "rank", "group_name"]].to_string(index=False)
        )

    if "source_stage" in df.columns:
        bad = ~df["source_stage"].astype(str).str.strip().str.lower().eq("inner_validation")
        if bad.any():
            raise ValueError("Frozen groups were not selected exclusively from inner_validation")

    if "used_final_test_for_selection" in df.columns:
        if bool(_to_bool_series(df["used_final_test_for_selection"]).any()):
            raise RuntimeError("Leakage guard: frozen groups used final-test data for selection")

    if "complete_model_fold_coverage" in df.columns:
        if not bool(_to_bool_series(df["complete_model_fold_coverage"]).all()):
            raise ValueError("Frozen groups do not have complete model-fold coverage")

    if "group_col" in df.columns:
        observed = set(df["group_col"].dropna().astype(str).str.strip())
        if observed and observed != {group_col}:
            raise ValueError(
                f"Frozen CSV group_col values {sorted(observed)} do not match requested {group_col}"
            )

    requested_horizons = set(map(int, horizons))
    available_horizons = set(df["horizon_month"].unique().tolist())
    missing_horizons = sorted(requested_horizons - available_horizons)
    if missing_horizons:
        raise ValueError(f"Frozen-group CSV is missing horizons: {missing_horizons}")

    selected_rows = []
    for horizon in sorted(requested_horizons):
        sub = df[df["horizon_month"] == horizon].sort_values("rank").copy()
        expected_ranks = list(range(1, int(top_k_max) + 1))
        actual_ranks = sub[sub["rank"] <= top_k_max]["rank"].tolist()
        if actual_ranks != expected_ranks:
            raise ValueError(
                f"H={horizon}m frozen ranks are {actual_ranks}; expected {expected_ranks}"
            )
        selected_rows.append(sub[sub["rank"] <= top_k_max])

    selected = pd.concat(selected_rows, ignore_index=True, sort=False)
    selected["frozen_groups_csv"] = str(Path(frozen_csv).resolve())
    selected["frozen_groups_sha256"] = file_sha256(frozen_csv)
    selected["selection_source"] = "inner_validation_frozen_csv"
    selected["used_final_test_for_selection"] = False
    return selected.sort_values(["horizon_month", "rank"]).reset_index(drop=True)



def build_feature_sets(feature_meta, selected_group_table, top_k_list, group_col):
    all_features = np.asarray(feature_meta["feature_idx"].astype(int).values, dtype=int)
    group_to_features = {
        str(group_name): np.asarray(sub["feature_idx"].astype(int).values, dtype=int)
        for group_name, sub in feature_meta.groupby(group_col)
    }

    ranked_groups = (
        selected_group_table.sort_values("rank")["group_name"].astype(str).tolist()
    )
    if len(ranked_groups) < max(top_k_list):
        raise ValueError(
            f"Only {len(ranked_groups)} frozen groups are available, "
            f"but Top-{max(top_k_list)} was requested"
        )

    missing_groups = [group for group in ranked_groups if group not in group_to_features]
    if missing_groups:
        raise ValueError(
            f"Frozen groups are absent from feature metadata column {group_col}: {missing_groups}"
        )

    rows = []
    masks = {"full_features": all_features}
    rows.append({
        "feature_set_name": "full_features",
        "feature_set_type": "full",
        "top_k_groups": 0,
        "groups_used": "ALL",
        "n_features": len(all_features),
    })

    for k in top_k_list:
        groups = ranked_groups[:k]
        feat = np.unique(
            np.concatenate([group_to_features[group] for group in groups])
        ).astype(int)
        if len(feat) == 0:
            raise ValueError(f"Top-{k} frozen groups produced an empty feature set")

        only_name = f"only_top{k}_groups"
        masks[only_name] = feat
        rows.append({
            "feature_set_name": only_name,
            "feature_set_type": "only_top_groups",
            "top_k_groups": int(k),
            "groups_used": "; ".join(groups),
            "n_features": len(feat),
        })

        remove_set = set(feat.tolist())
        remaining = np.asarray(
            [idx for idx in all_features if int(idx) not in remove_set], dtype=int
        )
        if len(remaining) == 0:
            raise ValueError(f"Removing Top-{k} groups removed every predictor")

        remove_name = f"remove_top{k}_groups"
        masks[remove_name] = remaining
        rows.append({
            "feature_set_name": remove_name,
            "feature_set_type": "remove_top_groups",
            "top_k_groups": int(k),
            "groups_used": "; ".join(groups),
            "n_features": len(remaining),
        })

    return masks, pd.DataFrame(rows)


def load_selected_configs_from_comparison(comparison_dir):
    p = os.path.join(comparison_dir, "selected_config_by_model.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r") as f:
            d = json.load(f)
        out = {}
        for cfg in d.get("selected_configs", []):
            if "model_name" in cfg:
                out[cfg["model_name"]] = {"config_name": cfg.get("config_name", "selected"), "params": cfg.get("params", {})}
        return out
    except Exception as e:
        print(f"[WARN] failed to read selected configs from {p}: {e}", flush=True)
        return {}


def get_model_config(model_name, selected_configs):
    if model_name in selected_configs and selected_configs[model_name].get("params"):
        return deepcopy(selected_configs[model_name])
    return deepcopy(DEFAULT_CONFIGS[model_name])


def make_model(model_name, params, seed, n_jobs):
    params = deepcopy(dict(params))
    if model_name == "ExtraTrees":
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
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(float)
        return proba.ravel().astype(float)
    pred = model.predict(X)
    return np.asarray(pred, dtype=float).ravel()


def safe_auc(y, score):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return np.nan


def safe_ap(y, score):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(average_precision_score(y, score))
    except Exception:
        return np.nan


def evaluate_binary(y_true, score, threshold):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    y_pred = (score >= float(threshold)).astype(int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    out = {
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true == 1)) if len(y_true) else np.nan,
        "threshold": float(threshold),
        "pred_positive_rate": float(np.mean(y_pred == 1)) if len(y_pred) else np.nan,
        "acc": float(accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f05": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "auc": safe_auc(y_true, score),
        "pr_auc": safe_ap(y_true, score),
        "brier": float(brier_score_loss(y_true, score)) if len(y_true) else np.nan,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
    return out, y_pred


def choose_threshold(y_true, score, thresholds, target="f1"):
    rows = []
    for th in thresholds:
        m, _ = evaluate_binary(y_true, score, th)
        rows.append(m)
    df = pd.DataFrame(rows)
    if target not in df.columns:
        raise ValueError(f"Unknown threshold target: {target}")
    df = df.sort_values([target, "threshold"], ascending=[False, True]).reset_index(drop=True)
    return float(df.loc[0, "threshold"]), df


def append_row_csv(path, row):
    df_new = pd.DataFrame([row])
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        cols = list(df_old.columns)
        for c in df_new.columns:
            if c not in cols:
                cols.append(c)
        df_old = df_old.reindex(columns=cols)
        df_new = df_new.reindex(columns=cols)
        pd.concat([df_old, df_new], ignore_index=True).to_csv(path, index=False)
    else:
        df_new.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
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
        default="outputs/feature_group_ablation",
    )

    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--models", default="extratrees,catboost,xgboost")
    parser.add_argument("--folds", default="1,2,3")
    parser.add_argument("--top-k-groups", default="1,2,3")
    parser.add_argument("--group-col", default="feature_group")
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument(
        "--allow-default-configs",
        action="store_true",
        help="Allow built-in model defaults when selected_config_by_model.json lacks a model.",
    )
    args = parser.parse_args()

    args.data_dir = expand_path(args.data_dir)
    args.frozen_groups_csv = expand_path(args.frozen_groups_csv)
    args.comparison_dir = expand_path(args.comparison_dir)
    args.out_dir = expand_path(args.out_dir)
    args.horizons = parse_int_list(args.horizons)
    args.models = [normalize_model_name(x) for x in parse_str_list(args.models)]
    args.folds = parse_int_list(args.folds)
    args.top_k_groups = parse_int_list(args.top_k_groups)
    args.threshold_grid = parse_float_list(args.threshold_grid)

    if not args.horizons:
        raise ValueError("No horizons were requested")
    if not args.models:
        raise ValueError("No models were requested")
    if not args.folds:
        raise ValueError("No inner folds were requested")
    if not args.top_k_groups or min(args.top_k_groups) < 1:
        raise ValueError("--top-k-groups must contain positive integers")
    if sorted(set(args.top_k_groups)) != sorted(args.top_k_groups):
        raise ValueError("--top-k-groups contains duplicate values")

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "predictions"))
    ensure_dir(os.path.join(args.out_dir, "models"))
    ensure_dir(os.path.join(args.out_dir, "threshold_curves"))

    frozen_groups = load_frozen_groups(
        args.frozen_groups_csv,
        horizons=args.horizons,
        group_col=args.group_col,
        top_k_max=max(args.top_k_groups),
    )
    frozen_hash = file_sha256(args.frozen_groups_csv)

    selected_configs = load_selected_configs_from_comparison(args.comparison_dir)
    missing_configs = [model for model in args.models if model not in selected_configs]
    if missing_configs and not args.allow_default_configs:
        raise FileNotFoundError(
            "selected_config_by_model.json does not contain configurations for: "
            f"{missing_configs}. Use --allow-default-configs only if this is intentional."
        )
    if missing_configs:
        print(f"[WARN] Use built-in defaults for models: {missing_configs}", flush=True)

    selected_group_path = os.path.join(args.out_dir, "frozen_selected_groups_used.csv")
    feature_set_path = os.path.join(args.out_dir, "ablation_feature_sets.csv")
    inner_metric_path = os.path.join(args.out_dir, "ablation_inner_valid_metrics.csv")
    threshold_path = os.path.join(args.out_dir, "ablation_thresholds.csv")
    final_metric_path = os.path.join(args.out_dir, "ablation_final_test_metrics.csv")
    delta_path = os.path.join(args.out_dir, "ablation_deltas_vs_full.csv")
    manifest_path = os.path.join(args.out_dir, "ablation_manifest.json")

    frozen_groups.to_csv(selected_group_path, index=False)

    manifest = {
        "experiment": "inner_validation_frozen_group_ablation",
        "data_dir": args.data_dir,
        "frozen_groups_csv": args.frozen_groups_csv,
        "frozen_groups_sha256": frozen_hash,
        "comparison_dir": args.comparison_dir,
        "out_dir": args.out_dir,
        "horizons": args.horizons,
        "models": args.models,
        "folds": args.folds,
        "top_k_groups": args.top_k_groups,
        "group_col": args.group_col,
        "threshold_target": args.threshold_target,
        "seed": args.seed,
        "selection_source": "inner_validation_frozen_csv",
        "used_final_test_for_selection": False,
        "paired_seed_within_model_horizon": True,
    }
    if os.path.exists(manifest_path) and args.resume:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            old_manifest = json.load(handle)
        keys = [
            "frozen_groups_sha256", "horizons", "models", "folds",
            "top_k_groups", "group_col", "threshold_target", "seed",
        ]
        mismatch = {
            key: (old_manifest.get(key), manifest.get(key))
            for key in keys
            if old_manifest.get(key) != manifest.get(key)
        }
        if mismatch:
            raise RuntimeError(
                "Refusing to resume because the experiment manifest changed: "
                + json.dumps(mismatch, ensure_ascii=False)
            )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print("=" * 100, flush=True)
    print("Frozen inner-validation feature-group ablation for binary M>=5", flush=True)
    print("data_dir:", args.data_dir, flush=True)
    print("frozen_groups_csv:", args.frozen_groups_csv, flush=True)
    print("frozen_groups_sha256:", frozen_hash, flush=True)
    print("out_dir:", args.out_dir, flush=True)
    print("horizons:", args.horizons, flush=True)
    print("models:", args.models, flush=True)
    print("group_col:", args.group_col, flush=True)
    print("top_k_groups:", args.top_k_groups, flush=True)
    print("threshold_target:", args.threshold_target, flush=True)
    print("=" * 100, flush=True)

    selected_group_rows = []
    feature_set_rows_all = []

    for horizon in args.horizons:
        print("\n" + "#" * 100, flush=True)
        print(f"Load H={horizon}m data", flush=True)
        X, y, meta, payload = load_dataset(args.data_dir, horizon)
        feature_meta = load_feature_metadata(args.data_dir, X.shape[1])
        if args.group_col not in feature_meta.columns:
            raise ValueError(f"Feature metadata has no group column: {args.group_col}")
        if "split" not in meta.columns:
            raise ValueError("Dataset metadata is missing required column: split")

        top_group_table = (
            frozen_groups[frozen_groups["horizon_month"] == int(horizon)]
            .sort_values("rank")
            .head(max(args.top_k_groups))
            .copy()
        )
        selected_group_rows.append(top_group_table)

        feature_masks, feature_set_df = build_feature_sets(
            feature_meta,
            top_group_table,
            args.top_k_groups,
            args.group_col,
        )
        feature_set_df["horizon_month"] = int(horizon)
        feature_set_df["selection_source"] = "inner_validation_frozen_csv"
        feature_set_df["frozen_groups_sha256"] = frozen_hash
        feature_set_rows_all.append(feature_set_df)

        test_mask = meta["split"].astype(str).eq("test").values
        train_pool_mask = meta["split"].astype(str).eq("train_pool").values
        if not np.any(test_mask) or not np.any(train_pool_mask):
            raise ValueError(
                f"H={horizon}m has empty train_pool or test split: "
                f"train_pool={int(train_pool_mask.sum())}, test={int(test_mask.sum())}"
            )

        X_test_all = X[test_mask]
        y_test = y[test_mask]
        test_meta = meta.loc[test_mask].reset_index(drop=True).copy()
        X_train_pool_all = X[train_pool_mask]
        y_train_pool = y[train_pool_mask]

        print(
            "X:", X.shape,
            "train_pool:", X_train_pool_all.shape,
            "test:", X_test_all.shape,
            flush=True,
        )
        print(
            "test positives:", int(np.sum(y_test == 1)),
            "positive_rate:", float(np.mean(y_test == 1)),
            flush=True,
        )
        print(
            "Frozen Top groups:",
            "; ".join(top_group_table["group_name"].astype(str).tolist()),
            flush=True,
        )

        for model_index, model_name in enumerate(args.models):
            config = get_model_config(model_name, selected_configs)
            config_name = config["config_name"]
            params = config["params"]
            paired_final_seed = int(args.seed + 10000 * int(horizon) + 1000 * model_index)

            print("\n" + "=" * 100, flush=True)
            print(
                f"Model={model_name}, config={config_name}, H={horizon}m, "
                f"paired_final_seed={paired_final_seed}",
                flush=True,
            )

            for feature_set_name, feature_idx in feature_masks.items():
                feature_idx = np.asarray(feature_idx, dtype=int)
                if len(feature_idx) == 0:
                    raise ValueError(f"Empty feature set: {feature_set_name}")

                if args.resume and os.path.exists(final_metric_path):
                    old = pd.read_csv(final_metric_path)
                    query = old[
                        (old["horizon_month"] == horizon)
                        & (old["model_name"] == model_name)
                        & (old["feature_set_name"] == feature_set_name)
                    ]
                    if len(query) > 0:
                        print(
                            f"[RESUME] Skip H={horizon} {model_name} {feature_set_name}",
                            flush=True,
                        )
                        continue

                print(
                    f"Feature set={feature_set_name}, n_features={len(feature_idx)}",
                    flush=True,
                )

                valid_y_all = []
                valid_score_all = []
                for fold in args.folds:
                    role_col = f"fold{fold}_role"
                    if role_col not in meta.columns:
                        message = f"Missing fold role column: {role_col}"
                        if args.skip_missing:
                            print("[SKIP]", message, flush=True)
                            continue
                        raise ValueError(message)

                    train_mask = meta[role_col].astype(str).eq("inner_train").values
                    valid_mask = meta[role_col].astype(str).eq("inner_valid").values
                    if not np.any(train_mask) or not np.any(valid_mask):
                        message = (
                            f"H={horizon} fold={fold} has empty inner_train/inner_valid"
                        )
                        if args.skip_missing:
                            print("[SKIP]", message, flush=True)
                            continue
                        raise ValueError(message)
                    if bool(meta.loc[valid_mask, "split"].astype(str).eq("test").any()):
                        raise RuntimeError(
                            f"Leakage guard: H={horizon} fold={fold} inner_valid contains test rows"
                        )

                    X_train = X[train_mask][:, feature_idx]
                    y_train = y[train_mask]
                    X_valid = X[valid_mask][:, feature_idx]
                    y_valid = y[valid_mask]
                    fold_seed = paired_final_seed + int(fold)

                    model = make_model(
                        model_name,
                        params,
                        seed=fold_seed,
                        n_jobs=args.n_jobs,
                    )
                    model.fit(X_train, y_train)
                    valid_score = get_score(model, X_valid)
                    valid_y_all.append(y_valid)
                    valid_score_all.append(valid_score)

                    fold_metrics, _ = evaluate_binary(y_valid, valid_score, 0.5)
                    row = {
                        "horizon_month": horizon,
                        "model_name": model_name,
                        "config_name": config_name,
                        "feature_set_name": feature_set_name,
                        "fold": fold,
                        "paired_final_seed": paired_final_seed,
                        "fold_seed": fold_seed,
                        "n_features": int(len(feature_idx)),
                        "selection_source": "inner_validation_frozen_csv",
                        "frozen_groups_sha256": frozen_hash,
                    }
                    row.update({f"valid05_{key}": value for key, value in fold_metrics.items()})
                    append_row_csv(inner_metric_path, row)

                    del model, X_train, y_train, X_valid, y_valid, valid_score
                    gc.collect()

                if not valid_y_all:
                    raise RuntimeError(
                        f"No valid inner folds for H={horizon} {model_name} {feature_set_name}"
                    )

                pooled_valid_y = np.concatenate(valid_y_all)
                pooled_valid_score = np.concatenate(valid_score_all)
                threshold, curve = choose_threshold(
                    pooled_valid_y,
                    pooled_valid_score,
                    args.threshold_grid,
                    target=args.threshold_target,
                )
                curve_file = os.path.join(
                    args.out_dir,
                    "threshold_curves",
                    f"threshold_curve_{safe_name(model_name)}_H{horizon}m_{feature_set_name}.csv",
                )
                curve.to_csv(curve_file, index=False)
                valid_metrics, _ = evaluate_binary(
                    pooled_valid_y, pooled_valid_score, threshold
                )
                threshold_row = {
                    "horizon_month": horizon,
                    "model_name": model_name,
                    "config_name": config_name,
                    "feature_set_name": feature_set_name,
                    "n_features": int(len(feature_idx)),
                    "paired_final_seed": paired_final_seed,
                    "threshold_target": args.threshold_target,
                    "selected_threshold": float(threshold),
                    "threshold_curve_csv": curve_file,
                    "selection_source": "inner_validation_frozen_csv",
                    "frozen_groups_sha256": frozen_hash,
                }
                threshold_row.update(
                    {f"pooled_valid_{key}": value for key, value in valid_metrics.items()}
                )
                append_row_csv(threshold_path, threshold_row)

                final_model = make_model(
                    model_name,
                    params,
                    seed=paired_final_seed,
                    n_jobs=args.n_jobs,
                )
                final_model.fit(X_train_pool_all[:, feature_idx], y_train_pool)
                test_score = get_score(final_model, X_test_all[:, feature_idx])
                test_metrics, test_prediction = evaluate_binary(
                    y_test, test_score, threshold
                )

                prediction_file = os.path.join(
                    args.out_dir,
                    "predictions",
                    f"predictions_test_{safe_name(model_name)}_H{horizon}m_{feature_set_name}.csv",
                )
                prediction_table = test_meta.copy()
                prediction_table.insert(0, "dataset_test_row", np.arange(len(y_test), dtype=int))
                prediction_table["y_true"] = y_test.astype(int)
                prediction_table["score_m5"] = test_score.astype(float)
                prediction_table["y_pred"] = test_prediction.astype(int)
                prediction_table["threshold"] = float(threshold)
                prediction_table.to_csv(prediction_file, index=False)

                model_file = ""
                if args.save_models:
                    model_file = os.path.join(
                        args.out_dir,
                        "models",
                        f"model_{safe_name(model_name)}_H{horizon}m_{feature_set_name}.joblib",
                    )
                    joblib.dump(final_model, model_file)

                feature_set_meta = feature_set_df[
                    feature_set_df["feature_set_name"] == feature_set_name
                ].iloc[0].to_dict()
                final_row = {
                    "horizon_month": horizon,
                    "model_name": model_name,
                    "config_name": config_name,
                    "feature_set_name": feature_set_name,
                    "feature_set_type": feature_set_meta.get("feature_set_type"),
                    "top_k_groups": feature_set_meta.get("top_k_groups"),
                    "groups_used": feature_set_meta.get("groups_used"),
                    "n_features": int(len(feature_idx)),
                    "paired_final_seed": paired_final_seed,
                    "threshold_target": args.threshold_target,
                    "threshold": float(threshold),
                    "selection_source": "inner_validation_frozen_csv",
                    "frozen_groups_csv": args.frozen_groups_csv,
                    "frozen_groups_sha256": frozen_hash,
                    "used_final_test_for_selection": False,
                    "prediction_csv": prediction_file,
                    "model_file": model_file,
                }
                final_row.update(test_metrics)
                append_row_csv(final_metric_path, final_row)

                print(
                    f"FINAL H={horizon} {model_name} {feature_set_name}: "
                    f"AUC={test_metrics['auc']:.6f} PR-AUC={test_metrics['pr_auc']:.6f} "
                    f"threshold={threshold:.3f} F1={test_metrics['f1']:.4f}",
                    flush=True,
                )

                del final_model, test_score, test_prediction
                gc.collect()

        del X, y, meta, payload, X_test_all, X_train_pool_all, test_meta
        gc.collect()

    selected_groups_all = (
        pd.concat(selected_group_rows, ignore_index=True, sort=False)
        if selected_group_rows else pd.DataFrame()
    )
    feature_sets_all = (
        pd.concat(feature_set_rows_all, ignore_index=True, sort=False)
        if feature_set_rows_all else pd.DataFrame()
    )
    selected_groups_all.to_csv(selected_group_path, index=False)
    feature_sets_all.to_csv(feature_set_path, index=False)

    final_df = pd.read_csv(final_metric_path) if os.path.exists(final_metric_path) else pd.DataFrame()
    delta_df = pd.DataFrame()
    if not final_df.empty:
        full = final_df[final_df["feature_set_name"] == "full_features"][
            ["horizon_month", "model_name", "auc", "pr_auc", "brier"]
        ].rename(
            columns={"auc": "full_auc", "pr_auc": "full_pr_auc", "brier": "full_brier"}
        )
        delta_df = final_df.merge(full, on=["horizon_month", "model_name"], how="left")
        delta_df["delta_auc_vs_full"] = delta_df["auc"] - delta_df["full_auc"]
        delta_df["delta_pr_auc_vs_full"] = delta_df["pr_auc"] - delta_df["full_pr_auc"]
        delta_df["delta_brier_vs_full"] = delta_df["brier"] - delta_df["full_brier"]
        delta_df.to_csv(delta_path, index=False)

    report_path = os.path.join(args.out_dir, "ablation_feature_group_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Inner-validation frozen feature-group ablation\n\n")
        handle.write(
            "Groups were read directly from the frozen inner-validation CSV. "
            "No SHAP result or final-test metric was used to reselect groups in this run.\n\n"
        )
        handle.write("## Configuration\n\n```json\n")
        handle.write(json.dumps(manifest, indent=2, ensure_ascii=False))
        handle.write("\n```\n\n")
        handle.write("## Frozen groups used\n\n")
        group_columns = [
            column for column in [
                "horizon_month", "rank", "group_name", "n_features",
                "mean_group_importance", "n_models", "n_folds", "n_runs",
            ] if column in selected_groups_all.columns
        ]
        handle.write(selected_groups_all[group_columns].to_markdown(index=False))
        handle.write("\n\n## Key outputs\n\n")
        handle.write(f"- frozen groups used: `{selected_group_path}`\n")
        handle.write(f"- feature sets: `{feature_set_path}`\n")
        handle.write(f"- final test metrics: `{final_metric_path}`\n")
        handle.write(f"- deltas versus matched full: `{delta_path}`\n")
        handle.write(f"- manifest: `{manifest_path}`\n")

        if not delta_df.empty:
            handle.write("\n## ROC-AUC changes versus matched full model\n\n")
            show = delta_df[
                [
                    "horizon_month", "model_name", "feature_set_name",
                    "n_features", "groups_used", "auc", "full_auc",
                    "delta_auc_vs_full",
                ]
            ].sort_values(["horizon_month", "model_name", "feature_set_name"])
            handle.write(show.to_markdown(index=False))
            handle.write("\n")

    print("\n" + "=" * 100, flush=True)
    print("FROZEN FEATURE-GROUP ABLATION FINISHED", flush=True)
    print("Saved:", selected_group_path, flush=True)
    print("Saved:", feature_set_path, flush=True)
    print("Saved:", final_metric_path, flush=True)
    print("Saved:", delta_path, flush=True)
    print("Saved report:", report_path, flush=True)


if __name__ == "__main__":
    main()
