#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_inner_validation_shap_discovery.py

Leakage-safe discovery of forecast-relevant feature groups.

Scientific role
---------------
This script discovers SHAP-ranked feature groups exclusively from the inner
rolling-validation folds. It never uses the final test split to select feature
groups. The frozen group list produced here is intended for confirmatory
retention/removal experiments on the final test period.

Workflow
--------
For each forecast horizon:
  1. For each inner fold, fit each selected tree ensemble on inner_train.
  2. Compute SHAP values on a random sample of inner_valid only.
  3. Aggregate feature SHAP values into seismologically defined groups.
  4. Average normalized group importance equally across folds and models.
  5. Freeze the Top-K groups for downstream confirmatory ablation.

Main outputs
------------
- all_inner_valid_feature_importance.csv
- all_inner_valid_group_importance.csv
- group_consensus_inner_validation.csv
- selected_groups_inner_validation.csv
- inner_validation_shap_manifest.csv
- inner_validation_shap_report.md

The final test set is not loaded into any SHAP calculation in this script.
"""

import argparse
import gc
import json
import os
import pickle
import warnings
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier

try:
    import shap
    HAS_SHAP = True
    SHAP_IMPORT_ERROR = ""
except Exception as exc:
    shap = None
    HAS_SHAP = False
    SHAP_IMPORT_ERROR = repr(exc)

try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
    CATBOOST_IMPORT_ERROR = ""
except Exception as exc:
    CatBoostClassifier = None
    Pool = None
    HAS_CATBOOST = False
    CATBOOST_IMPORT_ERROR = repr(exc)

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
    XGBOOST_IMPORT_ERROR = ""
except Exception as exc:
    xgb = None
    XGBClassifier = None
    HAS_XGBOOST = False
    XGBOOST_IMPORT_ERROR = repr(exc)

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


def expand_path(path):
    return str(Path(path).expanduser().resolve())


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def parse_int_list(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_str_list(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def normalize_model_name(name):
    key = str(name).strip().lower().replace(" ", "_")
    if key not in MODEL_ALIAS:
        raise ValueError(f"Unknown model name: {name}")
    return MODEL_ALIAS[key]


def safe_name(text):
    return str(text).lower().replace(" ", "_").replace("/", "__").replace("+", "plus")


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

    meta = payload["meta"].copy()
    if "t0" in meta.columns:
        meta["t0"] = pd.to_datetime(meta["t0"])

    return X, y, meta, payload


def load_feature_metadata(data_dir, n_features):
    path = os.path.join(data_dir, "feature_metadata_282_relative_lag.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature metadata is required for group selection but was not found: {path}"
        )

    meta = pd.read_csv(path)
    if "feature_idx" not in meta.columns:
        meta.insert(0, "feature_idx", np.arange(len(meta), dtype=int))

    meta = meta.sort_values("feature_idx").reset_index(drop=True)
    if len(meta) != n_features:
        raise ValueError(
            f"Feature metadata length {len(meta)} != dataset feature count {n_features}"
        )

    if "feature_name" not in meta.columns:
        meta["feature_name"] = [f"feature_{idx:03d}" for idx in range(n_features)]
    if "time_group" not in meta.columns:
        raise ValueError("Feature metadata is missing required column: time_group")
    if "signal_group" not in meta.columns:
        raise ValueError("Feature metadata is missing required column: signal_group")
    if "feature_group" not in meta.columns:
        meta["feature_group"] = (
            meta["time_group"].fillna("unknown").astype(str)
            + "/"
            + meta["signal_group"].fillna("unknown").astype(str)
        )

    return meta


def load_selected_configs(comparison_dir):
    path = os.path.join(comparison_dir, "selected_config_by_model.json")
    if not os.path.exists(path):
        print(f"[WARN] Selected-config file not found; use script defaults: {path}", flush=True)
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}; use defaults.", flush=True)
        return {}

    out = {}
    for item in raw.get("selected_configs", []):
        model_name = item.get("model_name")
        params = item.get("params", {})
        if model_name and params:
            out[str(model_name)] = {
                "config_name": item.get("config_name", "selected"),
                "params": params,
            }
    return out


def get_model_config(model_name, selected_configs):
    if model_name in selected_configs and selected_configs[model_name].get("params"):
        return deepcopy(selected_configs[model_name])
    return deepcopy(DEFAULT_CONFIGS[model_name])


def make_model(model_name, params, seed, n_jobs):
    params = deepcopy(dict(params))

    if model_name == "ExtraTrees":
        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", n_jobs)
        return ExtraTreesClassifier(**params)

    if model_name == "CatBoost":
        if not HAS_CATBOOST:
            raise ImportError(f"CatBoost is unavailable: {CATBOOST_IMPORT_ERROR}")
        params.setdefault("loss_function", "Logloss")
        params.setdefault("eval_metric", "Logloss")
        params.setdefault("random_seed", seed)
        params.setdefault("thread_count", n_jobs)
        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
        return CatBoostClassifier(**params)

    if model_name == "XGBoost":
        if not HAS_XGBOOST:
            raise ImportError(f"XGBoost is unavailable: {XGBOOST_IMPORT_ERROR}")
        params.setdefault("objective", "binary:logistic")
        params.setdefault("eval_metric", "logloss")
        params.setdefault("tree_method", "hist")
        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", n_jobs)
        return XGBClassifier(**params)

    raise ValueError(model_name)


def extract_positive_shap_array(shap_values, n_samples, n_features):
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[1] if len(shap_values) >= 2 else shap_values[0])
    else:
        arr = np.asarray(shap_values)

    if arr.ndim == 2:
        if arr.shape != (n_samples, n_features):
            raise ValueError(
                f"Unexpected 2D SHAP shape {arr.shape}; expected {(n_samples, n_features)}"
            )
        return arr.astype(np.float32)

    if arr.ndim == 3:
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            class_idx = 1 if arr.shape[2] > 1 else 0
            return arr[:, :, class_idx].astype(np.float32)
        if arr.shape[1] == n_samples and arr.shape[2] == n_features:
            class_idx = 1 if arr.shape[0] > 1 else 0
            return arr[class_idx, :, :].astype(np.float32)

    arr = np.squeeze(arr)
    if arr.ndim == 2 and arr.shape == (n_samples, n_features):
        return arr.astype(np.float32)

    raise ValueError(f"Cannot convert SHAP array with shape {np.asarray(shap_values).shape}")


def compute_shap_values(model, model_name, X_sample):
    if not HAS_SHAP:
        raise ImportError(f"shap import failed: {SHAP_IMPORT_ERROR}")

    n_samples, n_features = X_sample.shape

    if model_name == "CatBoost" and HAS_CATBOOST:
        try:
            values = np.asarray(
                model.get_feature_importance(Pool(X_sample), type="ShapValues")
            )
            if values.ndim == 2 and values.shape == (n_samples, n_features + 1):
                return values[:, :-1].astype(np.float32)
            if (
                values.ndim == 3
                and values.shape[0] == n_samples
                and values.shape[2] == n_features + 1
            ):
                class_idx = 1 if values.shape[1] > 1 else 0
                return values[:, class_idx, :-1].astype(np.float32)
        except Exception as exc:
            print(
                f"[WARN] CatBoost native SHAP failed; use TreeExplainer: {repr(exc)}",
                flush=True,
            )

    if model_name == "XGBoost" and HAS_XGBOOST:
        try:
            booster = model.get_booster() if hasattr(model, "get_booster") else model
            dmatrix = xgb.DMatrix(X_sample)
            values = np.asarray(booster.predict(dmatrix, pred_contribs=True))
            if values.ndim == 2 and values.shape == (n_samples, n_features + 1):
                return values[:, :-1].astype(np.float32)
            if (
                values.ndim == 3
                and values.shape[0] == n_samples
                and values.shape[2] == n_features + 1
            ):
                class_idx = 1 if values.shape[1] > 1 else 0
                return values[:, class_idx, :-1].astype(np.float32)
        except Exception as exc:
            print(
                f"[WARN] XGBoost native SHAP failed; use TreeExplainer: {repr(exc)}",
                flush=True,
            )

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_sample)
    return extract_positive_shap_array(values, n_samples, n_features)


def choose_validation_sample(meta, valid_mask, max_samples, seed):
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) == 0:
        raise ValueError("Inner-validation mask is empty")

    if "split" in meta.columns:
        bad = meta.iloc[valid_indices]["split"].astype(str).eq("test")
        if bool(bad.any()):
            raise RuntimeError(
                "Leakage guard triggered: inner_valid contains rows labeled as final test"
            )

    rng = np.random.default_rng(seed)
    if max_samples > 0 and len(valid_indices) > max_samples:
        selected = np.sort(
            rng.choice(valid_indices, size=max_samples, replace=False).astype(int)
        )
    else:
        selected = np.sort(valid_indices.astype(int))
    return selected


def feature_importance_table(
    shap_array,
    feature_meta,
    model_name,
    horizon,
    fold,
    config_name,
    n_inner_train,
    n_inner_valid_full,
):
    mean_abs = np.mean(np.abs(shap_array), axis=0)
    mean_signed = np.mean(shap_array, axis=0)
    total = float(mean_abs.sum())
    normalized = mean_abs / total if total > 0 else np.zeros_like(mean_abs)

    out = feature_meta.copy()
    out["model_name"] = model_name
    out["config_name"] = config_name
    out["horizon_month"] = int(horizon)
    out["fold"] = int(fold)
    out["source_stage"] = "inner_validation"
    out["n_inner_train"] = int(n_inner_train)
    out["n_inner_valid_full"] = int(n_inner_valid_full)
    out["n_inner_valid_shap"] = int(shap_array.shape[0])
    out["mean_abs_shap"] = mean_abs.astype(float)
    out["mean_signed_shap"] = mean_signed.astype(float)
    out["normalized_importance"] = normalized.astype(float)
    out = out.sort_values(
        ["normalized_importance", "feature_idx"], ascending=[False, True]
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def group_importance_table(feature_table, group_col):
    if group_col not in feature_table.columns:
        raise ValueError(f"Feature table is missing group column: {group_col}")

    group_table = (
        feature_table.groupby(group_col, dropna=False)
        .agg(
            n_features=("feature_idx", "count"),
            group_mean_abs_shap=("mean_abs_shap", "sum"),
            group_mean_signed_shap=("mean_signed_shap", "sum"),
            group_normalized_importance=("normalized_importance", "sum"),
        )
        .reset_index()
        .rename(columns={group_col: "group_name"})
    )

    first = feature_table.iloc[0]
    for column in [
        "model_name",
        "config_name",
        "horizon_month",
        "fold",
        "source_stage",
        "n_inner_train",
        "n_inner_valid_full",
        "n_inner_valid_shap",
    ]:
        group_table[column] = first[column]

    group_table["group_col"] = group_col
    group_table = group_table.sort_values(
        ["group_normalized_importance", "group_name"],
        ascending=[False, True],
    ).reset_index(drop=True)
    group_table["rank"] = np.arange(1, len(group_table) + 1, dtype=int)
    return group_table


def build_group_consensus(group_all, feature_meta, group_col):
    selected = group_all[group_all["group_col"].eq(group_col)].copy()
    if selected.empty:
        raise ValueError(f"No grouped SHAP rows found for group_col={group_col}")

    group_sizes = (
        feature_meta.groupby(group_col)
        .size()
        .rename("n_features")
        .reset_index()
        .rename(columns={group_col: "group_name"})
    )

    consensus = (
        selected.groupby(["horizon_month", "group_name"], as_index=False)
        .agg(
            mean_group_importance=(
                "group_normalized_importance",
                "mean",
            ),
            median_group_importance=(
                "group_normalized_importance",
                "median",
            ),
            std_group_importance=(
                "group_normalized_importance",
                "std",
            ),
            min_group_importance=(
                "group_normalized_importance",
                "min",
            ),
            max_group_importance=(
                "group_normalized_importance",
                "max",
            ),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            best_rank=("rank", "min"),
            worst_rank=("rank", "max"),
            n_runs=("model_name", "size"),
            n_models=("model_name", "nunique"),
            n_folds=("fold", "nunique"),
            models=("model_name", lambda x: ",".join(sorted(set(map(str, x))))),
            folds=("fold", lambda x: ",".join(map(str, sorted(set(map(int, x)))))),
        )
    )

    consensus["std_group_importance"] = consensus[
        "std_group_importance"
    ].fillna(0.0)
    consensus = consensus.merge(group_sizes, on="group_name", how="left")
    consensus["source_stage"] = "inner_validation"
    consensus["group_col"] = group_col

    consensus = consensus.sort_values(
        [
            "horizon_month",
            "mean_group_importance",
            "median_group_importance",
            "mean_rank",
            "group_name",
        ],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)

    consensus["rank"] = (
        consensus.groupby("horizon_month").cumcount() + 1
    ).astype(int)
    return consensus


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="data/raw282",
    )
    parser.add_argument(
        "--comparison-dir",
        default="outputs/discovery_models",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/inner_validation_shap",
    )
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--models", default="extratrees,catboost,xgboost")
    parser.add_argument("--folds", default="1,2,3")
    parser.add_argument("--group-cols", default="feature_group,signal_group,time_group")
    parser.add_argument("--primary-group-col", default="feature_group")
    parser.add_argument("--top-k-groups", type=int, default=3)
    parser.add_argument("--max-valid-samples", type=int, default=1500)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    if args.top_k_groups < 1:
        raise ValueError("--top-k-groups must be >= 1")
    if args.max_valid_samples == 0:
        raise ValueError(
            "--max-valid-samples must be positive, or negative to use all inner-valid rows"
        )

    args.data_dir = expand_path(args.data_dir)
    args.comparison_dir = expand_path(args.comparison_dir)
    args.out_dir = expand_path(args.out_dir)
    horizons = parse_int_list(args.horizons)
    models = [normalize_model_name(x) for x in parse_str_list(args.models)]
    folds = parse_int_list(args.folds)
    group_cols = parse_str_list(args.group_cols)

    if args.primary_group_col not in group_cols:
        group_cols.insert(0, args.primary_group_col)

    if not HAS_SHAP:
        raise ImportError(f"shap is unavailable: {SHAP_IMPORT_ERROR}")

    feature_dir = ensure_dir(os.path.join(args.out_dir, "per_run_feature"))
    group_dir = ensure_dir(os.path.join(args.out_dir, "per_run_group"))
    sample_dir = ensure_dir(os.path.join(args.out_dir, "validation_samples"))
    ensure_dir(args.out_dir)

    selected_configs = load_selected_configs(args.comparison_dir)

    print("=" * 100, flush=True)
    print("Leakage-safe inner-validation SHAP discovery", flush=True)
    print("data_dir:", args.data_dir, flush=True)
    print("comparison_dir:", args.comparison_dir, flush=True)
    print("out_dir:", args.out_dir, flush=True)
    print("horizons:", horizons, flush=True)
    print("models:", models, flush=True)
    print("folds:", folds, flush=True)
    print("max_valid_samples:", args.max_valid_samples, flush=True)
    print("primary_group_col:", args.primary_group_col, flush=True)
    print("=" * 100, flush=True)

    feature_tables = []
    group_tables = []
    manifest_rows = []
    feature_meta_reference = None

    for horizon in horizons:
        print("\n" + "#" * 100, flush=True)
        print(f"Horizon H={horizon} months", flush=True)

        X, y, meta, payload = load_dataset(args.data_dir, horizon)
        feature_meta = load_feature_metadata(args.data_dir, X.shape[1])
        if feature_meta_reference is None:
            feature_meta_reference = feature_meta.copy()

        if "split" not in meta.columns:
            raise ValueError("Dataset metadata is missing required column: split")

        final_test_count = int(meta["split"].astype(str).eq("test").sum())
        print(
            f"Dataset: X={X.shape}, positives={int(y.sum())}, "
            f"final_test_rows_present_but_unused={final_test_count}",
            flush=True,
        )

        for fold in folds:
            train_col = f"fold{fold}_role"
            if train_col not in meta.columns:
                message = f"Missing fold role column: {train_col}"
                if args.skip_missing:
                    print("[SKIP]", message, flush=True)
                    continue
                raise ValueError(message)

            inner_train_mask = meta[train_col].astype(str).eq("inner_train").values
            inner_valid_mask = meta[train_col].astype(str).eq("inner_valid").values

            n_train = int(inner_train_mask.sum())
            n_valid_full = int(inner_valid_mask.sum())
            if n_train == 0 or n_valid_full == 0:
                message = (
                    f"Empty fold H={horizon}, fold={fold}: "
                    f"inner_train={n_train}, inner_valid={n_valid_full}"
                )
                if args.skip_missing:
                    print("[SKIP]", message, flush=True)
                    continue
                raise ValueError(message)

            sample_seed = args.seed + 10000 * int(horizon) + 100 * int(fold)
            valid_indices = choose_validation_sample(
                meta,
                inner_valid_mask,
                args.max_valid_samples,
                sample_seed,
            )

            if np.any(meta.iloc[valid_indices]["split"].astype(str).eq("test")):
                raise RuntimeError("Leakage guard: sampled SHAP rows include final test")

            sample_table = meta.iloc[valid_indices].copy()
            sample_table.insert(0, "dataset_row_index", valid_indices.astype(int))
            sample_table["horizon_month"] = int(horizon)
            sample_table["fold"] = int(fold)
            sample_table["source_stage"] = "inner_validation"
            sample_path = os.path.join(
                sample_dir,
                f"inner_valid_sample_H{horizon}m_fold{fold}.csv",
            )
            sample_table.to_csv(sample_path, index=False)

            X_train = X[inner_train_mask]
            y_train = y[inner_train_mask]
            X_valid_sample = X[valid_indices]

            print(
                f"H={horizon} fold={fold}: inner_train={X_train.shape}, "
                f"inner_valid_full={n_valid_full}, SHAP_sample={X_valid_sample.shape}",
                flush=True,
            )

            for model_name in models:
                config = get_model_config(model_name, selected_configs)
                config_name = config["config_name"]
                params = config["params"]
                model_seed = args.seed + 10000 * int(horizon) + 100 * int(fold)

                print(
                    f"  Fit {model_name}, config={config_name}, "
                    f"H={horizon}, fold={fold}",
                    flush=True,
                )

                try:
                    model = make_model(
                        model_name,
                        params,
                        seed=model_seed,
                        n_jobs=args.n_jobs,
                    )
                    model.fit(X_train, y_train)
                    shap_values = compute_shap_values(
                        model,
                        model_name,
                        X_valid_sample,
                    )
                except Exception as exc:
                    if args.skip_missing:
                        print(
                            f"[SKIP model] {model_name} H={horizon} fold={fold}: "
                            f"{repr(exc)}",
                            flush=True,
                        )
                        continue
                    raise

                if shap_values.shape != X_valid_sample.shape:
                    raise ValueError(
                        f"SHAP shape {shap_values.shape} != "
                        f"validation sample shape {X_valid_sample.shape}"
                    )

                feature_table = feature_importance_table(
                    shap_values,
                    feature_meta=feature_meta,
                    model_name=model_name,
                    horizon=horizon,
                    fold=fold,
                    config_name=config_name,
                    n_inner_train=n_train,
                    n_inner_valid_full=n_valid_full,
                )

                feature_path = os.path.join(
                    feature_dir,
                    (
                        f"inner_valid_feature_importance_"
                        f"{safe_name(model_name)}_H{horizon}m_fold{fold}.csv"
                    ),
                )
                feature_table.to_csv(feature_path, index=False)
                feature_tables.append(feature_table)

                for group_col in group_cols:
                    group_table = group_importance_table(
                        feature_table,
                        group_col=group_col,
                    )
                    group_path = os.path.join(
                        group_dir,
                        (
                            f"inner_valid_group_importance_"
                            f"{safe_name(model_name)}_H{horizon}m_fold{fold}_"
                            f"{safe_name(group_col)}.csv"
                        ),
                    )
                    group_table.to_csv(group_path, index=False)
                    group_tables.append(group_table)

                manifest_rows.append(
                    {
                        "horizon_month": int(horizon),
                        "fold": int(fold),
                        "model_name": model_name,
                        "config_name": config_name,
                        "source_stage": "inner_validation",
                        "n_inner_train": n_train,
                        "n_inner_valid_full": n_valid_full,
                        "n_inner_valid_shap": int(len(valid_indices)),
                        "sample_csv": sample_path,
                        "feature_importance_csv": feature_path,
                        "used_final_test_rows": False,
                    }
                )

                del model, shap_values, feature_table
                gc.collect()

            del X_train, y_train, X_valid_sample
            gc.collect()

        del X, y, meta, payload
        gc.collect()

    if not feature_tables or not group_tables:
        raise RuntimeError("No SHAP tables were generated")

    feature_all = pd.concat(feature_tables, ignore_index=True, sort=False)
    group_all = pd.concat(group_tables, ignore_index=True, sort=False)
    manifest = pd.DataFrame(manifest_rows)

    feature_all_path = os.path.join(
        args.out_dir,
        "all_inner_valid_feature_importance.csv",
    )
    group_all_path = os.path.join(
        args.out_dir,
        "all_inner_valid_group_importance.csv",
    )
    manifest_path = os.path.join(
        args.out_dir,
        "inner_validation_shap_manifest.csv",
    )

    feature_all.to_csv(feature_all_path, index=False)
    group_all.to_csv(group_all_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    consensus = build_group_consensus(
        group_all,
        feature_meta_reference,
        group_col=args.primary_group_col,
    )
    expected_runs = len(models) * len(folds)
    consensus["expected_runs_per_horizon"] = expected_runs
    consensus["complete_model_fold_coverage"] = (
        consensus["n_runs"].astype(int) == expected_runs
    )

    consensus_path = os.path.join(
        args.out_dir,
        "group_consensus_inner_validation.csv",
    )
    consensus.to_csv(consensus_path, index=False)

    selected = (
        consensus[consensus["rank"] <= args.top_k_groups]
        .copy()
        .sort_values(["horizon_month", "rank"])
        .reset_index(drop=True)
    )
    selected["selection_rule"] = (
        "rank by equal-weight mean normalized group SHAP across "
        "inner-validation folds and models"
    )
    selected["used_final_test_for_selection"] = False

    selected_path = os.path.join(
        args.out_dir,
        "selected_groups_inner_validation.csv",
    )
    selected.to_csv(selected_path, index=False)

    report_path = os.path.join(
        args.out_dir,
        "inner_validation_shap_report.md",
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Leakage-safe inner-validation SHAP discovery\n\n")
        handle.write(
            "Feature groups in this directory were selected exclusively from "
            "inner rolling-validation folds. The final test split was not used "
            "for SHAP computation or group selection.\n\n"
        )
        handle.write("## Configuration\n\n```json\n")
        handle.write(
            json.dumps(
                {
                    "data_dir": args.data_dir,
                    "comparison_dir": args.comparison_dir,
                    "out_dir": args.out_dir,
                    "horizons": horizons,
                    "models": models,
                    "folds": folds,
                    "max_valid_samples": args.max_valid_samples,
                    "primary_group_col": args.primary_group_col,
                    "top_k_groups": args.top_k_groups,
                    "seed": args.seed,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        handle.write("\n```\n\n")
        handle.write("## Frozen groups\n\n")
        show_columns = [
            "horizon_month",
            "rank",
            "group_name",
            "n_features",
            "mean_group_importance",
            "median_group_importance",
            "mean_rank",
            "n_models",
            "n_folds",
            "n_runs",
        ]
        handle.write(
            selected[show_columns].to_markdown(index=False)
        )
        handle.write("\n\n")
        handle.write("## Leakage audit\n\n")
        handle.write(
            f"- Manifest rows: {len(manifest)}\n"
            f"- Any run using final-test rows: "
            f"{bool(manifest['used_final_test_rows'].astype(bool).any())}\n"
            f"- Selection source: inner_validation\n"
        )

    print("\n" + "=" * 100, flush=True)
    print("INNER-VALIDATION SHAP DISCOVERY FINISHED", flush=True)
    print("Saved:", feature_all_path, flush=True)
    print("Saved:", group_all_path, flush=True)
    print("Saved:", consensus_path, flush=True)
    print("Saved frozen groups:", selected_path, flush=True)
    print("Saved manifest:", manifest_path, flush=True)
    print("Saved report:", report_path, flush=True)


if __name__ == "__main__":
    main()
