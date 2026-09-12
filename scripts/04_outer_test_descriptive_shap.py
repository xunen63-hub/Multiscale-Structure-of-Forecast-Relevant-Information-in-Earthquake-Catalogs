#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_outer_test_descriptive_shap.py

Purpose
-------
Cross-model SHAP analysis for binary M>=5 regional earthquake forecasting.

This script is designed to run after:
    02_train_discovery_models.py

It loads final-test models and prediction files for selected tree ensembles
(CatBoost, ExtraTrees, XGBoost by default), computes SHAP values on controlled
samples of the final test set, and compares model explanations across horizons.

Main outputs
------------
1. Per model / horizon / subset feature SHAP importance.
2. Per model / horizon / subset grouped SHAP importance.
3. Cross-model Top-N feature overlap and rank consistency.
4. Cross-model grouped importance tables.
5. Simple publication-ready CSV tables and PNG bar plots.

Recommended interpretation
--------------------------
Use ExtraTrees as the main explanation model if the goal is overall detection
and ranking performance, but use CatBoost and XGBoost as robustness checks.
Focus on cross-model consistent feature groups rather than single-feature ranks.

Author: generated for hym63 HPC workflow
"""

import os
import gc
import json
import math
import pickle
import argparse
import warnings
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except Exception as e:
    shap = None
    HAS_SHAP = False
    SHAP_IMPORT_ERROR = repr(e)
else:
    SHAP_IMPORT_ERROR = ""

try:
    from catboost import Pool
    HAS_CATBOOST_POOL = True
except Exception:
    Pool = None
    HAS_CATBOOST_POOL = False

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# basic utilities
# ============================================================

MODEL_NAME_MAP = {
    "catboost": "CatBoost",
    "cat": "CatBoost",
    "extratrees": "ExtraTrees",
    "extra_trees": "ExtraTrees",
    "et": "ExtraTrees",
    "xgboost": "XGBoost",
    "xgb": "XGBoost",
    "rf": "RF",
    "randomforest": "RF",
    "lightgbm": "LightGBM",
    "lgbm": "LightGBM",
}


def expand_path(p):
    return str(Path(p).expanduser().resolve())


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def parse_int_list(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_model_list(s):
    out = []
    for x in str(s).split(","):
        key = x.strip().lower()
        if not key:
            continue
        if key not in MODEL_NAME_MAP:
            raise ValueError(f"Unknown model key: {x}")
        name = MODEL_NAME_MAP[key]
        if name not in out:
            out.append(name)
    return out


def safe_model_name(model_name):
    return str(model_name).lower().replace(" ", "_")


def model_path(comparison_dir, model_name, horizon):
    safe = safe_model_name(model_name)
    return os.path.join(comparison_dir, "models", f"{safe}_binary_m5_H{horizon}m.joblib")


def prediction_path(comparison_dir, model_name, horizon):
    safe = safe_model_name(model_name)
    return os.path.join(comparison_dir, "predictions", f"predictions_test_{safe}_H{horizon}m.csv")


def dataset_path(data_dir, horizon):
    return os.path.join(data_dir, f"china_raw282_dataset_horizon_{horizon}m.pkl")


def load_dataset(data_dir, horizon):
    path = dataset_path(data_dir, horizon)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        d = pickle.load(f)

    X = np.asarray(d["X"], dtype=np.float32)

    if "y_m5" in d:
        y = np.asarray(d["y_m5"], dtype=np.int64)
    else:
        y_class = np.asarray(d["y_class"], dtype=np.int64)
        y = (y_class > 0).astype(np.int64)

    meta = d["meta"].copy()
    if "t0" in meta.columns:
        meta["t0"] = pd.to_datetime(meta["t0"])
    if "future_end_inclusive" in meta.columns:
        meta["future_end_inclusive"] = pd.to_datetime(meta["future_end_inclusive"])

    return X, y, meta, d


# ============================================================
# feature metadata and grouping
# ============================================================


def make_generic_feature_metadata(n_features):
    return pd.DataFrame({
        "feature_idx": np.arange(n_features, dtype=int),
        "feature_name": [f"feature_{i:03d}" for i in range(n_features)],
        "time_group": "unknown",
        "signal_group": "unknown",
        "feature_group": "unknown/unknown",
    })


def load_feature_metadata(data_dir, n_features):
    p = os.path.join(data_dir, "feature_metadata_282_relative_lag.csv")
    if not os.path.exists(p):
        print(f"[WARN] feature metadata not found: {p}", flush=True)
        return make_generic_feature_metadata(n_features)

    meta = pd.read_csv(p)
    if "feature_idx" not in meta.columns:
        meta.insert(0, "feature_idx", np.arange(len(meta), dtype=int))
    meta = meta.sort_values("feature_idx").reset_index(drop=True)

    if len(meta) != n_features:
        print(
            f"[WARN] feature metadata length {len(meta)} != n_features {n_features}. Use generic metadata.",
            flush=True,
        )
        return make_generic_feature_metadata(n_features)

    if "feature_name" not in meta.columns:
        meta["feature_name"] = [f"feature_{i:03d}" for i in range(n_features)]
    if "time_group" not in meta.columns:
        meta["time_group"] = "unknown"
    if "signal_group" not in meta.columns:
        meta["signal_group"] = "unknown"

    meta["time_group"] = meta["time_group"].fillna("unknown").astype(str)
    meta["signal_group"] = meta["signal_group"].fillna("unknown").astype(str)
    meta["feature_group"] = meta["time_group"] + "/" + meta["signal_group"]

    return meta


# ============================================================
# sample construction
# ============================================================


def sample_indices(indices, max_n, rng, mode="random", score=None):
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= max_n:
        return indices

    if mode == "top_score":
        if score is None:
            raise ValueError("score is required for top_score sampling")
        order = np.argsort(-score[indices])
        return indices[order[:max_n]]

    chosen = rng.choice(indices, size=max_n, replace=False)
    return np.sort(chosen)


def build_model_horizon_sample(pred_df, args, rng):
    n = len(pred_df)
    y_true = pred_df["y_true"].astype(int).values
    y_pred = pred_df["y_pred"].astype(int).values
    score = pred_df["score_m5"].astype(float).values

    all_idx = np.arange(n, dtype=int)
    pos_pred = np.where(y_pred == 1)[0]
    tp = np.where((y_true == 1) & (y_pred == 1))[0]
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    fn = np.where((y_true == 1) & (y_pred == 0))[0]
    tn = np.where((y_true == 0) & (y_pred == 0))[0]

    top5_n = max(1, int(math.ceil(n * 0.05)))
    top10_n = max(1, int(math.ceil(n * 0.10)))
    score_order = np.argsort(-score)
    top5 = score_order[:top5_n]
    top10 = score_order[:top10_n]

    raw_subsets = {
        "all_test": all_idx,
        "predicted_positive": pos_pred,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "high_score_top5pct": top5,
        "high_score_top10pct": top10,
    }

    subset_sampled = {}
    for name, idx in raw_subsets.items():
        if name.startswith("high_score"):
            subset_sampled[name] = sample_indices(
                idx, args.max_samples_per_subset, rng, mode="top_score", score=score
            )
        else:
            subset_sampled[name] = sample_indices(
                idx, args.max_samples_per_subset, rng, mode="random"
            )

    union = np.unique(np.concatenate([v for v in subset_sampled.values() if len(v) > 0]))
    union = np.sort(union)
    union_pos = {int(idx): i for i, idx in enumerate(union)}

    subset_pos = {}
    for name, idx in subset_sampled.items():
        subset_pos[name] = np.asarray([union_pos[int(i)] for i in idx if int(i) in union_pos], dtype=int)

    sample_info = pd.DataFrame({
        "test_row_index": union.astype(int),
        "y_true": y_true[union].astype(int),
        "y_pred": y_pred[union].astype(int),
        "score_m5": score[union].astype(float),
    })

    subset_count_rows = []
    for name, full_idx in raw_subsets.items():
        subset_count_rows.append({
            "subset_name": name,
            "n_full_subset": int(len(full_idx)),
            "n_sampled_subset": int(len(subset_pos[name])),
        })

    subset_counts = pd.DataFrame(subset_count_rows)
    return union, subset_pos, sample_info, subset_counts


# ============================================================
# SHAP computation
# ============================================================


def extract_positive_shap_array(shap_values, n_samples, n_features):
    """Return a 2D array (n_samples, n_features) for positive-class SHAP values."""
    if isinstance(shap_values, list):
        if len(shap_values) >= 2:
            arr = np.asarray(shap_values[1])
        else:
            arr = np.asarray(shap_values[0])
    else:
        arr = np.asarray(shap_values)

    if arr.ndim == 2:
        return arr.astype(np.float32)

    if arr.ndim == 3:
        # Common shapes:
        #   (n_samples, n_features, n_outputs)
        #   (n_outputs, n_samples, n_features)
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            out_idx = 1 if arr.shape[2] > 1 else 0
            return arr[:, :, out_idx].astype(np.float32)
        if arr.shape[1] == n_samples and arr.shape[2] == n_features:
            out_idx = 1 if arr.shape[0] > 1 else 0
            return arr[out_idx, :, :].astype(np.float32)

    arr = np.squeeze(arr)
    if arr.ndim == 2 and arr.shape[0] == n_samples and arr.shape[1] == n_features:
        return arr.astype(np.float32)

    raise ValueError(f"Cannot convert shap_values with shape {np.asarray(shap_values).shape} to 2D positive-class SHAP")


def compute_shap_values(model, model_name, X_sample):
    if not HAS_SHAP:
        raise ImportError(f"shap is not available: {SHAP_IMPORT_ERROR}")

    n_samples, n_features = X_sample.shape

    # CatBoost native SHAP is often robust and fast.
    if model_name == "CatBoost" and HAS_CATBOOST_POOL:
        try:
            sv = model.get_feature_importance(Pool(X_sample), type="ShapValues")
            sv = np.asarray(sv)
            # Binary CatBoost usually returns (n_samples, n_features + 1).
            # Multidim variants may return 3D; fall back to generic extractor.
            if sv.ndim == 2 and sv.shape[1] == n_features + 1:
                return sv[:, :-1].astype(np.float32)
            if sv.ndim == 3:
                # Try positive class, excluding expected-value column.
                if sv.shape[0] == n_samples and sv.shape[2] == n_features + 1:
                    cls = 1 if sv.shape[1] > 1 else 0
                    return sv[:, cls, :-1].astype(np.float32)
        except Exception as e:
            print(f"[WARN] CatBoost native SHAP failed, fallback to shap.TreeExplainer: {repr(e)}", flush=True)

    # XGBoost fallback: some SHAP versions cannot parse newer XGBoost binary/UBJSON
    # model buffers and fail with UnicodeDecodeError. XGBoost native pred_contribs
    # returns TreeSHAP contributions plus the bias term, so we drop the last column.
    if model_name == "XGBoost":
        try:
            import xgboost as xgb
            booster = model.get_booster() if hasattr(model, "get_booster") else model
            dmat = xgb.DMatrix(X_sample)
            sv = booster.predict(dmat, pred_contribs=True)
            sv = np.asarray(sv)
            if sv.ndim == 2 and sv.shape[0] == n_samples and sv.shape[1] == n_features + 1:
                return sv[:, :-1].astype(np.float32)
            if sv.ndim == 3:
                # multiclass safety: (n_samples, n_classes, n_features + 1)
                if sv.shape[0] == n_samples and sv.shape[2] == n_features + 1:
                    cls = 1 if sv.shape[1] > 1 else 0
                    return sv[:, cls, :-1].astype(np.float32)
            print(f"[WARN] Unexpected XGBoost pred_contribs shape {sv.shape}; fallback to shap.TreeExplainer", flush=True)
        except Exception as e:
            print(f"[WARN] XGBoost native pred_contribs failed, fallback to shap.TreeExplainer: {repr(e)}", flush=True)

    # Generic TreeSHAP for scikit-learn tree ensembles.
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    return extract_positive_shap_array(sv, n_samples, n_features)


# ============================================================
# importance tables and plots
# ============================================================


def make_feature_importance_table(shap_arr, feature_meta, model_name, horizon, subset_name, n_subset):
    mean_abs = np.mean(np.abs(shap_arr), axis=0)
    mean_signed = np.mean(shap_arr, axis=0)
    total = float(np.sum(mean_abs))
    norm = mean_abs / total if total > 0 else np.zeros_like(mean_abs)

    out = feature_meta.copy()
    out["model_name"] = model_name
    out["horizon_month"] = int(horizon)
    out["subset_name"] = subset_name
    out["n_subset_sampled"] = int(n_subset)
    out["mean_abs_shap"] = mean_abs.astype(float)
    out["mean_signed_shap"] = mean_signed.astype(float)
    out["normalized_importance"] = norm.astype(float)
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)

    ordered = [
        "model_name", "horizon_month", "subset_name", "n_subset_sampled",
        "rank", "feature_idx", "feature_name", "time_group", "signal_group", "feature_group",
        "mean_abs_shap", "mean_signed_shap", "normalized_importance",
    ]
    ordered = [c for c in ordered if c in out.columns] + [c for c in out.columns if c not in ordered]
    return out[ordered]


def make_group_importance_table(feature_imp, group_col, model_name, horizon, subset_name, n_subset):
    if group_col not in feature_imp.columns:
        raise ValueError(f"Missing group column: {group_col}")

    g = (
        feature_imp.groupby(group_col, dropna=False)
        .agg(
            n_features=("feature_idx", "count"),
            group_mean_abs_shap=("mean_abs_shap", "sum"),
            group_mean_signed_shap=("mean_signed_shap", "sum"),
            group_normalized_importance=("normalized_importance", "sum"),
        )
        .reset_index()
        .rename(columns={group_col: "group_name"})
    )
    g["group_col"] = group_col
    g["model_name"] = model_name
    g["horizon_month"] = int(horizon)
    g["subset_name"] = subset_name
    g["n_subset_sampled"] = int(n_subset)
    g = g.sort_values("group_normalized_importance", ascending=False).reset_index(drop=True)
    g["rank"] = np.arange(1, len(g) + 1, dtype=int)

    ordered = [
        "model_name", "horizon_month", "subset_name", "n_subset_sampled",
        "group_col", "rank", "group_name", "n_features",
        "group_mean_abs_shap", "group_mean_signed_shap", "group_normalized_importance",
    ]
    return g[ordered]


def plot_top_feature_bar(feature_imp, out_png, top_n=20):
    d = feature_imp.head(top_n).copy()
    if len(d) == 0:
        return
    d = d.iloc[::-1]

    labels = d["feature_name"].astype(str).values
    values = d["normalized_importance"].astype(float).values

    fig_h = max(5.0, 0.28 * len(d) + 1.5)
    plt.figure(figsize=(9.0, fig_h))
    plt.barh(np.arange(len(d)), values)
    plt.yticks(np.arange(len(d)), labels, fontsize=8)
    title = f"{feature_imp['model_name'].iloc[0]} H={feature_imp['horizon_month'].iloc[0]}m {feature_imp['subset_name'].iloc[0]}"
    plt.title(title)
    plt.xlabel("Normalized mean |SHAP|")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_group_bar(group_imp, out_png, top_n=15):
    d = group_imp.head(top_n).copy()
    if len(d) == 0:
        return
    d = d.iloc[::-1]

    labels = d["group_name"].astype(str).values
    values = d["group_normalized_importance"].astype(float).values

    fig_h = max(4.5, 0.35 * len(d) + 1.5)
    plt.figure(figsize=(8.5, fig_h))
    plt.barh(np.arange(len(d)), values)
    plt.yticks(np.arange(len(d)), labels, fontsize=8)
    title = f"Grouped SHAP: {group_imp['model_name'].iloc[0]} H={group_imp['horizon_month'].iloc[0]}m {group_imp['subset_name'].iloc[0]}"
    plt.title(title)
    plt.xlabel("Group normalized mean |SHAP|")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


# ============================================================
# consistency analysis
# ============================================================


def rank_correlation(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    ra = pd.Series(a).rank(method="average").values
    rb = pd.Series(b).rank(method="average").values
    return float(np.corrcoef(ra, rb)[0, 1])


def build_consistency_tables(feature_tables, group_tables, args):
    top_n = int(args.top_n_consistency)

    if len(feature_tables) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    feat_all = pd.concat(feature_tables, ignore_index=True, sort=False)
    group_all = pd.concat(group_tables, ignore_index=True, sort=False) if group_tables else pd.DataFrame()

    overlap_rows = []
    stable_feature_rows = []

    keys = feat_all[["horizon_month", "subset_name"]].drop_duplicates()
    for _, key_row in keys.iterrows():
        H = int(key_row["horizon_month"])
        subset = str(key_row["subset_name"])
        sub = feat_all[(feat_all["horizon_month"] == H) & (feat_all["subset_name"] == subset)].copy()
        models = sorted(sub["model_name"].unique())

        for m1, m2 in combinations(models, 2):
            d1 = sub[sub["model_name"] == m1].sort_values("rank")
            d2 = sub[sub["model_name"] == m2].sort_values("rank")
            top1 = set(d1.head(top_n)["feature_idx"].astype(int).tolist())
            top2 = set(d2.head(top_n)["feature_idx"].astype(int).tolist())
            union = top1 | top2
            inter = top1 & top2

            merged = d1[["feature_idx", "normalized_importance"]].merge(
                d2[["feature_idx", "normalized_importance"]],
                on="feature_idx",
                suffixes=("_m1", "_m2"),
            )
            rho = rank_correlation(
                merged["normalized_importance_m1"].values,
                merged["normalized_importance_m2"].values,
            )

            overlap_rows.append({
                "horizon_month": H,
                "subset_name": subset,
                "model_1": m1,
                "model_2": m2,
                "top_n": top_n,
                "n_intersection": int(len(inter)),
                "n_union": int(len(union)),
                "jaccard_top_n": float(len(inter) / len(union)) if len(union) > 0 else np.nan,
                "spearman_rank_corr_all_features": rho,
            })

        # Stable features across all available models.
        top_sub = sub[sub["rank"] <= top_n].copy()
        stable = (
            top_sub.groupby(["feature_idx", "feature_name", "time_group", "signal_group", "feature_group"])
            .agg(
                n_models_in_top_n=("model_name", "nunique"),
                mean_rank=("rank", "mean"),
                min_rank=("rank", "min"),
                mean_normalized_importance=("normalized_importance", "mean"),
                models=("model_name", lambda x: ",".join(sorted(set(map(str, x))))),
            )
            .reset_index()
        )
        stable["horizon_month"] = H
        stable["subset_name"] = subset
        stable["top_n"] = top_n
        stable = stable.sort_values(
            ["n_models_in_top_n", "mean_rank", "mean_normalized_importance"],
            ascending=[False, True, False],
        )
        stable_feature_rows.append(stable)

    overlap_df = pd.DataFrame(overlap_rows)
    stable_features_df = pd.concat(stable_feature_rows, ignore_index=True, sort=False) if stable_feature_rows else pd.DataFrame()

    # Group-level consistency, feature_group only by default.
    group_overlap_rows = []
    stable_group_rows = []
    if len(group_all) > 0:
        g0 = group_all[group_all["group_col"] == args.primary_group_col].copy()
        keys = g0[["horizon_month", "subset_name"]].drop_duplicates()
        for _, key_row in keys.iterrows():
            H = int(key_row["horizon_month"])
            subset = str(key_row["subset_name"])
            sub = g0[(g0["horizon_month"] == H) & (g0["subset_name"] == subset)].copy()
            models = sorted(sub["model_name"].unique())

            for m1, m2 in combinations(models, 2):
                d1 = sub[sub["model_name"] == m1].sort_values("rank")
                d2 = sub[sub["model_name"] == m2].sort_values("rank")
                top1 = set(d1.head(top_n)["group_name"].astype(str).tolist())
                top2 = set(d2.head(top_n)["group_name"].astype(str).tolist())
                union = top1 | top2
                inter = top1 & top2
                merged = d1[["group_name", "group_normalized_importance"]].merge(
                    d2[["group_name", "group_normalized_importance"]],
                    on="group_name",
                    suffixes=("_m1", "_m2"),
                )
                rho = rank_correlation(
                    merged["group_normalized_importance_m1"].values,
                    merged["group_normalized_importance_m2"].values,
                )

                group_overlap_rows.append({
                    "horizon_month": H,
                    "subset_name": subset,
                    "model_1": m1,
                    "model_2": m2,
                    "group_col": args.primary_group_col,
                    "top_n": top_n,
                    "n_intersection": int(len(inter)),
                    "n_union": int(len(union)),
                    "jaccard_top_n": float(len(inter) / len(union)) if len(union) > 0 else np.nan,
                    "spearman_rank_corr_all_groups": rho,
                })

            top_sub = sub[sub["rank"] <= top_n].copy()
            stable = (
                top_sub.groupby(["group_name"])
                .agg(
                    n_models_in_top_n=("model_name", "nunique"),
                    mean_rank=("rank", "mean"),
                    mean_group_normalized_importance=("group_normalized_importance", "mean"),
                    models=("model_name", lambda x: ",".join(sorted(set(map(str, x))))),
                )
                .reset_index()
            )
            stable["horizon_month"] = H
            stable["subset_name"] = subset
            stable["group_col"] = args.primary_group_col
            stable["top_n"] = top_n
            stable = stable.sort_values(
                ["n_models_in_top_n", "mean_rank", "mean_group_normalized_importance"],
                ascending=[False, True, False],
            )
            stable_group_rows.append(stable)

    group_overlap_df = pd.DataFrame(group_overlap_rows)
    stable_groups_df = pd.concat(stable_group_rows, ignore_index=True, sort=False) if stable_group_rows else pd.DataFrame()

    return overlap_df, stable_features_df, group_overlap_df, stable_groups_df


def plot_group_heatmaps(group_all, args, plot_dir):
    if len(group_all) == 0:
        return []

    paths = []
    g0 = group_all[group_all["group_col"] == args.primary_group_col].copy()
    for H in sorted(g0["horizon_month"].unique()):
        for subset in args.heatmap_subsets:
            sub = g0[(g0["horizon_month"] == H) & (g0["subset_name"] == subset)].copy()
            if len(sub) == 0:
                continue
            pivot = sub.pivot_table(
                index="group_name",
                columns="model_name",
                values="group_normalized_importance",
                aggfunc="mean",
                fill_value=0.0,
            )
            if len(pivot) == 0:
                continue
            pivot["_mean"] = pivot.mean(axis=1)
            pivot = pivot.sort_values("_mean", ascending=False).drop(columns=["_mean"])
            pivot = pivot.head(args.heatmap_top_groups)

            plt.figure(figsize=(1.3 * len(pivot.columns) + 4, 0.35 * len(pivot.index) + 2.2))
            plt.imshow(pivot.values, aspect="auto")
            plt.colorbar(label="Group normalized mean |SHAP|")
            plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
            plt.yticks(np.arange(len(pivot.index)), pivot.index, fontsize=8)
            plt.title(f"Group SHAP consistency H={int(H)}m {subset}")
            plt.tight_layout()
            out_png = os.path.join(plot_dir, f"heatmap_group_consistency_H{int(H)}m_{subset}.png")
            plt.savefig(out_png, dpi=180)
            plt.close()
            paths.append(out_png)
    return paths


# ============================================================
# main routine
# ============================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", default="data/raw282")
    parser.add_argument("--comparison-dir", default="outputs/discovery_models")
    parser.add_argument("--out-dir", default="outputs/shap_outer_test_9095")

    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--models", default="extratrees,catboost,xgboost")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-samples-per-subset", type=int, default=1000)
    parser.add_argument("--min-subset-samples", type=int, default=20)
    parser.add_argument("--top-n-features-plot", type=int, default=25)
    parser.add_argument("--top-n-consistency", type=int, default=20)
    parser.add_argument("--group-cols", default="feature_group,signal_group,time_group")
    parser.add_argument("--primary-group-col", default="feature_group")
    parser.add_argument("--heatmap-subsets", default="all_test,high_score_top10pct,true_positive,false_positive")
    parser.add_argument("--heatmap-top-groups", type=int, default=20)

    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--only-consistency", action="store_true", help="Skip SHAP computation and rebuild consistency from existing CSV files.")

    args = parser.parse_args()

    args.data_dir = expand_path(args.data_dir)
    args.comparison_dir = expand_path(args.comparison_dir)
    args.out_dir = expand_path(args.out_dir)
    args.horizons = parse_int_list(args.horizons)
    args.models = parse_model_list(args.models)
    args.group_cols = [x.strip() for x in args.group_cols.split(",") if x.strip()]
    args.heatmap_subsets = [x.strip() for x in args.heatmap_subsets.split(",") if x.strip()]

    ensure_dir(args.out_dir)
    feature_dir = ensure_dir(os.path.join(args.out_dir, "feature_importance"))
    group_dir = ensure_dir(os.path.join(args.out_dir, "group_importance"))
    sample_dir = ensure_dir(os.path.join(args.out_dir, "samples"))
    plot_dir = ensure_dir(os.path.join(args.out_dir, "plots"))
    consistency_dir = ensure_dir(os.path.join(args.out_dir, "consistency"))

    if not HAS_SHAP:
        raise ImportError(f"shap import failed: {SHAP_IMPORT_ERROR}")

    print("=" * 100, flush=True)
    print("Cross-model SHAP analysis for binary M>=5", flush=True)
    print("data_dir:", args.data_dir, flush=True)
    print("comparison_dir:", args.comparison_dir, flush=True)
    print("out_dir:", args.out_dir, flush=True)
    print("horizons:", args.horizons, flush=True)
    print("models:", args.models, flush=True)
    print("max_samples_per_subset:", args.max_samples_per_subset, flush=True)
    print("group_cols:", args.group_cols, flush=True)
    print("primary_group_col:", args.primary_group_col, flush=True)
    print("SHAP version:", getattr(shap, "__version__", "unknown"), flush=True)
    print("=" * 100, flush=True)

    rng = np.random.default_rng(args.seed)
    feature_tables = []
    group_tables = []
    run_rows = []

    if args.only_consistency:
        print("only_consistency=True: loading existing importance CSV files", flush=True)
        for p in sorted(Path(feature_dir).glob("shap_feature_importance_*.csv")):
            feature_tables.append(pd.read_csv(p))
        for p in sorted(Path(group_dir).glob("shap_group_importance_*.csv")):
            group_tables.append(pd.read_csv(p))
    else:
        for H in args.horizons:
            print("\n" + "#" * 100, flush=True)
            print(f"Load test data for H={H}m", flush=True)
            X, y, meta, payload = load_dataset(args.data_dir, H)
            test_mask = meta["split"].eq("test").values
            X_test = X[test_mask]
            y_test = y[test_mask]
            n_features = X.shape[1]
            feature_meta = load_feature_metadata(args.data_dir, n_features)

            print("X_test:", X_test.shape, "y_test positives:", int(np.sum(y_test == 1)), flush=True)

            for model_name in args.models:
                mpath = model_path(args.comparison_dir, model_name, H)
                ppath = prediction_path(args.comparison_dir, model_name, H)

                if not os.path.exists(mpath) or not os.path.exists(ppath):
                    msg = f"Missing model or prediction for {model_name} H={H}m: model={mpath}, pred={ppath}"
                    if args.skip_missing:
                        print("[SKIP]", msg, flush=True)
                        continue
                    raise FileNotFoundError(msg)

                print("\n" + "=" * 100, flush=True)
                print(f"SHAP model={model_name}, H={H}m", flush=True)
                print("model_path:", mpath, flush=True)
                print("prediction_path:", ppath, flush=True)

                model = joblib.load(mpath)
                pred_df = pd.read_csv(ppath)
                if len(pred_df) != len(X_test):
                    raise ValueError(
                        f"Prediction length {len(pred_df)} != X_test length {len(X_test)} for {model_name} H={H}m"
                    )

                union_idx, subset_pos, sample_info, subset_counts = build_model_horizon_sample(pred_df, args, rng)
                X_sample = X_test[union_idx]

                sample_info_path = os.path.join(sample_dir, f"sample_rows_{safe_model_name(model_name)}_H{H}m.csv")
                subset_count_path = os.path.join(sample_dir, f"subset_counts_{safe_model_name(model_name)}_H{H}m.csv")
                sample_info.to_csv(sample_info_path, index=False)
                subset_counts.to_csv(subset_count_path, index=False)

                print("Union SHAP sample:", X_sample.shape, flush=True)
                print(subset_counts.to_string(index=False), flush=True)

                shap_arr = compute_shap_values(model, model_name, X_sample)
                if shap_arr.shape != X_sample.shape:
                    raise ValueError(f"SHAP shape {shap_arr.shape} != X_sample shape {X_sample.shape}")
                print("SHAP array:", shap_arr.shape, flush=True)

                for subset_name, positions in subset_pos.items():
                    if len(positions) < args.min_subset_samples:
                        print(
                            f"[SKIP subset] {model_name} H={H}m {subset_name}: n={len(positions)} < {args.min_subset_samples}",
                            flush=True,
                        )
                        continue

                    sub_shap = shap_arr[positions]
                    fimp = make_feature_importance_table(
                        sub_shap,
                        feature_meta=feature_meta,
                        model_name=model_name,
                        horizon=H,
                        subset_name=subset_name,
                        n_subset=len(positions),
                    )

                    fimp_path = os.path.join(
                        feature_dir,
                        f"shap_feature_importance_{safe_model_name(model_name)}_H{H}m_{subset_name}.csv",
                    )
                    fimp.to_csv(fimp_path, index=False)
                    feature_tables.append(fimp)

                    if subset_name in args.heatmap_subsets or subset_name in ("all_test", "true_positive", "false_positive"):
                        png_path = os.path.join(
                            plot_dir,
                            f"top_features_{safe_model_name(model_name)}_H{H}m_{subset_name}.png",
                        )
                        plot_top_feature_bar(fimp, png_path, top_n=args.top_n_features_plot)

                    for group_col in args.group_cols:
                        if group_col not in fimp.columns:
                            print(f"[WARN] Missing group column {group_col}, skip", flush=True)
                            continue
                        gimp = make_group_importance_table(
                            fimp,
                            group_col=group_col,
                            model_name=model_name,
                            horizon=H,
                            subset_name=subset_name,
                            n_subset=len(positions),
                        )
                        gimp_path = os.path.join(
                            group_dir,
                            f"shap_group_importance_{safe_model_name(model_name)}_H{H}m_{subset_name}_{group_col}.csv",
                        )
                        gimp.to_csv(gimp_path, index=False)
                        group_tables.append(gimp)

                        if group_col == args.primary_group_col and subset_name in args.heatmap_subsets:
                            gpng_path = os.path.join(
                                plot_dir,
                                f"top_groups_{safe_model_name(model_name)}_H{H}m_{subset_name}.png",
                            )
                            plot_group_bar(gimp, gpng_path, top_n=args.heatmap_top_groups)

                    run_rows.append({
                        "model_name": model_name,
                        "horizon_month": H,
                        "subset_name": subset_name,
                        "n_subset_sampled": int(len(positions)),
                        "feature_importance_csv": fimp_path,
                    })

                del model, pred_df, X_sample, shap_arr
                gc.collect()

            del X, y, meta, payload, X_test, y_test
            gc.collect()

    # Concatenate and save all tables.
    all_feature_path = os.path.join(args.out_dir, "all_shap_feature_importance.csv")
    all_group_path = os.path.join(args.out_dir, "all_shap_group_importance.csv")

    feat_all = pd.concat(feature_tables, ignore_index=True, sort=False) if feature_tables else pd.DataFrame()
    group_all = pd.concat(group_tables, ignore_index=True, sort=False) if group_tables else pd.DataFrame()

    feat_all.to_csv(all_feature_path, index=False)
    group_all.to_csv(all_group_path, index=False)

    overlap_df, stable_features_df, group_overlap_df, stable_groups_df = build_consistency_tables(
        feature_tables, group_tables, args
    )

    feature_overlap_path = os.path.join(consistency_dir, "cross_model_top_feature_overlap.csv")
    stable_feature_path = os.path.join(consistency_dir, "cross_model_stable_top_features.csv")
    group_overlap_path = os.path.join(consistency_dir, "cross_model_top_group_overlap.csv")
    stable_group_path = os.path.join(consistency_dir, "cross_model_stable_top_groups.csv")

    overlap_df.to_csv(feature_overlap_path, index=False)
    stable_features_df.to_csv(stable_feature_path, index=False)
    group_overlap_df.to_csv(group_overlap_path, index=False)
    stable_groups_df.to_csv(stable_group_path, index=False)

    heatmaps = plot_group_heatmaps(group_all, args, plot_dir)

    run_log_path = os.path.join(args.out_dir, "shap_run_manifest.csv")
    pd.DataFrame(run_rows).to_csv(run_log_path, index=False)

    report_path = os.path.join(args.out_dir, "shap_cross_model_report.md")
    with open(report_path, "w") as f:
        f.write("# Cross-model SHAP report for binary M>=5 forecasting\n\n")
        f.write("## Configuration\n\n")
        f.write("```json\n")
        f.write(json.dumps({
            "data_dir": args.data_dir,
            "comparison_dir": args.comparison_dir,
            "out_dir": args.out_dir,
            "horizons": args.horizons,
            "models": args.models,
            "max_samples_per_subset": args.max_samples_per_subset,
            "min_subset_samples": args.min_subset_samples,
            "top_n_consistency": args.top_n_consistency,
            "group_cols": args.group_cols,
            "primary_group_col": args.primary_group_col,
            "heatmap_subsets": args.heatmap_subsets,
        }, indent=2))
        f.write("\n```\n\n")
        f.write("## Key output files\n\n")
        f.write(f"- all feature SHAP importance: `{all_feature_path}`\n")
        f.write(f"- all group SHAP importance: `{all_group_path}`\n")
        f.write(f"- feature overlap: `{feature_overlap_path}`\n")
        f.write(f"- stable top features: `{stable_feature_path}`\n")
        f.write(f"- group overlap: `{group_overlap_path}`\n")
        f.write(f"- stable top groups: `{stable_group_path}`\n")
        f.write(f"- run manifest: `{run_log_path}`\n")
        f.write("\n## Suggested interpretation\n\n")
        f.write(
            "Use the feature- and group-level overlap tables to identify signals that are robust across "
            "ExtraTrees, CatBoost and XGBoost. For earthquake forecasting, prioritize high-score and "
            "true-positive subsets when discussing reported-event reliability, and compare false positives "
            "to identify features that may drive alarms without corresponding M>=5 events.\n"
        )
        if heatmaps:
            f.write("\n## Heatmap plots\n\n")
            for p in heatmaps:
                f.write(f"- `{p}`\n")

    print("\n" + "=" * 100, flush=True)
    print("SHAP ANALYSIS FINISHED", flush=True)
    print("Saved:", all_feature_path, flush=True)
    print("Saved:", all_group_path, flush=True)
    print("Saved:", feature_overlap_path, flush=True)
    print("Saved:", stable_feature_path, flush=True)
    print("Saved:", group_overlap_path, flush=True)
    print("Saved:", stable_group_path, flush=True)
    print("Saved report:", report_path, flush=True)


if __name__ == "__main__":
    main()
