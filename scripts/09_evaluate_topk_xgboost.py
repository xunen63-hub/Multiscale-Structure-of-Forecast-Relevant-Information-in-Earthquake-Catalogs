import os
import gc
import json
import pickle
import argparse
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


CFG4 = {
    "config_name": "cfg4_strong_reg",
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.030,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_lambda": 8.0,
    "reg_alpha": 0.8,
    "min_child_weight": 5.0,
    "gamma": 0.20,
}


def expand_path(p):
    return str(Path(p).expanduser().resolve())


def parse_list(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def load_dataset(data_dir, horizon):
    path = os.path.join(data_dir, f"china_raw282_dataset_horizon_{horizon}m.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        d = pickle.load(f)

    X = np.asarray(d["X"], dtype=np.float32)

    if "y_m5" not in d:
        raise ValueError(f"Dataset does not contain y_m5: {path}")

    y = np.asarray(d["y_m5"], dtype=np.int64)

    meta = d["meta"].copy()
    meta["t0"] = pd.to_datetime(meta["t0"])

    if "future_end_inclusive" in meta.columns:
        meta["future_end_inclusive"] = pd.to_datetime(meta["future_end_inclusive"])

    return X, y, meta


def load_patch_orders(patch_csv):
    if patch_csv is None or str(patch_csv).strip() == "":
        return None

    patch_csv = expand_path(patch_csv)
    if not os.path.exists(patch_csv):
        raise FileNotFoundError(patch_csv)

    p = pd.read_csv(patch_csv)
    if "patch_order" not in p.columns:
        raise ValueError("patch_csv must contain patch_order.")

    return set(p["patch_order"].astype(int).tolist())


def region_mask(meta, patch_orders):
    if patch_orders is None:
        return np.ones(len(meta), dtype=bool)

    return meta["patch_order"].astype(int).isin(patch_orders).values


def make_model(seed, n_jobs):
    params = {k: v for k, v in CFG4.items() if k != "config_name"}

    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=seed,
        verbosity=1,
        **params,
    )


def safe_auc(y, score):
    try:
        if len(np.unique(y)) < 2:
            return np.nan
        return float(roc_auc_score(y, score))
    except Exception:
        return np.nan


def safe_ap(y, score):
    try:
        if len(np.unique(y)) < 2:
            return np.nan
        return float(average_precision_score(y, score))
    except Exception:
        return np.nan


def threshold_search(y_true, score):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)

    rows = []

    for th in np.linspace(0.01, 0.99, 99):
        pred = (score >= th).astype(int)

        rows.append({
            "threshold": float(th),
            "pred_positive_rate": float(np.mean(pred)),
            "acc": float(accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["f1", "precision", "threshold"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return float(df.loc[0, "threshold"]), df


def eval_binary(y_true, score, threshold):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)

    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    return {
        "positive_rate": float(np.mean(y_true)),
        "threshold": float(threshold),
        "pred_positive_rate": float(np.mean(pred)),
        "acc": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": safe_auc(y_true, score),
        "pr_auc": safe_ap(y_true, score),
        "brier": float(brier_score_loss(y_true, score)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }, pred


def role_masks(meta, fold_id):
    col = f"fold{fold_id}_role"
    if col not in meta.columns:
        raise ValueError(f"Missing fold role column: {col}")

    role = meta[col].astype(str).str.lower()

    train_mask = role.isin(["train", "inner_train", "fit", "training"]).values
    valid_mask = role.isin(["valid", "validation", "inner_valid", "val"]).values

    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        raise ValueError(
            f"Fold {fold_id} has empty train or valid mask. "
            f"Unique roles: {sorted(role.unique().tolist())}"
        )

    return train_mask, valid_mask


def inner_fold_thresholds(
    X,
    y,
    meta,
    rmask,
    folds,
    horizon,
    variant_name,
    out_dir,
    seed,
    n_jobs,
):
    valid_pred_rows = []
    fold_metric_rows = []

    for fold_id in folds:
        print(f"  Inner fold {fold_id}", flush=True)

        fold_train_mask, fold_valid_mask = role_masks(meta, fold_id)

        train_mask = rmask & fold_train_mask
        valid_mask = rmask & fold_valid_mask

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        meta_valid = meta[valid_mask].copy().reset_index(drop=True)

        if len(np.unique(y_train)) < 2:
            raise ValueError(
                f"Only one class in inner training data: "
                f"variant={variant_name}, H={horizon}, fold={fold_id}"
            )

        print(
            f"    train={X_train.shape}, valid={X_valid.shape}, "
            f"train_count={Counter(y_train)}, valid_count={Counter(y_valid)}",
            flush=True,
        )

        model = make_model(
            seed=seed + horizon * 10000 + fold_id * 100,
            n_jobs=n_jobs,
        )
        model.fit(X_train, y_train)

        score = model.predict_proba(X_valid)[:, 1]

        auc = safe_auc(y_valid, score)
        ap = safe_ap(y_valid, score)
        brier = float(brier_score_loss(y_valid, score))

        # Fold-local best threshold, only for diagnostics.
        fold_th, _ = threshold_search(y_valid, score)
        fold_metrics, fold_pred = eval_binary(y_valid, score, fold_th)

        fold_metric_rows.append({
            "variant_name": variant_name,
            "horizon_month": int(horizon),
            "fold_id": int(fold_id),
            "n_train": int(len(y_train)),
            "n_valid": int(len(y_valid)),
            "train_positive_rate": float(np.mean(y_train)),
            "valid_positive_rate": float(np.mean(y_valid)),
            "fold_best_threshold": float(fold_th),
            "auc": auc,
            "pr_auc": ap,
            "brier": brier,
            "precision": fold_metrics["precision"],
            "recall": fold_metrics["recall"],
            "f1": fold_metrics["f1"],
        })

        vdf = meta_valid.copy()
        vdf["variant_name"] = variant_name
        vdf["horizon_month"] = int(horizon)
        vdf["fold_id"] = int(fold_id)
        vdf["y_true"] = y_valid
        vdf["score"] = score
        valid_pred_rows.append(vdf)

        del X_train, y_train, X_valid, y_valid, model, score, vdf
        gc.collect()

    valid_pred_df = pd.concat(valid_pred_rows, ignore_index=True)
    fold_metric_df = pd.DataFrame(fold_metric_rows)

    pooled_y = valid_pred_df["y_true"].astype(int).values
    pooled_score = valid_pred_df["score"].astype(float).values

    pooled_th, threshold_curve = threshold_search(pooled_y, pooled_score)
    pooled_metrics, _ = eval_binary(pooled_y, pooled_score, pooled_th)

    threshold_summary = {
        "variant_name": variant_name,
        "horizon_month": int(horizon),
        "pooled_inner_valid_threshold": float(pooled_th),
        "pooled_inner_valid_positive_rate": float(np.mean(pooled_y)),
        "pooled_inner_valid_auc": safe_auc(pooled_y, pooled_score),
        "pooled_inner_valid_pr_auc": safe_ap(pooled_y, pooled_score),
        "pooled_inner_valid_brier": float(brier_score_loss(pooled_y, pooled_score)),
        "pooled_inner_valid_precision": pooled_metrics["precision"],
        "pooled_inner_valid_recall": pooled_metrics["recall"],
        "pooled_inner_valid_f1": pooled_metrics["f1"],
    }

    pred_dir = os.path.join(out_dir, "inner_valid_predictions")
    curve_dir = os.path.join(out_dir, "threshold_curves")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(curve_dir, exist_ok=True)

    valid_pred_df.to_csv(
        os.path.join(pred_dir, f"inner_valid_predictions_H{horizon}m.csv"),
        index=False,
    )

    threshold_curve.to_csv(
        os.path.join(curve_dir, f"threshold_curve_H{horizon}m.csv"),
        index=False,
    )

    return float(pooled_th), threshold_summary, fold_metric_df


def compare_with_full_baseline(final_df, full_baseline_metrics):
    if full_baseline_metrics is None or str(full_baseline_metrics).strip() == "":
        return final_df

    full_baseline_metrics = expand_path(full_baseline_metrics)
    if not os.path.exists(full_baseline_metrics):
        print(f"Warning: full baseline not found: {full_baseline_metrics}", flush=True)
        return final_df

    base = pd.read_csv(full_baseline_metrics)

    # The discovery-model table contains CatBoost, ExtraTrees, and XGBoost.
    # Keep the matched XGBoost row so one baseline remains per horizon.
    if "model_name" in base.columns:
        model_name = base["model_name"].astype(str).str.lower()
        base = base[model_name.str.replace("_", "", regex=False).eq("xgboost")]

    keep_cols = [
        "horizon_month",
        "f1",
        "precision",
        "recall",
        "auc",
        "pr_auc",
        "brier",
    ]

    missing = set(keep_cols) - set(base.columns)
    if missing:
        print(f"Warning: full baseline missing columns: {missing}", flush=True)
        return final_df

    if base["horizon_month"].duplicated().any():
        raise ValueError(
            "Full baseline must contain exactly one XGBoost row per horizon."
        )

    base = base[keep_cols].copy()
    base = base.rename(columns={c: f"full_{c}" for c in keep_cols if c != "horizon_month"})

    out = final_df.merge(base, on="horizon_month", how="left")

    for c in ["f1", "precision", "recall", "auc", "pr_auc", "brier"]:
        if c in out.columns and f"full_{c}" in out.columns:
            out[f"delta_{c}_minus_full"] = out[c] - out[f"full_{c}"]

    return out


def run_one_variant(args):
    data_dir = expand_path(args.data_dir)
    out_dir = expand_path(args.out_dir)

    os.makedirs(out_dir, exist_ok=True)

    model_dir = os.path.join(out_dir, "models")
    pred_dir = os.path.join(out_dir, "predictions")
    metric_dir = os.path.join(out_dir, "metrics")
    threshold_dir = os.path.join(out_dir, "thresholds")

    for d in [model_dir, pred_dir, metric_dir, threshold_dir]:
        os.makedirs(d, exist_ok=True)

    horizons = [int(x) for x in parse_list(args.horizons)]
    folds = [int(x) for x in parse_list(args.folds)]
    patch_orders = load_patch_orders(args.patch_csv)

    all_final_rows = []
    all_threshold_rows = []
    all_fold_metrics = []

    print("=" * 100, flush=True)
    print("Binary M>=5 reduced-feature ablation train + final test", flush=True)
    print("variant_name:", args.variant_name, flush=True)
    print("data_dir:", data_dir, flush=True)
    print("out_dir:", out_dir, flush=True)
    print("region_name:", args.region_name, flush=True)
    print("patch_csv:", args.patch_csv, flush=True)
    print("horizons:", horizons, flush=True)
    print("folds:", folds, flush=True)
    print("cfg:", CFG4, flush=True)
    print("=" * 100, flush=True)

    for H in horizons:
        print("\n" + "#" * 100, flush=True)
        print(f"Variant={args.variant_name} | Horizon={H}m", flush=True)

        X, y, meta = load_dataset(data_dir, H)

        rmask = region_mask(meta, patch_orders)

        print("Loaded X:", X.shape, flush=True)
        print("All y count:", Counter(y), flush=True)
        print("Region samples:", int(rmask.sum()), flush=True)

        threshold, th_summary, fold_metric_df = inner_fold_thresholds(
            X=X,
            y=y,
            meta=meta,
            rmask=rmask,
            folds=folds,
            horizon=H,
            variant_name=args.variant_name,
            out_dir=out_dir,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )

        all_threshold_rows.append(th_summary)
        all_fold_metrics.append(fold_metric_df)

        train_mask = rmask & meta["split"].eq("train_pool").values
        test_mask = rmask & meta["split"].eq("test").values

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        meta_test = meta[test_mask].copy().reset_index(drop=True)

        if len(np.unique(y_train)) < 2:
            raise ValueError(f"Only one class in final training data: H={H}")

        print(
            f"Final train={X_train.shape}, test={X_test.shape}, "
            f"train_count={Counter(y_train)}, test_count={Counter(y_test)}",
            flush=True,
        )

        model = make_model(seed=args.seed + H * 10000 + 999, n_jobs=args.n_jobs)
        model.fit(X_train, y_train)

        score = model.predict_proba(X_test)[:, 1]
        metrics, pred = eval_binary(y_test, score, threshold)

        row = {
            "variant_name": args.variant_name,
            "region_name": args.region_name,
            "horizon_month": int(H),
            "n_features": int(X.shape[1]),
            "config_name": CFG4["config_name"],
            "objective": "binary:logistic",
            "balanced_weight": False,
            "n_train": int(len(y_train)),
            "train_negative": int(np.sum(y_train == 0)),
            "train_positive": int(np.sum(y_train == 1)),
            "n_test": int(len(y_test)),
            "test_negative": int(np.sum(y_test == 0)),
            "test_positive": int(np.sum(y_test == 1)),
        }

        for k, v in CFG4.items():
            if k != "config_name":
                row[f"param_{k}"] = v

        row.update(metrics)
        all_final_rows.append(row)

        pred_df = meta_test.copy()
        pred_df["variant_name"] = args.variant_name
        pred_df["horizon_month"] = int(H)
        pred_df["y_true"] = y_test
        pred_df["score_m5"] = score
        pred_df["threshold"] = threshold
        pred_df["y_pred"] = pred

        pred_path = os.path.join(pred_dir, f"predictions_test_H{H}m.csv")
        model_path = os.path.join(model_dir, f"xgb_binary_m5_cfg4_H{H}m.joblib")

        pred_df.to_csv(pred_path, index=False)
        joblib.dump(model, model_path)

        print("Final metrics:", json.dumps(metrics, indent=2), flush=True)
        print("Saved predictions:", pred_path, flush=True)
        print("Saved model:", model_path, flush=True)

        del X, y, meta, X_train, y_train, X_test, y_test, model, score, pred, pred_df
        gc.collect()

    final_df = pd.DataFrame(all_final_rows).sort_values("horizon_month")
    threshold_df = pd.DataFrame(all_threshold_rows).sort_values("horizon_month")
    fold_metric_all = pd.concat(all_fold_metrics, ignore_index=True)

    final_comp_df = compare_with_full_baseline(final_df, args.full_baseline_metrics)

    final_path = os.path.join(out_dir, "final_test_metrics_binary_m5_ablation.csv")
    final_comp_path = os.path.join(out_dir, "final_test_metrics_binary_m5_ablation_vs_full.csv")
    threshold_path = os.path.join(threshold_dir, "pooled_inner_valid_thresholds.csv")
    fold_path = os.path.join(metric_dir, "inner_fold_metrics.csv")

    final_df.to_csv(final_path, index=False)
    final_comp_df.to_csv(final_comp_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    fold_metric_all.to_csv(fold_path, index=False)

    manifest = {
        "variant_name": args.variant_name,
        "data_dir": data_dir,
        "out_dir": out_dir,
        "region_name": args.region_name,
        "patch_csv": expand_path(args.patch_csv) if args.patch_csv else "",
        "horizons": horizons,
        "folds": folds,
        "config": CFG4,
        "balanced_weight": False,
        "objective": "binary:logistic",
        "threshold_protocol": "fixed cfg4; pooled inner-validation threshold per horizon; no final-test threshold tuning",
        "full_baseline_metrics": expand_path(args.full_baseline_metrics) if args.full_baseline_metrics else "",
    }

    with open(os.path.join(out_dir, "ablation_run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    show_cols = [
        "horizon_month",
        "n_features",
        "positive_rate",
        "threshold",
        "pred_positive_rate",
        "precision",
        "recall",
        "f1",
        "auc",
        "pr_auc",
        "brier",
        "tp",
        "fp",
        "fn",
        "tn",
    ]

    print("\n" + "=" * 100, flush=True)
    print("FINAL SUMMARY", flush=True)
    print(final_df[show_cols].to_string(index=False), flush=True)

    if "delta_f1_minus_full" in final_comp_df.columns:
        comp_cols = [
            "horizon_month",
            "n_features",
            "f1",
            "full_f1",
            "delta_f1_minus_full",
            "auc",
            "full_auc",
            "delta_auc_minus_full",
            "pr_auc",
            "full_pr_auc",
            "delta_pr_auc_minus_full",
            "brier",
            "full_brier",
            "delta_brier_minus_full",
        ]
        print("\nCOMPARE WITH FULL 282", flush=True)
        print(final_comp_df[comp_cols].to_string(index=False), flush=True)

    print("Saved:", final_path, flush=True)
    print("Saved:", final_comp_path, flush=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--variant-name", required=True)

    parser.add_argument("--region-name", default="all")
    parser.add_argument("--patch-csv", default="")

    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--folds", default="1,2,3")

    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--full-baseline-metrics",
        default="outputs/discovery_models/final_test_metrics_catboost_extratrees_xgb_binary_m5.csv",
    )

    args = parser.parse_args()
    run_one_variant(args)


if __name__ == "__main__":
    main()
