#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train last-month-only, annual-only and last100-only binary M>=5 models
using the selected hyperparameters and pooled decision thresholds from the
full raw-282 experiment.

Scientific design
-----------------
This is a fixed-configuration feature-sufficiency experiment:
  * model hyperparameters are read from selected_config_by_model.json;
  * decision thresholds are read from pooled_threshold_by_model_horizon.csv;
  * train/test samples, labels and split metadata come from the restricted
    datasets generated from the full raw-282 payloads;
  * no new tuning or threshold search is performed on restricted datasets.

Expected restricted data layout
-------------------------------
DATA_ROOT/
  last_month_only/china_last_month_only_dataset_horizon_{H}m.pkl
  annual_only/china_annual_only_dataset_horizon_{H}m.pkl
  last100_only/china_last100_only_dataset_horizon_{H}m.pkl

Expected full-result files
--------------------------
FULL_RESULT_DIR/
  selected_config_by_model.json
  pooled_threshold_by_model_horizon.csv
  final_test_metrics_catboost_extratrees_xgb_binary_m5.csv   # optional,
                                                              # used for deltas
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    CatBoostClassifier = None
    HAS_CATBOOST = False


ALLOWED_SUBSETS = ("last_month_only", "annual_only", "last100_only")
CANONICAL_MODELS = ("CatBoost", "ExtraTrees", "XGBoost")


# ============================================================================
# utilities
# ============================================================================


def clean_path_text(value: str) -> str:
    """Remove accidental NUL characters before expanding a path."""
    return str(value).replace("\x00", "").strip()


def expand_path(value: str) -> Path:
    return Path(clean_path_text(value)).expanduser().resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("No forecast horizons were supplied.")
    if any(v <= 0 for v in values):
        raise ValueError(f"All horizons must be positive: {values}")
    return values


def parse_subset_list(text: str) -> List[str]:
    values = [x.strip() for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("No feature subsets were supplied.")
    unknown = [x for x in values if x not in ALLOWED_SUBSETS]
    if unknown:
        raise ValueError(
            f"Unknown subsets {unknown}. Allowed: {list(ALLOWED_SUBSETS)}"
        )
    return values


def canonical_model_name(token: str) -> str:
    key = str(token).strip().lower().replace("_", "").replace("-", "")
    mapping = {
        "catboost": "CatBoost",
        "cat": "CatBoost",
        "extratrees": "ExtraTrees",
        "et": "ExtraTrees",
        "xgboost": "XGBoost",
        "xgb": "XGBoost",
    }
    if key not in mapping:
        raise ValueError(f"Unknown model name: {token}")
    return mapping[key]


def parse_model_list(text: str) -> List[str]:
    values = [canonical_model_name(x) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("No models were supplied.")
    # Keep order while removing duplicates.
    return list(dict.fromkeys(values))


def safe_file_token(text: str) -> str:
    return str(text).strip().lower().replace(" ", "_")


# ============================================================================
# full-experiment configuration and threshold loading
# ============================================================================


def load_selected_configs(path: Path, requested_models: Sequence[str]) -> Tuple[List[dict], dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Selected-config JSON not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "selected_configs" in payload:
        configs = payload["selected_configs"]
        metadata = {k: v for k, v in payload.items() if k != "selected_configs"}
    elif isinstance(payload, list):
        configs = payload
        metadata = {}
    else:
        raise ValueError(
            "selected_config_by_model.json must be a list or contain "
            "a 'selected_configs' list."
        )

    if not isinstance(configs, list) or len(configs) == 0:
        raise ValueError("No selected configurations found in JSON.")

    by_model: Dict[str, dict] = {}
    for cfg in configs:
        required = {"model_name", "config_name", "params"}
        missing = required.difference(cfg)
        if missing:
            raise ValueError(f"Selected config missing fields {sorted(missing)}: {cfg}")
        model_name = canonical_model_name(cfg["model_name"])
        normalized = {
            "model_name": model_name,
            "config_name": str(cfg["config_name"]),
            "params": dict(cfg["params"]),
        }
        if model_name in by_model:
            raise ValueError(f"Duplicate selected config for {model_name}")
        by_model[model_name] = normalized

    missing_models = [m for m in requested_models if m not in by_model]
    if missing_models:
        raise ValueError(
            f"Requested models absent from selected config JSON: {missing_models}"
        )

    selected = [by_model[m] for m in requested_models]

    if any(c["model_name"] == "XGBoost" for c in selected) and not HAS_XGBOOST:
        raise ImportError("xgboost is not installed in the active Python environment.")
    if any(c["model_name"] == "CatBoost" for c in selected) and not HAS_CATBOOST:
        raise ImportError("catboost is not installed in the active Python environment.")

    return selected, metadata


def load_threshold_map(
    path: Path,
    selected_configs: Sequence[dict],
    horizons: Sequence[int],
) -> Tuple[Dict[Tuple[str, int], float], pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"Threshold CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"model_name", "config_name", "horizon_month", "pooled_threshold"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Threshold CSV missing columns: {sorted(missing)}")

    work = df.copy()
    work["model_name"] = work["model_name"].map(canonical_model_name)
    work["config_name"] = work["config_name"].astype(str)
    work["horizon_month"] = pd.to_numeric(
        work["horizon_month"], errors="raise"
    ).astype(int)
    work["pooled_threshold"] = pd.to_numeric(
        work["pooled_threshold"], errors="raise"
    ).astype(float)

    selected_name = {c["model_name"]: c["config_name"] for c in selected_configs}
    threshold_map: Dict[Tuple[str, int], float] = {}

    for model_name, config_name in selected_name.items():
        for horizon in horizons:
            hit = work[
                work["model_name"].eq(model_name)
                & work["config_name"].eq(config_name)
                & work["horizon_month"].eq(int(horizon))
            ]
            if len(hit) != 1:
                raise ValueError(
                    "Expected exactly one pooled threshold for "
                    f"model={model_name}, config={config_name}, H={horizon}; "
                    f"found {len(hit)}."
                )
            threshold = float(hit.iloc[0]["pooled_threshold"])
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    f"Invalid threshold {threshold} for {model_name}, H={horizon}"
                )
            threshold_map[(model_name, int(horizon))] = threshold

    return threshold_map, work


# ============================================================================
# restricted dataset loading
# ============================================================================


def restricted_dataset_path(data_root: Path, subset: str, horizon: int) -> Path:
    return (
        data_root
        / subset
        / f"china_{subset}_dataset_horizon_{int(horizon)}m.pkl"
    )


def load_restricted_dataset(
    data_root: Path,
    subset: str,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, dict, Path]:
    path = restricted_dataset_path(data_root, subset, horizon)
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(f"Dataset payload is not a dictionary: {path}")
    if "X" not in payload or "meta" not in payload:
        raise KeyError(f"Dataset payload must contain X and meta: {path}")

    X = np.asarray(payload["X"], dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got {X.shape}: {path}")

    if "y_m5" in payload:
        y = np.asarray(payload["y_m5"], dtype=np.int64)
    elif "y_class" in payload:
        y = (np.asarray(payload["y_class"], dtype=np.int64) > 0).astype(np.int64)
    else:
        raise KeyError(f"Dataset payload has neither y_m5 nor y_class: {path}")

    meta = payload["meta"].copy()
    if not isinstance(meta, pd.DataFrame):
        meta = pd.DataFrame(meta)

    if len(X) != len(y) or len(X) != len(meta):
        raise ValueError(
            f"Length mismatch in {path}: X={len(X)}, y={len(y)}, meta={len(meta)}"
        )
    if "split" not in meta.columns:
        raise ValueError(f"meta lacks 'split' column: {path}")

    payload_subset = payload.get("feature_subset")
    if payload_subset is not None and str(payload_subset) != subset:
        raise ValueError(
            f"feature_subset mismatch: file requested as {subset}, payload says "
            f"{payload_subset}: {path}"
        )

    payload_horizon = payload.get("horizon_month")
    if payload_horizon is not None and int(payload_horizon) != int(horizon):
        raise ValueError(
            f"horizon mismatch: requested H{horizon}, payload says H{payload_horizon}: "
            f"{path}"
        )

    if "t0" in meta.columns:
        meta["t0"] = pd.to_datetime(meta["t0"])
    if "future_end_inclusive" in meta.columns:
        meta["future_end_inclusive"] = pd.to_datetime(
            meta["future_end_inclusive"]
        )

    return X, y, meta, payload, path


def load_patch_orders(patch_csv: str) -> Optional[set]:
    if patch_csv is None or clean_path_text(patch_csv) == "":
        return None

    path = expand_path(patch_csv)
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if "patch_order" not in df.columns:
        raise ValueError("patch_csv must contain column 'patch_order'.")
    return set(df["patch_order"].astype(int).tolist())


def make_region_mask(meta: pd.DataFrame, patch_orders: Optional[set]) -> np.ndarray:
    if patch_orders is None:
        return np.ones(len(meta), dtype=bool)
    if "patch_order" not in meta.columns:
        raise ValueError("meta lacks patch_order, cannot apply patch filter.")
    return meta["patch_order"].astype(int).isin(patch_orders).to_numpy()


# ============================================================================
# models, weights and metrics
# ============================================================================


def make_model(cfg: dict, seed: int, n_jobs: int):
    model_name = cfg["model_name"]
    params = dict(cfg["params"])

    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(
            random_state=seed,
            n_jobs=n_jobs,
            **params,
        )

    if model_name == "XGBoost":
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed.")
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=seed,
            verbosity=1,
            **params,
        )

    if model_name == "CatBoost":
        if not HAS_CATBOOST:
            raise ImportError("catboost is not installed.")
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            thread_count=n_jobs,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            **params,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def sample_weight_for_mode(y: np.ndarray, mode: str) -> Optional[np.ndarray]:
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"Unknown sample_weight_mode: {mode}")

    y = np.asarray(y, dtype=int)
    n = len(y)
    n0 = int(np.sum(y == 0))
    n1 = int(np.sum(y == 1))
    if n0 == 0 or n1 == 0:
        return None
    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)
    return np.where(y == 1, w1, w0).astype(np.float32)


def fit_model(model, X: np.ndarray, y: np.ndarray, sample_weight=None):
    if sample_weight is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


def positive_probability(model, X: np.ndarray) -> np.ndarray:
    probability = model.predict_proba(X)
    classes = getattr(model, "classes_", None)

    if classes is None:
        array = np.asarray(probability)
        if array.ndim == 1:
            return array.astype(float)
        return array[:, -1].astype(float)

    classes = np.asarray(classes)
    hit = np.where(classes == 1)[0]
    if len(hit) == 0:
        return np.zeros(X.shape[0], dtype=float)
    return np.asarray(probability)[:, int(hit[0])].astype(float)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, score))
    except Exception:
        return float("nan")


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, score))
    except Exception:
        return float("nan")


def evaluate_binary(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> Tuple[dict, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= float(threshold)).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    metrics = {
        "positive_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
        "threshold": float(threshold),
        "pred_positive_rate": float(np.mean(pred)) if len(pred) else np.nan,
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
    }
    return metrics, pred


def config_param_columns(cfg: dict) -> dict:
    return {f"param_{key}": value for key, value in cfg["params"].items()}


# ============================================================================
# output and comparison helpers
# ============================================================================


def existing_done_keys(metrics_path: Path) -> set:
    if not metrics_path.is_file():
        return set()
    df = pd.read_csv(metrics_path)
    required = {"feature_subset", "model_name", "horizon_month"}
    if not required.issubset(df.columns):
        return set()
    return {
        (str(row["feature_subset"]), str(row["model_name"]), int(row["horizon_month"]))
        for _, row in df.iterrows()
    }


def write_metrics_atomic(df: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, path)


def build_full_comparison(
    subset_metrics: pd.DataFrame,
    full_metrics_path: Path,
) -> Optional[pd.DataFrame]:
    if not full_metrics_path.is_file():
        return None

    full = pd.read_csv(full_metrics_path)
    required = {"model_name", "horizon_month"}
    if not required.issubset(full.columns):
        print(
            f"WARNING: full metrics file lacks {sorted(required)}: {full_metrics_path}",
            flush=True,
        )
        return None

    full = full.copy()
    full["model_name"] = full["model_name"].map(canonical_model_name)
    full["horizon_month"] = pd.to_numeric(
        full["horizon_month"], errors="raise"
    ).astype(int)

    metric_names = [
        "acc",
        "precision",
        "recall",
        "f1",
        "auc",
        "pr_auc",
        "brier",
        "pred_positive_rate",
    ]
    keep = ["model_name", "horizon_month"]
    if "config_name" in full.columns:
        keep.append("config_name")
    keep += [m for m in metric_names if m in full.columns]
    full = full[keep].copy()

    rename = {
        col: f"full_{col}"
        for col in full.columns
        if col not in {"model_name", "horizon_month", "config_name"}
    }
    if "config_name" in full.columns:
        rename["config_name"] = "full_config_name"
    full = full.rename(columns=rename)

    merged = subset_metrics.merge(
        full,
        on=["model_name", "horizon_month"],
        how="left",
        validate="many_to_one",
    )

    for metric in metric_names:
        full_col = f"full_{metric}"
        if metric in merged.columns and full_col in merged.columns:
            merged[f"delta_{metric}_only_minus_full"] = (
                merged[metric] - merged[full_col]
            )
            if metric in {"acc", "precision", "recall", "f1", "auc", "pr_auc"}:
                denominator = merged[full_col].replace(0, np.nan)
                merged[f"retained_fraction_{metric}"] = merged[metric] / denominator

    return merged


def write_markdown_report(
    metrics: pd.DataFrame,
    comparison: Optional[pd.DataFrame],
    path: Path,
    manifest: dict,
) -> None:
    show_cols = [
        "feature_subset",
        "horizon_month",
        "model_name",
        "config_name",
        "feature_dim",
        "threshold",
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
    show_cols = [c for c in show_cols if c in metrics.columns]

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Timescale-only fixed-full-configuration experiment\n\n")
        handle.write("No hyperparameter tuning or threshold search was performed on the restricted datasets.\n\n")
        handle.write("## Run manifest\n\n```json\n")
        handle.write(json.dumps(manifest, indent=2, ensure_ascii=False))
        handle.write("\n```\n\n")
        handle.write("## Restricted-feature final-test metrics\n\n")
        if len(metrics):
            try:
                handle.write(metrics[show_cols].to_markdown(index=False))
            except Exception:
                handle.write(metrics[show_cols].to_csv(index=False))
        else:
            handle.write("No metrics generated.")
        handle.write("\n")

        if comparison is not None:
            comp_cols = [
                "feature_subset",
                "horizon_month",
                "model_name",
                "pr_auc",
                "full_pr_auc",
                "delta_pr_auc_only_minus_full",
                "precision",
                "full_precision",
                "delta_precision_only_minus_full",
                "recall",
                "full_recall",
                "delta_recall_only_minus_full",
                "brier",
                "full_brier",
                "delta_brier_only_minus_full",
            ]
            comp_cols = [c for c in comp_cols if c in comparison.columns]
            handle.write("\n## Comparison with full raw-282 models\n\n")
            try:
                handle.write(comparison[comp_cols].to_markdown(index=False))
            except Exception:
                handle.write(comparison[comp_cols].to_csv(index=False))
            handle.write("\n")


# ============================================================================
# main experiment
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train restricted timescale-only datasets with fixed hyperparameters "
            "and thresholds inherited from the full raw-282 experiment."
        )
    )
    parser.add_argument(
        "--data-root",
        default="data/representation_only",
    )
    parser.add_argument(
        "--full-result-dir",
        default="outputs/discovery_models",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/representation_only",
    )
    parser.add_argument(
        "--subsets",
        default="last_month_only,annual_only,last100_only",
    )
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument(
        "--models",
        default="catboost,extratrees,xgboost",
    )
    parser.add_argument(
        "--sample-weight-mode",
        choices=["inherit", "none", "balanced"],
        default="inherit",
        help=(
            "inherit reads sample_weight_mode from selected_config_by_model.json; "
            "use an explicit value only to override it."
        ),
    )
    parser.add_argument("--region-name", default=None)
    parser.add_argument("--patch-csv", default="")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Do not save fitted joblib models; predictions and metrics are still saved.",
    )
    args = parser.parse_args()

    data_root = expand_path(args.data_root)
    full_result_dir = expand_path(args.full_result_dir)
    out_dir = ensure_dir(expand_path(args.out_dir))
    subsets = parse_subset_list(args.subsets)
    horizons = parse_int_list(args.horizons)
    requested_models = parse_model_list(args.models)

    selected_json = full_result_dir / "selected_config_by_model.json"
    threshold_csv = full_result_dir / "pooled_threshold_by_model_horizon.csv"
    full_metrics_path = (
        full_result_dir
        / "final_test_metrics_catboost_extratrees_xgb_binary_m5.csv"
    )

    selected_configs, selected_metadata = load_selected_configs(
        selected_json, requested_models
    )
    threshold_map, threshold_df = load_threshold_map(
        threshold_csv, selected_configs, horizons
    )

    if args.sample_weight_mode == "inherit":
        sample_weight_mode = str(
            selected_metadata.get("sample_weight_mode", "none")
        ).strip().lower()
        if sample_weight_mode not in {"none", "balanced"}:
            raise ValueError(
                "Unsupported inherited sample_weight_mode "
                f"'{sample_weight_mode}' in {selected_json}"
            )
    else:
        sample_weight_mode = args.sample_weight_mode

    inherited_region_name = selected_metadata.get("region_name", "all")
    region_name = (
        str(args.region_name)
        if args.region_name is not None
        else str(inherited_region_name)
    )
    patch_orders = load_patch_orders(args.patch_csv)

    models_dir = ensure_dir(out_dir / "models")
    predictions_dir = ensure_dir(out_dir / "predictions")
    metrics_path = out_dir / "final_test_metrics_timescale_only_fixed_full_params.csv"
    comparison_path = out_dir / "comparison_timescale_only_vs_full.csv"
    manifest_path = out_dir / "timescale_only_fixed_full_params_manifest.json"
    report_path = out_dir / "timescale_only_fixed_full_params_report.md"

    if not data_root.is_dir():
        raise NotADirectoryError(data_root)
    if not full_result_dir.is_dir():
        raise NotADirectoryError(full_result_dir)

    print("=" * 110, flush=True)
    print("Timescale-only binary M>=5 experiment with fixed full-model settings", flush=True)
    print(f"data_root:          {data_root}", flush=True)
    print(f"full_result_dir:    {full_result_dir}", flush=True)
    print(f"out_dir:            {out_dir}", flush=True)
    print(f"subsets:            {subsets}", flush=True)
    print(f"horizons:           {horizons}", flush=True)
    print(f"models:             {requested_models}", flush=True)
    print(f"sample_weight_mode: {sample_weight_mode}", flush=True)
    print(f"region_name:        {region_name}", flush=True)
    print(f"patch_csv:          {clean_path_text(args.patch_csv)}", flush=True)
    print(f"n_jobs:             {args.n_jobs}", flush=True)
    print(f"seed:               {args.seed}", flush=True)
    print(f"HAS_CATBOOST:       {HAS_CATBOOST}", flush=True)
    print(f"HAS_XGBOOST:        {HAS_XGBOOST}", flush=True)
    print("Selected configurations:", flush=True)
    print(json.dumps(selected_configs, indent=2), flush=True)
    print("Fixed pooled thresholds:", flush=True)
    print(
        threshold_df[
            threshold_df["model_name"].isin(requested_models)
            & threshold_df["horizon_month"].isin(horizons)
        ][["model_name", "config_name", "horizon_month", "pooled_threshold"]]
        .sort_values(["model_name", "horizon_month"])
        .to_string(index=False),
        flush=True,
    )
    print("=" * 110, flush=True)

    old_metrics = pd.DataFrame()
    done = set()
    if args.resume and metrics_path.is_file():
        old_metrics = pd.read_csv(metrics_path)
        done = existing_done_keys(metrics_path)
        print(f"Resume enabled: {len(done)} completed runs found.", flush=True)

    new_rows: List[dict] = []
    skipped_files: List[str] = []

    # Use the selected-config order exactly as saved/read. This reproduces the
    # full-script seed formula for each model and horizon. The subset is not
    # included in the seed so all feature subsets use identical algorithmic seeds.
    for subset in subsets:
        for horizon in horizons:
            path = restricted_dataset_path(data_root, subset, horizon)
            if not path.is_file():
                if args.skip_missing:
                    print(f"[SKIP] Missing dataset: {path}", flush=True)
                    skipped_files.append(str(path))
                    continue
                raise FileNotFoundError(path)

            print("\n" + "#" * 110, flush=True)
            print(f"Load subset={subset}, H={horizon}m", flush=True)
            X, y, meta, payload, source_path = load_restricted_dataset(
                data_root, subset, horizon
            )
            region_mask = make_region_mask(meta, patch_orders)
            train_mask = region_mask & meta["split"].eq("train_pool").to_numpy()
            test_mask = region_mask & meta["split"].eq("test").to_numpy()

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
            meta_test = meta.loc[test_mask].copy().reset_index(drop=True)

            print(f"Source: {source_path}", flush=True)
            print(f"X full subset shape: {X.shape}", flush=True)
            print(f"Train: {X_train.shape}, labels={Counter(y_train)}", flush=True)
            print(f"Test:  {X_test.shape}, labels={Counter(y_test)}", flush=True)

            if len(y_train) == 0 or len(y_test) == 0:
                raise ValueError(f"Empty train/test for subset={subset}, H={horizon}")
            if len(np.unique(y_train)) < 2:
                raise ValueError(
                    f"Only one class in training data for subset={subset}, H={horizon}"
                )

            for cfg_idx, cfg in enumerate(selected_configs):
                model_name = cfg["model_name"]
                config_name = cfg["config_name"]
                key = (subset, model_name, int(horizon))
                if key in done:
                    print(f"[RESUME] Skip completed run: {key}", flush=True)
                    continue

                threshold = threshold_map[(model_name, int(horizon))]
                # Exact formula used in the full final-test stage.
                model_seed = (
                    int(args.seed)
                    + 900000
                    + 10000 * (cfg_idx + 1)
                    + int(horizon) * 100
                )

                print("\n" + "=" * 110, flush=True)
                print(
                    f"Train subset={subset}, model={model_name}, "
                    f"config={config_name}, H={horizon}m",
                    flush=True,
                )
                print(f"feature_dim={X.shape[1]}", flush=True)
                print(f"fixed threshold={threshold}", flush=True)
                print(f"model_seed={model_seed}", flush=True)

                model = make_model(cfg, seed=model_seed, n_jobs=args.n_jobs)
                sample_weight = sample_weight_for_mode(y_train, sample_weight_mode)
                fit_model(model, X_train, y_train, sample_weight=sample_weight)
                score = positive_probability(model, X_test)
                metrics, pred = evaluate_binary(y_test, score, threshold)

                row = {
                    "region_name": region_name,
                    "feature_subset": subset,
                    "feature_dim": int(X.shape[1]),
                    "model_name": model_name,
                    "config_name": config_name,
                    "horizon_month": int(horizon),
                    "objective": "binary_Mge5",
                    "comparison_design": "fixed_full_hyperparameters_and_threshold",
                    "sample_weight_mode": sample_weight_mode,
                    "model_seed": int(model_seed),
                    "source_dataset": str(source_path),
                    "selected_config_json": str(selected_json),
                    "threshold_csv": str(threshold_csv),
                    "n_train": int(len(y_train)),
                    "train_negative": int(np.sum(y_train == 0)),
                    "train_positive": int(np.sum(y_train == 1)),
                    "n_test": int(len(y_test)),
                    "test_negative": int(np.sum(y_test == 0)),
                    "test_positive": int(np.sum(y_test == 1)),
                }
                row.update(config_param_columns(cfg))
                row.update(metrics)
                new_rows.append(row)

                model_token = safe_file_token(model_name)
                subset_pred_dir = ensure_dir(predictions_dir / subset)
                subset_model_dir = ensure_dir(models_dir / subset)

                prediction = meta_test.copy()
                prediction["feature_subset"] = subset
                prediction["feature_dim"] = int(X.shape[1])
                prediction["model_name"] = model_name
                prediction["config_name"] = config_name
                prediction["horizon_month"] = int(horizon)
                prediction["y_true"] = y_test.astype(int)
                prediction["score_m5"] = score.astype(float)
                prediction["threshold_from_full_model"] = float(threshold)
                prediction["y_pred"] = pred.astype(int)
                pred_path = (
                    subset_pred_dir
                    / f"predictions_test_{model_token}_H{horizon}m.csv"
                )
                prediction.to_csv(pred_path, index=False)

                model_path = (
                    subset_model_dir
                    / f"{model_token}_binary_m5_H{horizon}m.joblib"
                )
                if not args.no_save_models:
                    joblib.dump(model, model_path)

                print("Metrics:", json.dumps(metrics, indent=2), flush=True)
                print(f"Saved predictions: {pred_path}", flush=True)
                if not args.no_save_models:
                    print(f"Saved model:       {model_path}", flush=True)

                # Save incrementally to protect long runs from interruption.
                current = pd.concat(
                    [old_metrics, pd.DataFrame(new_rows)], ignore_index=True
                )
                if len(current):
                    current = (
                        current.sort_values(
                            ["feature_subset", "horizon_month", "model_name"]
                        )
                        .drop_duplicates(
                            ["feature_subset", "horizon_month", "model_name"],
                            keep="last",
                        )
                        .reset_index(drop=True)
                    )
                write_metrics_atomic(current, metrics_path)

                del model, score, pred, prediction, sample_weight
                gc.collect()

            del X, y, meta, payload, X_train, y_train, X_test, y_test, meta_test
            gc.collect()

    final_metrics = pd.concat(
        [old_metrics, pd.DataFrame(new_rows)], ignore_index=True
    )
    if len(final_metrics):
        final_metrics = (
            final_metrics.sort_values(
                ["feature_subset", "horizon_month", "model_name"]
            )
            .drop_duplicates(
                ["feature_subset", "horizon_month", "model_name"],
                keep="last",
            )
            .reset_index(drop=True)
        )
    write_metrics_atomic(final_metrics, metrics_path)

    comparison = build_full_comparison(final_metrics, full_metrics_path)
    if comparison is not None:
        comparison = comparison.sort_values(
            ["feature_subset", "horizon_month", "model_name"]
        ).reset_index(drop=True)
        comparison.to_csv(comparison_path, index=False)

    manifest = {
        "data_root": str(data_root),
        "full_result_dir": str(full_result_dir),
        "out_dir": str(out_dir),
        "selected_config_json": str(selected_json),
        "threshold_csv": str(threshold_csv),
        "full_metrics_path": str(full_metrics_path),
        "subsets": subsets,
        "horizons": horizons,
        "models": requested_models,
        "selected_configs": selected_configs,
        "fixed_thresholds": [
            {
                "model_name": model_name,
                "horizon_month": horizon,
                "threshold": threshold,
            }
            for (model_name, horizon), threshold in sorted(threshold_map.items())
        ],
        "sample_weight_mode": sample_weight_mode,
        "region_name": region_name,
        "patch_csv": clean_path_text(args.patch_csv),
        "n_jobs": int(args.n_jobs),
        "seed": int(args.seed),
        "save_models": not args.no_save_models,
        "no_restricted_feature_tuning": True,
        "no_restricted_feature_threshold_search": True,
        "skipped_files": skipped_files,
        "n_completed_rows": int(len(final_metrics)),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    write_markdown_report(
        metrics=final_metrics,
        comparison=comparison,
        path=report_path,
        manifest=manifest,
    )

    show_cols = [
        "feature_subset",
        "horizon_month",
        "model_name",
        "feature_dim",
        "threshold",
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
    show_cols = [c for c in show_cols if c in final_metrics.columns]

    print("\n" + "=" * 110, flush=True)
    print("FINAL TIMESCALE-ONLY SUMMARY", flush=True)
    print("=" * 110, flush=True)
    if len(final_metrics):
        print(final_metrics[show_cols].to_string(index=False), flush=True)
    else:
        print("No metrics generated.", flush=True)
    print(f"Metrics:    {metrics_path}", flush=True)
    if comparison is not None:
        print(f"Comparison: {comparison_path}", flush=True)
    else:
        print(
            f"Comparison not written because full metrics were unavailable: "
            f"{full_metrics_path}",
            flush=True,
        )
    print(f"Manifest:   {manifest_path}", flush=True)
    print(f"Report:     {report_path}", flush=True)


if __name__ == "__main__":
    main()
