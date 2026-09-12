#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_validate_frozen_groups_heldout.py

Held-out-architecture validation of frozen feature groups.

Scientific design
-----------------
Discovery models:
    CatBoost + ExtraTrees + XGBoost
Discovery data:
    inner rolling-validation folds only
Discovery output:
    selected_groups_inner_validation.csv

Validation models:
    Random Forest + LightGBM
Validation rule:
    the frozen group list is loaded verbatim; this script never reads SHAP
    results and never re-ranks or re-selects groups.

For each held-out model and forecast horizon, the script fits:
    full_features
    only_top1_groups / only_top2_groups / only_top3_groups
    remove_top1_groups / remove_top2_groups / remove_top3_groups

Default evaluation mode is ``auc_only`` because ROC-AUC is the primary metric
and does not require threshold selection. Use ``full_metrics`` only when
threshold-dependent supplementary metrics are also needed.

Required inputs
---------------
1. Raw-282 datasets:
   china_raw282_dataset_horizon_{H}m.pkl
2. Feature metadata:
   feature_metadata_282_relative_lag.csv
3. Frozen discovery groups from script 05:
   selected_groups_inner_validation.csv
4. RF/LightGBM selected hyperparameters from the earlier comparison:
   selected_config_by_model.json

Main outputs
------------
- frozen_selected_groups_used.csv
- heldout_feature_sets.csv
- heldout_final_test_metrics.csv
- heldout_deltas_vs_full.csv
- heldout_summary_by_horizon_feature_set.csv
- heldout_summary_by_model_feature_set.csv
- heldout_summary_overall.csv
- heldout_validation_manifest.json
- heldout_validation_report.md
- predictions/*.csv
- models/*.joblib (optional)
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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

try:
    from lightgbm import LGBMClassifier
except Exception as exc:  # checked at runtime on HPC
    LGBMClassifier = None
    LIGHTGBM_IMPORT_ERROR = repr(exc)
else:
    LIGHTGBM_IMPORT_ERROR = ""

warnings.filterwarnings("ignore", category=UserWarning)


MODEL_ALIASES = {
    "rf": "RF",
    "randomforest": "RF",
    "randomforestclassifier": "RF",
    "random_forest": "RF",
    "random forest": "RF",
    "lightgbm": "LightGBM",
    "lgbm": "LightGBM",
    "lgbmclassifier": "LightGBM",
}

DISCOVERY_ALIASES = {
    "cat": "CatBoost",
    "catboost": "CatBoost",
    "et": "ExtraTrees",
    "extratrees": "ExtraTrees",
    "extra_trees": "ExtraTrees",
    "xgb": "XGBoost",
    "xgboost": "XGBoost",
}


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def expand_path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_int_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return list(dict.fromkeys(values))


def parse_float_list(text: str) -> List[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float.")
    return values


def normalize_token(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def canonical_heldout_model(value: str) -> str:
    raw = str(value).strip().lower()
    direct_key = raw.replace("-", "_")
    if direct_key in MODEL_ALIASES:
        return MODEL_ALIASES[direct_key]
    key = normalize_token(value)
    normalized_map = {normalize_token(k): v for k, v in MODEL_ALIASES.items()}
    if key not in normalized_map:
        raise ValueError(f"Unknown held-out model name: {value!r}")
    return normalized_map[key]


def canonical_discovery_model(value: str) -> str:
    raw = str(value).strip().lower().replace(" ", "_")
    if raw in DISCOVERY_ALIASES:
        return DISCOVERY_ALIASES[raw]
    key = normalize_token(value)
    normalized_map = {normalize_token(k): v for k, v in DISCOVERY_ALIASES.items()}
    if key not in normalized_map:
        raise ValueError(f"Unknown discovery model name: {value!r}")
    return normalized_map[key]


def parse_heldout_models(text: str) -> List[str]:
    models = [canonical_heldout_model(x) for x in str(text).split(",") if x.strip()]
    if not models:
        raise ValueError("No held-out validation models were supplied.")
    return list(dict.fromkeys(models))


def parse_discovery_models(text: str) -> List[str]:
    models = [canonical_discovery_model(x) for x in str(text).split(",") if x.strip()]
    if not models:
        raise ValueError("No expected discovery models were supplied.")
    return list(dict.fromkeys(models))


def safe_name(text: str) -> str:
    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "__")
        .replace("+", "plus")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_if_exists(paths: Iterable[Path], resume: bool) -> None:
    if resume:
        return
    for path in paths:
        if path.exists():
            path.unlink()


def append_row_csv(path: Path, row: Mapping[str, object]) -> None:
    new = pd.DataFrame([dict(row)])
    if path.exists():
        old = pd.read_csv(path)
        columns = list(old.columns)
        for column in new.columns:
            if column not in columns:
                columns.append(column)
        old = old.reindex(columns=columns)
        new = new.reindex(columns=columns)
        pd.concat([old, new], ignore_index=True).to_csv(path, index=False)
    else:
        new.to_csv(path, index=False)


# -----------------------------------------------------------------------------
# Data and feature metadata
# -----------------------------------------------------------------------------


def dataset_path(data_dir: Path, horizon: int) -> Path:
    return data_dir / f"china_raw282_dataset_horizon_{horizon}m.pkl"


def load_dataset(data_dir: Path, horizon: int):
    path = dataset_path(data_dir, horizon)
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("rb") as handle:
        payload = pickle.load(handle)

    X = np.asarray(payload["X"], dtype=np.float32)
    if "y_m5" in payload:
        y = np.asarray(payload["y_m5"], dtype=np.int64)
    else:
        y = (np.asarray(payload["y_class"], dtype=np.int64) > 0).astype(np.int64)

    meta = payload["meta"].copy()
    if "t0" in meta.columns:
        meta["t0"] = pd.to_datetime(meta["t0"])
    return X, y, meta, payload, path


def load_feature_metadata(data_dir: Path, n_features: int) -> Tuple[pd.DataFrame, Path]:
    path = data_dir / "feature_metadata_282_relative_lag.csv"
    if not path.is_file():
        raise FileNotFoundError(path)

    metadata = pd.read_csv(path)
    if "feature_idx" not in metadata.columns:
        metadata.insert(0, "feature_idx", np.arange(len(metadata), dtype=int))
    metadata = metadata.sort_values("feature_idx").reset_index(drop=True)

    if len(metadata) != n_features:
        raise ValueError(
            f"Feature metadata length {len(metadata)} != dataset feature count {n_features}"
        )
    expected_index = np.arange(n_features, dtype=int)
    actual_index = metadata["feature_idx"].astype(int).to_numpy()
    if not np.array_equal(actual_index, expected_index):
        raise ValueError("feature_idx must be a complete ordered sequence 0..n_features-1")

    if "time_group" not in metadata.columns or "signal_group" not in metadata.columns:
        raise ValueError("Feature metadata must contain time_group and signal_group.")
    if "feature_group" not in metadata.columns:
        metadata["feature_group"] = (
            metadata["time_group"].fillna("unknown").astype(str)
            + "/"
            + metadata["signal_group"].fillna("unknown").astype(str)
        )
    if "feature_name" not in metadata.columns:
        metadata["feature_name"] = [f"feature_{i:03d}" for i in range(n_features)]

    return metadata, path


# -----------------------------------------------------------------------------
# Frozen group loading and leakage guards
# -----------------------------------------------------------------------------


def split_model_field(value: object) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).replace(";", ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def truthy_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def load_frozen_groups(
    path: Path,
    group_col: str,
    horizons: Sequence[int],
    max_top_k: int,
    expected_discovery_models: Sequence[str],
    expected_folds: int,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    groups = pd.read_csv(path)
    required = {"horizon_month", "rank", "group_name", "source_stage"}
    missing = required.difference(groups.columns)
    if missing:
        raise ValueError(f"Frozen-group CSV is missing columns: {sorted(missing)}")

    groups["horizon_month"] = pd.to_numeric(
        groups["horizon_month"], errors="raise"
    ).astype(int)
    groups["rank"] = pd.to_numeric(groups["rank"], errors="raise").astype(int)
    groups["group_name"] = groups["group_name"].astype(str)

    source = groups["source_stage"].astype(str).str.strip().str.lower()
    if not bool(source.eq("inner_validation").all()):
        bad = sorted(groups.loc[~source.eq("inner_validation"), "source_stage"].astype(str).unique())
        raise ValueError(
            "Leakage guard: groups were not selected exclusively from inner validation. "
            f"Found source_stage values: {bad}"
        )

    if "group_col" in groups.columns:
        wrong = groups["group_col"].astype(str).ne(group_col)
        if bool(wrong.any()):
            bad = sorted(groups.loc[wrong, "group_col"].astype(str).unique())
            raise ValueError(
                f"Frozen group_col values {bad} do not match --group-col={group_col}"
            )

    for column in ["used_final_test_for_selection", "used_test_for_selection"]:
        if column in groups.columns and bool(truthy_series(groups[column]).any()):
            raise ValueError(f"Leakage guard: {column} indicates final-test use.")

    for column in ["shap_subset", "subset_name", "selection_subset"]:
        if column in groups.columns:
            contains_test = groups[column].astype(str).str.lower().str.contains("test", regex=False)
            if bool(contains_test.any()):
                raise ValueError(
                    f"Leakage guard: {column} contains test-derived selection labels."
                )

    duplicate = groups.duplicated(["horizon_month", "rank"], keep=False)
    if bool(duplicate.any()):
        bad = groups.loc[duplicate, ["horizon_month", "rank", "group_name"]]
        raise ValueError(
            "Duplicate frozen horizon/rank rows:\n" + bad.to_string(index=False)
        )

    expected_set = set(expected_discovery_models)
    if "models" not in groups.columns:
        raise ValueError(
            "Frozen-group CSV must contain a 'models' column so discovery-model "
            "provenance can be verified."
        )
    for row_index, value in groups["models"].items():
        row_models = {canonical_discovery_model(x) for x in split_model_field(value)}
        if row_models != expected_set:
            raise ValueError(
                f"Discovery-model audit failed at row {row_index}: "
                f"found {sorted(row_models)}, expected {sorted(expected_set)}"
            )

    if "n_models" in groups.columns:
        n_models = pd.to_numeric(groups["n_models"], errors="raise").astype(int)
        if not bool(n_models.eq(len(expected_set)).all()):
            raise ValueError("Frozen groups do not have complete discovery-model coverage.")

    if "n_folds" in groups.columns:
        n_folds = pd.to_numeric(groups["n_folds"], errors="raise").astype(int)
        if not bool(n_folds.eq(expected_folds).all()):
            raise ValueError(
                f"Frozen groups do not have the expected {expected_folds}-fold coverage."
            )

    if "complete_model_fold_coverage" in groups.columns:
        if not bool(truthy_series(groups["complete_model_fold_coverage"]).all()):
            raise ValueError("Frozen groups include incomplete model-fold coverage.")

    requested = groups[groups["horizon_month"].isin([int(h) for h in horizons])].copy()
    missing_horizons = sorted(set(map(int, horizons)).difference(requested["horizon_month"].unique()))
    if missing_horizons:
        raise ValueError(f"Frozen groups are missing horizons: {missing_horizons}")

    for horizon in horizons:
        subset = requested[requested["horizon_month"].eq(int(horizon))].sort_values("rank")
        actual = subset.head(max_top_k)["rank"].astype(int).tolist()
        expected = list(range(1, max_top_k + 1))
        if actual != expected:
            raise ValueError(
                f"H={horizon} frozen ranks are {actual}; expected {expected}."
            )

    return requested.sort_values(["horizon_month", "rank"]).reset_index(drop=True)


def groups_for_horizon(groups: pd.DataFrame, horizon: int, max_top_k: int) -> pd.DataFrame:
    subset = (
        groups[groups["horizon_month"].eq(int(horizon))]
        .sort_values("rank")
        .head(max_top_k)
        .copy()
    )
    if len(subset) != max_top_k:
        raise ValueError(f"H={horizon} has {len(subset)} groups; expected {max_top_k}.")
    return subset


def build_feature_sets(
    feature_metadata: pd.DataFrame,
    selected_groups: pd.DataFrame,
    top_k_list: Sequence[int],
    group_col: str,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    if group_col not in feature_metadata.columns:
        raise ValueError(f"Feature metadata has no group column {group_col!r}")

    all_features = feature_metadata["feature_idx"].astype(int).to_numpy()
    group_to_features = {
        str(group_name): sub["feature_idx"].astype(int).to_numpy()
        for group_name, sub in feature_metadata.groupby(group_col, sort=False)
    }

    ranked_groups = selected_groups.sort_values("rank")["group_name"].astype(str).tolist()
    unknown = [name for name in ranked_groups if name not in group_to_features]
    if unknown:
        raise ValueError(f"Frozen groups absent from feature metadata: {unknown}")

    masks: Dict[str, np.ndarray] = {"full_features": all_features}
    rows = [
        {
            "feature_set_name": "full_features",
            "feature_set_type": "full",
            "top_k_groups": 0,
            "groups_used": "ALL",
            "n_features": int(len(all_features)),
        }
    ]

    for top_k in top_k_list:
        group_names = ranked_groups[: int(top_k)]
        retained = np.unique(
            np.concatenate([group_to_features[name] for name in group_names])
        ).astype(int)

        only_name = f"only_top{top_k}_groups"
        masks[only_name] = retained
        rows.append(
            {
                "feature_set_name": only_name,
                "feature_set_type": "retain_only",
                "top_k_groups": int(top_k),
                "groups_used": "; ".join(group_names),
                "n_features": int(len(retained)),
            }
        )

        retained_set = set(retained.tolist())
        remaining = np.asarray(
            [index for index in all_features if int(index) not in retained_set],
            dtype=int,
        )
        remove_name = f"remove_top{top_k}_groups"
        masks[remove_name] = remaining
        rows.append(
            {
                "feature_set_name": remove_name,
                "feature_set_type": "remove",
                "top_k_groups": int(top_k),
                "groups_used": "; ".join(group_names),
                "n_features": int(len(remaining)),
            }
        )

    return masks, pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Held-out model configuration and fitting
# -----------------------------------------------------------------------------


def load_selected_configs(
    config_json: Path,
    requested_models: Sequence[str],
) -> Tuple[Dict[str, dict], dict]:
    if not config_json.is_file():
        raise FileNotFoundError(config_json)

    with config_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "selected_configs" in payload:
        raw_configs = payload["selected_configs"]
        metadata = {key: value for key, value in payload.items() if key != "selected_configs"}
    elif isinstance(payload, list):
        raw_configs = payload
        metadata = {}
    else:
        raise ValueError(
            "Config JSON must be a list or a dictionary containing selected_configs."
        )

    by_model: Dict[str, dict] = {}
    for item in raw_configs:
        required = {"model_name", "config_name", "params"}
        missing = required.difference(item)
        if missing:
            raise ValueError(f"Selected config missing fields {sorted(missing)}: {item}")
        try:
            model_name = canonical_heldout_model(item["model_name"])
        except ValueError:
            # The JSON may also contain the three discovery models. Ignore them.
            continue
        if model_name in by_model:
            raise ValueError(f"Duplicate selected configuration for {model_name}")
        by_model[model_name] = {
            "model_name": model_name,
            "config_name": str(item["config_name"]),
            "params": dict(item["params"]),
        }

    missing_models = [model for model in requested_models if model not in by_model]
    if missing_models:
        raise ValueError(
            f"Held-out config JSON lacks selected configurations for {missing_models}. "
            f"Available held-out models: {sorted(by_model)}"
        )

    if "selection_rule" not in metadata:
        print(
            "[WARN] selected-config JSON has no selection_rule metadata; "
            "hyperparameters will still be used as supplied.",
            flush=True,
        )

    return {model: by_model[model] for model in requested_models}, metadata


def make_model(model_name: str, params: Mapping[str, object], seed: int, n_jobs: int):
    clean = dict(params)
    for runtime_key in ["random_state", "random_seed", "n_jobs", "num_threads", "nthread"]:
        clean.pop(runtime_key, None)

    if model_name == "RF":
        return RandomForestClassifier(
            random_state=int(seed),
            n_jobs=int(n_jobs),
            **clean,
        )

    if model_name == "LightGBM":
        if LGBMClassifier is None:
            raise ImportError(f"lightgbm is unavailable: {LIGHTGBM_IMPORT_ERROR}")
        clean.setdefault("objective", "binary")
        clean.setdefault("verbosity", -1)
        return LGBMClassifier(
            random_state=int(seed),
            n_jobs=int(n_jobs),
            **clean,
        )

    raise ValueError(f"Unsupported held-out model: {model_name}")


def balanced_sample_weight(y: np.ndarray) -> np.ndarray | None:
    y = np.asarray(y, dtype=int)
    n = len(y)
    n0 = int(np.sum(y == 0))
    n1 = int(np.sum(y == 1))
    if n0 == 0 or n1 == 0:
        return None
    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)
    return np.where(y == 1, w1, w0).astype(np.float32)


def fit_model(model, X: np.ndarray, y: np.ndarray, sample_weight_mode: str):
    if sample_weight_mode == "none":
        model.fit(X, y)
    elif sample_weight_mode == "balanced":
        weights = balanced_sample_weight(y)
        if weights is None:
            model.fit(X, y)
        else:
            model.fit(X, y, sample_weight=weights)
    else:
        raise ValueError(f"Unsupported sample_weight_mode: {sample_weight_mode}")
    return model


def get_score(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probability = np.asarray(model.predict_proba(X))
        if probability.ndim == 2 and probability.shape[1] >= 2:
            return probability[:, 1].astype(float)
        return probability.ravel().astype(float)
    return np.asarray(model.predict(X), dtype=float).ravel()


# -----------------------------------------------------------------------------
# Metrics and optional threshold selection
# -----------------------------------------------------------------------------


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def safe_pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def threshold_free_metrics(y_true: np.ndarray, score: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    return {
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true == 1)),
        "auc": safe_auc(y_true, score),
        "pr_auc": safe_pr_auc(y_true, score),
        "brier": float(brier_score_loss(y_true, score)),
    }


def threshold_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    y_pred = (score >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "pred_positive_rate": float(np.mean(y_pred == 1)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f05": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def choose_threshold(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold_grid: Sequence[float],
    target: str,
) -> Tuple[float, pd.DataFrame]:
    rows = []
    for threshold in threshold_grid:
        row = threshold_free_metrics(y_true, score)
        row.update(threshold_metrics(y_true, score, threshold))
        rows.append(row)
    table = pd.DataFrame(rows)
    table = table.sort_values([target, "threshold"], ascending=[False, True]).reset_index(drop=True)
    return float(table.loc[0, "threshold"]), table


def estimate_threshold_from_inner_folds(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    feature_indices: np.ndarray,
    model_name: str,
    params: Mapping[str, object],
    folds: Sequence[int],
    threshold_grid: Sequence[float],
    threshold_target: str,
    sample_weight_mode: str,
    seed: int,
    n_jobs: int,
) -> Tuple[float, pd.DataFrame]:
    labels: List[np.ndarray] = []
    scores: List[np.ndarray] = []

    for fold in folds:
        role_column = f"fold{fold}_role"
        if role_column not in meta.columns:
            raise ValueError(f"Dataset metadata is missing {role_column}")
        train_mask = meta[role_column].astype(str).eq("inner_train").to_numpy()
        valid_mask = meta[role_column].astype(str).eq("inner_valid").to_numpy()
        if not np.any(train_mask) or not np.any(valid_mask):
            raise ValueError(f"Empty inner split for fold {fold}")
        if "split" in meta.columns:
            has_test = meta.loc[valid_mask, "split"].astype(str).eq("test").any()
            if bool(has_test):
                raise RuntimeError("Leakage guard: inner_valid contains final-test rows.")

        fold_model = make_model(
            model_name,
            params,
            seed=seed + 100 * int(fold),
            n_jobs=n_jobs,
        )
        fit_model(
            fold_model,
            X[train_mask][:, feature_indices],
            y[train_mask],
            sample_weight_mode,
        )
        valid_score = get_score(fold_model, X[valid_mask][:, feature_indices])
        labels.append(y[valid_mask])
        scores.append(valid_score)
        del fold_model, valid_score
        gc.collect()

    pooled_y = np.concatenate(labels)
    pooled_score = np.concatenate(scores)
    return choose_threshold(
        pooled_y,
        pooled_score,
        threshold_grid=threshold_grid,
        target=threshold_target,
    )


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------


def add_deltas_vs_full(final_metrics: pd.DataFrame) -> pd.DataFrame:
    if final_metrics.empty:
        return pd.DataFrame()

    metric_columns = [
        column
        for column in ["auc", "pr_auc", "brier", "acc", "precision", "recall", "f1", "f05"]
        if column in final_metrics.columns
    ]
    full = final_metrics[final_metrics["feature_set_name"].eq("full_features")][
        ["horizon_month", "model_name"] + metric_columns
    ].copy()
    full = full.rename(columns={metric: f"full_{metric}" for metric in metric_columns})

    merged = final_metrics.merge(
        full,
        on=["horizon_month", "model_name"],
        how="left",
        validate="many_to_one",
    )
    for metric in metric_columns:
        merged[f"delta_{metric}_vs_full"] = merged[metric] - merged[f"full_{metric}"]

    return merged.sort_values(
        ["horizon_month", "model_name", "feature_set_name"]
    ).reset_index(drop=True)


def summarize_deltas(deltas: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    if deltas.empty:
        return pd.DataFrame()
    work = deltas[~deltas["feature_set_name"].eq("full_features")].copy()
    aggregations = {
        "n_runs": ("delta_auc_vs_full", "size"),
        "mean_delta_auc": ("delta_auc_vs_full", "mean"),
        "std_delta_auc": ("delta_auc_vs_full", "std"),
        "min_delta_auc": ("delta_auc_vs_full", "min"),
        "max_delta_auc": ("delta_auc_vs_full", "max"),
        "mean_delta_pr_auc": ("delta_pr_auc_vs_full", "mean"),
        "mean_delta_brier": ("delta_brier_vs_full", "mean"),
        "all_auc_nonpositive": ("delta_auc_vs_full", lambda x: bool((x <= 0).all())),
    }
    result = work.groupby(list(group_columns), as_index=False).agg(**aggregations)
    result["std_delta_auc"] = result["std_delta_auc"].fillna(0.0)
    return result


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate frozen CatBoost/ExtraTrees/XGBoost SHAP groups in RF and LightGBM."
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw282",
    )
    parser.add_argument(
        "--selected-groups-csv",
        required=True,
        help="Frozen selected_groups_inner_validation.csv from script 05.",
    )
    parser.add_argument(
        "--heldout-config-json",
        required=True,
        help="selected_config_by_model.json from the prior RF/LightGBM comparison.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/heldout_group_validation",
    )
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--models", default="rf,lightgbm")
    parser.add_argument("--top-k-groups", default="1,2,3")
    parser.add_argument("--group-col", default="feature_group")
    parser.add_argument(
        "--expected-discovery-models",
        default="CatBoost,ExtraTrees,XGBoost",
    )
    parser.add_argument("--expected-discovery-folds", type=int, default=3)
    parser.add_argument(
        "--evaluation-mode",
        choices=["auc_only", "full_metrics"],
        default="auc_only",
        help=(
            "auc_only fits each final train_pool model once and reports ROC-AUC, "
            "PR-AUC, and Brier score. full_metrics additionally refits all inner "
            "folds to select a threshold for each feature set."
        ),
    )
    parser.add_argument("--folds", default="1,2,3")
    parser.add_argument(
        "--threshold-target",
        choices=["f1", "f05", "precision", "recall", "acc"],
        default="f1",
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
    parser.add_argument(
        "--sample-weight-mode",
        choices=["inherit", "none", "balanced"],
        default="inherit",
    )
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()

    data_dir = expand_path(args.data_dir)
    selected_groups_csv = expand_path(args.selected_groups_csv)
    heldout_config_json = expand_path(args.heldout_config_json)
    out_dir = ensure_dir(expand_path(args.out_dir))
    predictions_dir = ensure_dir(out_dir / "predictions")
    models_dir = ensure_dir(out_dir / "models")
    threshold_dir = ensure_dir(out_dir / "threshold_curves")

    horizons = parse_int_list(args.horizons)
    heldout_models = parse_heldout_models(args.models)
    top_k_list = sorted(parse_int_list(args.top_k_groups))
    folds = parse_int_list(args.folds)
    threshold_grid = parse_float_list(args.threshold_grid)
    expected_discovery_models = parse_discovery_models(args.expected_discovery_models)

    if min(top_k_list) < 1:
        raise ValueError("--top-k-groups must contain positive integers.")
    if set(heldout_models).intersection(expected_discovery_models):
        raise ValueError("Held-out validation models overlap discovery models.")
    if "LightGBM" in heldout_models and LGBMClassifier is None:
        raise ImportError(f"lightgbm is unavailable: {LIGHTGBM_IMPORT_ERROR}")

    frozen_groups = load_frozen_groups(
        selected_groups_csv,
        group_col=args.group_col,
        horizons=horizons,
        max_top_k=max(top_k_list),
        expected_discovery_models=expected_discovery_models,
        expected_folds=args.expected_discovery_folds,
    )
    selected_configs, config_metadata = load_selected_configs(
        heldout_config_json,
        requested_models=heldout_models,
    )

    if args.sample_weight_mode == "inherit":
        sample_weight_mode = str(config_metadata.get("sample_weight_mode", "none")).strip().lower()
        if sample_weight_mode not in {"none", "balanced"}:
            raise ValueError(
                f"Unsupported inherited sample_weight_mode={sample_weight_mode!r}"
            )
    else:
        sample_weight_mode = args.sample_weight_mode

    frozen_copy_path = out_dir / "frozen_selected_groups_used.csv"
    frozen_groups.to_csv(frozen_copy_path, index=False)

    feature_set_path = out_dir / "heldout_feature_sets.csv"
    final_metric_path = out_dir / "heldout_final_test_metrics.csv"
    delta_path = out_dir / "heldout_deltas_vs_full.csv"
    horizon_summary_path = out_dir / "heldout_summary_by_horizon_feature_set.csv"
    model_summary_path = out_dir / "heldout_summary_by_model_feature_set.csv"
    overall_summary_path = out_dir / "heldout_summary_overall.csv"
    manifest_path = out_dir / "heldout_validation_manifest.json"
    report_path = out_dir / "heldout_validation_report.md"

    remove_if_exists(
        [
            feature_set_path,
            final_metric_path,
            delta_path,
            horizon_summary_path,
            model_summary_path,
            overall_summary_path,
            manifest_path,
            report_path,
        ],
        resume=args.resume,
    )

    print("=" * 100, flush=True)
    print("Held-out architecture validation of frozen feature groups", flush=True)
    print("data_dir:", data_dir, flush=True)
    print("selected_groups_csv:", selected_groups_csv, flush=True)
    print("heldout_config_json:", heldout_config_json, flush=True)
    print("out_dir:", out_dir, flush=True)
    print("discovery_models:", expected_discovery_models, flush=True)
    print("heldout_models:", heldout_models, flush=True)
    print("horizons:", horizons, flush=True)
    print("top_k_groups:", top_k_list, flush=True)
    print("evaluation_mode:", args.evaluation_mode, flush=True)
    print("sample_weight_mode:", sample_weight_mode, flush=True)
    print("=" * 100, flush=True)

    feature_set_tables: List[pd.DataFrame] = []
    dataset_hashes: Dict[str, str] = {}
    feature_metadata_hash = ""

    for horizon in horizons:
        print("\n" + "#" * 100, flush=True)
        print(f"Horizon H={horizon} months", flush=True)

        X, y, meta, payload, source_dataset = load_dataset(data_dir, horizon)
        dataset_hashes[f"H{horizon}"] = sha256_file(source_dataset)
        feature_metadata, metadata_path = load_feature_metadata(data_dir, X.shape[1])
        if not feature_metadata_hash:
            feature_metadata_hash = sha256_file(metadata_path)

        selected_horizon_groups = groups_for_horizon(
            frozen_groups,
            horizon=horizon,
            max_top_k=max(top_k_list),
        )
        feature_masks, feature_set_table = build_feature_sets(
            feature_metadata,
            selected_groups=selected_horizon_groups,
            top_k_list=top_k_list,
            group_col=args.group_col,
        )
        feature_set_table["horizon_month"] = int(horizon)
        feature_set_table["selection_source"] = "inner_validation"
        feature_set_table["discovery_models"] = ",".join(expected_discovery_models)
        feature_set_table["validation_models"] = ",".join(heldout_models)
        feature_set_tables.append(feature_set_table)

        if "split" not in meta.columns:
            raise ValueError("Dataset metadata is missing split.")
        train_pool_mask = meta["split"].astype(str).eq("train_pool").to_numpy()
        test_mask = meta["split"].astype(str).eq("test").to_numpy()
        if not np.any(train_pool_mask) or not np.any(test_mask):
            raise ValueError(f"H={horizon} has an empty train_pool or test split.")

        X_train_pool = X[train_pool_mask]
        y_train_pool = y[train_pool_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        meta_test = meta.loc[test_mask].reset_index(drop=True).copy()

        print(
            f"X={X.shape}, train_pool={X_train_pool.shape}, test={X_test.shape}, "
            f"test_positives={int(y_test.sum())}",
            flush=True,
        )
        print(
            "Frozen groups:",
            "; ".join(selected_horizon_groups["group_name"].astype(str).tolist()),
            flush=True,
        )

        for model_index, model_name in enumerate(heldout_models):
            config = selected_configs[model_name]
            config_name = config["config_name"]
            params = config["params"]

            print("\n" + "=" * 100, flush=True)
            print(
                f"Held-out model={model_name}, config={config_name}, H={horizon}",
                flush=True,
            )

            for feature_set_name, feature_indices in feature_masks.items():
                feature_indices = np.asarray(feature_indices, dtype=int)
                if len(feature_indices) == 0:
                    raise ValueError(f"Empty feature set: H={horizon}, {feature_set_name}")

                if args.resume and final_metric_path.exists():
                    existing = pd.read_csv(final_metric_path)
                    hit = existing[
                        existing["horizon_month"].eq(int(horizon))
                        & existing["model_name"].eq(model_name)
                        & existing["feature_set_name"].eq(feature_set_name)
                    ]
                    if len(hit) > 0:
                        print(
                            f"[RESUME] Skip H={horizon} {model_name} {feature_set_name}",
                            flush=True,
                        )
                        continue

                print(
                    f"Feature set={feature_set_name}, n_features={len(feature_indices)}",
                    flush=True,
                )

                # Same paired seed for full, retain, and remove feature sets
                # within each model-horizon combination.
                run_seed = (
                        int(args.seed)
                        + 100000 * int(model_index + 1)
                        + 10000 * int(horizon)
                )

                threshold = float("nan")
                threshold_curve_csv = ""
                if args.evaluation_mode == "full_metrics":
                    threshold, threshold_curve = estimate_threshold_from_inner_folds(
                        X=X,
                        y=y,
                        meta=meta,
                        feature_indices=feature_indices,
                        model_name=model_name,
                        params=params,
                        folds=folds,
                        threshold_grid=threshold_grid,
                        threshold_target=args.threshold_target,
                        sample_weight_mode=sample_weight_mode,
                        seed=run_seed,
                        n_jobs=args.n_jobs,
                    )
                    threshold_curve_path = (
                        threshold_dir
                        / f"threshold_curve_{safe_name(model_name)}_H{horizon}m_{feature_set_name}.csv"
                    )
                    threshold_curve.to_csv(threshold_curve_path, index=False)
                    threshold_curve_csv = str(threshold_curve_path)

                model = make_model(
                    model_name,
                    params,
                    seed=run_seed,
                    n_jobs=args.n_jobs,
                )
                fit_model(
                    model,
                    X_train_pool[:, feature_indices],
                    y_train_pool,
                    sample_weight_mode,
                )
                score = get_score(model, X_test[:, feature_indices])

                metrics = threshold_free_metrics(y_test, score)
                if args.evaluation_mode == "full_metrics":
                    metrics.update(threshold_metrics(y_test, score, threshold))

                prediction_path = (
                    predictions_dir
                    / f"predictions_test_{safe_name(model_name)}_H{horizon}m_{feature_set_name}.csv"
                )
                prediction = meta_test.copy()
                prediction["model_name"] = model_name
                prediction["config_name"] = config_name
                prediction["feature_set_name"] = feature_set_name
                prediction["y_true"] = y_test.astype(int)
                prediction["score_m5"] = score.astype(float)
                if args.evaluation_mode == "full_metrics":
                    prediction["threshold"] = float(threshold)
                    prediction["y_pred"] = (score >= threshold).astype(int)
                prediction.to_csv(prediction_path, index=False)

                model_path = ""
                if args.save_models:
                    output_model_path = (
                        models_dir
                        / f"model_{safe_name(model_name)}_H{horizon}m_{feature_set_name}.joblib"
                    )
                    joblib.dump(model, output_model_path)
                    model_path = str(output_model_path)

                feature_set_metadata = feature_set_table[
                    feature_set_table["feature_set_name"].eq(feature_set_name)
                ].iloc[0]
                row = {
                    "horizon_month": int(horizon),
                    "model_name": model_name,
                    "config_name": config_name,
                    "validation_role": "held_out_architecture",
                    "feature_set_name": feature_set_name,
                    "feature_set_type": feature_set_metadata["feature_set_type"],
                    "top_k_groups": int(feature_set_metadata["top_k_groups"]),
                    "groups_used": feature_set_metadata["groups_used"],
                    "n_features": int(len(feature_indices)),
                    "selection_source": "inner_validation",
                    "discovery_models": ",".join(expected_discovery_models),
                    "selected_groups_csv": str(selected_groups_csv),
                    "selected_groups_sha256": sha256_file(selected_groups_csv),
                    "heldout_config_json": str(heldout_config_json),
                    "heldout_config_sha256": sha256_file(heldout_config_json),
                    "evaluation_mode": args.evaluation_mode,
                    "sample_weight_mode": sample_weight_mode,
                    "threshold_target": args.threshold_target if args.evaluation_mode == "full_metrics" else "not_used",
                    "threshold_curve_csv": threshold_curve_csv,
                    "prediction_csv": str(prediction_path),
                    "model_file": model_path,
                    "seed": int(run_seed),
                }
                row.update(metrics)
                append_row_csv(final_metric_path, row)

                print(
                    f"FINAL H={horizon} {model_name} {feature_set_name}: "
                    f"AUC={metrics['auc']:.6f}, PR-AUC={metrics['pr_auc']:.6f}, "
                    f"Brier={metrics['brier']:.6f}",
                    flush=True,
                )

                del model, score, prediction
                gc.collect()

        del X, y, meta, payload, X_train_pool, y_train_pool, X_test, y_test, meta_test
        gc.collect()

    feature_sets_all = pd.concat(feature_set_tables, ignore_index=True, sort=False)
    feature_sets_all.to_csv(feature_set_path, index=False)

    final_metrics = pd.read_csv(final_metric_path)
    deltas = add_deltas_vs_full(final_metrics)
    deltas.to_csv(delta_path, index=False)

    horizon_summary = summarize_deltas(
        deltas,
        group_columns=["horizon_month", "feature_set_name", "feature_set_type", "top_k_groups"],
    )
    model_summary = summarize_deltas(
        deltas,
        group_columns=["model_name", "feature_set_name", "feature_set_type", "top_k_groups"],
    )
    overall_summary = summarize_deltas(
        deltas,
        group_columns=["feature_set_name", "feature_set_type", "top_k_groups"],
    )
    horizon_summary.to_csv(horizon_summary_path, index=False)
    model_summary.to_csv(model_summary_path, index=False)
    overall_summary.to_csv(overall_summary_path, index=False)

    manifest = {
        "scientific_design": {
            "discovery_models": expected_discovery_models,
            "discovery_source": "inner_validation_SHAP",
            "validation_models": heldout_models,
            "validation_role": "held_out_architectures_without_reselection",
            "group_reranking_in_validation": False,
            "final_test_used_for_group_selection": False,
        },
        "inputs": {
            "data_dir": str(data_dir),
            "selected_groups_csv": str(selected_groups_csv),
            "selected_groups_sha256": sha256_file(selected_groups_csv),
            "heldout_config_json": str(heldout_config_json),
            "heldout_config_sha256": sha256_file(heldout_config_json),
            "feature_metadata_sha256": feature_metadata_hash,
            "dataset_sha256": dataset_hashes,
        },
        "run": {
            "horizons": horizons,
            "heldout_models": heldout_models,
            "top_k_groups": top_k_list,
            "group_col": args.group_col,
            "evaluation_mode": args.evaluation_mode,
            "sample_weight_mode": sample_weight_mode,
            "n_jobs": args.n_jobs,
            "seed": args.seed,
        },
        "outputs": {
            "final_metrics": str(final_metric_path),
            "deltas_vs_full": str(delta_path),
            "horizon_summary": str(horizon_summary_path),
            "model_summary": str(model_summary_path),
            "overall_summary": str(overall_summary_path),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Held-out RF/LightGBM validation of frozen feature groups\n\n")
        handle.write(
            "Groups were identified by CatBoost, ExtraTrees, and XGBoost using "
            "inner-validation SHAP, frozen before confirmation, and transferred "
            "without reselection to Random Forest and LightGBM. The validation "
            "script contains no SHAP ranking step.\n\n"
        )
        handle.write("## Frozen groups\n\n")
        frozen_columns = [
            column
            for column in [
                "horizon_month",
                "rank",
                "group_name",
                "n_features",
                "mean_group_importance",
                "mean_rank",
                "n_models",
                "n_folds",
                "source_stage",
            ]
            if column in frozen_groups.columns
        ]
        handle.write(frozen_groups[frozen_columns].to_markdown(index=False))
        handle.write("\n\n## Held-out changes relative to matched full reruns\n\n")
        show_columns = [
            "horizon_month",
            "model_name",
            "feature_set_name",
            "n_features",
            "delta_auc_vs_full",
            "delta_pr_auc_vs_full",
            "delta_brier_vs_full",
        ]
        handle.write(deltas[show_columns].to_markdown(index=False))
        handle.write("\n\n## Overall summary\n\n")
        handle.write(overall_summary.to_markdown(index=False))
        handle.write("\n")

    print("\n" + "=" * 100, flush=True)
    print("HELD-OUT ARCHITECTURE VALIDATION FINISHED", flush=True)
    print("Saved final metrics:", final_metric_path, flush=True)
    print("Saved deltas:", delta_path, flush=True)
    print("Saved summaries:", overall_summary_path, flush=True)
    print("Saved manifest:", manifest_path, flush=True)
    print("Saved report:", report_path, flush=True)


if __name__ == "__main__":
    main()
