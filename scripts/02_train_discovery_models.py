#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_train_discovery_models.py

Purpose
-------
Compare CatBoost, ExtraTrees and XGBoost baselines for binary M>=5 prediction
using the existing raw282 dataset files:
    china_raw282_dataset_horizon_{H}m.pkl

Data contract
-------------
Each pickle should contain:
    X:     raw 282-dim features
    y_m5:  binary label, 1 means future maximum magnitude >= 5
           If y_m5 is absent, y_class > 0 is used.
    meta:  DataFrame with split=train_pool/test and fold{1,2,3}_role columns

Workflow
--------
1. Inner rolling folds:
   tune/select candidate configurations using only inner_train/inner_valid.
2. Threshold selection:
   pool inner-validation scores of the selected configuration for each horizon,
   then choose the M>=5 threshold by F1, precision, then threshold.
3. Final test:
   train the selected model on train_pool and evaluate on final test split.
4. Save models, predictions, metrics and a markdown report.

Notes
-----
This script uses robust CSV writing with column alignment, so mixed model
parameter schemas will not corrupt the tuning metrics CSV.
"""

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

from sklearn.ensemble import ExtraTreesClassifier
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

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    xgb = None
    XGBClassifier = None
    HAS_XGBOOST = False

try:
    import catboost as cb
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    cb = None
    CatBoostClassifier = None
    HAS_CATBOOST = False


# ============================================================
# basic utils
# ============================================================

def expand_path(p):
    return str(Path(p).expanduser().resolve())


def parse_int_list(s):
    out = []
    for x in str(s).split(','):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def parse_str_list(s):
    return [x.strip().lower() for x in str(s).split(',') if x.strip()]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def append_row_csv(path, row):
    """Append one row while preserving a union of columns.

    This avoids the RF/LightGBM style schema bug where different model parameter
    columns produce rows with different field counts.
    """
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


def append_df_csv(path, df_new):
    """Append a DataFrame while preserving a union of columns."""
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


# ============================================================
# data loading and region filtering
# ============================================================

def dataset_path(data_dir, horizon):
    return os.path.join(data_dir, f'china_raw282_dataset_horizon_{horizon}m.pkl')


def load_dataset(data_dir, horizon):
    path = dataset_path(data_dir, horizon)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, 'rb') as f:
        d = pickle.load(f)

    X = np.asarray(d['X'], dtype=np.float32)

    if 'y_m5' in d:
        y = np.asarray(d['y_m5'], dtype=np.int64)
    else:
        y_class = np.asarray(d['y_class'], dtype=np.int64)
        y = (y_class > 0).astype(np.int64)

    meta = d['meta'].copy()
    if 't0' in meta.columns:
        meta['t0'] = pd.to_datetime(meta['t0'])
    if 'future_end_inclusive' in meta.columns:
        meta['future_end_inclusive'] = pd.to_datetime(meta['future_end_inclusive'])

    return X, y, meta, d


def load_patch_orders(patch_csv):
    if patch_csv is None or str(patch_csv).strip() == '':
        return None

    patch_csv = expand_path(patch_csv)
    if not os.path.exists(patch_csv):
        raise FileNotFoundError(patch_csv)

    p = pd.read_csv(patch_csv)
    if 'patch_order' not in p.columns:
        raise ValueError('patch_csv must contain column patch_order.')

    return set(p['patch_order'].astype(int).tolist())


def make_region_mask(meta, patch_orders):
    if patch_orders is None:
        return np.ones(len(meta), dtype=bool)

    if 'patch_order' not in meta.columns:
        raise ValueError('meta lacks patch_order column, cannot apply patch_csv region filter.')

    return meta['patch_order'].astype(int).isin(patch_orders).values


# ============================================================
# metrics and threshold search
# ============================================================

def safe_auc(y_true, score):
    try:
        y_true = np.asarray(y_true, dtype=int)
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, score))
    except Exception:
        return np.nan


def safe_ap(y_true, score):
    try:
        y_true = np.asarray(y_true, dtype=int)
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(average_precision_score(y_true, score))
    except Exception:
        return np.nan


def search_best_threshold(y_true, score):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)

    if len(y_true) == 0:
        raise ValueError('Empty validation labels for threshold search.')

    rows = []
    for th in np.linspace(0.01, 0.99, 99):
        pred = (score >= th).astype(int)
        rows.append({
            'threshold': float(th),
            'acc': float(accuracy_score(y_true, pred)),
            'precision': float(precision_score(y_true, pred, zero_division=0)),
            'recall': float(recall_score(y_true, pred, zero_division=0)),
            'f1': float(f1_score(y_true, pred, zero_division=0)),
            'pred_positive_rate': float(np.mean(pred)),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ['f1', 'precision', 'threshold'],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return float(df.loc[0, 'threshold']), df


def evaluate_binary(y_true, score, threshold):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)

    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    return {
        'positive_rate': float(np.mean(y_true)) if len(y_true) > 0 else np.nan,
        'threshold': float(threshold),
        'pred_positive_rate': float(np.mean(pred)) if len(pred) > 0 else np.nan,
        'acc': float(accuracy_score(y_true, pred)),
        'precision': float(precision_score(y_true, pred, zero_division=0)),
        'recall': float(recall_score(y_true, pred, zero_division=0)),
        'f1': float(f1_score(y_true, pred, zero_division=0)),
        'auc': safe_auc(y_true, score),
        'pr_auc': safe_ap(y_true, score),
        'brier': float(brier_score_loss(y_true, score)),
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1]),
    }, pred


def positive_probability(model, X):
    prob = model.predict_proba(X)
    classes = getattr(model, 'classes_', None)

    if classes is None:
        if np.asarray(prob).ndim == 1:
            return np.asarray(prob, dtype=float)
        return np.asarray(prob)[:, -1].astype(float)

    classes = np.asarray(classes)
    hit = np.where(classes == 1)[0]
    if len(hit) == 0:
        return np.zeros(X.shape[0], dtype=float)

    return np.asarray(prob)[:, int(hit[0])].astype(float)


def sample_weight_for_mode(y, mode):
    if mode == 'none':
        return None
    if mode != 'balanced':
        raise ValueError(f'Unknown sample_weight_mode: {mode}')

    y = np.asarray(y, dtype=int)
    n = len(y)
    n0 = int(np.sum(y == 0))
    n1 = int(np.sum(y == 1))
    if n0 == 0 or n1 == 0:
        return None
    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)
    return np.where(y == 1, w1, w0).astype(np.float32)


# ============================================================
# model definitions
# ============================================================

def default_candidate_configs(model_names):
    configs = []

    if 'extratrees' in model_names or 'et' in model_names:
        configs += [
            {
                'model_name': 'ExtraTrees',
                'config_name': 'et_cfg0_depth12_leaf5',
                'params': {
                    'n_estimators': 500,
                    'max_depth': 12,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'max_features': 'sqrt',
                    'bootstrap': False,
                },
            },
            {
                'model_name': 'ExtraTrees',
                'config_name': 'et_cfg1_depth16_leaf5',
                'params': {
                    'n_estimators': 500,
                    'max_depth': 16,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'max_features': 'sqrt',
                    'bootstrap': False,
                },
            },
            {
                'model_name': 'ExtraTrees',
                'config_name': 'et_cfg2_depth18_leaf10_mf050',
                'params': {
                    'n_estimators': 600,
                    'max_depth': 18,
                    'min_samples_split': 20,
                    'min_samples_leaf': 10,
                    'max_features': 0.50,
                    'bootstrap': False,
                },
            },
        ]

    if 'xgboost' in model_names or 'xgb' in model_names:
        configs += [
            {
                'model_name': 'XGBoost',
                'config_name': 'xgb_cfg4_strong_reg',
                'params': {
                    'n_estimators': 500,
                    'max_depth': 3,
                    'learning_rate': 0.030,
                    'subsample': 0.90,
                    'colsample_bytree': 0.90,
                    'reg_lambda': 8.0,
                    'reg_alpha': 0.8,
                    'min_child_weight': 5.0,
                    'gamma': 0.20,
                },
            },
            {
                'model_name': 'XGBoost',
                'config_name': 'xgb_cfg8_recall_depth3',
                'params': {
                    'n_estimators': 800,
                    'max_depth': 3,
                    'learning_rate': 0.025,
                    'subsample': 0.85,
                    'colsample_bytree': 0.85,
                    'reg_lambda': 4.0,
                    'reg_alpha': 0.3,
                    'min_child_weight': 2.0,
                    'gamma': 0.05,
                },
            },
            {
                'model_name': 'XGBoost',
                'config_name': 'xgb_cfg9_depth4_reg',
                'params': {
                    'n_estimators': 700,
                    'max_depth': 4,
                    'learning_rate': 0.020,
                    'subsample': 0.80,
                    'colsample_bytree': 0.80,
                    'reg_lambda': 5.0,
                    'reg_alpha': 0.5,
                    'min_child_weight': 3.0,
                    'gamma': 0.10,
                },
            },
        ]

    if 'catboost' in model_names or 'cat' in model_names:
        configs += [
            {
                'model_name': 'CatBoost',
                'config_name': 'cat_cfg0_depth4_lr003',
                'params': {
                    'iterations': 600,
                    'depth': 4,
                    'learning_rate': 0.030,
                    'l2_leaf_reg': 8.0,
                    'random_strength': 1.0,
                    'bootstrap_type': 'Bernoulli',
                    'subsample': 0.85,
                },
            },
            {
                'model_name': 'CatBoost',
                'config_name': 'cat_cfg1_depth5_lr002',
                'params': {
                    'iterations': 800,
                    'depth': 5,
                    'learning_rate': 0.020,
                    'l2_leaf_reg': 10.0,
                    'random_strength': 1.0,
                    'bootstrap_type': 'Bernoulli',
                    'subsample': 0.85,
                },
            },
            {
                'model_name': 'CatBoost',
                'config_name': 'cat_cfg2_depth6_lr003',
                'params': {
                    'iterations': 600,
                    'depth': 6,
                    'learning_rate': 0.030,
                    'l2_leaf_reg': 12.0,
                    'random_strength': 1.5,
                    'bootstrap_type': 'Bernoulli',
                    'subsample': 0.80,
                },
            },
        ]

    return configs


def load_configs(args, model_names):
    if args.config_json is None or str(args.config_json).strip() == '':
        configs = default_candidate_configs(model_names)
    else:
        with open(expand_path(args.config_json), 'r') as f:
            configs = json.load(f)

    allowed = set()
    for m in model_names:
        if m in ('extratrees', 'et'):
            allowed.add('ExtraTrees')
        if m in ('xgboost', 'xgb'):
            allowed.add('XGBoost')
        if m in ('catboost', 'cat'):
            allowed.add('CatBoost')

    configs = [c for c in configs if c['model_name'] in allowed]

    if any(c['model_name'] == 'XGBoost' for c in configs) and not HAS_XGBOOST:
        raise ImportError('xgboost is not installed in the current Python environment.')
    if any(c['model_name'] == 'CatBoost' for c in configs) and not HAS_CATBOOST:
        raise ImportError('catboost is not installed in the current Python environment. Install with: python -m pip install catboost')

    return configs


def make_model(cfg, seed, n_jobs):
    model_name = cfg['model_name']
    params = dict(cfg['params'])

    if model_name == 'ExtraTrees':
        return ExtraTreesClassifier(
            random_state=seed,
            n_jobs=n_jobs,
            **params,
        )

    if model_name == 'XGBoost':
        if not HAS_XGBOOST:
            raise ImportError('xgboost is not installed.')
        return XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            n_jobs=n_jobs,
            random_state=seed,
            verbosity=1,
            **params,
        )

    if model_name == 'CatBoost':
        if not HAS_CATBOOST:
            raise ImportError('catboost is not installed.')
        return CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='AUC',
            thread_count=n_jobs,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            **params,
        )

    raise ValueError(f'Unknown model_name: {model_name}')


def fit_model(model, X_train, y_train, sample_weight=None):
    if sample_weight is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def config_param_columns(cfg):
    out = {}
    for k, v in cfg['params'].items():
        out[f'param_{k}'] = v
    return out


# ============================================================
# tuning stage
# ============================================================

def get_done_keys(metric_path):
    if not os.path.exists(metric_path):
        return set()

    df = pd.read_csv(metric_path)
    need = {'model_name', 'config_name', 'horizon_month', 'fold'}
    if not need.issubset(df.columns):
        return set()

    keys = set()
    for _, r in df.iterrows():
        keys.add((str(r['model_name']), str(r['config_name']), int(r['horizon_month']), int(r['fold'])))
    return keys


def tune_configs(args, configs, horizons, folds, patch_orders):
    metric_path = os.path.join(args.out_dir, 'tuning_fold_metrics_catboost_extratrees_xgb.csv')
    score_path = os.path.join(args.out_dir, 'valid_prediction_scores_catboost_extratrees_xgb.csv')
    threshold_dir = ensure_dir(os.path.join(args.out_dir, 'threshold_search'))

    done = get_done_keys(metric_path) if args.resume else set()

    for H in horizons:
        path = dataset_path(args.data_dir, H)
        if not os.path.exists(path):
            if args.skip_missing:
                print(f'[SKIP] Missing dataset for H={H}m: {path}', flush=True)
                continue
            raise FileNotFoundError(path)

        print('\n' + '#' * 100, flush=True)
        print(f'Load horizon {H}m dataset', flush=True)
        X, y, meta, payload = load_dataset(args.data_dir, H)
        rmask = make_region_mask(meta, patch_orders)

        print('X:', X.shape, flush=True)
        print('region samples:', int(rmask.sum()), flush=True)
        print('y all:', Counter(y), flush=True)
        print('y region:', Counter(y[rmask]), flush=True)

        for cfg_idx, cfg in enumerate(configs):
            model_name = cfg['model_name']
            cfg_name = cfg['config_name']

            for fold in folds:
                key = (model_name, cfg_name, H, fold)
                if key in done:
                    print(f'Skip existing: {key}', flush=True)
                    continue

                role_col = f'fold{fold}_role'
                if role_col not in meta.columns:
                    raise ValueError(f'Missing column {role_col} in meta.')

                train_mask = rmask & meta[role_col].eq('inner_train').values
                valid_mask = rmask & meta[role_col].eq('inner_valid').values

                X_train, y_train = X[train_mask], y[train_mask]
                X_valid, y_valid = X[valid_mask], y[valid_mask]

                print('\n' + '=' * 100, flush=True)
                print(f'Model={model_name}, Config={cfg_name}, H={H}m, fold={fold}', flush=True)
                print('Train:', X_train.shape, Counter(y_train), flush=True)
                print('Valid:', X_valid.shape, Counter(y_valid), flush=True)

                if len(y_train) == 0 or len(y_valid) == 0:
                    raise ValueError(f'Empty train/valid: model={model_name}, cfg={cfg_name}, H={H}, fold={fold}')
                if len(np.unique(y_train)) < 2:
                    raise ValueError(f'Only one class in training data: model={model_name}, H={H}, fold={fold}')

                model_seed = args.seed + 100000 * (cfg_idx + 1) + 1000 * H + 10 * fold
                model = make_model(cfg, seed=model_seed, n_jobs=args.n_jobs)
                sw = sample_weight_for_mode(y_train, args.sample_weight_mode)

                fit_model(model, X_train, y_train, sample_weight=sw)
                score_valid = positive_probability(model, X_valid)

                best_th, th_df = search_best_threshold(y_valid, score_valid)
                th_path = os.path.join(
                    threshold_dir,
                    f'threshold_search_{model_name}_{cfg_name}_H{H}m_fold{fold}.csv',
                )
                th_df.to_csv(th_path, index=False)

                metrics, pred_valid = evaluate_binary(y_valid, score_valid, best_th)

                row = {
                    'region_name': args.region_name,
                    'model_name': model_name,
                    'config_name': cfg_name,
                    'horizon_month': H,
                    'fold': fold,
                    'n_train': int(len(y_train)),
                    'n_valid': int(len(y_valid)),
                    'train_negative': int(np.sum(y_train == 0)),
                    'train_positive': int(np.sum(y_train == 1)),
                    'valid_negative': int(np.sum(y_valid == 0)),
                    'valid_positive': int(np.sum(y_valid == 1)),
                    'sample_weight_mode': args.sample_weight_mode,
                }
                row.update(config_param_columns(cfg))
                for k, v in metrics.items():
                    row[f'valid_{k}'] = v

                append_row_csv(metric_path, row)

                score_df = pd.DataFrame({
                    'region_name': args.region_name,
                    'model_name': model_name,
                    'config_name': cfg_name,
                    'horizon_month': H,
                    'fold': fold,
                    'y_true': y_valid.astype(int),
                    'score': score_valid.astype(float),
                })
                append_df_csv(score_path, score_df)

                print('Valid metrics:', json.dumps(metrics, indent=2), flush=True)
                print('Best threshold:', best_th, flush=True)
                print('Saved:', th_path, flush=True)

                del model, X_train, X_valid, y_train, y_valid, score_valid, pred_valid, sw
                gc.collect()

        del X, y, meta, payload
        gc.collect()

    return metric_path, score_path


# ============================================================
# selection and threshold pooling
# ============================================================

def summarize_and_select(args, configs, metric_path, score_path):
    if not os.path.exists(metric_path):
        raise FileNotFoundError(metric_path)
    if not os.path.exists(score_path):
        raise FileNotFoundError(score_path)

    df = pd.read_csv(metric_path)
    score_df = pd.read_csv(score_path)

    if len(df) == 0:
        raise ValueError('No tuning metrics found.')

    by_cfg_horizon = (
        df.groupby(['model_name', 'config_name', 'horizon_month'])
        .agg(
            mean_valid_auc=('valid_auc', 'mean'),
            std_valid_auc=('valid_auc', 'std'),
            mean_valid_pr_auc=('valid_pr_auc', 'mean'),
            std_valid_pr_auc=('valid_pr_auc', 'std'),
            mean_valid_brier=('valid_brier', 'mean'),
            std_valid_brier=('valid_brier', 'std'),
            mean_valid_f1=('valid_f1', 'mean'),
            std_valid_f1=('valid_f1', 'std'),
            mean_valid_precision=('valid_precision', 'mean'),
            mean_valid_recall=('valid_recall', 'mean'),
            mean_valid_acc=('valid_acc', 'mean'),
            median_fold_threshold=('valid_threshold', 'median'),
            n_runs=('fold', 'count'),
        )
        .reset_index()
    )
    by_cfg_horizon_path = os.path.join(args.out_dir, 'tuning_summary_by_model_config_horizon.csv')
    by_cfg_horizon.to_csv(by_cfg_horizon_path, index=False)

    summary = (
        df.groupby(['model_name', 'config_name'])
        .agg(
            mean_valid_auc=('valid_auc', 'mean'),
            std_valid_auc=('valid_auc', 'std'),
            mean_valid_pr_auc=('valid_pr_auc', 'mean'),
            std_valid_pr_auc=('valid_pr_auc', 'std'),
            mean_valid_brier=('valid_brier', 'mean'),
            std_valid_brier=('valid_brier', 'std'),
            mean_valid_f1=('valid_f1', 'mean'),
            std_valid_f1=('valid_f1', 'std'),
            mean_valid_precision=('valid_precision', 'mean'),
            mean_valid_recall=('valid_recall', 'mean'),
            mean_valid_acc=('valid_acc', 'mean'),
            n_runs=('fold', 'count'),
        )
        .reset_index()
    )

    cfg_rows = []
    for c in configs:
        row = {'model_name': c['model_name'], 'config_name': c['config_name']}
        row.update(config_param_columns(c))
        cfg_rows.append(row)
    cfg_params = pd.DataFrame(cfg_rows)
    summary = summary.merge(cfg_params, on=['model_name', 'config_name'], how='left')

    # Rank configurations within each algorithm, not across different algorithms.
    summary['rank_brier'] = summary.groupby('model_name')['mean_valid_brier'].rank(
        ascending=True, method='min', na_option='bottom'
    )
    summary['rank_f1'] = summary.groupby('model_name')['mean_valid_f1'].rank(
        ascending=False, method='min', na_option='bottom'
    )
    summary['rank_pr_auc'] = summary.groupby('model_name')['mean_valid_pr_auc'].rank(
        ascending=False, method='min', na_option='bottom'
    )
    summary['rank_auc'] = summary.groupby('model_name')['mean_valid_auc'].rank(
        ascending=False, method='min', na_option='bottom'
    )
    summary['rank_acc'] = summary.groupby('model_name')['mean_valid_acc'].rank(
        ascending=False, method='min', na_option='bottom'
    )

    summary['composite_rank'] = (
        0.35 * summary['rank_brier']
        + 0.30 * summary['rank_f1']
        + 0.20 * summary['rank_pr_auc']
        + 0.10 * summary['rank_auc']
        + 0.05 * summary['rank_acc']
    )

    summary = summary.sort_values(
        ['model_name', 'composite_rank', 'rank_brier', 'rank_f1', 'rank_pr_auc', 'rank_auc'],
        ascending=[True, True, True, True, True, True],
    ).reset_index(drop=True)

    summary_path = os.path.join(args.out_dir, 'tuning_summary_by_model_config.csv')
    summary.to_csv(summary_path, index=False)

    selected = (
        summary.sort_values(
            ['model_name', 'composite_rank', 'rank_brier', 'rank_f1', 'rank_pr_auc', 'rank_auc'],
            ascending=[True, True, True, True, True, True],
        )
        .groupby('model_name', as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    selected_configs = []
    for _, r in selected.iterrows():
        model_name = str(r['model_name'])
        cfg_name = str(r['config_name'])
        cfg = [c for c in configs if c['model_name'] == model_name and c['config_name'] == cfg_name][0]
        selected_configs.append(cfg)

    selected_path = os.path.join(args.out_dir, 'selected_config_by_model.json')
    with open(selected_path, 'w') as f:
        json.dump(
            {
                'region_name': args.region_name,
                'sample_weight_mode': args.sample_weight_mode,
                'selection_rule': (
                    'For each algorithm, select one common hyperparameter configuration across horizons '
                    'using inner rolling backtests only. Composite rank = 0.35*Brier + 0.30*F1 '
                    '+ 0.20*PR-AUC + 0.10*AUC + 0.05*accuracy ranks.'
                ),
                'selected_configs': selected_configs,
            },
            f,
            indent=2,
        )

    threshold_rows = []
    threshold_dir = ensure_dir(os.path.join(args.out_dir, 'pooled_thresholds'))

    for cfg in selected_configs:
        model_name = cfg['model_name']
        cfg_name = cfg['config_name']
        model_scores = score_df[
            (score_df['model_name'] == model_name)
            & (score_df['config_name'] == cfg_name)
        ].copy()

        for H in sorted(model_scores['horizon_month'].unique()):
            g = model_scores[model_scores['horizon_month'] == H].copy()
            if len(g) == 0:
                continue

            th, curve = search_best_threshold(g['y_true'].values, g['score'].values)
            curve_path = os.path.join(
                threshold_dir,
                f'pooled_threshold_curve_{model_name}_{cfg_name}_H{int(H)}m.csv',
            )
            curve.to_csv(curve_path, index=False)

            metrics, pred = evaluate_binary(g['y_true'].values, g['score'].values, th)
            row = {
                'model_name': model_name,
                'config_name': cfg_name,
                'horizon_month': int(H),
                'pooled_threshold': float(th),
            }
            for k, v in metrics.items():
                row[f'pooled_inner_valid_{k}'] = v
            threshold_rows.append(row)

    threshold_df = pd.DataFrame(threshold_rows).sort_values(['model_name', 'horizon_month'])
    threshold_path = os.path.join(args.out_dir, 'pooled_threshold_by_model_horizon.csv')
    threshold_df.to_csv(threshold_path, index=False)

    print('\n' + '#' * 100, flush=True)
    print('TUNING SUMMARY', flush=True)
    print('#' * 100, flush=True)
    show_cols = [
        'model_name', 'config_name', 'composite_rank', 'mean_valid_brier',
        'mean_valid_f1', 'mean_valid_pr_auc', 'mean_valid_auc',
        'mean_valid_acc', 'n_runs',
    ]
    print(summary[show_cols].to_string(index=False), flush=True)
    print('\nSelected configs:', flush=True)
    print(json.dumps(selected_configs, indent=2), flush=True)
    print('\nPooled thresholds:', flush=True)
    print(threshold_df.to_string(index=False), flush=True)

    return selected_configs, threshold_df, summary_path, by_cfg_horizon_path, selected_path, threshold_path


# ============================================================
# final test stage
# ============================================================

def get_threshold_map(threshold_df):
    out = {}
    for _, r in threshold_df.iterrows():
        out[(str(r['model_name']), int(r['horizon_month']))] = float(r['pooled_threshold'])
    return out


def final_test(args, selected_configs, threshold_df, horizons, patch_orders):
    final_rows = []
    model_dir = ensure_dir(os.path.join(args.out_dir, 'models'))
    pred_dir = ensure_dir(os.path.join(args.out_dir, 'predictions'))

    threshold_map = get_threshold_map(threshold_df)

    for H in horizons:
        path = dataset_path(args.data_dir, H)
        if not os.path.exists(path):
            if args.skip_missing:
                print(f'[SKIP] Missing dataset for H={H}m: {path}', flush=True)
                continue
            raise FileNotFoundError(path)

        print('\n' + '#' * 100, flush=True)
        print(f'Final test: H={H}m', flush=True)
        X, y, meta, payload = load_dataset(args.data_dir, H)
        rmask = make_region_mask(meta, patch_orders)

        train_mask = rmask & meta['split'].eq('train_pool').values
        test_mask = rmask & meta['split'].eq('test').values

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        meta_test = meta[test_mask].copy().reset_index(drop=True)

        print('Train:', X_train.shape, Counter(y_train), flush=True)
        print('Test:', X_test.shape, Counter(y_test), flush=True)

        if len(y_train) == 0 or len(y_test) == 0:
            raise ValueError(f'Empty final train/test for H={H}m.')
        if len(np.unique(y_train)) < 2:
            raise ValueError(f'Only one class in final training data for H={H}m.')

        for cfg_idx, cfg in enumerate(selected_configs):
            model_name = cfg['model_name']
            cfg_name = cfg['config_name']
            th_key = (model_name, H)
            if th_key not in threshold_map:
                raise ValueError(f'Missing pooled threshold for {th_key}')
            threshold = threshold_map[th_key]

            print('\n' + '=' * 100, flush=True)
            print(f'Final model={model_name}, config={cfg_name}, H={H}m', flush=True)
            print('threshold:', threshold, flush=True)

            model_seed = args.seed + 900000 + 10000 * (cfg_idx + 1) + H * 100
            model = make_model(cfg, seed=model_seed, n_jobs=args.n_jobs)
            sw = sample_weight_for_mode(y_train, args.sample_weight_mode)
            fit_model(model, X_train, y_train, sample_weight=sw)
            score = positive_probability(model, X_test)
            metrics, pred = evaluate_binary(y_test, score, threshold)

            row = {
                'region_name': args.region_name,
                'model_name': model_name,
                'config_name': cfg_name,
                'horizon_month': H,
                'objective': 'binary_Mge5',
                'sample_weight_mode': args.sample_weight_mode,
                'n_train': int(len(y_train)),
                'train_negative': int(np.sum(y_train == 0)),
                'train_positive': int(np.sum(y_train == 1)),
                'n_test': int(len(y_test)),
                'test_negative': int(np.sum(y_test == 0)),
                'test_positive': int(np.sum(y_test == 1)),
            }
            row.update(config_param_columns(cfg))
            row.update(metrics)
            final_rows.append(row)

            safe_model_name = model_name.lower().replace(' ', '_')
            pred_df = meta_test.copy()
            pred_df['model_name'] = model_name
            pred_df['config_name'] = cfg_name
            pred_df['y_true'] = y_test.astype(int)
            pred_df['score_m5'] = score.astype(float)
            pred_df['threshold'] = float(threshold)
            pred_df['y_pred'] = pred.astype(int)

            pred_path = os.path.join(pred_dir, f'predictions_test_{safe_model_name}_H{H}m.csv')
            model_path = os.path.join(model_dir, f'{safe_model_name}_binary_m5_H{H}m.joblib')
            pred_df.to_csv(pred_path, index=False)
            joblib.dump(model, model_path)

            print('Metrics:', json.dumps(metrics, indent=2), flush=True)
            print('Saved:', pred_path, flush=True)
            print('Saved:', model_path, flush=True)

            del model, score, pred, pred_df, sw
            gc.collect()

        del X, y, meta, payload, X_train, y_train, X_test, y_test, meta_test
        gc.collect()

    final_df = pd.DataFrame(final_rows)
    final_path = os.path.join(args.out_dir, 'final_test_metrics_catboost_extratrees_xgb_binary_m5.csv')
    final_df.to_csv(final_path, index=False)

    return final_df, final_path


# ============================================================
# comparison and report
# ============================================================

def write_report(args, final_df, paths):
    report_path = os.path.join(args.out_dir, 'catboost_extratrees_xgb_comparison_report.md')

    show_cols = [
        'horizon_month', 'model_name', 'config_name', 'positive_rate',
        'threshold', 'pred_positive_rate', 'acc', 'precision', 'recall',
        'f1', 'auc', 'pr_auc', 'brier', 'tp', 'fp', 'fn', 'tn',
    ]
    show_cols = [c for c in show_cols if c in final_df.columns]

    with open(report_path, 'w') as f:
        f.write('# CatBoost, ExtraTrees and XGBoost binary M>=5 comparison report\n\n')
        f.write('## Run configuration\n\n')
        f.write('```json\n')
        f.write(json.dumps({
            'data_dir': args.data_dir,
            'out_dir': args.out_dir,
            'horizons': args.horizons,
            'folds': args.folds,
            'models': args.models,
            'region_name': args.region_name,
            'patch_csv': args.patch_csv,
            'sample_weight_mode': args.sample_weight_mode,
            'n_jobs': args.n_jobs,
            'seed': args.seed,
        }, indent=2))
        f.write('\n```\n\n')

        f.write('## Output files\n\n')
        for k, v in paths.items():
            f.write(f'- {k}: `{v}`\n')

        f.write('\n## Final comparison metrics\n\n')
        if len(final_df) > 0:
            f.write(final_df[show_cols].to_markdown(index=False))
            f.write('\n')
        else:
            f.write('No final metrics generated.\n')

    return report_path


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', default='data/raw282')
    parser.add_argument('--out-dir', default='outputs/discovery_models')

    parser.add_argument('--horizons', default='1,3,6,12')
    parser.add_argument('--folds', default='1,2,3')
    parser.add_argument('--models', default='catboost,extratrees,xgboost', help='comma-separated: catboost,extratrees,xgboost')
    parser.add_argument('--config-json', default=None)

    parser.add_argument('--region-name', default='all')
    parser.add_argument('--patch-csv', default='')

    parser.add_argument('--sample-weight-mode', choices=['none', 'balanced'], default='none')
    parser.add_argument('--n-jobs', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-missing', action='store_true')

    args = parser.parse_args()

    args.data_dir = expand_path(args.data_dir)
    args.out_dir = expand_path(args.out_dir)
    ensure_dir(args.out_dir)

    horizons = parse_int_list(args.horizons)
    folds = parse_int_list(args.folds)
    model_names = parse_str_list(args.models)
    patch_orders = load_patch_orders(args.patch_csv)
    configs = load_configs(args, model_names)

    config_path = os.path.join(args.out_dir, 'candidate_configs_catboost_extratrees_xgb.json')
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=2)

    print('=' * 100, flush=True)
    print('CatBoost, ExtraTrees and XGBoost binary M>=5 comparison on raw282 datasets', flush=True)
    print('data_dir:', args.data_dir, flush=True)
    print('out_dir:', args.out_dir, flush=True)
    print('horizons:', horizons, flush=True)
    print('folds:', folds, flush=True)
    print('models:', model_names, flush=True)
    print('region_name:', args.region_name, flush=True)
    print('patch_csv:', args.patch_csv, flush=True)
    print('sample_weight_mode:', args.sample_weight_mode, flush=True)
    print('n_jobs:', args.n_jobs, flush=True)
    print('HAS_XGBOOST:', HAS_XGBOOST, flush=True)
    print('HAS_CATBOOST:', HAS_CATBOOST, flush=True)
    print('candidate configs:', config_path, flush=True)
    print('=' * 100, flush=True)

    metric_path, score_path = tune_configs(args, configs, horizons, folds, patch_orders)

    (
        selected_configs,
        threshold_df,
        summary_path,
        by_cfg_horizon_path,
        selected_path,
        threshold_path,
    ) = summarize_and_select(args, configs, metric_path, score_path)

    final_df, final_path = final_test(args, selected_configs, threshold_df, horizons, patch_orders)

    comp_path = os.path.join(args.out_dir, 'model_comparison_final_metrics_binary_m5.csv')
    final_df.to_csv(comp_path, index=False)

    paths = {
        'candidate_configs': config_path,
        'tuning_fold_metrics': metric_path,
        'valid_prediction_scores': score_path,
        'tuning_summary_by_model_config': summary_path,
        'tuning_summary_by_model_config_horizon': by_cfg_horizon_path,
        'selected_config_by_model': selected_path,
        'pooled_threshold_by_model_horizon': threshold_path,
        'final_test_metrics': final_path,
        'model_comparison_final_metrics': comp_path,
    }
    report_path = write_report(args, final_df, paths)

    show_cols = [
        'horizon_month', 'model_name', 'config_name', 'positive_rate',
        'threshold', 'pred_positive_rate', 'acc', 'precision', 'recall',
        'f1', 'auc', 'pr_auc', 'brier', 'tp', 'fp', 'fn', 'tn',
    ]
    show_cols = [c for c in show_cols if c in final_df.columns]

    print('\n' + '=' * 100, flush=True)
    print('FINAL MODEL COMPARISON SUMMARY', flush=True)
    print('=' * 100, flush=True)
    print(final_df[show_cols].to_string(index=False), flush=True)
    print('\nSaved final metrics:', final_path, flush=True)
    print('Saved comparison metrics:', comp_path, flush=True)
    print('Saved report:', report_path, flush=True)


if __name__ == '__main__':
    main()
