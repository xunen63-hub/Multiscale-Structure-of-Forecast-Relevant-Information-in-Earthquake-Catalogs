#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_train_heldout_models.py

Purpose
-------
Add Random Forest and LightGBM baselines for binary M>=5 final-test comparison
using the existing raw282 dataset files:
    china_raw282_dataset_horizon_{H}m.pkl

The script follows the same data contract as the existing XGBoost workflow:
    - X:       raw 282-dim features
    - y_m5:    binary label, 1 means future max magnitude >= 5
    - meta:    contains split=train_pool/test and fold{1,2,3}_role columns

Workflow
--------
1. Inner rolling folds:
   tune/select RF and LightGBM configurations using only inner_train/inner_valid.
2. Threshold selection:
   pool inner-validation scores of the selected configuration for each horizon
   and select the M>=5 threshold by F1, then precision, then threshold.
3. Final test:
   train the selected model on train_pool and evaluate on the final test split.
4. Optional comparison:
   if --xgb-final-metrics exists, merge existing XGBoost final metrics with
   RF/LightGBM metrics into one comparison CSV.

Author: generated for hym63 HPC workflow
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

from sklearn.ensemble import RandomForestClassifier
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
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    lgb = None
    LGBMClassifier = None
    HAS_LIGHTGBM = False


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
    df_new = pd.DataFrame([row])

    if os.path.exists(path):
        df_old = pd.read_csv(path)

        all_cols = list(df_old.columns)
        for c in df_new.columns:
            if c not in all_cols:
                all_cols.append(c)

        df_old = df_old.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)

        df_out = pd.concat([df_old, df_new], ignore_index=True)
        df_out.to_csv(path, index=False)
    else:
        df_new.to_csv(path, index=False)


def append_df_csv(path, df):
    if os.path.exists(path):
        df.to_csv(path, mode='a', index=False, header=False)
    else:
        df.to_csv(path, index=False)


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
        if prob.ndim == 1:
            return prob.astype(float)
        return prob[:, -1].astype(float)

    classes = np.asarray(classes)
    hit = np.where(classes == 1)[0]
    if len(hit) == 0:
        return np.zeros(X.shape[0], dtype=float)

    return prob[:, int(hit[0])].astype(float)


# ============================================================
# model definitions
# ============================================================

def default_candidate_configs(model_names):
    configs = []

    if 'rf' in model_names:
        configs += [
            {
                'model_name': 'RF',
                'config_name': 'rf_cfg0_depth12_leaf5',
                'params': {
                    'n_estimators': 300,
                    'max_depth': 12,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                },
            },
            {
                'model_name': 'RF',
                'config_name': 'rf_cfg1_depth16_leaf5',
                'params': {
                    'n_estimators': 300,
                    'max_depth': 16,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                },
            },
            {
                'model_name': 'RF',
                'config_name': 'rf_cfg2_depth18_leaf10',
                'params': {
                    'n_estimators': 400,
                    'max_depth': 18,
                    'min_samples_split': 20,
                    'min_samples_leaf': 10,
                    'max_features': 0.50,
                    'bootstrap': True,
                },
            },
        ]

    if 'lightgbm' in model_names or 'lgbm' in model_names:
        configs += [
            {
                'model_name': 'LightGBM',
                'config_name': 'lgbm_cfg0_xgb_like',
                'params': {
                    'n_estimators': 500,
                    'learning_rate': 0.030,
                    'num_leaves': 15,
                    'max_depth': 3,
                    'subsample': 0.90,
                    'colsample_bytree': 0.90,
                    'reg_lambda': 8.0,
                    'reg_alpha': 0.8,
                    'min_child_samples': 50,
                    'subsample_freq': 1,
                },
            },
            {
                'model_name': 'LightGBM',
                'config_name': 'lgbm_cfg1_conservative',
                'params': {
                    'n_estimators': 700,
                    'learning_rate': 0.025,
                    'num_leaves': 15,
                    'max_depth': 4,
                    'subsample': 0.85,
                    'colsample_bytree': 0.85,
                    'reg_lambda': 10.0,
                    'reg_alpha': 1.0,
                    'min_child_samples': 80,
                    'subsample_freq': 1,
                },
            },
            {
                'model_name': 'LightGBM',
                'config_name': 'lgbm_cfg2_leaf31',
                'params': {
                    'n_estimators': 600,
                    'learning_rate': 0.030,
                    'num_leaves': 31,
                    'max_depth': -1,
                    'subsample': 0.80,
                    'colsample_bytree': 0.80,
                    'reg_lambda': 8.0,
                    'reg_alpha': 0.8,
                    'min_child_samples': 100,
                    'subsample_freq': 1,
                },
            },
            {
                'model_name': 'LightGBM',
                'config_name': 'lgbm_cfg3_shallow_fast',
                'params': {
                    'n_estimators': 400,
                    'learning_rate': 0.050,
                    'num_leaves': 15,
                    'max_depth': 3,
                    'subsample': 0.90,
                    'colsample_bytree': 0.90,
                    'reg_lambda': 6.0,
                    'reg_alpha': 0.5,
                    'min_child_samples': 50,
                    'subsample_freq': 1,
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
        if m == 'rf':
            allowed.add('RF')
        if m in ('lightgbm', 'lgbm'):
            allowed.add('LightGBM')

    configs = [c for c in configs if c['model_name'] in allowed]

    if any(c['model_name'] == 'LightGBM' for c in configs) and not HAS_LIGHTGBM:
        raise ImportError(
            'lightgbm is not installed in the current Python environment. '\
            'Install it in the conda environment, for example: pip install lightgbm'
        )

    return configs


def make_model(cfg, seed, n_jobs, class_weight_mode):
    model_name = cfg['model_name']
    params = dict(cfg['params'])

    if class_weight_mode == 'balanced':
        class_weight = 'balanced'
    else:
        class_weight = None

    if model_name == 'RF':
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=n_jobs,
            class_weight=class_weight,
            **params,
        )

    if model_name == 'LightGBM':
        if not HAS_LIGHTGBM:
            raise ImportError('lightgbm is not installed.')
        return LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            random_state=seed,
            n_jobs=n_jobs,
            class_weight=class_weight,
            force_col_wise=True,
            verbosity=-1,
            **params,
        )

    raise ValueError(f'Unknown model_name: {model_name}')


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
    metric_path = os.path.join(args.out_dir, 'tuning_fold_metrics_rf_lightgbm.csv')
    score_path = os.path.join(args.out_dir, 'valid_prediction_scores_rf_lightgbm.csv')
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
                model = make_model(
                    cfg,
                    seed=model_seed,
                    n_jobs=args.n_jobs,
                    class_weight_mode=args.class_weight_mode,
                )

                model.fit(X_train, y_train)
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
                    'class_weight_mode': args.class_weight_mode,
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

                del model, X_train, X_valid, y_train, y_valid, score_valid, pred_valid
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
        row = {
            'model_name': c['model_name'],
            'config_name': c['config_name'],
        }
        row.update(config_param_columns(c))
        cfg_rows.append(row)
    cfg_params = pd.DataFrame(cfg_rows)
    summary = summary.merge(cfg_params, on=['model_name', 'config_name'], how='left')

    # Rank configurations within each model type, not across different algorithms.
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
                'class_weight_mode': args.class_weight_mode,
                'selection_rule': (
                    'For each algorithm, select one common hyperparameter configuration across horizons '\
                    'using inner rolling backtests only. Composite rank = 0.35*Brier + 0.30*F1 '\
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
        'model_name',
        'config_name',
        'composite_rank',
        'mean_valid_brier',
        'mean_valid_f1',
        'mean_valid_pr_auc',
        'mean_valid_auc',
        'mean_valid_acc',
        'n_runs',
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
            model = make_model(
                cfg,
                seed=model_seed,
                n_jobs=args.n_jobs,
                class_weight_mode=args.class_weight_mode,
            )
            model.fit(X_train, y_train)
            score = positive_probability(model, X_test)
            metrics, pred = evaluate_binary(y_test, score, threshold)

            row = {
                'region_name': args.region_name,
                'model_name': model_name,
                'config_name': cfg_name,
                'horizon_month': H,
                'objective': 'binary_Mge5',
                'class_weight_mode': args.class_weight_mode,
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

            del model, score, pred, pred_df
            gc.collect()

        del X, y, meta, payload, X_train, y_train, X_test, y_test, meta_test
        gc.collect()

    final_df = pd.DataFrame(final_rows)
    final_path = os.path.join(args.out_dir, 'final_test_metrics_rf_lightgbm_binary_m5.csv')
    final_df.to_csv(final_path, index=False)

    return final_df, final_path


# ============================================================
# comparison with existing XGBoost final metrics
# ============================================================

def maybe_add_xgb_metrics(args, final_df):
    rows = [final_df]

    xgb_path = args.xgb_final_metrics
    if xgb_path is not None and str(xgb_path).strip() != '':
        xgb_path = expand_path(xgb_path)
        if os.path.exists(xgb_path):
            xgb_df = pd.read_csv(xgb_path)
            if 'model_name' not in xgb_df.columns:
                xgb_df.insert(0, 'model_name', 'XGBoost')
            else:
                xgb_df['model_name'] = xgb_df['model_name'].fillna('XGBoost')
            if 'class_weight_mode' not in xgb_df.columns:
                xgb_df['class_weight_mode'] = 'none'
            rows.append(xgb_df)
            print('Merged existing XGBoost metrics:', xgb_path, flush=True)
        else:
            print(f'[WARN] xgb-final-metrics not found, skip merge: {xgb_path}', flush=True)

    comp = pd.concat(rows, ignore_index=True, sort=False)

    preferred = [
        'region_name',
        'model_name',
        'config_name',
        'horizon_month',
        'objective',
        'class_weight_mode',
        'n_train',
        'n_test',
        'test_positive',
        'test_negative',
        'positive_rate',
        'threshold',
        'pred_positive_rate',
        'acc',
        'precision',
        'recall',
        'f1',
        'auc',
        'pr_auc',
        'brier',
        'tp',
        'fp',
        'fn',
        'tn',
    ]
    ordered = [c for c in preferred if c in comp.columns] + [c for c in comp.columns if c not in preferred]
    comp = comp[ordered]
    comp = comp.sort_values(['horizon_month', 'model_name'], na_position='last').reset_index(drop=True)

    comp_path = os.path.join(args.out_dir, 'model_comparison_final_metrics_binary_m5.csv')
    comp.to_csv(comp_path, index=False)

    return comp, comp_path


def write_report(args, final_df, comp_df, paths):
    report_path = os.path.join(args.out_dir, 'rf_lightgbm_comparison_report.md')

    show_cols = [
        'horizon_month',
        'model_name',
        'config_name',
        'positive_rate',
        'threshold',
        'pred_positive_rate',
        'acc',
        'precision',
        'recall',
        'f1',
        'auc',
        'pr_auc',
        'brier',
        'tp',
        'fp',
        'fn',
        'tn',
    ]
    show_cols = [c for c in show_cols if c in comp_df.columns]

    with open(report_path, 'w') as f:
        f.write('# RF and LightGBM binary M>=5 comparison report\n\n')
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
            'class_weight_mode': args.class_weight_mode,
            'n_jobs': args.n_jobs,
            'seed': args.seed,
            'xgb_final_metrics': args.xgb_final_metrics,
        }, indent=2))
        f.write('\n```\n\n')

        f.write('## Output files\n\n')
        for k, v in paths.items():
            f.write(f'- {k}: `{v}`\n')

        f.write('\n## Final comparison metrics\n\n')
        if len(comp_df) > 0:
            f.write(comp_df[show_cols].to_markdown(index=False))
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
    parser.add_argument('--out-dir', default='outputs/heldout_models')

    parser.add_argument('--horizons', default='1,3,6,12')
    parser.add_argument('--folds', default='1,2,3')
    parser.add_argument('--models', default='rf,lightgbm', help='comma-separated: rf,lightgbm')
    parser.add_argument('--config-json', default=None)

    parser.add_argument('--region-name', default='all')
    parser.add_argument('--patch-csv', default='')

    parser.add_argument('--class-weight-mode', choices=['none', 'balanced'], default='none')
    parser.add_argument('--n-jobs', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-missing', action='store_true')

    parser.add_argument('--xgb-final-metrics', default='', help='optional existing XGBoost final metrics CSV')

    args = parser.parse_args()

    args.data_dir = expand_path(args.data_dir)
    args.out_dir = expand_path(args.out_dir)
    ensure_dir(args.out_dir)

    horizons = parse_int_list(args.horizons)
    folds = parse_int_list(args.folds)
    model_names = parse_str_list(args.models)
    patch_orders = load_patch_orders(args.patch_csv)
    configs = load_configs(args, model_names)

    config_path = os.path.join(args.out_dir, 'candidate_configs_rf_lightgbm.json')
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=2)

    print('=' * 100, flush=True)
    print('RF and LightGBM binary M>=5 comparison on raw282 datasets', flush=True)
    print('data_dir:', args.data_dir, flush=True)
    print('out_dir:', args.out_dir, flush=True)
    print('horizons:', horizons, flush=True)
    print('folds:', folds, flush=True)
    print('models:', model_names, flush=True)
    print('region_name:', args.region_name, flush=True)
    print('patch_csv:', args.patch_csv, flush=True)
    print('class_weight_mode:', args.class_weight_mode, flush=True)
    print('n_jobs:', args.n_jobs, flush=True)
    print('HAS_LIGHTGBM:', HAS_LIGHTGBM, flush=True)
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
    comp_df, comp_path = maybe_add_xgb_metrics(args, final_df)

    paths = {
        'candidate_configs': config_path,
        'tuning_fold_metrics': metric_path,
        'valid_prediction_scores': score_path,
        'tuning_summary_by_model_config': summary_path,
        'tuning_summary_by_model_config_horizon': by_cfg_horizon_path,
        'selected_config_by_model': selected_path,
        'pooled_threshold_by_model_horizon': threshold_path,
        'final_test_metrics_rf_lightgbm': final_path,
        'model_comparison_final_metrics': comp_path,
    }
    report_path = write_report(args, final_df, comp_df, paths)

    show_cols = [
        'horizon_month',
        'model_name',
        'config_name',
        'positive_rate',
        'threshold',
        'pred_positive_rate',
        'acc',
        'precision',
        'recall',
        'f1',
        'auc',
        'pr_auc',
        'brier',
        'tp',
        'fp',
        'fn',
        'tn',
    ]
    show_cols = [c for c in show_cols if c in comp_df.columns]

    print('\n' + '=' * 100, flush=True)
    print('FINAL MODEL COMPARISON SUMMARY', flush=True)
    print('=' * 100, flush=True)
    print(comp_df[show_cols].to_string(index=False), flush=True)
    print('\nSaved final RF/LightGBM metrics:', final_path, flush=True)
    print('Saved comparison metrics:', comp_path, flush=True)
    print('Saved report:', report_path, flush=True)


if __name__ == '__main__':
    main()
