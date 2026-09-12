#!/usr/bin/env python3
"""Paired calendar-year block bootstrap for saved outer-test predictions.

Each bootstrap draw samples the nine observed calendar-year blocks
(2012--2020) with replacement.  A selected block always contains every
available monthly origin and all 85 spatial cells.  The same draw is used
for every paired model/feature-set comparison and every forecast horizon.
No model is refitted.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISCOVERY_PRED = REPO_ROOT / "outputs" / "feature_group_ablation" / "predictions"
DEFAULT_HELDOUT_PRED = REPO_ROOT / "outputs" / "heldout_group_validation" / "predictions"
DEFAULT_OUT = REPO_ROOT / "outputs" / "year_block_bootstrap"

MODEL_LABELS = {
    "catboost": "CatBoost",
    "extratrees": "ExtraTrees",
    "xgboost": "XGBoost",
    "rf": "RF",
    "lightgbm": "LightGBM",
}
MODEL_ORDER = ["CatBoost", "ExtraTrees", "XGBoost", "RF", "LightGBM"]
DISCOVERY_MODELS = ["CatBoost", "ExtraTrees", "XGBoost"]
HELDOUT_MODELS = ["RF", "LightGBM"]
HORIZONS = [1, 3, 6, 12]
FEATURE_SETS = [
    "full_features",
    "only_top3_groups",
    "remove_top1_groups",
    "remove_top2_groups",
    "remove_top3_groups",
]
METRICS = ["roc_auc", "pr_auc", "brier"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--discovery-predictions-dir",
        type=Path,
        default=DEFAULT_DISCOVERY_PRED,
        help="Row-level prediction CSVs from script 06.",
    )
    parser.add_argument(
        "--heldout-predictions-dir",
        type=Path,
        default=DEFAULT_HELDOUT_PRED,
        help="Row-level prediction CSVs from script 07.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--equivalence-margin", type=float, default=0.01)
    parser.add_argument("--ap-batch-size", type=int, default=125)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def prediction_files(
    discovery_predictions_dir: Path,
    heldout_predictions_dir: Path,
) -> dict[tuple[str, int, str], Path]:
    pattern = re.compile(
        r"predictions_test_(?P<model>.+)_H(?P<horizon>\d+)m_(?P<feature>.+)\.csv$"
    )
    found: dict[tuple[str, int, str], Path] = {}
    for directory in [discovery_predictions_dir, heldout_predictions_dir]:
        if not directory.is_dir():
            raise NotADirectoryError(f"Prediction directory not found: {directory}")
        for path in sorted(directory.glob("predictions_test_*.csv")):
            match = pattern.match(path.name)
            if not match:
                continue
            model = MODEL_LABELS[match.group("model")]
            horizon = int(match.group("horizon"))
            feature_set = match.group("feature")
            if feature_set not in FEATURE_SETS:
                continue
            key = (model, horizon, feature_set)
            if key in found:
                raise ValueError(f"Duplicate prediction file for {key}: {path}")
            found[key] = path

    expected = {
        (model, horizon, feature_set)
        for model in MODEL_ORDER
        for horizon in HORIZONS
        for feature_set in FEATURE_SETS
    }
    missing = sorted(expected - set(found))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} prediction files: {missing[:8]}")
    return found


def load_predictions(
    files: dict[tuple[str, int, str], Path],
) -> tuple[dict[tuple[str, int, str], dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    loaded: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    reference_by_horizon: dict[int, pd.DataFrame] = {}
    common_years: np.ndarray | None = None
    common_t0: np.ndarray | None = None

    for key in sorted(files, key=lambda x: (HORIZONS.index(x[1]), MODEL_ORDER.index(x[0]), x[2])):
        model, horizon, feature_set = key
        frame = pd.read_csv(
            files[key],
            usecols=["t0", "patch_order", "y_true", "score_m5"],
            parse_dates=["t0"],
        ).sort_values(["t0", "patch_order"], kind="mergesort").reset_index(drop=True)

        if len(frame) != 9095:
            raise ValueError(f"{files[key]} has {len(frame)} rows, expected 9095")
        if frame.duplicated(["t0", "patch_order"]).any():
            raise ValueError(f"Duplicate cell-origin keys in {files[key]}")
        per_origin = frame.groupby("t0", sort=True)["patch_order"].nunique()
        if len(per_origin) != 107 or not (per_origin == 85).all():
            raise ValueError(f"Incomplete monthly spatial slabs in {files[key]}")

        keys_and_y = frame[["t0", "patch_order", "y_true"]]
        if horizon not in reference_by_horizon:
            reference_by_horizon[horizon] = keys_and_y
        elif not keys_and_y.equals(reference_by_horizon[horizon]):
            raise ValueError(f"Pairing mismatch in {files[key]}")

        years = frame["t0"].dt.year.to_numpy(dtype=int)
        if common_years is None:
            common_years = years
            common_t0 = frame["t0"].to_numpy()
        elif not np.array_equal(years, common_years) or not np.array_equal(
            frame["t0"].to_numpy(), common_t0
        ):
            raise ValueError(f"Calendar keys differ across horizons in {files[key]}")

        loaded[key] = {
            "y": frame["y_true"].to_numpy(dtype=np.int8),
            "score": frame["score_m5"].to_numpy(dtype=float),
        }

    assert common_years is not None and common_t0 is not None
    unique_years = np.unique(common_years)
    if unique_years.tolist() != list(range(2012, 2021)):
        raise ValueError(f"Unexpected test years: {unique_years.tolist()}")
    return loaded, common_years, common_t0


def point_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    # ROC-AUC as the tie-adjusted Mann--Whitney concordance probability.
    ascending = np.argsort(score, kind="mergesort")
    y_ascending = y[ascending]
    score_ascending = score[ascending]
    group_ends_ascending = np.r_[
        np.flatnonzero(score_ascending[:-1] != score_ascending[1:]),
        len(score_ascending) - 1,
    ]
    positive_cumulative = np.cumsum(y_ascending)
    total_cumulative = np.arange(1, len(y_ascending) + 1)
    positive_at_end = positive_cumulative[group_ends_ascending]
    negative_at_end = total_cumulative[group_ends_ascending] - positive_at_end
    positive_in_group = np.diff(np.r_[0, positive_at_end])
    negative_in_group = np.diff(np.r_[0, negative_at_end])
    negative_before_group = np.r_[0, negative_at_end[:-1]]
    concordant = np.sum(
        positive_in_group * (negative_before_group + 0.5 * negative_in_group)
    )
    n_positive = float(y.sum())
    n_negative = float(len(y) - y.sum())
    roc_auc = float(concordant / (n_positive * n_negative))

    # Average precision using the standard non-interpolated step integral.
    descending = np.argsort(-score, kind="mergesort")
    y_descending = y[descending]
    score_descending = score[descending]
    group_ends_descending = np.r_[
        np.flatnonzero(score_descending[:-1] != score_descending[1:]),
        len(score_descending) - 1,
    ]
    tp_at_end = np.cumsum(y_descending)[group_ends_descending]
    total_at_end = group_ends_descending + 1
    positive_in_group_descending = np.diff(np.r_[0, tp_at_end])
    precision_at_end = tp_at_end / total_at_end
    pr_auc = float(
        np.sum(positive_in_group_descending * precision_at_end) / n_positive
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": float(np.mean((score - y) ** 2)),
    }


def roc_bootstrap(
    y: np.ndarray,
    score: np.ndarray,
    year_index: np.ndarray,
    weights: np.ndarray,
    n_years: int,
) -> np.ndarray:
    positive_counts = np.bincount(year_index[y == 1], minlength=n_years).astype(float)
    negative_counts = np.bincount(year_index[y == 0], minlength=n_years).astype(float)
    concordance = np.zeros((n_years, n_years), dtype=float)

    for pos_year in range(n_years):
        pos_score = score[(year_index == pos_year) & (y == 1)]
        for neg_year in range(n_years):
            neg_score = np.sort(score[(year_index == neg_year) & (y == 0)])
            left = np.searchsorted(neg_score, pos_score, side="left")
            right = np.searchsorted(neg_score, pos_score, side="right")
            concordance[pos_year, neg_year] = np.sum(
                left + 0.5 * (right - left), dtype=float
            )

    numerator = np.einsum("bi,ij,bj->b", weights, concordance, weights, optimize=True)
    denominator = (weights @ positive_counts) * (weights @ negative_counts)
    return numerator / denominator


def pr_bootstrap(
    y: np.ndarray,
    score: np.ndarray,
    year_index: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_y = y[order].astype(float)
    sorted_year = year_index[order]
    group_ends = np.r_[
        np.flatnonzero(sorted_score[:-1] != sorted_score[1:]),
        len(sorted_score) - 1,
    ]
    result = np.empty(len(weights), dtype=float)

    for start in range(0, len(weights), batch_size):
        stop = min(start + batch_size, len(weights))
        row_weight = weights[start:stop, sorted_year].astype(float, copy=False)
        cumulative_total = np.cumsum(row_weight, axis=1)
        cumulative_positive = np.cumsum(row_weight * sorted_y, axis=1)
        total_at_threshold = cumulative_total[:, group_ends]
        positive_at_threshold = cumulative_positive[:, group_ends]
        positive_in_group = np.diff(
            np.concatenate(
                [np.zeros((stop - start, 1)), positive_at_threshold], axis=1
            ),
            axis=1,
        )
        precision = np.divide(
            positive_at_threshold,
            total_at_threshold,
            out=np.zeros_like(positive_at_threshold),
            where=total_at_threshold > 0,
        )
        total_positive = positive_at_threshold[:, -1]
        result[start:stop] = np.sum(positive_in_group * precision, axis=1) / total_positive
    return result


def brier_bootstrap(
    y: np.ndarray,
    score: np.ndarray,
    year_index: np.ndarray,
    weights: np.ndarray,
    n_years: int,
) -> np.ndarray:
    squared_error = (score - y) ** 2
    error_sum = np.bincount(year_index, weights=squared_error, minlength=n_years)
    row_count = np.bincount(year_index, minlength=n_years).astype(float)
    return (weights @ error_sum) / (weights @ row_count)


def bootstrap_metrics(
    y: np.ndarray,
    score: np.ndarray,
    year_index: np.ndarray,
    weights: np.ndarray,
    n_years: int,
    ap_batch_size: int,
) -> dict[str, np.ndarray]:
    result = {
        "roc_auc": roc_bootstrap(y, score, year_index, weights, n_years),
        "pr_auc": pr_bootstrap(y, score, year_index, weights, ap_batch_size),
        "brier": brier_bootstrap(y, score, year_index, weights, n_years),
    }

    # Exact checks for the unweighted original sample.
    observed = point_metrics(y, score)
    ones = np.ones((1, n_years), dtype=int)
    checks = {
        "roc_auc": roc_bootstrap(y, score, year_index, ones, n_years)[0],
        "pr_auc": pr_bootstrap(y, score, year_index, ones, 1)[0],
        "brier": brier_bootstrap(y, score, year_index, ones, n_years)[0],
    }
    for metric in METRICS:
        if not np.isclose(observed[metric], checks[metric], atol=1e-12, rtol=1e-10):
            raise RuntimeError(
                f"Metric implementation mismatch for {metric}: "
                f"{observed[metric]} vs {checks[metric]}"
            )
    return result


def interval(values: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return float(low), float(high)


def paired_row(
    *,
    comparison: str,
    model: str,
    comparator_model: str | None,
    horizon: int | str,
    candidate: str,
    reference: str,
    metric: str,
    point_delta: float,
    bootstrap_delta: np.ndarray,
    ci: float,
    equivalence_margin: float,
    expected: str,
    model_set: str = "individual",
) -> dict[str, object]:
    ci_low, ci_high = interval(bootstrap_delta, ci)
    if expected == "candidate_worse":
        direction_supported = ci_high < 0 if metric != "brier" else ci_low > 0
    elif expected == "candidate_better":
        direction_supported = ci_low > 0 if metric != "brier" else ci_high < 0
    else:
        direction_supported = False
    return {
        "comparison": comparison,
        "model_set": model_set,
        "model": model,
        "comparator_model": comparator_model or "",
        "horizon_month": horizon,
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "point_delta_candidate_minus_reference": point_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": ci,
        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
        "expected_direction": expected,
        "expected_direction_supported": bool(direction_supported),
        "equivalence_low": -equivalence_margin if metric == "roc_auc" else np.nan,
        "equivalence_high": equivalence_margin if metric == "roc_auc" else np.nan,
        "point_within_equivalence_margin": (
            bool(-equivalence_margin <= point_delta <= equivalence_margin)
            if metric == "roc_auc"
            else False
        ),
        "ci_fully_within_equivalence_margin": (
            bool(ci_low >= -equivalence_margin and ci_high <= equivalence_margin)
            if metric == "roc_auc"
            else False
        ),
        "ci_above_noninferiority_margin": (
            bool(ci_low >= -equivalence_margin)
            if metric == "roc_auc"
            else False
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = prediction_files(
        args.discovery_predictions_dir,
        args.heldout_predictions_dir,
    )
    predictions, row_years, row_t0 = load_predictions(files)

    years = np.unique(row_years)
    year_to_index = {year: index for index, year in enumerate(years)}
    year_index = np.array([year_to_index[year] for year in row_years], dtype=np.int8)
    rng = np.random.default_rng(args.seed)
    bootstrap_weights = rng.multinomial(
        len(years),
        np.repeat(1.0 / len(years), len(years)),
        size=args.n_bootstrap,
    ).astype(np.int16)

    metric_point: dict[tuple[str, int, str, str], float] = {}
    metric_boot: dict[tuple[str, int, str, str], np.ndarray] = {}
    task_rows: list[dict[str, object]] = []

    for model in MODEL_ORDER:
        for horizon in HORIZONS:
            for feature_set in FEATURE_SETS:
                data = predictions[(model, horizon, feature_set)]
                points = point_metrics(data["y"], data["score"])
                boots = bootstrap_metrics(
                    data["y"],
                    data["score"],
                    year_index,
                    bootstrap_weights,
                    len(years),
                    args.ap_batch_size,
                )
                for metric in METRICS:
                    key = (model, horizon, feature_set, metric)
                    metric_point[key] = points[metric]
                    metric_boot[key] = boots[metric]
                    ci_low, ci_high = interval(boots[metric], args.ci)
                    task_rows.append(
                        {
                            "model": model,
                            "horizon_month": horizon,
                            "feature_set": feature_set,
                            "metric": metric,
                            "point_estimate": points[metric],
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "ci_level": args.ci,
                        }
                    )

    paired_rows: list[dict[str, object]] = []
    paired_draws: dict[str, np.ndarray] = {}

    # Compact three-group core versus its matched full model.
    for model in MODEL_ORDER:
        for horizon in HORIZONS:
            for metric in METRICS:
                candidate_key = (model, horizon, "only_top3_groups", metric)
                reference_key = (model, horizon, "full_features", metric)
                delta = metric_boot[candidate_key] - metric_boot[reference_key]
                point_delta = metric_point[candidate_key] - metric_point[reference_key]
                paired_rows.append(
                    paired_row(
                        comparison="compact_core_vs_full",
                        model=model,
                        comparator_model=None,
                        horizon=horizon,
                        candidate="only_top3_groups",
                        reference="full_features",
                        metric=metric,
                        point_delta=point_delta,
                        bootstrap_delta=delta,
                        ci=args.ci,
                        equivalence_margin=args.equivalence_margin,
                        expected="equivalence",
                    )
                )
                paired_draws[f"compact__{model}__H{horizon}__{metric}"] = delta

    # Cumulative removal versus matched full model.
    for feature_set in ["remove_top1_groups", "remove_top2_groups", "remove_top3_groups"]:
        for model in MODEL_ORDER:
            for horizon in HORIZONS:
                for metric in METRICS:
                    candidate_key = (model, horizon, feature_set, metric)
                    reference_key = (model, horizon, "full_features", metric)
                    delta = metric_boot[candidate_key] - metric_boot[reference_key]
                    point_delta = metric_point[candidate_key] - metric_point[reference_key]
                    paired_rows.append(
                        paired_row(
                            comparison="cumulative_removal_vs_full",
                            model=model,
                            comparator_model=None,
                            horizon=horizon,
                            candidate=feature_set,
                            reference="full_features",
                            metric=metric,
                            point_delta=point_delta,
                            bootstrap_delta=delta,
                            ci=args.ci,
                            equivalence_margin=args.equivalence_margin,
                            expected="candidate_worse",
                        )
                    )
                    paired_draws[f"removal__{feature_set}__{model}__H{horizon}__{metric}"] = delta

    # ExtraTrees full-model advantage relative to every other algorithm.
    for comparator in [m for m in MODEL_ORDER if m != "ExtraTrees"]:
        for horizon in HORIZONS:
            for metric in METRICS:
                candidate_key = ("ExtraTrees", horizon, "full_features", metric)
                reference_key = (comparator, horizon, "full_features", metric)
                delta = metric_boot[candidate_key] - metric_boot[reference_key]
                point_delta = metric_point[candidate_key] - metric_point[reference_key]
                paired_rows.append(
                    paired_row(
                        comparison="extratrees_full_vs_other_full",
                        model="ExtraTrees",
                        comparator_model=comparator,
                        horizon=horizon,
                        candidate="ExtraTrees/full_features",
                        reference=f"{comparator}/full_features",
                        metric=metric,
                        point_delta=point_delta,
                        bootstrap_delta=delta,
                        ci=args.ci,
                        equivalence_margin=args.equivalence_margin,
                        expected="candidate_better",
                    )
                )
                paired_draws[f"extratrees__vs__{comparator}__H{horizon}__{metric}"] = delta

    # Algorithm-mean paired effects.  The same temporal draws are retained,
    # so averaging does not treat algorithms as independent observations.
    model_sets = {
        "all5": MODEL_ORDER,
        "discovery3": DISCOVERY_MODELS,
        "heldout2": HELDOUT_MODELS,
    }
    for model_set_name, models in model_sets.items():
        for horizon in HORIZONS:
            for metric in METRICS:
                deltas = np.stack(
                    [
                        metric_boot[(m, horizon, "only_top3_groups", metric)]
                        - metric_boot[(m, horizon, "full_features", metric)]
                        for m in models
                    ]
                )
                point_deltas = [
                    metric_point[(m, horizon, "only_top3_groups", metric)]
                    - metric_point[(m, horizon, "full_features", metric)]
                    for m in models
                ]
                paired_rows.append(
                    paired_row(
                        comparison="compact_core_vs_full_model_mean",
                        model="algorithm_mean",
                        comparator_model=None,
                        horizon=horizon,
                        candidate="only_top3_groups",
                        reference="full_features",
                        metric=metric,
                        point_delta=float(np.mean(point_deltas)),
                        bootstrap_delta=deltas.mean(axis=0),
                        ci=args.ci,
                        equivalence_margin=args.equivalence_margin,
                        expected="equivalence",
                        model_set=model_set_name,
                    )
                )

        for feature_set in ["remove_top1_groups", "remove_top2_groups", "remove_top3_groups"]:
            for horizon in HORIZONS:
                for metric in METRICS:
                    deltas = np.stack(
                        [
                            metric_boot[(m, horizon, feature_set, metric)]
                            - metric_boot[(m, horizon, "full_features", metric)]
                            for m in models
                        ]
                    )
                    point_deltas = [
                        metric_point[(m, horizon, feature_set, metric)]
                        - metric_point[(m, horizon, "full_features", metric)]
                        for m in models
                    ]
                    paired_rows.append(
                        paired_row(
                            comparison="cumulative_removal_vs_full_model_mean",
                            model="algorithm_mean",
                            comparator_model=None,
                            horizon=horizon,
                            candidate=feature_set,
                            reference="full_features",
                            metric=metric,
                            point_delta=float(np.mean(point_deltas)),
                            bootstrap_delta=deltas.mean(axis=0),
                            ci=args.ci,
                            equivalence_margin=args.equivalence_margin,
                            expected="candidate_worse",
                            model_set=model_set_name,
                        )
                    )

    # Cross-horizon average for each ExtraTrees comparison, preserving the
    # dependence among nested horizons through common year draws.
    for comparator in [m for m in MODEL_ORDER if m != "ExtraTrees"]:
        for metric in METRICS:
            deltas = np.stack(
                [
                    metric_boot[("ExtraTrees", h, "full_features", metric)]
                    - metric_boot[(comparator, h, "full_features", metric)]
                    for h in HORIZONS
                ]
            )
            point_deltas = [
                metric_point[("ExtraTrees", h, "full_features", metric)]
                - metric_point[(comparator, h, "full_features", metric)]
                for h in HORIZONS
            ]
            paired_rows.append(
                paired_row(
                    comparison="extratrees_full_vs_other_full_horizon_mean",
                    model="ExtraTrees",
                    comparator_model=comparator,
                    horizon="mean_1_3_6_12",
                    candidate="ExtraTrees/full_features",
                    reference=f"{comparator}/full_features",
                    metric=metric,
                    point_delta=float(np.mean(point_deltas)),
                    bootstrap_delta=deltas.mean(axis=0),
                    ci=args.ci,
                    equivalence_margin=args.equivalence_margin,
                    expected="candidate_better",
                )
            )

    task_frame = pd.DataFrame(task_rows)
    paired_frame = pd.DataFrame(paired_rows)
    task_frame.to_csv(args.output_dir / "bootstrap_task_metric_intervals.csv", index=False)
    paired_frame.to_csv(args.output_dir / "paired_block_bootstrap_differences.csv", index=False)

    compact = paired_frame[
        paired_frame["comparison"].isin(
            ["compact_core_vs_full", "compact_core_vs_full_model_mean"]
        )
        & paired_frame["metric"].eq("roc_auc")
    ].copy()
    compact.to_csv(args.output_dir / "compact_core_roc_equivalence.csv", index=False)

    removal = paired_frame[
        paired_frame["comparison"].str.startswith("cumulative_removal")
    ].copy()
    removal.to_csv(args.output_dir / "cumulative_removal_intervals.csv", index=False)

    extratrees = paired_frame[
        paired_frame["comparison"].str.startswith("extratrees")
    ].copy()
    extratrees.to_csv(args.output_dir / "extratrees_pairwise_intervals.csv", index=False)

    np.savez_compressed(
        args.output_dir / "paired_bootstrap_draws.npz",
        bootstrap_year_counts=bootstrap_weights,
        bootstrap_year_labels=years,
        **paired_draws,
    )

    individual_compact = compact[compact["comparison"].eq("compact_core_vs_full")]
    individual_removal = removal[
        removal["comparison"].eq("cumulative_removal_vs_full")
        & removal["metric"].eq("roc_auc")
    ]
    individual_et = extratrees[
        extratrees["comparison"].eq("extratrees_full_vs_other_full")
        & extratrees["metric"].eq("roc_auc")
    ]

    summary = {
        "method": "paired calendar-year block bootstrap",
        "years": years.tolist(),
        "n_year_blocks_per_draw": len(years),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "ci_level": args.ci,
        "equivalence_margin_roc_auc": [
            -args.equivalence_margin,
            args.equivalence_margin,
        ],
        "test_origins": {
            "first": str(pd.Timestamp(row_t0.min()).date()),
            "last": str(pd.Timestamp(row_t0.max()).date()),
            "n_months": int(len(np.unique(row_t0))),
            "n_cells_per_month": 85,
        },
        "compact_individual_tasks_equivalent": int(
            individual_compact["ci_fully_within_equivalence_margin"].sum()
        ),
        "compact_individual_tasks_noninferior": int(
            individual_compact["ci_above_noninferiority_margin"].sum()
        ),
        "compact_individual_tasks_total": int(len(individual_compact)),
        "removal_individual_roc_losses_supported": {
            feature_set: int(
                group["expected_direction_supported"].sum()
            )
            for feature_set, group in individual_removal.groupby("candidate")
        },
        "removal_individual_roc_tasks_per_feature_set": 20,
        "extratrees_individual_roc_advantages_supported": int(
            individual_et["expected_direction_supported"].sum()
        ),
        "extratrees_individual_roc_comparisons_total": int(len(individual_et)),
        "unavailable_for_block_bootstrap": [
            "single-history latest-month, Annual, and Latest-100 models: only aggregate metrics are local",
            "individual-feature Top-K models: no local row-level outer-test predictions were found",
        ],
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    readme = f"""# Paired calendar-year block bootstrap

- Outer-test origins: {summary['test_origins']['first']} to {summary['test_origins']['last']} ({summary['test_origins']['n_months']} monthly origins; all 85 cells retained per origin).
- Resampling unit: calendar-year slab; the nine observed years 2012--2020 were sampled with replacement, nine blocks per draw.
- Bootstrap draws: {args.n_bootstrap:,}; seed: {args.seed}; percentile {args.ci:.0%} intervals.
- Paired differences use the same resampled rows for candidate and matched full predictions.
- Compact-core ROC-AUC equivalence margin: [{-args.equivalence_margin:.3f}, {args.equivalence_margin:.3f}].

## Headline counts

- Individual compact-core model--horizon tasks whose entire ROC-AUC interval is inside the equivalence margin: {summary['compact_individual_tasks_equivalent']}/{summary['compact_individual_tasks_total']}.
- Individual compact-core model--horizon tasks whose ROC-AUC interval is above the -0.010 noninferiority margin: {summary['compact_individual_tasks_noninferior']}/{summary['compact_individual_tasks_total']}.
- Individual cumulative-removal ROC-AUC intervals supporting a loss: {summary['removal_individual_roc_losses_supported']} (20 model--horizon tasks per removal depth).
- Individual ExtraTrees-versus-other full-model ROC-AUC intervals supporting an advantage: {summary['extratrees_individual_roc_advantages_supported']}/{summary['extratrees_individual_roc_comparisons_total']}.

## Data gaps

The local workspace does not contain row-level predictions for the three single-history experiments or the individual-feature Top-K experiments. Their aggregate point estimates cannot be converted into paired block-bootstrap intervals. Saved row-level predictions or fresh inference from saved fitted models are required; retraining is not required.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
