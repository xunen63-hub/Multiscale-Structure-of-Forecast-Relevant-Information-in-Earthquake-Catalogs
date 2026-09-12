#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build three restricted-feature datasets from existing raw-282 dataset files:

1. last_month_only: all lag-1 monthly features (21 features)
2. annual_only: all annual features (15 features)
3. last100_only: all last-100-event features (15 features)

This script DOES NOT recompute catalog features. It slices the already-built
raw-282 matrices, preserving labels, sample metadata, train/test splits, and
inner-fold columns exactly. This guarantees paired comparison with the full
282-feature experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


EXPECTED_INPUT_DIM = 282
EXPECTED_SUBSET_DIMS = {
    "last_month_only": 21,
    "annual_only": 15,
    "last100_only": 15,
}


def expand_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def parse_horizons(text: str) -> List[int]:
    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Forecast horizon must be positive, got {value}")
        values.append(value)

    if not values:
        raise ValueError("No valid horizons were supplied.")

    return values


def load_feature_metadata(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Feature metadata not found: {path}")

    df = pd.read_csv(path)
    required = {"feature_idx", "feature_name", "time_group", "month_lag"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Feature metadata is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["feature_idx"] = pd.to_numeric(df["feature_idx"], errors="raise").astype(int)
    df["month_lag"] = pd.to_numeric(df["month_lag"], errors="coerce")
    df = df.sort_values("feature_idx").reset_index(drop=True)

    expected_indices = np.arange(EXPECTED_INPUT_DIM, dtype=int)
    actual_indices = df["feature_idx"].to_numpy(dtype=int)

    if len(df) != EXPECTED_INPUT_DIM:
        raise ValueError(
            f"Expected {EXPECTED_INPUT_DIM} metadata rows, got {len(df)}"
        )
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(
            "feature_idx must be exactly 0..281 in the raw-282 metadata file."
        )
    if df["feature_name"].duplicated().any():
        duplicated = df.loc[df["feature_name"].duplicated(), "feature_name"].tolist()
        raise ValueError(f"Duplicated feature names: {duplicated[:10]}")

    return df


def build_subset_metadata(metadata: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    masks = {
        "last_month_only": (
            metadata["time_group"].eq("monthly_lag")
            & metadata["month_lag"].eq(1)
        ),
        "annual_only": metadata["time_group"].eq("annual"),
        "last100_only": metadata["time_group"].eq("last100"),
    }

    subsets: Dict[str, pd.DataFrame] = {}
    for subset_name, mask in masks.items():
        sub = metadata.loc[mask].copy()
        sub = sub.sort_values("feature_idx").reset_index(drop=True)
        sub.insert(0, "subset_feature_idx", np.arange(len(sub), dtype=int))

        expected_dim = EXPECTED_SUBSET_DIMS[subset_name]
        if len(sub) != expected_dim:
            raise ValueError(
                f"{subset_name}: expected {expected_dim} features, got {len(sub)}"
            )

        subsets[subset_name] = sub

    # The three groups must be mutually exclusive.
    all_indices: List[int] = []
    for sub in subsets.values():
        all_indices.extend(sub["feature_idx"].astype(int).tolist())
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("Feature subsets overlap unexpectedly.")

    return subsets


def validate_payload(payload: dict, source_path: Path) -> np.ndarray:
    if not isinstance(payload, dict):
        raise TypeError(f"Dataset payload must be a dict: {source_path}")
    if "X" not in payload:
        raise KeyError(f"Dataset payload has no 'X': {source_path}")

    X = np.asarray(payload["X"])
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}: {source_path}")
    if X.shape[1] != EXPECTED_INPUT_DIM:
        raise ValueError(
            f"Expected X with {EXPECTED_INPUT_DIM} columns, got {X.shape[1]}: "
            f"{source_path}"
        )

    for key in ("y_class", "y_m5"):
        if key in payload and len(payload[key]) != X.shape[0]:
            raise ValueError(
                f"Length mismatch for {key}: {len(payload[key])} vs {X.shape[0]}"
            )

    if "meta" in payload and len(payload["meta"]) != X.shape[0]:
        raise ValueError(
            f"Length mismatch for meta: {len(payload['meta'])} vs {X.shape[0]}"
        )

    return X


def copy_optional_sidecar(source_dir: Path, out_dir: Path, horizon: int) -> None:
    """Copy sample metadata/summary files once for convenient inspection."""
    candidates = [
        source_dir / f"china_raw282_dataset_horizon_{horizon}m_meta.csv",
        source_dir / f"summary_{horizon}m.csv",
    ]
    for source in candidates:
        if source.is_file():
            target = out_dir / source.name
            shutil.copy2(source, target)


def build_one_subset_payload(
    source_payload: dict,
    X_full: np.ndarray,
    subset_name: str,
    subset_meta: pd.DataFrame,
    source_path: Path,
    horizon: int,
) -> dict:
    original_indices = subset_meta["feature_idx"].to_numpy(dtype=int)
    feature_names = subset_meta["feature_name"].astype(str).tolist()

    # Advanced indexing creates an independent compact array.
    X_subset = np.ascontiguousarray(X_full[:, original_indices], dtype=np.float32)

    if X_subset.shape[1] != EXPECTED_SUBSET_DIMS[subset_name]:
        raise RuntimeError(
            f"Internal dimension error for {subset_name}: {X_subset.shape}"
        )

    payload = dict(source_payload)
    payload["X"] = X_subset
    payload["feature_dim"] = int(X_subset.shape[1])
    payload["feature_subset"] = subset_name
    payload["feature_names"] = feature_names
    payload["feature_indices_original_282"] = original_indices.tolist()
    payload["source_full_dataset"] = str(source_path)
    payload["source_full_feature_dim"] = EXPECTED_INPUT_DIM
    payload["horizon_month"] = int(horizon)
    payload["subset_definition"] = {
        "last_month_only": (
            "All raw-282 features with time_group='monthly_lag' and "
            "month_lag=1; lag1=[t0-1 month,t0)."
        ),
        "annual_only": (
            "All raw-282 features with time_group='annual'; annual window "
            "inherits the source definition [t0-365 days,t0)."
        ),
        "last100_only": (
            "All raw-282 features with time_group='last100'; based on the "
            "latest up to 100 events before t0."
        ),
    }[subset_name]
    payload["comparison_design"] = (
        "Restricted-feature sufficiency experiment. Labels, sample order, "
        "metadata, train/test split and fold annotations are copied unchanged "
        "from the full raw-282 dataset."
    )

    return payload


def save_pickle_atomic(payload: dict, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slice raw-282 datasets into last-month, annual and last100 subsets."
    )
    parser.add_argument(
        "--source-dir",
        default="data/raw282",
        help="Directory containing full raw-282 pkl files and feature metadata.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/representation_only",
        help="Output root for the three subset families.",
    )
    parser.add_argument(
        "--horizons",
        default="1,3,6,12",
        help="Comma-separated forecast horizons, e.g. 1,3,6,12.",
    )
    parser.add_argument(
        "--metadata-file",
        default="feature_metadata_282_relative_lag.csv",
        help="Feature metadata filename inside source-dir.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip a missing horizon pkl instead of stopping.",
    )
    args = parser.parse_args()

    source_dir = expand_path(args.source_dir)
    out_root = expand_path(args.out_dir)
    horizons = parse_horizons(args.horizons)

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory not found: {source_dir}")

    out_root.mkdir(parents=True, exist_ok=True)

    metadata_path = source_dir / args.metadata_file
    metadata = load_feature_metadata(metadata_path)
    subset_metadata = build_subset_metadata(metadata)

    print("=" * 100, flush=True)
    print("Build restricted-feature datasets from existing raw-282 files", flush=True)
    print(f"source_dir: {source_dir}", flush=True)
    print(f"out_dir:    {out_root}", flush=True)
    print(f"horizons:   {horizons}", flush=True)
    print("=" * 100, flush=True)

    for subset_name, sub_meta in subset_metadata.items():
        subset_dir = out_root / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)
        sub_meta.to_csv(
            subset_dir / f"feature_metadata_{subset_name}.csv", index=False
        )

        print(
            f"{subset_name}: {len(sub_meta)} features; original indices = "
            f"{sub_meta['feature_idx'].astype(int).tolist()}",
            flush=True,
        )

    generated = []
    skipped = []

    for horizon in horizons:
        source_path = source_dir / f"china_raw282_dataset_horizon_{horizon}m.pkl"
        if not source_path.is_file():
            message = f"Missing source dataset: {source_path}"
            if args.skip_missing:
                print(f"WARNING: {message}; skipped.", flush=True)
                skipped.append(str(source_path))
                continue
            raise FileNotFoundError(message)

        print("\n" + "-" * 100, flush=True)
        print(f"Loading H{horizon}: {source_path}", flush=True)
        with source_path.open("rb") as handle:
            source_payload = pickle.load(handle)

        X_full = validate_payload(source_payload, source_path)
        print(f"Full X shape: {X_full.shape}, dtype={X_full.dtype}", flush=True)

        # Copy sidecars once to the common root. They are identical for all subsets.
        horizon_common_dir = out_root / "common_sample_metadata"
        horizon_common_dir.mkdir(parents=True, exist_ok=True)
        copy_optional_sidecar(source_dir, horizon_common_dir, horizon)

        for subset_name, sub_meta in subset_metadata.items():
            subset_dir = out_root / subset_name
            output_name = f"china_{subset_name}_dataset_horizon_{horizon}m.pkl"
            output_path = subset_dir / output_name

            payload = build_one_subset_payload(
                source_payload=source_payload,
                X_full=X_full,
                subset_name=subset_name,
                subset_meta=sub_meta,
                source_path=source_path,
                horizon=horizon,
            )
            save_pickle_atomic(payload, output_path)

            # Read-back verification catches incomplete or corrupted output early.
            with output_path.open("rb") as handle:
                check = pickle.load(handle)
            X_check = np.asarray(check["X"])
            expected_shape = (X_full.shape[0], EXPECTED_SUBSET_DIMS[subset_name])
            if X_check.shape != expected_shape:
                raise RuntimeError(
                    f"Read-back shape mismatch for {output_path}: "
                    f"{X_check.shape} vs {expected_shape}"
                )
            if not np.array_equal(
                X_check,
                X_full[:, sub_meta["feature_idx"].to_numpy(dtype=int)].astype(np.float32),
                equal_nan=True,
            ):
                raise RuntimeError(f"Read-back value mismatch: {output_path}")

            record = {
                "subset": subset_name,
                "horizon_month": int(horizon),
                "n_samples": int(X_check.shape[0]),
                "feature_dim": int(X_check.shape[1]),
                "output_path": str(output_path),
                "source_path": str(source_path),
            }
            generated.append(record)
            print(
                f"Saved {subset_name}, H{horizon}: {output_path} "
                f"shape={X_check.shape}",
                flush=True,
            )

        # Release the full matrix before loading the next horizon.
        del X_full, source_payload

    manifest = {
        "source_dir": str(source_dir),
        "out_dir": str(out_root),
        "requested_horizons": horizons,
        "generated": generated,
        "skipped": skipped,
        "subset_dimensions": EXPECTED_SUBSET_DIMS,
        "selection_rules": {
            "last_month_only": "time_group == monthly_lag AND month_lag == 1",
            "annual_only": "time_group == annual",
            "last100_only": "time_group == last100",
        },
        "important_note": (
            "This script only slices X. All labels and split metadata are preserved "
            "from the full raw-282 payload."
        ),
    }
    manifest_path = out_root / "timescale_only_dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    generated_df = pd.DataFrame(generated)
    generated_df.to_csv(out_root / "timescale_only_dataset_inventory.csv", index=False)

    print("\n" + "=" * 100, flush=True)
    print(f"Generated {len(generated)} dataset files.", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    if skipped:
        print(f"Skipped {len(skipped)} missing source files.", flush=True)
    print("All done.", flush=True)


if __name__ == "__main__":
    main()
