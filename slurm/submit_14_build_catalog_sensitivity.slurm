#!/bin/bash
#SBATCH --job-name=14_build_sensitivity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/14_%x_%j.out
#SBATCH --error=logs/14_%x_%j.err

set -euo pipefail

# REQUIRED: set PROJECT_ROOT to the root of your cloned repository.
# Example (replace the placeholder with your own path before submitting):
#   export PROJECT_ROOT=/path/to/Multiscale-Structure-Forecast-Relevant-Information
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the root of your cloned repository}"
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data}
CATALOG_PATH=${CATALOG_PATH:-${DATA_ROOT}/catalog.csv}
PATCH_GRID_PATH=${PATCH_GRID_PATH:-${DATA_ROOT}/patch_grid.csv}
FEATURE_MIN_MAGNITUDE=${FEATURE_MIN_MAGNITUDE-3}
FEATURE_DEPTH_MIN_KM=${FEATURE_DEPTH_MIN_KM-}
FEATURE_DEPTH_MAX_KM=${FEATURE_DEPTH_MAX_KM-}
OUT_DIR=${OUT_DIR:-${DATA_ROOT}/raw282_mge3}
FILTER_ARGS=()

if [[ -n "${FEATURE_MIN_MAGNITUDE}" ]]; then
  FILTER_ARGS+=(--feature-min-magnitude "${FEATURE_MIN_MAGNITUDE}")
fi
if [[ -n "${FEATURE_DEPTH_MIN_KM}" ]]; then
  FILTER_ARGS+=(--feature-depth-min-km "${FEATURE_DEPTH_MIN_KM}")
fi
if [[ -n "${FEATURE_DEPTH_MAX_KM}" ]]; then
  FILTER_ARGS+=(--feature-depth-max-km "${FEATURE_DEPTH_MAX_KM}")
fi

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUT_DIR}"

"${PYTHON}" -u scripts/14_build_catalog_sensitivity.py \
  --catalog-path "${CATALOG_PATH}" \
  --patch-list-path "${PATCH_GRID_PATH}" \
  --out-dir "${OUT_DIR}" \
  --horizons 1,3,6,12 \
  --max-horizon-month 12 \
  --feature-window-days 365 \
  --train-label-end 2010-12-31 \
  --test-start 2012-01-01 \
  --anchor-day 1 \
  "${FILTER_ARGS[@]}"
