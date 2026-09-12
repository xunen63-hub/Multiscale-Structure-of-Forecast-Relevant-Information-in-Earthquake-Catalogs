#!/bin/bash
#SBATCH --job-name=13_annual_t3t5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=logs/13_%x_%j.out
#SBATCH --error=logs/13_%x_%j.err

set -euo pipefail

# REQUIRED: set PROJECT_ROOT to the root of your cloned repository.
# Example (replace the placeholder with your own path before submitting):
#   export PROJECT_ROOT=/path/to/Multiscale-Structure-Forecast-Relevant-Information
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the root of your cloned repository}"
PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-${PROJECT_ROOT}/data/raw282}
FROZEN_GROUPS=${FROZEN_GROUPS:-${PROJECT_ROOT}/outputs/inner_validation_shap/selected_groups_inner_validation.csv}
COMPARISON_DIR=${COMPARISON_DIR:-${PROJECT_ROOT}/outputs/discovery_models}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/outputs/annual_top3_vs_top5}

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${PYTHON}" -u scripts/13_annual_top3_vs_top5.py \
  --data-dir "${DATA_DIR}" \
  --frozen-groups-csv "${FROZEN_GROUPS}" \
  --comparison-dir "${COMPARISON_DIR}" \
  --out-dir "${OUT_DIR}" \
  --horizons 1,3,6,12 \
  --models extratrees,catboost,xgboost \
  --folds 1,2,3 \
  --frozen-top-k 3 \
  --annual-group annual/magnitude_extreme \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --seed 42
