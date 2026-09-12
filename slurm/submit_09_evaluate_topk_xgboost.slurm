#!/bin/bash
#SBATCH --job-name=09_eval_topk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=36:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/09_%x_%A_%a.out
#SBATCH --error=logs/09_%x_%A_%a.err

set -euo pipefail

# REQUIRED: set PROJECT_ROOT to the root of your cloned repository.
# Example (replace the placeholder with your own path before submitting):
#   export PROJECT_ROOT=/path/to/Multiscale-Structure-Forecast-Relevant-Information
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the root of your cloned repository}"
PYTHON=${PYTHON:-python}
TOPK_DATA_ROOT=${TOPK_DATA_ROOT:-${PROJECT_ROOT}/data/topk}
OUT_ROOT=${OUT_ROOT:-${PROJECT_ROOT}/outputs/topk_xgboost}
FULL_BASELINE_METRICS=${FULL_BASELINE_METRICS:-${PROJECT_ROOT}/outputs/discovery_models/final_test_metrics_catboost_extratrees_xgb_binary_m5.csv}
K_VALUES=(25 50 75 100)
K=${K_VALUES[${SLURM_ARRAY_TASK_ID}]}
VARIANT=top${K}_innerfrozen_consensus

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUT_ROOT}/${VARIANT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${PYTHON}" -u scripts/09_evaluate_topk_xgboost.py \
  --data-dir "${TOPK_DATA_ROOT}/${VARIANT}" \
  --out-dir "${OUT_ROOT}/${VARIANT}" \
  --variant-name "${VARIANT}" \
  --horizons 1,3,6,12 \
  --folds 1,2,3 \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --seed 42 \
  --full-baseline-metrics "${FULL_BASELINE_METRICS}"
