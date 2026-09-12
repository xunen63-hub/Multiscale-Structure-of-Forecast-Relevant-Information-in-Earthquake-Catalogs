#!/bin/bash
#SBATCH --job-name=04_shap9095
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=logs/04_%x_%j.out
#SBATCH --error=logs/04_%x_%j.err

set -euo pipefail

# REQUIRED: set PROJECT_ROOT to the root of your cloned repository.
# Example (replace the placeholder with your own path before submitting):
#   export PROJECT_ROOT=/path/to/Multiscale-Structure-Forecast-Relevant-Information
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the root of your cloned repository}"
PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-${PROJECT_ROOT}/data/raw282}
COMPARISON_DIR=${COMPARISON_DIR:-${PROJECT_ROOT}/outputs/discovery_models}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/outputs/shap_outer_test_9095}

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${PYTHON}" -u scripts/04_outer_test_descriptive_shap.py \
  --data-dir "${DATA_DIR}" \
  --comparison-dir "${COMPARISON_DIR}" \
  --out-dir "${OUT_DIR}" \
  --horizons 1,3,6,12 \
  --models extratrees,catboost,xgboost \
  --max-samples-per-subset 9095 \
  --seed 42
