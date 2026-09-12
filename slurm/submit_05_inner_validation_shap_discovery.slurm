#!/bin/bash
#SBATCH --job-name=05_inner_shap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=logs/05_%x_%j.out
#SBATCH --error=logs/05_%x_%j.err

set -euo pipefail

# REQUIRED: set PROJECT_ROOT to the root of your cloned repository.
# Example (replace the placeholder with your own path before submitting):
#   export PROJECT_ROOT=/path/to/Multiscale-Structure-Forecast-Relevant-Information
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the root of your cloned repository}"
PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-${PROJECT_ROOT}/data/raw282}
COMPARISON_DIR=${COMPARISON_DIR:-${PROJECT_ROOT}/outputs/discovery_models}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/outputs/inner_validation_shap}

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${PYTHON}" -u scripts/05_inner_validation_shap_discovery.py \
  --data-dir "${DATA_DIR}" \
  --comparison-dir "${COMPARISON_DIR}" \
  --out-dir "${OUT_DIR}" \
  --horizons 1,3,6,12 \
  --models catboost,extratrees,xgboost \
  --folds 1,2,3 \
  --primary-group-col feature_group \
  --top-k-groups 3 \
  --max-valid-samples 1500 \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --seed 42
