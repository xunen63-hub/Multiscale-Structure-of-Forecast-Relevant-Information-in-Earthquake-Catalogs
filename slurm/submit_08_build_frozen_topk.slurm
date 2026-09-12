#!/bin/bash
#SBATCH --job-name=08_build_topk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/08_%x_%j.out
#SBATCH --error=logs/08_%x_%j.err

set -euo pipefail

# REQUIRED: set PROJECT_ROOT to the root of your cloned repository.
# Example (replace the placeholder with your own path before submitting):
#   export PROJECT_ROOT=/path/to/Multiscale-Structure-Forecast-Relevant-Information
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the root of your cloned repository}"
PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-${PROJECT_ROOT}/data/raw282}
IMPORTANCE_CSV=${IMPORTANCE_CSV:-${PROJECT_ROOT}/outputs/inner_validation_shap/all_inner_valid_feature_importance.csv}
OUT_ROOT=${OUT_ROOT:-${PROJECT_ROOT}/data/topk}
AUDIT_DIR=${AUDIT_DIR:-${PROJECT_ROOT}/outputs/topk_ranking}

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUT_ROOT}" "${AUDIT_DIR}"

"${PYTHON}" -u scripts/08_build_frozen_topk.py \
  --importance-csv "${IMPORTANCE_CSV}" \
  --base-data-dir "${DATA_DIR}" \
  --out-root "${OUT_ROOT}" \
  --audit-out-dir "${AUDIT_DIR}" \
  --models CatBoost,ExtraTrees,XGBoost \
  --horizons 1,3,6,12 \
  --folds 1,2,3 \
  --topk-list 25,50,75,100 \
  --expected-n-features 282
