#!/bin/bash
#SBATCH --job-name=12_eval_repr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/12_%x_%j.out
#SBATCH --error=logs/12_%x_%j.err

set -euo pipefail

# REQUIRED: set PROJECT_ROOT to the root of your cloned repository.
# Example (replace the placeholder with your own path before submitting):
#   export PROJECT_ROOT=/path/to/Multiscale-Structure-Forecast-Relevant-Information
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the root of your cloned repository}"
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/representation_only}
FULL_RESULT_DIR=${FULL_RESULT_DIR:-${PROJECT_ROOT}/outputs/discovery_models}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/outputs/representation_only}

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${PYTHON}" -u scripts/12_evaluate_representation_only.py \
  --data-root "${DATA_ROOT}" \
  --full-result-dir "${FULL_RESULT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --subsets last_month_only,annual_only,last100_only \
  --horizons 1,3,6,12 \
  --models catboost,extratrees,xgboost \
  --sample-weight-mode inherit \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --seed 42
