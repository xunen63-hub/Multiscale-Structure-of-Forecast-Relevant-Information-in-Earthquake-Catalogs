#!/bin/bash
#SBATCH --job-name=03_heldout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=logs/03_%x_%j.out
#SBATCH --error=logs/03_%x_%j.err

set -euo pipefail

# REQUIRED: set PROJECT_ROOT to the root of your cloned repository.
# Example (replace the placeholder with your own path before submitting):
#   export PROJECT_ROOT=/path/to/Multiscale-Structure-Forecast-Relevant-Information
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the root of your cloned repository}"
PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-${PROJECT_ROOT}/data/raw282}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/outputs/heldout_models}

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${PYTHON}" -u scripts/03_train_heldout_models.py \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --horizons 1,3,6,12 \
  --folds 1,2,3 \
  --models rf,lightgbm \
  --class-weight-mode none \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --seed 42
