#!/bin/bash
#SBATCH --job-name=07_heldout_groups
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=36:00:00
#SBATCH --output=logs/07_%x_%j.out
#SBATCH --error=logs/07_%x_%j.err

set -euo pipefail

# REQUIRED: set PROJECT_ROOT to the root of your cloned repository.
# Example (replace the placeholder with your own path before submitting):
#   export PROJECT_ROOT=/path/to/Multiscale-Structure-Forecast-Relevant-Information
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the root of your cloned repository}"
PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-${PROJECT_ROOT}/data/raw282}
SELECTED_GROUPS=${SELECTED_GROUPS:-${PROJECT_ROOT}/outputs/inner_validation_shap/selected_groups_inner_validation.csv}
HELDOUT_CONFIG=${HELDOUT_CONFIG:-${PROJECT_ROOT}/outputs/heldout_models/selected_config_by_model.json}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/outputs/heldout_group_validation}

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${PYTHON}" -u scripts/07_validate_frozen_groups_heldout.py \
  --data-dir "${DATA_DIR}" \
  --selected-groups-csv "${SELECTED_GROUPS}" \
  --heldout-config-json "${HELDOUT_CONFIG}" \
  --out-dir "${OUT_DIR}" \
  --horizons 1,3,6,12 \
  --models rf,lightgbm \
  --top-k-groups 1,2,3 \
  --evaluation-mode auc_only \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --seed 42
