#!/bin/bash
#SBATCH --job-name=15_magnitude_sens
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/15_%x_%j.out
#SBATCH --error=logs/15_%x_%j.err

set -euo pipefail

# REQUIRED: set PROJECT_ROOT to the root of your cloned repository.
# Example (replace the placeholder with your own path before submitting):
#   export PROJECT_ROOT=/path/to/Multiscale-Structure-Forecast-Relevant-Information
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the root of your cloned repository}"
PYTHON=${PYTHON:-python}
REFERENCE_DIR=${REFERENCE_DIR:-${PROJECT_ROOT}/outputs/inner_validation_shap}
SENSITIVITY_DIR=${SENSITIVITY_DIR:-${PROJECT_ROOT}/outputs/inner_validation_shap_mge3}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/outputs/magnitude_threshold_sensitivity}

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUT_DIR}"

"${PYTHON}" -u scripts/15_magnitude_threshold_sensitivity.py \
  --reference-dir "${REFERENCE_DIR}" \
  --sensitivity-dir "${SENSITIVITY_DIR}" \
  --out-dir "${OUT_DIR}" \
  --horizons 1,3,6,12 \
  --top-k 3 \
  --expected-groups 11 \
  --expected-models 3 \
  --expected-folds 3
