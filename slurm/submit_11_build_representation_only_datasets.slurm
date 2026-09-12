#!/bin/bash
#SBATCH --job-name=11_build_repr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/11_%x_%j.out
#SBATCH --error=logs/11_%x_%j.err

set -euo pipefail

# REQUIRED: set PROJECT_ROOT to the root of your cloned repository.
# Example (replace the placeholder with your own path before submitting):
#   export PROJECT_ROOT=/path/to/Multiscale-Structure-Forecast-Relevant-Information
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the root of your cloned repository}"
PYTHON=${PYTHON:-python}
SOURCE_DIR=${SOURCE_DIR:-${PROJECT_ROOT}/data/raw282}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/data/representation_only}

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUT_DIR}"

"${PYTHON}" -u scripts/11_build_representation_only_datasets.py \
  --source-dir "${SOURCE_DIR}" \
  --out-dir "${OUT_DIR}" \
  --horizons 1,3,6,12
