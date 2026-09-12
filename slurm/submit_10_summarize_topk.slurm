#!/bin/bash
#SBATCH --job-name=10_sum_topk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/10_%x_%j.out
#SBATCH --error=logs/10_%x_%j.err

set -euo pipefail

# REQUIRED: set PROJECT_ROOT to the root of your cloned repository.
# Example (replace the placeholder with your own path before submitting):
#   export PROJECT_ROOT=/path/to/Multiscale-Structure-Forecast-Relevant-Information
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the root of your cloned repository}"
PYTHON=${PYTHON:-python}
BASE_OUT=${BASE_OUT:-${PROJECT_ROOT}/outputs/topk_xgboost}
OUT_CSV=${OUT_CSV:-${BASE_OUT}/summary_topk_binary_m5_vs_full.csv}

cd "${PROJECT_ROOT}"
mkdir -p logs "${BASE_OUT}"

"${PYTHON}" -u scripts/10_summarize_topk.py \
  --base-out "${BASE_OUT}" \
  --topk-list 25,50,75,100 \
  --variant-pattern 'top{k}_innerfrozen_consensus' \
  --out-csv "${OUT_CSV}"
