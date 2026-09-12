# Multiscale Structure of Forecast-Relevant Information in Earthquake Catalogs

Code associated with the manuscript **“Multiscale Structure of Forecast-Relevant Information in Earthquake Catalogs”** and its Supporting Information.

## Repository structure

```text
Multiscale-Structure-Forecast-Relevant-Information/
├── README.md
├── requirements.txt
├── LICENSE
├── scripts/    # analysis workflow
├── slurm/      # matching SLURM submission scripts
└── figures/    # scripts and source data for reproducing manuscript figures
```

## Data

The seismic catalogs for China provided by the **China Earthquake Networks Center (CENC)** are available at:

https://codeocean.com/capsule/7316858/tree

The repository itself does not redistribute the earthquake catalog.

## Analysis scripts

| Step | Script | Purpose |
| ---: | --- | --- |
| 01 | `01_build_raw282_dataset.py` | Build the 282-predictor datasets, temporal split, and rolling inner folds. |
| 02 | `02_train_discovery_models.py` | Select and fit the CatBoost, ExtraTrees, and XGBoost discovery models. |
| 03 | `03_train_heldout_models.py` | Select and fit the Random Forest and LightGBM held-out models. |
| 04 | `04_outer_test_descriptive_shap.py` | Compute descriptive SHAP values on the outer test set. |
| 05 | `05_inner_validation_shap_discovery.py` | Identify forecast-relevant feature groups using inner-validation SHAP only. |
| 06 | `06_feature_group_ablation.py` | Evaluate frozen feature-group retention and removal in the discovery models. |
| 07 | `07_validate_frozen_groups_heldout.py` | Validate the frozen feature groups in Random Forest and LightGBM without reranking. |
| 08 | `08_build_frozen_topk.py` | Freeze the individual-feature consensus ranking and construct Top-K datasets. |
| 09 | `09_evaluate_topk_xgboost.py` | Evaluate Top-K feature subsets (`K = 25, 50, 75, 100`) with fixed XGBoost settings. |
| 10 | `10_summarize_topk.py` | Summarize Top-K performance relative to the matched full-feature model. |
| 11 | `11_build_representation_only_datasets.py` | Construct Monthly, Annual, and Latest-100 representation-only datasets. |
| 12 | `12_evaluate_representation_only.py` | Evaluate the three prespecified single-representation models. |
| 13 | `13_annual_top3_vs_top5.py` | Perform the matched Annual Top-3 versus Top-5 analysis. |
| 14 | `14_build_catalog_sensitivity.py` | Rebuild predictors under catalog-magnitude or focal-depth filtering while retaining the original targets. |
| 15 | `15_magnitude_threshold_sensitivity.py` | Compare inner-validation SHAP structure after raising the feature-catalog magnitude threshold to `M >= 3`. |

Matching submission scripts are provided in `slurm/`.

## Figure reproduction

The `figures/` directory contains the scripts and processed source data used to reproduce the figures in the manuscript and Supporting Information.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## License

The code is released under the MIT License. See `LICENSE`.
