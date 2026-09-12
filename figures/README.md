# Figure reproduction

`generate_9095sample_figures.py` reproduces the main-text 9,095-sample versions
of Figures 2, 3, and 4, including the audit variant of Figure 4 with every
model-level point shown.

Run from any directory:

```bash
python /path/to/repository/figures/generate_9095sample_figures.py
```

Outputs are written to `figures/generated/`; intermediate audit tables are
written to `figures/derived_tables/`.

## Bundled inputs

| Path | Role |
| --- | --- |
| `source_data/logs/discovery_models_final_metrics.out` | Final CatBoost, ExtraTrees, and XGBoost metrics parsed for Figure 2. |
| `source_data/logs/heldout_models_final_metrics.out` | Final RF and LightGBM metrics parsed for Figure 2. |
| `source_data/shap_outer_test_9095/all_shap_feature_importance.csv` | Feature-level descriptive SHAP summaries; every `all_test` task has `n_subset_sampled = 9095`. |
| `source_data/shap_outer_test_9095/all_shap_group_importance.csv` | Group-level descriptive SHAP summaries from the same 9,095-sample run. |
| `source_data/feature_group_ablation/ablation_deltas_vs_full.csv` | Discovery-architecture frozen-group retraining results. |
| `source_data/heldout_group_validation/heldout_deltas_vs_full.csv` | Held-out-architecture frozen-group results. |
| `source_data/comparison_timescale_only_vs_full.csv` | Single-representation comparisons used in Figure 4. |

The plotting module keeps the original publication functions but its public
entry point calls only the fully bundled Figure 2--4 workflow. Figure styling
uses a consistent semantic hierarchy across panels; dense heat-map cells are
the documented smaller-annotation exception.

The files in `published/` are the reference exports used for the manuscript.
They are not overwritten when the reproduction command is run.
