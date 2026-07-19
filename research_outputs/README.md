# Final Data Snapshot - 2026-07-19

This folder is a versioned copy of the final thesis data artifacts. It keeps the latest final model artifacts and test results separate from the older root-level `data/` snapshot.

## What This Snapshot Contains

- `models/`
  - `final_compact_ridge/`: compact Ridge model trained without hard-cap penalty rows.
  - `final_compact_mlp/`: compact MLP model trained without hard-cap penalty rows, using project-level validation split and seed `42`.
  - `hard_cap_compact_ridge/`: compact Ridge model trained with hard-cap penalty data.
  - `hard_cap_compact_mlp/`: compact MLP model trained with hard-cap penalty data, using project-level validation split and seed `42`.

- `test_pipeline/`
  - Current copied state for test pipeline `pipeline_5df0ad97ef5a`.
  - This state contains 84 runs: 12 baseline runs and 72 model-selected test runs.
  - The copied test state includes 6 tested models total, including older featurewise comparison models and the four final compact models.

- `training_data/`
  - Existing clean Training Data snapshot with 653 rows.
  - This was copied unchanged.

- `training_data_with_hard_cap_penalty/`
  - Existing hard-cap-penalty Training Data snapshot with 690 rows.
  - This was copied unchanged.

- `offline_pipeline/`
  - Existing copied state and registry for offline data pipeline `pipeline_55776ebd244e`.
  - This was copied unchanged.

- `analysis_outputs/`
  - `thesis_results_analysis_data.csv`: regenerated from the current test-pipeline run folders for this snapshot.
  - Ridge coefficient and descriptor analysis files copied from the previous data snapshot because the final Ridge models did not change.

## Final Compact Model IDs

- Clean compact Ridge: `model_compact-ridge_20260712_131550_final_offline_data_june_27-training-data-compact-ridge`
- Clean compact MLP: `model_compact-mlp_20260718_194410_compact_mlp_july18_final_offline_data_june_27-training-data-compact-featurewise-mlp`
- Hard-cap compact Ridge: `model_compact-ridge_20260713_095334_final_compact_ridge_with_hard_cap_data_july13`
- Hard-cap compact MLP: `model_compact-mlp_20260718_194519_compact_mlp_july18_with_hardcap_data_final_offline_data_june_27-training-data`

## Notes

- The MLP models in this snapshot are the newly trained versions that use project-level validation rows.
- The Training Data files were not rebuilt or changed for this snapshot.
- `thesis_results_analysis_data.csv` uses PSNR and SSIM improvement as `model - baseline`, and LPIPS improvement as `baseline - model` because lower LPIPS is better.
