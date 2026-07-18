# Thesis Result Data Index

This folder contains copied source artifacts used for preparing the Results, Discussion, and Conclusion chapters. These files are copies only; the working platform files remain in their original locations.

## Offline Data Pipeline

Folder: `offline_pipeline/`

- `pipeline_55776ebd244e_registry.json`  
  Registry pointer for the offline data pipeline `pipeline_55776ebd244e`.

- `pipeline_55776ebd244e_state.json`  
  Full offline pipeline state copied from `E:\Thesis\PipelineProjects\Final_offline_data_June_27\pipeline_state.json`.  
  Use this for offline run counts, project list, pipeline configuration, hard-cap count, and runtime summary.

Key values used in the report:

- 46 model-development projects
- 736 official total runs
- 699 normally completed runs
- 37 hard-cap runs
- 0 failed runs
- 15 planned exploratory runs per project
- 5,000 training iterations per run
- 1,000 evaluation interval
- 100 log interval
- 4,000 densification-until iteration
- 6,000,000 Gaussian hard cap
- 1,400 px image max size

## Test Pipeline

Folder: `test_pipeline/`

- `pipeline_5df0ad97ef5a_registry.json`  
  Registry pointer for the final test pipeline `pipeline_5df0ad97ef5a`.

- `pipeline_5df0ad97ef5a_state.json`  
  Full test pipeline state copied from `E:\Thesis\PipelineTests\Final_test_June_27\pipeline_state.json`.  
  Use this for test project list, tested model IDs, candidate grid settings, test run counts, prediction previews, and final test-pipeline status.

Key values used in the report:

- 12 final test projects
- 6 tested models
- 12 baseline runs
- 72 model-selected test runs
- 84 total test runs
- 0 failed runs
- 0 hard-cap runs
- 30 test candidate points

## Training Data

Folder: `training_data/`

- `training_data_manifest.json`  
  Manifest for `training_data_20260710_183008_final_offline_data_june_27-training-data`.

- `training_data_rows.json`  
  Final training rows used by the clean final compact Ridge and MLP models.

Key values used in the report:

- 653 usable exploratory training rows
- 46 represented projects
- score reference step: 5,000
- hard-cap penalty rows not included
- source rows before excluding baseline rows: 699

Folder: `training_data_with_hard_cap_penalty/`

- `training_data_manifest.json`  
  Manifest for the same Training Data artifact after rebuilding with hard-cap penalty rows included.

- `training_data_rows.json`  
  Training rows after including hard-cap penalty rows.

Key values for the hard-cap-penalty rebuild:

- 690 total rows
- 653 normally completed exploratory rows
- 37 hard-cap penalty rows
- hard-cap penalty value: -0.5373449585805301
- Gaussian hard cap recorded in the data: 6,000,000

Note on hard-cap counting:

- The offline pipeline summary reports 37 hard-cap runs. This is the final pipeline-slot count shown in the interface.
- The stored run list contains 44 hard-cap attempts because some hard-cap slots were retried and previous attempts remained in the run history.
- The hard-cap-penalty Training Data rebuild now follows the pipeline summary rule: previous retry attempts are kept in raw history but are not duplicated as Training Data rows.

## Final Models

Folder: `models/final_compact_ridge/`

- `model.json`  
  Registry manifest for the final compact Ridge model.

- `compact_featurewise_ridge_model.json`  
  Saved Ridge model artifact.

- `compact_featurewise_ridge_metadata.json`  
  Additional metadata for the Ridge model.

Folder: `models/final_compact_mlp/`

- `model.json`  
  Registry manifest for the final compact MLP model.

- `compact_featurewise_20260712_151631.pt`  
  Saved MLP checkpoint.

- `compact_featurewise_20260712_151631_metadata.json`  
  Additional metadata for the MLP model.

## Hard-Cap Comparison Models

Folder: `models/hard_cap_compact_ridge/`

- `model.json`
- `compact_featurewise_ridge_model.json`
- `compact_featurewise_ridge_metadata.json`

Folder: `models/hard_cap_compact_mlp/`

- `model.json`
- `compact_featurewise_20260713_115539.pt`
- `compact_featurewise_20260713_115539_metadata.json`

These hard-cap model artifacts support the short comparison discussion. They were not used as final models because their final test results were weaker than the clean final compact Ridge model.

## Analysis Outputs

Folder: `analysis_outputs/`

- `thesis_results_analysis_data.csv`  
  Derived per-project test-result table. It contains baseline-vs-model differences for PSNR, SSIM, LPIPS, runtime, Gaussian count, and selected multiplier fields.

This CSV was generated from the copied/saved pipeline and run analytics files and is useful for preparing final charts and result tables.

- `final_compact_ridge_coefficient_summary.csv`  
  Ranked coefficient table reconstructed from the saved final compact Ridge model. It is useful for discussing which standardized action and descriptor-interaction terms had the strongest influence in the Ridge scoring model.

- `final_compact_ridge_test_project_descriptors.csv`  
  Test-project descriptor table joined with the final compact Ridge PSNR, SSIM, and LPIPS improvements. It is useful for cautious project-level discussion, such as comparing high-improvement and low-improvement projects against their descriptor values.
