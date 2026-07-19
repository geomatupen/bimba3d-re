This folder stores prepared data artifacts used by the standalone notebooks.

Each training-data folder should contain:

- `manifest.json`: dataset summary, descriptor schema, multiplier bounds, base parameter values, and build metadata.
- `rows.json`: prepared training rows used by the Ridge and MLP notebooks.

The file `test_project_descriptors_final_compact_models.csv` contains descriptors for the test projects used by the prediction-preview and candidate-curve charts. The notebooks train from the training-data folder, then use this CSV only to preview how the trained scoring model behaves on test projects. The `project_index` column follows the same stable project-name order used for the final metric charts.

The notebooks are configured to read from this folder so they can run without depending on the application data directory.
