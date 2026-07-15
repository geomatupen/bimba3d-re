# Featurewise Quality Scoring Notebooks

This folder contains two standalone notebooks for training and inspecting scoring models for Gaussian Splatting parameter selection.

## Notebooks

- `featurewise_ridge_quality_scoring_model.ipynb` trains the Featurewise Ridge quality scoring model.
- `featurewise_mlp_quality_scoring_model.ipynb` trains the Featurewise MLP quality scoring model.

Both notebooks use the same prepared training-data folder and can be run independently.

## Required Data

The bundled reference dataset is:

`notebooks/data/training_data_20260710_183008_final_offline_data_june_27-training-data/`

It contains:

- `manifest.json`: dataset summary, descriptor schema, multiplier bounds, base parameter values, and build metadata.
- `rows.json`: prepared training rows used by the notebooks.

The same reference data can also be viewed on GitHub:

[reference training-data folder](https://github.com/geomatupen/bimba3d_re/tree/main/notebooks/data/training_data_20260710_183008_final_offline_data_june_27-training-data)

## How To Run

1. Open one of the two notebooks.
2. Run the first setup cell to install required Python packages.
3. Run the cells from top to bottom.
4. Outputs are written under `notebooks/_outputs/`.

To use a different prepared dataset, copy its folder into `notebooks/data/` and update `TRAINING_DATA_DIR` in the notebook configuration cell.

## Notes

The notebooks train scoring models and save preview predictions/charts. Preview predictions are for review; real reconstruction testing should be done by uploading the trained model into Bimba3D-re and running a testing pipeline.
