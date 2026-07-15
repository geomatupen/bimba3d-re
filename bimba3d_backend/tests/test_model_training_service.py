from pathlib import Path

import pytest

from bimba3d_backend.app.services.model_training import ModelTrainingOptions, train_model_from_training_data
from bimba3d_backend.app.services.training_data_registry import create_manifest, replace_rows
from bimba3d_backend.app.services.workflow_model_registry import read_model
from bimba3d_backend.app.services.workflow_paths import WorkflowPaths


def test_train_featurewise_ridge_registers_shared_model(tmp_path: Path):
    paths = WorkflowPaths(data_root=tmp_path)
    manifest = create_manifest(
        name="Prepared Data",
        source_pipeline_id="pipeline_abc",
        feature_schema="mode3_exif_flight_scene_v1",
        training_data_id="training_data_target",
        paths=paths,
    )
    replace_rows(
        manifest.training_data_id,
        [
            _training_row("project_1", "run_1", 1.1, 0.10),
            _training_row("project_2", "run_2", 0.9, -0.05),
        ],
        paths=paths,
    )

    model = train_model_from_training_data(
        ModelTrainingOptions(
            model_family="featurewise_ridge_regression",
            model_name="Ridge Test Model",
            source_training_data_id=manifest.training_data_id,
            source_pipeline_id="pipeline_abc",
            lambda_ridge=2.0,
            candidate_points=7,
        ),
        paths=paths,
    )

    assert model.model_family == "featurewise_ridge_regression"
    assert model.source_training_data_id == manifest.training_data_id
    assert model.training_samples == 2
    assert Path(model.artifact_path).exists()
    assert read_model(model.model_id, paths=paths).model_name == "Ridge Test Model"


def test_train_model_requires_usable_training_rows(tmp_path: Path):
    paths = WorkflowPaths(data_root=tmp_path)
    manifest = create_manifest(
        name="Prepared Data",
        source_pipeline_id="pipeline_abc",
        feature_schema="mode3_exif_flight_scene_v1",
        training_data_id="training_data_target",
        paths=paths,
    )

    with pytest.raises(ValueError, match="No valid Training Data rows"):
        train_model_from_training_data(
            ModelTrainingOptions(
                model_family="featurewise_ridge_regression",
                model_name="Empty Model",
                source_training_data_id=manifest.training_data_id,
            ),
            paths=paths,
        )


def _training_row(project_id: str, run_id: str, multiplier: float, score: float) -> dict:
    return {
        "project_id": project_id,
        "project_name": project_id,
        "run_id": run_id,
        "source_pipeline_id": "pipeline_abc",
        "phase": 2,
        "x_features": {
            "focal_length_mm": 24.0,
            "iso": 200.0,
            "img_width_median": 4000.0,
            "img_height_median": 3000.0,
            "gsd_median": 0.05,
            "overlap_proxy": 0.5,
            "coverage_spread": 0.4,
            "camera_angle_bucket": 1,
            "heading_consistency": 0.8,
            "texture_density": 0.6,
            "blur_motion_risk": 0.2,
            "terrain_roughness_proxy": 0.3,
            "vegetation_cover": 0.4,
            "vegetation_complexity": 0.5,
        },
        "selected_multipliers": {
            "geometry_lr_mult": multiplier,
            "appearance_lr_mult": multiplier,
            "densification_mult": 1.0,
        },
        "selected_log_multipliers": {
            "geometry_lr_mult": 0.0,
            "appearance_lr_mult": 0.0,
            "densification_mult": 0.0,
        },
        "relative_quality_score": score,
    }

