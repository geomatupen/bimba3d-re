from bimba3d_backend.worker.ai_input_modes.feature_schema import (
    FEATURE_SCHEMA_NAME,
    GROUP_KEYS,
    MULTIPLIER_SCHEMA_NAME,
)
from bimba3d_backend.worker.ai_input_modes.featurewise_ridge_regression import train_featurewise_ridge_model


def test_feature_schema_names_are_report_aligned():
    assert FEATURE_SCHEMA_NAME == "mode3_exif_flight_scene_v1"
    assert MULTIPLIER_SCHEMA_NAME == "geometry_appearance_densification_v1"
    assert GROUP_KEYS == ["geometry_lr_mult", "appearance_lr_mult", "densification_mult"]


def test_featurewise_ridge_wrapper_trains_candidate():
    rows = [
        _row("project_1", 1.1, 0.10),
        _row("project_2", 0.9, -0.05),
    ]

    model, metrics, theta_norms = train_featurewise_ridge_model(
        rows=rows,
        score_key="relative_quality_score",
        lambda_ridge=2.0,
        candidate_points=7,
    )

    assert model["model_family"] == "featurewise_ridge_regression"
    assert model["runs"] == 2
    assert "avg_val_mse" in metrics
    assert sorted(theta_norms) == sorted(GROUP_KEYS)


def _row(project_name: str, multiplier: float, score: float) -> dict:
    return {
        "project_name": project_name,
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
        "relative_quality_score": score,
    }

