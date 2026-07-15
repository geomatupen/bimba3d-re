from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bimba3d_backend.app.api import model_training as model_training_api
from bimba3d_backend.app.services.workflow_model_registry import register_model
from bimba3d_backend.app.services.training_data_registry import create_manifest, replace_rows
from bimba3d_backend.app.services.workflow_paths import WorkflowPaths


def make_client(tmp_path: Path) -> TestClient:
    model_training_api.WORKFLOW_PATHS = WorkflowPaths(data_root=tmp_path)
    app = FastAPI()
    app.include_router(model_training_api.router, prefix="/api/workflow/model-training")
    return TestClient(app)


def test_model_training_api_trains_ridge_from_training_data(tmp_path: Path):
    paths = WorkflowPaths(data_root=tmp_path)
    model_training_api.WORKFLOW_PATHS = paths
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
    client = make_client(tmp_path)

    res = client.post(
        "/api/workflow/model-training/train",
        json={
            "model_family": "featurewise_ridge_regression",
            "model_name": "Ridge API Model",
            "source_training_data_id": manifest.training_data_id,
            "source_pipeline_id": "pipeline_abc",
            "lambda_ridge": 2.0,
            "candidate_points": 7,
        },
    )

    assert res.status_code == 200
    payload = res.json()
    assert payload["model_family"] == "featurewise_ridge_regression"
    assert payload["source_training_data_id"] == manifest.training_data_id
    assert payload["training_samples"] == 2


def test_model_training_api_lists_usable_training_data_sources(tmp_path: Path):
    paths = WorkflowPaths(data_root=tmp_path)
    model_training_api.WORKFLOW_PATHS = paths
    empty = create_manifest(
        name="Empty Data",
        source_pipeline_id="pipeline_empty",
        feature_schema="mode3_exif_flight_scene_v1",
        training_data_id="training_data_empty",
        paths=paths,
    )
    ready = create_manifest(
        name="Ready Data",
        source_pipeline_id="pipeline_ready",
        feature_schema="mode3_exif_flight_scene_v1",
        training_data_id="training_data_ready",
        paths=paths,
    )
    replace_rows(
        ready.training_data_id,
        [
            _training_row("project_1", "run_1", 1.1, 0.10),
        ],
        paths=paths,
    )
    client = make_client(tmp_path)

    usable_res = client.get("/api/workflow/model-training/training-data-sources")
    assert usable_res.status_code == 200
    usable_payload = usable_res.json()
    assert usable_payload["total"] == 1
    assert usable_payload["items"][0]["training_data_id"] == ready.training_data_id

    all_res = client.get("/api/workflow/model-training/training-data-sources?usable_only=false")
    assert all_res.status_code == 200
    all_ids = {item["training_data_id"] for item in all_res.json()["items"]}
    assert all_ids == {empty.training_data_id, ready.training_data_id}


def test_model_training_api_returns_summary(tmp_path: Path):
    paths = WorkflowPaths(data_root=tmp_path)
    model_training_api.WORKFLOW_PATHS = paths
    artifact = tmp_path / "ridge.pkl"
    metadata = tmp_path / "ridge.json"
    artifact.write_text("model", encoding="utf-8")
    metadata.write_text("{}", encoding="utf-8")
    register_model(
        model_id="model_ridge",
        model_name="Ridge Model",
        model_family="featurewise_ridge_regression",
        source_training_data_id="training_data_ready",
        source_pipeline_id="pipeline_abc",
        artifact_path=artifact,
        metadata_path=metadata,
        training_samples=12,
        metrics={"selected_lambda": 2.0},
        paths=paths,
    )
    client = make_client(tmp_path)

    res = client.get("/api/workflow/model-training/summary")

    assert res.status_code == 200
    payload = res.json()
    assert payload["total_models"] == 1
    assert payload["ridge_count"] == 1
    assert payload["mlp_count"] == 0
    assert payload["latest_model"]["selected_lambda"] == 2.0
    assert payload["total_training_samples"] == 12


def test_model_training_api_returns_structured_missing_training_data_error(tmp_path: Path):
    client = make_client(tmp_path)

    res = client.post(
        "/api/workflow/model-training/train",
        json={
            "model_family": "featurewise_ridge_regression",
            "model_name": "Missing Data Model",
            "source_training_data_id": "missing",
        },
    )

    assert res.status_code == 404
    assert res.json()["detail"]["code"] == "training_data_not_found"


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

