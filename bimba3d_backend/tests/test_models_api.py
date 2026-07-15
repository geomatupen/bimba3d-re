from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bimba3d_backend.app.api import models as models_api
from bimba3d_backend.app.services.workflow_model_registry import register_model
from bimba3d_backend.app.services.workflow_paths import WorkflowPaths


def make_client(tmp_path: Path) -> TestClient:
    models_api.WORKFLOW_PATHS = WorkflowPaths(data_root=tmp_path)
    app = FastAPI()
    app.include_router(models_api.router, prefix="/api/models")
    return TestClient(app)


def test_models_api_lists_and_reads_models(tmp_path: Path):
    paths = WorkflowPaths(data_root=tmp_path)
    models_api.WORKFLOW_PATHS = paths
    artifact = tmp_path / "ridge.json"
    artifact.write_text('{"model": "ridge"}', encoding="utf-8")
    register_model(
        model_id="model_ridge_001",
        model_name="Featurewise Ridge",
        model_family="featurewise_ridge_regression",
        source_training_data_id="training_data_001",
        artifact_path=artifact,
        paths=paths,
    )
    client = make_client(tmp_path)

    list_res = client.get("/api/models")
    assert list_res.status_code == 200
    assert list_res.json()["total"] == 1
    assert list_res.json()["items"][0]["model_id"] == "model_ridge_001"

    get_res = client.get("/api/models/model_ridge_001")
    assert get_res.status_code == 200
    assert get_res.json()["model_family"] == "featurewise_ridge_regression"


def test_models_api_filters_by_source_pipeline_and_training_data(tmp_path: Path):
    paths = WorkflowPaths(data_root=tmp_path)
    models_api.WORKFLOW_PATHS = paths
    artifact_a = tmp_path / "ridge_a.json"
    artifact_b = tmp_path / "ridge_b.json"
    artifact_a.write_text('{"model": "a"}', encoding="utf-8")
    artifact_b.write_text('{"model": "b"}', encoding="utf-8")
    register_model(
        model_id="model_a",
        model_name="Model A",
        model_family="featurewise_ridge_regression",
        source_training_data_id="training_data_a",
        source_pipeline_id="pipeline_a",
        artifact_path=artifact_a,
        paths=paths,
    )
    register_model(
        model_id="model_b",
        model_name="Model B",
        model_family="featurewise_mlp",
        source_training_data_id="training_data_b",
        source_pipeline_id="pipeline_b",
        artifact_path=artifact_b,
        paths=paths,
    )
    client = make_client(tmp_path)

    by_pipeline = client.get("/api/models?source_pipeline_id=pipeline_a")
    assert by_pipeline.status_code == 200
    assert by_pipeline.json()["total"] == 1
    assert by_pipeline.json()["items"][0]["model_id"] == "model_a"

    by_training_data = client.get("/api/models?source_training_data_id=training_data_b")
    assert by_training_data.status_code == 200
    assert by_training_data.json()["total"] == 1
    assert by_training_data.json()["items"][0]["model_id"] == "model_b"


def test_models_api_returns_structured_not_found(tmp_path: Path):
    client = make_client(tmp_path)

    res = client.get("/api/models/missing")

    assert res.status_code == 404
    assert res.json()["detail"]["code"] == "workflow_model_not_found"
