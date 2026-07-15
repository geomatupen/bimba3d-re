from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bimba3d_backend.app.api import training_data as training_data_api
from bimba3d_backend.app.services.workflow_paths import WorkflowPaths


def make_client(tmp_path: Path) -> TestClient:
    training_data_api.WORKFLOW_PATHS = WorkflowPaths(data_root=tmp_path)
    app = FastAPI()
    app.include_router(training_data_api.router, prefix="/api/workflow/training-data")
    return TestClient(app)


def test_training_data_api_create_list_and_read(tmp_path: Path):
    client = make_client(tmp_path)

    create_res = client.post(
        "/api/workflow/training-data",
        json={
            "name": "Prepared Data",
            "source_pipeline_id": "pipeline_123",
            "feature_schema": "mode3_exif_flight_scene_v1",
            "training_data_id": "training_data_test",
        },
    )
    assert create_res.status_code == 200
    payload = create_res.json()
    assert payload["training_data_id"] == "training_data_test"
    assert payload["status"] == "empty"

    list_res = client.get("/api/workflow/training-data")
    assert list_res.status_code == 200
    assert list_res.json()["total"] == 1

    get_res = client.get("/api/workflow/training-data/training_data_test")
    assert get_res.status_code == 200
    assert get_res.json()["source_pipeline_id"] == "pipeline_123"

    rows_res = client.get("/api/workflow/training-data/training_data_test/rows")
    assert rows_res.status_code == 200
    assert rows_res.json() == {
        "training_data_id": "training_data_test",
        "rows": [],
        "total": 0,
    }


def test_training_data_api_returns_structured_not_found(tmp_path: Path):
    client = make_client(tmp_path)

    res = client.get("/api/workflow/training-data/missing")

    assert res.status_code == 404
    assert res.json()["detail"]["code"] == "training_data_not_found"


def test_training_data_api_builds_rows_from_learning_rows(tmp_path: Path):
    client = make_client(tmp_path)
    client.post(
        "/api/workflow/training-data",
        json={
            "name": "Prepared Data",
            "source_pipeline_id": "pipeline_123",
            "feature_schema": "mode3_exif_flight_scene_v1",
            "training_data_id": "training_data_test",
        },
    )

    res = client.post(
        "/api/workflow/training-data/training_data_test/build-from-learning-rows",
        json={
            "source_pipeline_id": "pipeline_123",
            "rows": [
                {
                    "project_id": "project_1",
                    "run_id": "run_1",
                    "x_features": {"image_count": 10},
                    "selected_multipliers": {"geometry_lr_mult": 1.1},
                    "selected_log_multipliers": {"geometry_lr_mult": 0.09531},
                    "relative_quality_score": 0.25,
                }
            ],
        },
    )

    assert res.status_code == 200
    payload = res.json()
    assert payload["imported_rows"] == 1
    assert payload["manifest"]["status"] == "ready"


def test_training_data_api_lists_usable_only_and_reports_validity(tmp_path: Path):
    client = make_client(tmp_path)
    client.post(
        "/api/workflow/training-data",
        json={
            "name": "Empty Data",
            "source_pipeline_id": "pipeline_empty",
            "training_data_id": "training_data_empty",
        },
    )
    client.post(
        "/api/workflow/training-data",
        json={
            "name": "Ready Data",
            "source_pipeline_id": "pipeline_ready",
            "training_data_id": "training_data_ready",
        },
    )
    client.post(
        "/api/workflow/training-data/training_data_ready/build-from-learning-rows",
        json={
            "source_pipeline_id": "pipeline_ready",
            "rows": [
                {
                    "project_id": "project_1",
                    "run_id": "run_1",
                    "x_features": {"image_count": 10},
                    "selected_multipliers": {"geometry_lr_mult": 1.1},
                    "selected_log_multipliers": {"geometry_lr_mult": 0.09531},
                    "relative_quality_score": 0.25,
                }
            ],
        },
    )

    list_res = client.get("/api/workflow/training-data?usable_only=true")
    assert list_res.status_code == 200
    payload = list_res.json()
    assert payload["total"] == 1
    assert payload["items"][0]["training_data_id"] == "training_data_ready"

    ready_validity = client.get("/api/workflow/training-data/training_data_ready/validity")
    assert ready_validity.status_code == 200
    assert ready_validity.json()["usable_for_model_training"] is True
    assert ready_validity.json()["row_count"] == 1

    empty_validity = client.get("/api/workflow/training-data/training_data_empty/validity")
    assert empty_validity.status_code == 200
    assert empty_validity.json()["usable_for_model_training"] is False


def test_training_data_api_lists_by_source_pipeline(tmp_path: Path):
    client = make_client(tmp_path)
    client.post(
        "/api/workflow/training-data",
        json={
            "name": "Pipeline A Data",
            "source_pipeline_id": "pipeline_a",
            "training_data_id": "training_data_a",
        },
    )
    client.post(
        "/api/workflow/training-data",
        json={
            "name": "Pipeline B Data",
            "source_pipeline_id": "pipeline_b",
            "training_data_id": "training_data_b",
        },
    )

    res = client.get("/api/workflow/training-data/by-source-pipeline/pipeline_a")

    assert res.status_code == 200
    payload = res.json()
    assert payload["source_pipeline_id"] == "pipeline_a"
    assert payload["total"] == 1
    assert payload["items"][0]["training_data_id"] == "training_data_a"


def test_training_data_api_build_marks_manifest_failed_for_bad_rows(tmp_path: Path):
    client = make_client(tmp_path)
    client.post(
        "/api/workflow/training-data",
        json={
            "name": "Prepared Data",
            "source_pipeline_id": "pipeline_123",
            "feature_schema": "mode3_exif_flight_scene_v1",
            "training_data_id": "training_data_test",
        },
    )

    res = client.post(
        "/api/workflow/training-data/training_data_test/build-from-learning-rows",
        json={
            "source_pipeline_id": "pipeline_123",
            "rows": [{"project_id": "project_1", "run_id": "run_1"}],
        },
    )

    assert res.status_code == 200
    payload = res.json()
    assert payload["imported_rows"] == 0
    assert payload["skipped_rows"] == 1
    assert "x_features" in payload["errors"][0]
    assert payload["manifest"]["status"] == "failed"


def test_training_data_api_builds_rows_from_pipeline(monkeypatch, tmp_path: Path):
    client = make_client(tmp_path)
    client.post(
        "/api/workflow/training-data",
        json={
            "name": "Prepared Data",
            "source_pipeline_id": "pipeline_123",
            "feature_schema": "mode3_exif_flight_scene_v1",
            "training_data_id": "training_data_test",
        },
    )

    def fake_collect_pipeline_learning_rows(pipeline_id: str):
        assert pipeline_id == "pipeline_123"
        return {
            "rows": [
                {
                    "project_id": "project_1",
                    "run_id": "run_1",
                    "x_features": {"image_count": 10},
                    "selected_multipliers": {"geometry_lr_mult": 1.1},
                    "selected_log_multipliers": {"geometry_lr_mult": 0.09531},
                    "relative_quality_score": 0.25,
                }
            ]
        }

    from bimba3d_backend.app.api import training_data as training_data_api

    monkeypatch.setattr(
        training_data_api.pipeline_learning_rows,
        "collect_pipeline_learning_rows",
        fake_collect_pipeline_learning_rows,
    )

    res = client.post("/api/workflow/training-data/training_data_test/build-from-pipeline/pipeline_123")

    assert res.status_code == 200
    payload = res.json()
    assert payload["imported_rows"] == 1
    assert payload["manifest"]["status"] == "ready"

