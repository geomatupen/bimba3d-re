from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bimba3d_backend.app.api.workflow_pipelines import router
from bimba3d_backend.app.services import workflow_pipeline_service


def make_client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/workflow/pipelines")
    return TestClient(app)


def sample_pipeline(pipeline_id: str, *, pipeline_type: str = "train", pipeline_folder: str | None = None) -> dict:
    return {
        "id": pipeline_id,
        "name": "Pipeline",
        "status": "completed",
        "pipeline_type": pipeline_type,
        "created_at": "2026-01-01T00:00:00Z",
        "started_at": "2026-01-01T00:01:00Z",
        "completed_at": "2026-01-01T00:10:00Z",
        "total_runs": 2,
        "completed_runs": 2,
        "failed_runs": 0,
        "current_phase": 1,
        "current_pass": 1,
        "current_project_index": 0,
        "mean_relative_score": 0.2,
        "best_relative_score": 0.3,
        "success_rate": 100.0,
        "last_error": None,
        "cooldown_active": False,
        "next_run_scheduled_at": None,
        "config": {
            "pipeline_type": pipeline_type,
            "pipeline_folder": pipeline_folder,
            "pre_generated_log_multipliers": {"geometry": [0.1]},
            "multiplier_current_index": 1,
        },
        "runs": [{"run_id": "run_1", "status": "success"}],
    }


class PipelineStore:
    def __init__(self, pipeline: dict):
        self.pipeline = pipeline

    def get_pipeline(self, pipeline_id: str):
        if self.pipeline.get("id") != pipeline_id:
            return None
        return dict(self.pipeline)

    def update_pipeline(self, pipeline_id: str, updates: dict):
        if self.pipeline.get("id") != pipeline_id:
            return None
        self.pipeline.update(updates)
        return dict(self.pipeline)


def test_workflow_pipelines_api_lists_and_filters(monkeypatch):
    client = make_client()

    monkeypatch.setattr(
        workflow_pipeline_service.training_pipeline_storage,
        "list_pipelines",
        lambda limit=100: [
            sample_pipeline("pipeline_data", pipeline_type="train"),
            sample_pipeline("pipeline_test", pipeline_type="test"),
        ],
    )

    all_res = client.get("/api/workflow/pipelines")
    assert all_res.status_code == 200
    assert all_res.json()["total"] == 2

    data_res = client.get("/api/workflow/pipelines?stage=offline_data_preparation")
    assert data_res.status_code == 200
    payload = data_res.json()
    assert payload["total"] == 1
    assert payload["items"][0]["id"] == "pipeline_data"
    assert payload["items"][0]["workflow_stage"] == "offline_data_preparation"


def test_workflow_pipelines_api_scans_dataset_directory(tmp_path: Path):
    dataset_b = tmp_path / "dataset_b"
    dataset_a = tmp_path / "dataset_a"
    empty_dataset = tmp_path / "empty"
    dataset_b.mkdir()
    dataset_a.mkdir()
    empty_dataset.mkdir()
    (dataset_b / "image_2.JPG").write_bytes(b"bb")
    (dataset_a / "image_1.png").write_bytes(b"a")
    (empty_dataset / "notes.txt").write_text("ignore", encoding="utf-8")

    client = make_client()
    res = client.post(
        "/api/workflow/pipelines/scan-directory",
        json={"base_directory": str(tmp_path)},
    )

    assert res.status_code == 200
    payload = res.json()
    assert payload["total"] == 2
    assert [item["name"] for item in payload["datasets"]] == ["dataset_a", "dataset_b"]
    assert payload["datasets"][0]["image_count"] == 1


def test_workflow_pipelines_api_scan_directory_reports_missing(tmp_path: Path):
    client = make_client()
    res = client.post(
        "/api/workflow/pipelines/scan-directory",
        json={"base_directory": str(tmp_path / "missing")},
    )

    assert res.status_code == 404
    assert res.json()["detail"]["code"] == "workflow_scan_directory_not_found"


def test_workflow_pipelines_api_batch_creates_projects(monkeypatch, tmp_path: Path):
    data_root = tmp_path / "data"
    existing_dir = data_root / "existing_dataset"
    existing_dir.mkdir(parents=True)
    (existing_dir / "config.json").write_text('{"id": "existing_project"}', encoding="utf-8")
    monkeypatch.setattr(workflow_pipeline_service, "DATA_DIR", data_root)

    client = make_client()
    res = client.post(
        "/api/workflow/pipelines/batch-create-projects",
        json={
            "datasets": [
                {
                    "name": "new_dataset",
                    "path": "D:/datasets/new_dataset",
                    "image_count": 4,
                    "size_mb": 1.0,
                    "has_images": True,
                },
                {
                    "name": "existing_dataset",
                    "path": "D:/datasets/existing_dataset",
                    "image_count": 2,
                    "size_mb": 1.0,
                    "has_images": True,
                },
            ],
            "shared_config": {"camera_model": "OPENCV"},
        },
    )

    assert res.status_code == 200
    payload = res.json()
    assert len(payload["created"]) == 1
    assert len(payload["existing"]) == 1
    assert payload["failed"] == []
    assert payload["created"][0]["name"] == "new_dataset"
    assert payload["existing"][0]["project_id"] == "existing_project"
    assert (data_root / "new_dataset" / "config.json").exists()
    assert "OPENCV" in (data_root / "new_dataset" / "config.json").read_text(encoding="utf-8")


def test_workflow_pipelines_api_creates_offline_data_pipeline(monkeypatch):
    captured: dict[str, object] = {}

    def fake_create_pipeline(config: dict):
        captured.update(config)
        return sample_pipeline("pipeline_created", pipeline_type=config["pipeline_type"])

    monkeypatch.setattr(
        workflow_pipeline_service.training_pipeline_storage,
        "create_pipeline",
        fake_create_pipeline,
    )

    client = make_client()
    res = client.post(
        "/api/workflow/pipelines",
        json={
            "name": "Prepare Offline Data",
            "workflow_stage": "offline_data_preparation",
            "base_directory": "D:/datasets",
            "projects": [],
            "shared_config": {
                "geometry_log_multiplier_min": 0.5,
                "geometry_log_multiplier_max": 2.0,
                "appearance_log_multiplier_min": 0.5,
                "appearance_log_multiplier_max": 2.0,
                "densification_log_multiplier_min": 0.7,
                "densification_log_multiplier_max": 1.42,
            },
            "phases": [
                {"phase_number": 1, "exploration_runs_per_project": 1},
                {"phase_number": 2, "exploration_runs_per_project": 3, "context_jitter": True},
            ],
        },
    )

    assert res.status_code == 200
    assert res.json()["workflow_stage"] == "offline_data_preparation"
    assert captured["pipeline_type"] == "offline_data"
    assert captured["source_model_id"] is None
    assert captured["multiplier_current_index"] == 0
    assert len(captured["pre_generated_log_multipliers"]["geometry_lr"]) == 3
    assert len(captured["pre_generated_log_multipliers"]["appearance_lr"]) == 3
    assert len(captured["pre_generated_log_multipliers"]["scale_lr"]) == 3
    assert all(0.5 <= value <= 2.0 for value in captured["pre_generated_log_multipliers"]["geometry_lr"])
    assert all(0.5 <= value <= 2.0 for value in captured["pre_generated_log_multipliers"]["appearance_lr"])
    assert all(0.7 <= value <= 1.42 for value in captured["pre_generated_log_multipliers"]["scale_lr"])
    assert isinstance(captured["restart_token"], str)


def test_workflow_pipelines_api_creates_testing_pipeline_with_model_ids(monkeypatch):
    captured: dict[str, object] = {}

    def fake_create_pipeline(config: dict):
        captured.update(config)
        return sample_pipeline("pipeline_test", pipeline_type=config["pipeline_type"])

    monkeypatch.setattr(
        workflow_pipeline_service.training_pipeline_storage,
        "create_pipeline",
        fake_create_pipeline,
    )

    client = make_client()
    res = client.post(
        "/api/workflow/pipelines",
        json={
            "name": "Run Tests",
            "workflow_stage": "testing_pipeline",
            "source_model_ids": ["model_a", "model_a", "model_b"],
            "projects": [],
            "shared_config": {},
            "phases": [{"phase_number": 1, "exploration_runs_per_project": 1}],
        },
    )

    assert res.status_code == 200
    assert res.json()["workflow_stage"] == "testing_pipeline"
    assert captured["pipeline_type"] == "test"
    assert captured["source_model_ids"] == ["model_a", "model_b"]
    assert captured["source_model_id"] == "model_a"


def test_workflow_pipelines_api_rejects_unsupported_create_stage():
    client = make_client()

    res = client.post(
        "/api/workflow/pipelines",
        json={
            "name": "Train Models",
            "workflow_stage": "model_training",
        },
    )

    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "workflow_pipeline_create_invalid"


def test_workflow_pipelines_api_reads_detail(monkeypatch):
    client = make_client()

    monkeypatch.setattr(
        workflow_pipeline_service.training_pipeline_storage,
        "get_pipeline",
        lambda pipeline_id: sample_pipeline(pipeline_id, pipeline_type="test"),
    )

    res = client.get("/api/workflow/pipelines/pipeline_test")

    assert res.status_code == 200
    payload = res.json()
    assert payload["workflow_stage"] == "testing_pipeline"
    assert payload["fixed_log_space_schedule"]["multiplier_current_index"] == 1
    assert payload["runs"][0]["run_id"] == "run_1"


def test_workflow_pipelines_api_updates_config_preserving_fixed_schedule(monkeypatch):
    pipeline = sample_pipeline("pipeline_data")
    pipeline["config"].update(
        {
            "name": "Old Name",
            "pipeline_folder": "D:/existing/pipeline",
            "restart_version": 4,
            "restart_token": "restart-token",
            "last_restart_at": "2026-01-01T00:00:00Z",
            "pre_generated_log_multipliers": {"geometry_lr": [1.0, 0.8]},
            "multiplier_current_index": 1,
            "phases": [{"phase_number": 1, "exploration_runs_per_project": 2}],
            "projects": [{"name": "Project"}],
        }
    )
    store = PipelineStore(pipeline)

    monkeypatch.setattr(workflow_pipeline_service.training_pipeline_storage, "get_pipeline", store.get_pipeline)
    monkeypatch.setattr(workflow_pipeline_service.training_pipeline_storage, "update_pipeline", store.update_pipeline)

    client = make_client()
    res = client.put(
        "/api/workflow/pipelines/pipeline_data/config",
        json={
            "name": "Updated Name",
            "workflow_stage": "offline_data_preparation",
            "projects": [{"name": "Project"}],
            "shared_config": {"geometry_jitter_factor": 0.5},
            "phases": [{"phase_number": 1, "exploration_runs_per_project": 3}],
        },
    )

    assert res.status_code == 200
    payload = res.json()
    assert payload["total_runs"] == 3
    assert payload["pipeline"]["name"] == "Updated Name"
    assert store.pipeline["config"]["pipeline_folder"] == "D:/existing/pipeline"
    assert store.pipeline["config"]["restart_version"] == 4
    assert store.pipeline["config"]["restart_token"] == "restart-token"
    assert store.pipeline["config"]["pre_generated_log_multipliers"] == {"geometry_lr": [1.0, 0.8]}
    assert store.pipeline["config"]["multiplier_current_index"] == 1


def test_workflow_pipelines_api_rejects_config_update_while_running(monkeypatch):
    pipeline = sample_pipeline("pipeline_data")
    pipeline["status"] = "running"
    store = PipelineStore(pipeline)

    monkeypatch.setattr(workflow_pipeline_service.training_pipeline_storage, "get_pipeline", store.get_pipeline)

    client = make_client()
    res = client.put(
        "/api/workflow/pipelines/pipeline_data/config",
        json={"name": "Updated", "workflow_stage": "offline_data_preparation"},
    )

    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "workflow_pipeline_action_invalid"


def test_workflow_pipelines_api_config_update_marks_completed_pipeline_resumable(monkeypatch):
    pipeline = sample_pipeline("pipeline_data")
    pipeline["status"] = "completed"
    pipeline["completed_at"] = "2026-01-01T00:10:00Z"
    pipeline["config"].update(
        {
            "phases": [{"phase_number": 1, "exploration_runs_per_project": 1}],
            "projects": [{"name": "Project"}],
            "pre_generated_log_multipliers": {"geometry_lr": [1.0]},
        }
    )
    store = PipelineStore(pipeline)

    monkeypatch.setattr(workflow_pipeline_service.training_pipeline_storage, "get_pipeline", store.get_pipeline)
    monkeypatch.setattr(workflow_pipeline_service.training_pipeline_storage, "update_pipeline", store.update_pipeline)

    client = make_client()
    res = client.put(
        "/api/workflow/pipelines/pipeline_data/config",
        json={
            "name": "Updated",
            "workflow_stage": "offline_data_preparation",
            "projects": [{"name": "Project"}],
            "phases": [{"phase_number": 1, "exploration_runs_per_project": 2}],
        },
    )

    assert res.status_code == 200
    assert res.json()["resumable"] is True
    assert store.pipeline["status"] == "stopped"
    assert store.pipeline["completed_at"] is None


def test_workflow_pipelines_api_reads_worker_logs(monkeypatch, tmp_path: Path):
    pipeline_root = tmp_path / "pipeline"
    project_dir = pipeline_root / "project_one"
    run_dir = project_dir / "runs" / "run_001"
    run_dir.mkdir(parents=True)
    (project_dir / "processing.log").write_text("project log\n", encoding="utf-8")
    (run_dir / "processing.log").write_text("run log\n", encoding="utf-8")

    monkeypatch.setattr(
        workflow_pipeline_service.training_pipeline_storage,
        "get_pipeline",
        lambda pipeline_id: sample_pipeline(pipeline_id, pipeline_folder=str(pipeline_root)),
    )

    client = make_client()
    res = client.get("/api/workflow/pipelines/pipeline_data/worker-logs")

    assert res.status_code == 200
    payload = res.json()
    assert payload["pipeline_id"] == "pipeline_data"
    assert payload["total_projects"] == 2
    assert payload["logs"][0]["logs"] == "project log"
    assert payload["logs"][1]["project"] == "project_one / run_001"


def test_workflow_pipelines_api_returns_fixed_log_space_schedule(monkeypatch):
    pipeline = sample_pipeline("pipeline_data", pipeline_type="train")
    pipeline["config"].update(
        {
            "pre_generated_log_multipliers": {
                "geometry_lr": [1.0, 0.8, 0.64],
                "appearance_lr": [1.0, 0.9],
            },
            "multiplier_current_index": 2,
            "restart_version": 3,
            "restart_token": "token",
            "last_restart_at": "2026-01-01T00:00:00Z",
        }
    )
    monkeypatch.setattr(
        workflow_pipeline_service.training_pipeline_storage,
        "get_pipeline",
        lambda pipeline_id: pipeline,
    )

    client = make_client()
    res = client.get("/api/workflow/pipelines/pipeline_data/fixed-log-space-schedule")

    assert res.status_code == 200
    payload = res.json()
    assert payload["restart_version"] == 3
    assert payload["groups"]["geometry_lr"]["last_value"] == 0.8
    assert payload["groups"]["geometry_lr"]["next_value"] == 0.64
    assert payload["groups"]["appearance_lr"]["next_value"] is None


def test_workflow_pipelines_api_returns_offline_data_summary(monkeypatch):
    pipeline = sample_pipeline("pipeline_data", pipeline_type="train")
    pipeline["config"]["projects"] = [{"name": "project_1"}]
    pipeline["config"]["pre_generated_log_multipliers"] = {"geometry_lr": [1.0, 0.8]}
    monkeypatch.setattr(
        workflow_pipeline_service.training_pipeline_storage,
        "get_pipeline",
        lambda pipeline_id: pipeline,
    )

    def fake_collect_pipeline_learning_rows(pipeline_id: str):
        return {
            "rows": [
                {
                    "project_id": "project_1",
                    "project_name": "Project One",
                    "run_id": "baseline",
                    "is_baseline_row": True,
                },
                {
                    "project_id": "project_1",
                    "project_name": "Project One",
                    "run_id": "run_1",
                    "is_baseline_row": False,
                    "phase": 2,
                    "selected_log_multipliers": {
                        "geometry_lr_mult": 0.1,
                        "appearance_lr_mult": -0.2,
                        "densification_mult": 0.3,
                    },
                    "selected_multipliers": {
                        "geometry_lr_mult": 1.1,
                        "appearance_lr_mult": 0.8,
                        "densification_mult": 1.3,
                    },
                    "relative_quality_score": 0.25,
                },
            ]
        }

    monkeypatch.setattr(
        "bimba3d_backend.app.services.workflow_summaries.pipeline_learning_rows.collect_pipeline_learning_rows",
        fake_collect_pipeline_learning_rows,
    )

    client = make_client()
    res = client.get("/api/workflow/pipelines/pipeline_data/offline-data-summary")

    assert res.status_code == 200
    payload = res.json()
    assert payload["learning_rows"] == 2
    assert payload["baseline_rows"] == 1
    assert payload["non_baseline_rows"] == 1
    assert payload["mean_relative_score"] == 0.25
    assert len(payload["multiplier_score_distribution"]["geometry"]) == 1
    assert payload["multiplier_score_distribution"]["appearance"][0]["relative_score"] == 0.25
    assert payload["multiplier_score_distribution"]["densification"][0]["multiplier"] == 1.3


def test_workflow_pipelines_api_returns_testing_summary(monkeypatch):
    pipeline = sample_pipeline("pipeline_test", pipeline_type="test")
    pipeline["config"]["projects"] = [{"name": "project_1"}, {"name": "project_2"}]
    pipeline["config"]["source_model_ids"] = ["model_a", "model_b"]
    pipeline["runs"] = [
        {"run_id": "run_1", "model_id": "model_a", "status": "success"},
        {"run_id": "run_2", "model_id": "model_a", "status": "failed"},
        {"run_id": "run_3", "model_id": "model_b", "status": "success"},
    ]
    pipeline["prediction_previews"] = {
        "preview_1": {
            "created_at": "2026-01-01T00:00:00Z",
            "rows": [
                {
                    "project_id": "project_1",
                    "model_id": "model_a",
                    "candidate_score_checks": {
                        "geometry": [{"candidate": 1}, {"candidate": 2}],
                        "appearance": [{"candidate": 1}],
                    },
                }
            ],
        }
    }
    pipeline["latest_prediction_preview_key"] = "preview_1"

    monkeypatch.setattr(
        workflow_pipeline_service.training_pipeline_storage,
        "get_pipeline",
        lambda pipeline_id: pipeline,
    )

    client = make_client()
    res = client.get("/api/workflow/pipelines/pipeline_test/testing-summary")

    assert res.status_code == 200
    payload = res.json()
    assert payload["total_test_projects"] == 2
    assert payload["models_tested"] == 2
    assert payload["per_model_status"][0]["completed"] == 1
    assert payload["per_model_status"][0]["failed"] == 1
    assert payload["prediction_preview"]["candidate_curve_rows"] == 1
    assert payload["prediction_preview"]["candidate_points"] == 3


def test_workflow_pipelines_api_returns_prediction_preview(monkeypatch):
    pipeline = sample_pipeline("pipeline_test", pipeline_type="test")
    pipeline["prediction_previews"] = {
        "preview_old": {
            "generated_at": "2026-01-01T00:00:00Z",
            "results": [{"project_id": "old"}],
        },
        "preview_latest": {
            "generated_at": "2026-01-02T00:00:00Z",
            "results": [{"project_id": "latest"}],
        },
    }
    pipeline["latest_prediction_preview_key"] = "preview_latest"
    pipeline["prediction_preview_artifacts"] = {
        "preview_latest": {"dir": "D:/preview"},
    }

    monkeypatch.setattr(
        workflow_pipeline_service.training_pipeline_storage,
        "get_pipeline",
        lambda pipeline_id: pipeline,
    )

    client = make_client()
    res = client.get("/api/workflow/pipelines/pipeline_test/prediction-preview")

    assert res.status_code == 200
    payload = res.json()
    assert payload["preview_key"] == "preview_latest"
    assert payload["total_rows"] == 1
    assert payload["rows"][0]["project_id"] == "latest"
    assert payload["artifact"]["dir"] == "D:/preview"


def test_workflow_pipelines_api_returns_testing_candidate_curves(monkeypatch):
    pipeline = sample_pipeline("pipeline_test", pipeline_type="test")
    pipeline["prediction_previews"] = {
        "preview_1": {
            "generated_at": "2026-01-02T00:00:00Z",
            "results": [
                {
                    "project_id": "project_1",
                    "project_name": "Project One",
                    "model_id": "model_a",
                    "mode": "featurewise_ridge",
                    "candidate_score_checks": {
                        "geometry": [
                            {
                                "candidate_log_multiplier": -0.1,
                                "candidate_multiplier": 0.9,
                                "predicted_score": 0.1,
                            },
                            {
                                "candidate_log_multiplier": 0.0,
                                "candidate_multiplier": 1.0,
                                "predicted_score": 0.3,
                                "selected": True,
                            },
                        ],
                        "appearance": [
                            {
                                "candidate_log_multiplier": -0.2,
                                "candidate_multiplier": 0.8,
                                "predicted_relative_score": 0.2,
                            },
                            {
                                "candidate_log_multiplier": 0.2,
                                "candidate_multiplier": 1.2,
                                "predicted_relative_score": 0.5,
                            },
                        ],
                    },
                }
            ],
        }
    }
    pipeline["latest_prediction_preview_key"] = "preview_1"
    monkeypatch.setattr(
        workflow_pipeline_service.training_pipeline_storage,
        "get_pipeline",
        lambda pipeline_id: pipeline,
    )

    client = make_client()
    res = client.get("/api/workflow/pipelines/pipeline_test/testing-candidate-curves")

    assert res.status_code == 200
    payload = res.json()
    assert payload["preview_key"] == "preview_1"
    assert payload["total_curves"] == 1
    assert payload["total_points"] == 4
    curve = payload["curves"][0]
    assert curve["project_id"] == "project_1"
    assert curve["highest_point_by_group"]["geometry"]["selected"] is True
    assert curve["highest_point_by_group"]["appearance"]["candidate_multiplier"] == 1.2


def test_workflow_pipelines_api_predict_multipliers_bridge(monkeypatch):
    called: dict[str, object] = {}

    async def fake_predict_multipliers(pipeline_id: str, request_payload: dict):
        called["pipeline_id"] = pipeline_id
        called["request_payload"] = request_payload
        return {"pipeline_id": pipeline_id, "ok": 2, "failed": 0, "results": []}

    monkeypatch.setattr(workflow_pipeline_service, "predict_multipliers", fake_predict_multipliers)

    client = make_client()
    res = client.post(
        "/api/workflow/pipelines/pipeline_test/predict-multipliers",
        json={"model_ids": ["model_a", "model_b"]},
    )

    assert res.status_code == 200
    assert res.json()["ok"] == 2
    assert called["pipeline_id"] == "pipeline_test"
    assert called["request_payload"]["model_ids"] == ["model_a", "model_b"]


def test_workflow_pipelines_api_export_current_test_bridge(monkeypatch):
    called: dict[str, str] = {}

    async def fake_export_current_test(pipeline_id: str):
        called["pipeline_id"] = pipeline_id
        return {"pipeline_id": pipeline_id, "exported": True}

    monkeypatch.setattr(workflow_pipeline_service, "export_current_test", fake_export_current_test)

    client = make_client()
    res = client.get("/api/workflow/pipelines/pipeline_test/export-current-test")

    assert res.status_code == 200
    assert res.json()["exported"] is True
    assert called == {"pipeline_id": "pipeline_test"}


def test_workflow_pipelines_api_rejects_testing_summary_for_non_test_pipeline(monkeypatch):
    monkeypatch.setattr(
        workflow_pipeline_service.training_pipeline_storage,
        "get_pipeline",
        lambda pipeline_id: sample_pipeline(pipeline_id, pipeline_type="train"),
    )

    client = make_client()
    res = client.get("/api/workflow/pipelines/pipeline_data/testing-summary")

    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "workflow_pipeline_action_invalid"


def test_workflow_pipelines_api_returns_structured_not_found(monkeypatch):
    monkeypatch.setattr(
        workflow_pipeline_service.training_pipeline_storage,
        "get_pipeline",
        lambda pipeline_id: None,
    )

    client = make_client()
    res = client.get("/api/workflow/pipelines/missing")

    assert res.status_code == 404
    assert res.json()["detail"]["code"] == "workflow_pipeline_not_found"


def test_workflow_pipelines_api_starts_pipeline(monkeypatch):
    store = PipelineStore(sample_pipeline("pipeline_data"))
    started: list[str] = []

    monkeypatch.setattr(workflow_pipeline_service.training_pipeline_storage, "get_pipeline", store.get_pipeline)
    monkeypatch.setattr(workflow_pipeline_service.training_pipeline_storage, "update_pipeline", store.update_pipeline)
    monkeypatch.setattr(
        workflow_pipeline_service.training_pipeline_orchestrator,
        "start_pipeline_orchestrator",
        lambda pipeline_id: started.append(pipeline_id),
    )

    client = make_client()
    res = client.post("/api/workflow/pipelines/pipeline_data/start")

    assert res.status_code == 200
    assert res.json()["status"] == "running"
    assert store.pipeline["current_phase"] == 1
    assert started == ["pipeline_data"]


def test_workflow_pipelines_api_rejects_start_when_running(monkeypatch):
    pipeline = sample_pipeline("pipeline_data")
    pipeline["status"] = "running"
    store = PipelineStore(pipeline)

    monkeypatch.setattr(workflow_pipeline_service.training_pipeline_storage, "get_pipeline", store.get_pipeline)

    client = make_client()
    res = client.post("/api/workflow/pipelines/pipeline_data/start")

    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "workflow_pipeline_action_invalid"


def test_workflow_pipelines_api_pauses_and_resumes_pipeline(monkeypatch):
    pipeline = sample_pipeline("pipeline_data")
    pipeline["status"] = "running"
    store = PipelineStore(pipeline)
    started: list[str] = []
    stopped: list[str] = []

    monkeypatch.setattr(workflow_pipeline_service.training_pipeline_storage, "get_pipeline", store.get_pipeline)
    monkeypatch.setattr(workflow_pipeline_service.training_pipeline_storage, "update_pipeline", store.update_pipeline)
    monkeypatch.setattr(
        workflow_pipeline_service.training_pipeline_orchestrator,
        "start_pipeline_orchestrator",
        lambda pipeline_id: started.append(pipeline_id),
    )
    monkeypatch.setattr(
        workflow_pipeline_service.training_pipeline_orchestrator,
        "stop_pipeline_orchestrator",
        lambda pipeline_id: stopped.append(pipeline_id),
    )

    client = make_client()
    pause_res = client.post("/api/workflow/pipelines/pipeline_data/pause")
    resume_res = client.post("/api/workflow/pipelines/pipeline_data/resume")

    assert pause_res.status_code == 200
    assert pause_res.json()["status"] == "paused"
    assert resume_res.status_code == 200
    assert resume_res.json()["status"] == "running"
    assert stopped == ["pipeline_data"]
    assert started == ["pipeline_data"]


def test_workflow_pipelines_api_stops_pipeline(monkeypatch):
    pipeline = sample_pipeline("pipeline_data")
    pipeline["status"] = "running"
    store = PipelineStore(pipeline)
    stopped: list[str] = []

    monkeypatch.setattr(workflow_pipeline_service.training_pipeline_storage, "get_pipeline", store.get_pipeline)
    monkeypatch.setattr(workflow_pipeline_service.training_pipeline_storage, "update_pipeline", store.update_pipeline)
    monkeypatch.setattr(
        workflow_pipeline_service.training_pipeline_orchestrator,
        "stop_pipeline_orchestrator",
        lambda pipeline_id: stopped.append(pipeline_id),
    )

    import bimba3d_backend.app.services.colmap as colmap_service

    monkeypatch.setattr(colmap_service, "stop_all_local_workers", lambda: 2)

    client = make_client()
    res = client.post("/api/workflow/pipelines/pipeline_data/stop")

    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] == "stopped"
    assert payload["killed_workers"] == 2
    assert stopped == ["pipeline_data"]


def test_workflow_pipelines_api_restarts_pipeline(monkeypatch):
    called: dict[str, object] = {}

    async def fake_restart_pipeline(pipeline_id: str, *, keep_baseline: bool = False):
        called["pipeline_id"] = pipeline_id
        called["keep_baseline"] = keep_baseline
        return {
            "status": "restarted_and_running",
            "message": "Pipeline restarted",
            "restart_version": 2,
        }

    monkeypatch.setattr(workflow_pipeline_service, "restart_pipeline", fake_restart_pipeline)

    client = make_client()
    res = client.post("/api/workflow/pipelines/pipeline_data/restart?keep_baseline=true")

    assert res.status_code == 200
    assert res.json()["restart_version"] == 2
    assert called == {"pipeline_id": "pipeline_data", "keep_baseline": True}


def test_workflow_pipelines_api_retries_failed_runs(monkeypatch):
    called: dict[str, object] = {}

    async def fake_retry_failed_runs(pipeline_id: str, *, auto_start: bool = True):
        called["pipeline_id"] = pipeline_id
        called["auto_start"] = auto_start
        return {
            "status": "stopped",
            "message": "Failed runs prepared for retry",
            "retry_count": 3,
        }

    monkeypatch.setattr(workflow_pipeline_service, "retry_failed_runs", fake_retry_failed_runs)

    client = make_client()
    res = client.post("/api/workflow/pipelines/pipeline_data/retry-failed?auto_start=false")

    assert res.status_code == 200
    assert res.json()["retry_count"] == 3
    assert called == {"pipeline_id": "pipeline_data", "auto_start": False}


