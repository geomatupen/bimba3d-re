import json
from pathlib import Path

from bimba3d_backend.app.services import pipeline_learning_rows


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_pipeline_learning_rows_extracts_training_data_fields(monkeypatch, tmp_path: Path):
    pipeline_root = tmp_path / "pipeline"
    project_dir = pipeline_root / "project_one"
    run_dir = project_dir / "runs" / "run_001"

    write_json(
        project_dir / "config.json",
        {
            "id": "project_1",
            "name": "Project One",
            "ai_selector_strategy": "featurewise_ridge",
        },
    )
    write_json(
        run_dir / "analytics" / "run_analytics_v1.json",
        {
            "summary": {
                "mode": "mode3",
                "run_name": "Run 001",
                "metrics": {"final_loss": 0.3},
                "major_params": {"total_steps_completed": 5000},
            },
            "ai": {
                "input_mode_insights": {
                    "ai_input_mode": "exif_flight_scene",
                    "initial_params": {"run_jitter_multiplier": 1.0},
                    "x_features": {"image_count": 24},
                },
                "input_mode_learning": {
                    "phase": "explore",
                    "x_features": {"image_count": 24, "focal_length_mm": 35.0},
                    "selected_multipliers": {
                        "geometry_lr_mult": 1.1,
                        "appearance_lr_mult": 0.9,
                        "densification_lr_mult": 1.25,
                    },
                    "selected_log_multipliers": {
                        "geometry_lr_mult": 0.09531,
                        "appearance_lr_mult": -0.10536,
                        "densification_lr_mult": 0.22314,
                    },
                    "baseline_comparison": {
                        "r_quality": 0.2,
                        "r_convergence": 0.05,
                        "score_reference_step": 7000,
                        "loss_at_7000_run": 0.3,
                        "loss_at_7000_base": 0.35,
                    },
                },
            },
        },
    )

    monkeypatch.setattr(
        pipeline_learning_rows.training_pipeline_storage,
        "get_pipeline",
        lambda pipeline_id: {
            "id": pipeline_id,
            "name": "Offline Data Pipeline",
            "config": {
                "pipeline_folder": str(pipeline_root),
                "pre_generated_log_multipliers": {"geometry": [0.1]},
                "multiplier_current_index": 1,
            },
        },
    )

    payload = pipeline_learning_rows.collect_pipeline_learning_rows("pipeline_123")

    assert payload["pipeline_id"] == "pipeline_123"
    assert payload["total_rows"] == 1
    row = payload["rows"][0]
    assert row["project_id"] == "project_1"
    assert row["project_name"] == "Project One"
    assert row["x_features"]["focal_length_mm"] == 35.0
    assert row["selected_multipliers"]["geometry_lr_mult"] == 1.1
    assert row["selected_log_multipliers"]["densification_lr_mult"] == 0.22314
    assert row["relative_quality_score"] == 0.2


def test_collect_pipeline_learning_rows_reports_missing_pipeline(monkeypatch):
    monkeypatch.setattr(
        pipeline_learning_rows.training_pipeline_storage,
        "get_pipeline",
        lambda pipeline_id: None,
    )

    try:
        pipeline_learning_rows.collect_pipeline_learning_rows("missing")
    except FileNotFoundError as exc:
        assert "Pipeline not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")

