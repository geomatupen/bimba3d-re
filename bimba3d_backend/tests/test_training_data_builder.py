from pathlib import Path

from bimba3d_backend.app.services.training_data_builder import (
    build_from_learning_rows,
    convert_learning_row,
)
from bimba3d_backend.app.services.training_data_registry import create_manifest, read_manifest, read_rows
from bimba3d_backend.app.services.workflow_paths import WorkflowPaths


def test_convert_learning_row_keeps_report_fields():
    row = convert_learning_row(
        {
            "project_id": "project_1",
            "project_name": "Project One",
            "run_id": "run_1",
            "phase": 2,
            "x_features": {"image_count": 12},
            "selected_multipliers": {
                "geometry_lr_mult": 1.2,
                "appearance_lr_mult": 0.9,
                "densification_mult": 1.05,
            },
            "selected_log_multipliers": {
                "geometry_lr_mult": 0.182321,
                "appearance_lr_mult": -0.105361,
                "densification_mult": 0.04879,
            },
            "relative_quality_score": 0.33,
            "convergence_score": 0.12,
            "loss_at_5000_run": 0.5,
            "loss_at_5000_base": 0.7,
        },
        source_pipeline_id="pipeline_abc",
    )

    assert row.project_id == "project_1"
    assert row.source_pipeline_id == "pipeline_abc"
    assert row.relative_quality_score == 0.33
    assert row.selected_log_multipliers["appearance_lr_mult"] == -0.105361
    assert row.source is not None
    assert row.source.pipeline_id == "pipeline_abc"


def test_build_from_learning_rows_replaces_registry_rows(tmp_path: Path):
    paths = WorkflowPaths(data_root=tmp_path)
    manifest = create_manifest(
        name="Prepared Data",
        source_pipeline_id="pipeline_abc",
        feature_schema="mode3_exif_flight_scene_v1",
        training_data_id="training_data_target",
        paths=paths,
    )

    result = build_from_learning_rows(
        training_data_id=manifest.training_data_id,
        source_pipeline_id="pipeline_abc",
        rows=[
            {
                "project_id": "project_1",
                "run_id": "run_1",
                "x_features": {"image_count": 12},
                "selected_multipliers": {"geometry_lr_mult": 1.2},
                "selected_log_multipliers": {"geometry_lr_mult": 0.182321},
                "relative_quality_score": 0.33,
            }
        ],
        paths=paths,
    )

    assert result.imported_rows == 1
    assert result.errors == []
    assert read_manifest(manifest.training_data_id, paths=paths).status == "ready"
    assert len(read_rows(manifest.training_data_id, paths=paths)) == 1


def test_build_from_learning_rows_marks_manifest_failed_on_bad_row(tmp_path: Path):
    paths = WorkflowPaths(data_root=tmp_path)
    manifest = create_manifest(
        name="Prepared Data",
        source_pipeline_id="pipeline_abc",
        feature_schema="mode3_exif_flight_scene_v1",
        training_data_id="training_data_target",
        paths=paths,
    )

    result = build_from_learning_rows(
        training_data_id=manifest.training_data_id,
        source_pipeline_id="pipeline_abc",
        rows=[
            {
                "project_id": "project_1",
                "run_id": "run_1",
                "selected_multipliers": {"geometry_lr_mult": 1.2},
            }
        ],
        paths=paths,
    )

    updated = read_manifest(manifest.training_data_id, paths=paths)
    assert result.imported_rows == 0
    assert result.skipped_rows == 1
    assert "x_features" in result.errors[0]
    assert updated.status == "failed"
    assert updated.schema_valid is False

