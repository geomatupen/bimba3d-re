from pathlib import Path

from bimba3d_backend.app.services.training_data_registry import (
    create_manifest,
    list_manifests,
    read_rows,
    replace_rows,
)
from bimba3d_backend.app.services.workflow_paths import WorkflowPaths


def test_training_data_registry_creates_manifest_and_replaces_rows(tmp_path: Path):
    paths = WorkflowPaths(data_root=tmp_path)
    manifest = create_manifest(
        name="Prepared Training Data",
        source_pipeline_id="pipeline_abc",
        feature_schema="mode3_exif_flight_scene_v1",
        paths=paths,
    )

    assert manifest.training_data_id.startswith("training_data_")
    assert manifest.status == "empty"
    assert manifest.row_count == 0
    assert read_rows(manifest.training_data_id, paths=paths) == []

    updated = replace_rows(
        manifest.training_data_id,
        [
            {
                "project_id": "project_1",
                "project_name": "Project One",
                "run_id": "run_1",
                "source_pipeline_id": "pipeline_abc",
                "x_features": {"image_count": 42},
                "selected_multipliers": {"geometry_lr_mult": 1.1},
                "selected_log_multipliers": {"geometry_lr_mult": 0.09531},
                "relative_quality_score": 0.25,
                "relative_quality_score": 0.25,
            }
        ],
        paths=paths,
    )

    assert updated.status == "ready"
    assert updated.row_count == 1
    assert updated.schema_valid is True

    rows = read_rows(manifest.training_data_id, paths=paths)
    assert len(rows) == 1
    assert rows[0].project_id == "project_1"
    assert rows[0].selected_multipliers["geometry_lr_mult"] == 1.1


def test_training_data_registry_lists_newest_first(tmp_path: Path):
    paths = WorkflowPaths(data_root=tmp_path)
    older = create_manifest(
        name="Older Data",
        source_pipeline_id="pipeline_old",
        feature_schema="mode3_exif_flight_scene_v1",
        training_data_id="training_data_old",
        paths=paths,
    )
    newer = create_manifest(
        name="Newer Data",
        source_pipeline_id="pipeline_new",
        feature_schema="mode3_exif_flight_scene_v1",
        training_data_id="training_data_new",
        paths=paths,
    )

    replace_rows(
        newer.training_data_id,
        [
            {
                "project_id": "project_2",
                "run_id": "run_2",
                "source_pipeline_id": "pipeline_new",
                "x_features": {"image_count": 1},
            }
        ],
        paths=paths,
    )

    manifests = list_manifests(paths=paths)
    assert [item.training_data_id for item in manifests] == [
        newer.training_data_id,
        older.training_data_id,
    ]

