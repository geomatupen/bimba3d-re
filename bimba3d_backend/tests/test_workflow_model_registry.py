from pathlib import Path

import pytest

from bimba3d_backend.app.services.workflow_model_registry import list_models, read_model, register_model
from bimba3d_backend.app.services.workflow_paths import WorkflowPaths


def test_register_model_writes_manifest_and_index(tmp_path: Path):
    paths = WorkflowPaths(data_root=tmp_path)
    artifact = tmp_path / "ridge_model.json"
    artifact.write_text('{"model": "ridge"}', encoding="utf-8")
    metadata = tmp_path / "ridge_metadata.json"
    metadata.write_text('{"lambda": 2.0}', encoding="utf-8")

    manifest = register_model(
        model_id="model_ridge_001",
        model_name="Featurewise Ridge 001",
        model_family="featurewise_ridge_regression",
        source_training_data_id="training_data_001",
        source_pipeline_id="pipeline_001",
        artifact_path=artifact,
        metadata_path=metadata,
        training_samples=12,
        metrics={"lambda_selected": 2.0},
        config={"candidate_points": 30},
        paths=paths,
    )

    assert manifest.model_id == "model_ridge_001"
    assert manifest.training_samples == 12
    assert read_model("model_ridge_001", paths=paths).model_name == "Featurewise Ridge 001"

    models = list_models(paths=paths)
    assert [model.model_id for model in models] == ["model_ridge_001"]


def test_register_model_rejects_missing_artifact(tmp_path: Path):
    paths = WorkflowPaths(data_root=tmp_path)

    with pytest.raises(FileNotFoundError):
        register_model(
            model_id="model_missing",
            model_name="Missing",
            model_family="featurewise_mlp",
            source_training_data_id="training_data_001",
            artifact_path=tmp_path / "missing.pt",
            paths=paths,
        )


def test_register_model_upserts_existing_record(tmp_path: Path):
    paths = WorkflowPaths(data_root=tmp_path)
    artifact = tmp_path / "mlp.pt"
    artifact.write_bytes(b"model")

    register_model(
        model_id="model_mlp_001",
        model_name="MLP old",
        model_family="featurewise_mlp",
        source_training_data_id="training_data_001",
        artifact_path=artifact,
        paths=paths,
    )
    register_model(
        model_id="model_mlp_001",
        model_name="MLP updated",
        model_family="featurewise_mlp",
        source_training_data_id="training_data_001",
        artifact_path=artifact,
        training_samples=20,
        paths=paths,
    )

    models = list_models(paths=paths)
    assert len(models) == 1
    assert models[0].model_name == "MLP updated"
    assert models[0].training_samples == 20
