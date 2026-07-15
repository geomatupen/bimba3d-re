"""Canonical filesystem layout for thesis workflow artifacts.

The current backend still stores many workflow artifacts beside projects or
inside pipeline folders. New backend code should use this module so each
artifact type has one owner and other pages/services reference it by id.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bimba3d_backend.app.config import DATA_ROOT


@dataclass(frozen=True)
class WorkflowPaths:
    """Resolved directories for report-aligned workflow artifacts."""

    data_root: Path = DATA_ROOT

    @property
    def workflow_root(self) -> Path:
        return self.data_root / "workflow"

    @property
    def offline_data_preparation_root(self) -> Path:
        return self.workflow_root / "offline_data_preparation"

    @property
    def training_data_root(self) -> Path:
        return self.workflow_root / "training_data"

    @property
    def model_training_root(self) -> Path:
        return self.workflow_root / "model_training"

    @property
    def testing_root(self) -> Path:
        return self.workflow_root / "testing"

    @property
    def models_root(self) -> Path:
        return self.workflow_root / "models"

    def offline_data_preparation_dir(self, pipeline_id: str) -> Path:
        return self.offline_data_preparation_root / _clean_id(pipeline_id)

    def training_data_dir(self, training_data_id: str) -> Path:
        return self.training_data_root / _clean_id(training_data_id)

    def model_training_dir(self, pipeline_id: str) -> Path:
        return self.model_training_root / _clean_id(pipeline_id)

    def testing_dir(self, pipeline_id: str) -> Path:
        return self.testing_root / _clean_id(pipeline_id)

    def model_dir(self, model_id: str) -> Path:
        return self.models_root / _clean_id(model_id)

    def ensure_roots(self) -> None:
        for path in (
            self.workflow_root,
            self.offline_data_preparation_root,
            self.training_data_root,
            self.model_training_root,
            self.testing_root,
            self.models_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _clean_id(value: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned or any(part in cleaned for part in ("/", "\\", "..")):
        raise ValueError(f"Invalid workflow artifact id: {value!r}")
    return cleaned


DEFAULT_WORKFLOW_PATHS = WorkflowPaths()
