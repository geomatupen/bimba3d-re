"""Report-aligned workflow data contracts.

These schemas describe the backend contracts used by workflow services.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


WorkflowStage = Literal[
    "offline_data_preparation",
    "training_data",
    "model_training",
    "testing",
]

ModelFamily = Literal[
    "featurewise_ridge_regression",
    "featurewise_mlp",
    "compact_featurewise_ridge_regression",
    "compact_featurewise_mlp",
    "compact_descriptor_mlp",
]


class ArtifactReference(BaseModel):
    """Reference to a source artifact without copying its contents."""

    project_id: str | None = None
    run_id: str | None = None
    pipeline_id: str | None = None
    stage: WorkflowStage | None = None
    path: str | None = None
    artifact_version: str | None = None


class TrainingDataRow(BaseModel):
    """One normalized row consumed by thesis model training."""

    project_id: str
    project_name: str | None = None
    run_id: str
    source_pipeline_id: str
    phase: int | None = None
    is_baseline_row: bool = False
    x_features: dict[str, Any]
    selected_multipliers: dict[str, float] = Field(default_factory=dict)
    selected_log_multipliers: dict[str, float] = Field(default_factory=dict)
    relative_quality_score: float | None = None
    convergence_score: float | None = None
    score_reference_step: int | None = None
    loss_at_reference_step_run: float | None = None
    loss_at_reference_step_base: float | None = None
    source: ArtifactReference | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingDataManifest(BaseModel):
    """Manifest for a reusable prepared Training Data artifact."""

    training_data_id: str
    name: str
    status: Literal["empty", "building", "ready", "failed"]
    source_pipeline_id: str
    dataset_version: str
    feature_schema: str
    multiplier_schema: str = "geometry_appearance_densification_v1"
    row_count: int = 0
    schema_valid: bool = False
    rows_path: str
    manifest_path: str
    created_at: str
    last_built_at: str | None = None
    errors: list[str] = Field(default_factory=list)
    build_options: dict[str, Any] = Field(default_factory=dict)
    build_summary: dict[str, Any] = Field(default_factory=dict)


class FixedLogSpaceSchedule(BaseModel):
    """Seed-fixed multiplier schedule used by preparation/testing pipelines."""

    seed: int
    restart_version: int = 0
    restart_token: str | None = None
    config_hash: str | None = None
    generated_at: str
    current_index: int = 0
    values_by_group: dict[str, list[float]]


class CandidateScorePoint(BaseModel):
    """One candidate point from model-predicted score/relative-score curves."""

    candidate_index: int
    group: str
    candidate_log_multiplier: float
    candidate_multiplier: float
    predicted_relative_score: float | None = None
    predicted_score: float | None = None
    selected: bool = False


class TestingCandidateCurve(BaseModel):
    """Candidate exploration surface for one project/model prediction."""

    project_id: str
    project_name: str | None = None
    model_id: str
    model_family: ModelFamily
    candidate_count: int
    highest_point_by_group: dict[str, CandidateScorePoint] = Field(default_factory=dict)
    points: list[CandidateScorePoint] = Field(default_factory=list)


class WorkflowModelManifest(BaseModel):
    """Shared registry record for a trained thesis model."""

    model_id: str
    model_name: str
    model_family: ModelFamily
    source_training_data_id: str
    source_pipeline_id: str | None = None
    artifact_path: str
    metadata_path: str | None = None
    training_report_path: str | None = None
    trained_at: str
    training_samples: int | None = None
    model_evaluation_step: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)

