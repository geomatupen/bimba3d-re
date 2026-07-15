"""Relative quality score helpers for thesis data preparation.

The report scoring term compares a predicted/hyperparameter run against its
baseline run and keeps the configured reference-step convergence term as a
separate inspectable value. This module is the report-named import path used
by the featurewise model code.
"""
from __future__ import annotations

from typing import Any

from bimba3d_backend.worker.ai_input_modes.relative_quality_score_runtime import compute_score_summary as _runtime_score_summary


def compute_relative_quality_summary(
    *,
    eval_rows: list[dict[str, Any]],
    baseline_eval_history: list[dict[str, Any]] | None,
    loss_by_step: dict[int, float],
    elapsed_by_step: dict[int, float],
    t_eval_best: int,
    t_end: int,
    prefer_quality_best: bool = False,
    include_breakdown: bool = False,
    baseline_loss_by_step_override: dict[int, float] | None = None,
    score_reference_step: int | None = None,
) -> dict[str, Any]:
    """Return report scoring summary for one run against its baseline."""
    return _runtime_score_summary(
        eval_rows=eval_rows,
        baseline_eval_history=baseline_eval_history,
        loss_by_step=loss_by_step,
        elapsed_by_step=elapsed_by_step,
        t_eval_best=t_eval_best,
        t_end=t_end,
        prefer_quality_best=prefer_quality_best,
        include_breakdown=include_breakdown,
        baseline_loss_by_step_override=baseline_loss_by_step_override,
        score_reference_step=score_reference_step,
    )


__all__ = ["compute_relative_quality_summary"]

