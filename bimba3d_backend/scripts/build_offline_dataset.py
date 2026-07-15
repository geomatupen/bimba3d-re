#!/usr/bin/env python3
"""Build offline training dataset for contextual-continuous learner.

Scans a pipeline project folder (or a list of them) and extracts one
training example per completed run from run_analytics_v1.json.

Each row contains:
  - project_id / project_name
  - run_id
  - x_features  (raw feature dict from ai.input_mode_learning.feature_details)
  - selected_multipliers  (3 group multipliers that were applied)
  - selected_log_multipliers  (log-space, what the ridge model predicted)
  - relative_quality_score    (R_quality = s_run - s_base, or s_run if no baseline)
  - convergence_score  (R_conv  = AUC_baseline - AUC_run, or 0.0 if unavailable)
  - relative_score     (combined score used by online learner)
  - auc_loss_run      (AUC of training loss to the configured score reference step)
  - auc_loss_base     (AUC of baseline training loss, if available)
  - s_run / s_best / s_end  (quality scores)
  - max_steps
  - is_baseline_run   (True when run_jitter_only=True or is_baseline_row=True)

Usage:
    python -m bimba3d_backend.scripts.build_offline_dataset \\
        --train-dir  "E:\\Thesis\\PipelineProjects\\New_First_Training_Pipeline" \\
        --out        "bimba3d_backend/data/_offline_training/offline_dataset_v1.json"
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _safe_float(v: Any) -> float | None:
    if isinstance(v, (int, float)) and not math.isnan(float(v)):
        return float(v)
    return None


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Failed reading %s: %s", path, exc)
        return None


GROUP_MULTIPLIERS_MAP: dict[str, list[str]] = {
    "geometry_lr_mult": ["position_lr_init_mult", "scaling_lr_mult", "rotation_lr_mult", "geometry_lr_multiplier"],
    "appearance_lr_mult": ["feature_lr_mult", "opacity_lr_mult", "lambda_dssim_mult", "appearance_lr_multiplier"],
    "densification_mult": ["densify_grad_threshold_mult", "opacity_threshold_mult", "scale_lr_multiplier", "densification_multiplier"],
}


def _float_if_valid(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _pick_group_value(values: dict[str, Any], group_key: str) -> float | None:
    direct = _float_if_valid(values.get(group_key))
    if direct is not None and direct > 0:
        return direct

    candidates: list[float] = []
    for key in GROUP_MULTIPLIERS_MAP.get(group_key, []):
        candidate = _float_if_valid(values.get(key))
        if candidate is not None and candidate > 0:
            candidates.append(candidate)
    if not candidates:
        return None
    return candidates[0]


def _extract_group_multipliers(iml: dict[str, Any], learning_param_rows: Any) -> dict[str, float]:
    source_dicts: list[dict[str, Any]] = []
    if isinstance(learning_param_rows, list) and learning_param_rows:
        final_values: dict[str, float] = {}
        selected_values: dict[str, float] = {}
        for param_row in learning_param_rows:
            if not isinstance(param_row, dict):
                continue
            key = param_row.get("key")
            if not key:
                continue
            mult_key = f"{key}_mult"
            selected_raw = _float_if_valid(param_row.get("selected_multiplier_raw"))
            selected = _float_if_valid(param_row.get("selected_multiplier"))
            final_mult = _float_if_valid(param_row.get("final_multiplier"))
            if selected_raw is not None and selected_raw > 0:
                selected_values[mult_key] = selected_raw
            elif selected is not None and selected > 0:
                selected_values[mult_key] = selected
            if final_mult is not None and final_mult > 0:
                final_values[mult_key] = final_mult
        if final_values:
            source_dicts.append(final_values)
        if selected_values:
            source_dicts.append(selected_values)

    for key in ("selected_multipliers", "selected_multipliers_raw", "yhat_scores"):
        candidate = iml.get(key)
        if isinstance(candidate, dict) and candidate:
            source_dicts.append(candidate)

    grouped: dict[str, float] = {}
    for group_key in GROUP_MULTIPLIERS_MAP:
        for source in source_dicts:
            value = _pick_group_value(source, group_key)
            if value is not None:
                grouped[group_key] = value
                break
    return grouped


def _extract_parameter_multipliers(iml: dict[str, Any], learning_param_rows: Any) -> dict[str, float]:
    parameter_values: dict[str, float] = {}
    if isinstance(learning_param_rows, list) and learning_param_rows:
        for param_row in learning_param_rows:
            if not isinstance(param_row, dict):
                continue
            key = param_row.get("key")
            if not key:
                continue
            mult_key = f"{key}_mult"
            selected_raw = _float_if_valid(param_row.get("selected_multiplier_raw"))
            selected = _float_if_valid(param_row.get("selected_multiplier"))
            final_mult = _float_if_valid(param_row.get("final_multiplier"))
            if final_mult is not None and final_mult > 0:
                parameter_values[mult_key] = final_mult
            elif selected_raw is not None and selected_raw > 0:
                parameter_values[mult_key] = selected_raw
            elif selected is not None and selected > 0:
                parameter_values[mult_key] = selected

    if parameter_values:
        return parameter_values

    for key in ("selected_multipliers", "selected_multipliers_raw", "yhat_scores"):
        candidate = iml.get(key)
        if isinstance(candidate, dict) and candidate:
            return {
                item_key: float(item_value)
                for item_key, item_value in candidate.items()
                if _float_if_valid(item_value) is not None
            }

    return {}


def _auc_trapezoid(loss_by_step: dict[str, Any], max_step: int) -> float | None:
    """Trapezoid-rule AUC of training loss over [0, max_step]."""
    points: list[tuple[int, float]] = []
    for k, v in (loss_by_step or {}).items():
        try:
            s, lv = int(k), float(v)
        except Exception:
            continue
        if 0 <= s <= max_step:
            points.append((s, lv))
    if not points:
        return None
    points.sort()
    if points[0][0] > 0:
        points.insert(0, (0, points[0][1]))
    if points[-1][0] < max_step:
        points.append((max_step, points[-1][1]))
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x1 != x0:
            auc += (x1 - x0) * (y0 + y1) / 2.0
    return float(auc)


def _reference_step(baseline_comparison: dict[str, Any], max_steps: int | None) -> int:
    for key in ("score_reference_step", "auc_max_step"):
        value = baseline_comparison.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and int(value) > 0:
            return int(value)
    if isinstance(max_steps, int) and max_steps > 0:
        return int(max_steps)
    for key in baseline_comparison:
        if not isinstance(key, str) or not key.startswith("loss_at_") or not key.endswith("_run"):
            continue
        try:
            step = int(key[len("loss_at_") : -len("_run")])
        except Exception:
            continue
        if step > 0:
            return step
    return 1


def _baseline_loss_value(baseline_comparison: dict[str, Any], side: str, reference_step: int) -> float | None:
    value = _safe_float(baseline_comparison.get(f"loss_at_{reference_step}_{side}"))
    if value is not None:
        return value
    for key, raw in baseline_comparison.items():
        if isinstance(key, str) and key.startswith("loss_at_") and key.endswith(f"_{side}"):
            value = _safe_float(raw)
            if value is not None:
                return value
    return None


# â”€â”€ core extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _extract_row(
    analytics: dict[str, Any],
    project_name: str,
    run_dir: Path,
) -> dict[str, Any] | None:
    """Extract one training row from a run_analytics_v1.json dict.

    Returns None if the run lacks the minimum data needed.
    """
    project_id = str(analytics.get("project_id") or project_name)
    run_id = str(analytics.get("run_id") or run_dir.name)
    ai = analytics.get("ai") if isinstance(analytics.get("ai"), dict) else {}
    iml = ai.get("input_mode_learning") if isinstance(ai.get("input_mode_learning"), dict) else {}

    # â”€â”€ features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    x_features: dict[str, Any] = {}
    for key in ("feature_details", "features"):
        candidate = iml.get(key)
        if isinstance(candidate, dict) and candidate:
            x_features = candidate
            break
    # fall back to transition.x
    if not x_features:
        transition = iml.get("transition") if isinstance(iml.get("transition"), dict) else {}
        x = transition.get("x")
        if isinstance(x, dict) and x:
            x_features = x

    if not x_features:
        logger.debug("run %s/%s: no x_features â€” skipping", project_name, run_id)
        return None

    # Strip null and indicator features (e.g. iso_missing) from offline training vectors.
    # These flags are metadata for extraction quality and are not training inputs.
    x_features = {
        k: v
        for k, v in x_features.items()
        if v is not None
        and isinstance(v, (int, float))
        and not str(k).endswith("_missing")
    }

    # â”€â”€ multipliers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Store the report-level action groups, not every individual parameter.
    learning_param_rows = iml.get("learning_param_rows")
    selected_multipliers: dict[str, float] = _extract_group_multipliers(iml, learning_param_rows)
    selected_parameter_multipliers: dict[str, float] = _extract_parameter_multipliers(iml, learning_param_rows)

    selected_log_multipliers: dict[str, float] = {}
    candidate_log = iml.get("selected_log_multipliers")
    if isinstance(candidate_log, dict) and candidate_log:
        for group_key in GROUP_MULTIPLIERS_MAP:
            value = _pick_group_value(candidate_log, group_key)
            if value is not None:
                selected_log_multipliers[group_key] = value
    # derive log multipliers from linear multipliers when absent
    for k, v in selected_multipliers.items():
        if k not in selected_log_multipliers:
            try:
                selected_log_multipliers[k] = math.log(max(v, 1e-9))
            except Exception:
                pass

    # â”€â”€ score signals â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    relative_score = _safe_float(iml.get("relative_score"))
    s_best = _safe_float(iml.get("s_best"))
    s_end = _safe_float(iml.get("s_end"))
    s_run = _safe_float(iml.get("s_run"))

    baseline_comparison: dict[str, Any] = {}
    # Try transition.baseline_comparison
    transition = iml.get("transition") if isinstance(iml.get("transition"), dict) else {}
    bc = transition.get("baseline_comparison")
    if isinstance(bc, dict):
        baseline_comparison = bc
    # Also try learn_snapshot.baseline_comparison (insights path)
    insights = ai.get("input_mode_insights") if isinstance(ai.get("input_mode_insights"), dict) else {}
    ls = insights.get("learn_snapshot") if isinstance(insights.get("learn_snapshot"), dict) else {}
    bc2 = ls.get("baseline_comparison")
    if isinstance(bc2, dict) and not baseline_comparison:
        baseline_comparison = bc2

    r_quality = _safe_float(baseline_comparison.get("r_quality")) if baseline_comparison else None
    r_convergence = _safe_float(baseline_comparison.get("r_convergence")) if baseline_comparison else None
    auc_loss_run = _safe_float(baseline_comparison.get("auc_loss_run")) if baseline_comparison else None
    auc_loss_base = _safe_float(baseline_comparison.get("auc_loss_base")) if baseline_comparison else None

    summary = analytics.get("summary") if isinstance(analytics.get("summary"), dict) else {}
    major_params = summary.get("major_params") if isinstance(summary.get("major_params"), dict) else {}
    max_steps = major_params.get("max_steps")
    if not isinstance(max_steps, (int, float)):
        max_steps = None
    reference_step = _reference_step(baseline_comparison, int(max_steps) if max_steps is not None else None)
    loss_at_reference_step_run = _baseline_loss_value(baseline_comparison, "run", reference_step) if baseline_comparison else None
    loss_at_reference_step_base = _baseline_loss_value(baseline_comparison, "base", reference_step) if baseline_comparison else None

    # Recompute AUC from persisted loss_by_step when available (more accurate).
    raw_lbs = analytics.get("loss_by_step")
    if isinstance(raw_lbs, dict) and raw_lbs:
        auc_from_file = _auc_trapezoid(raw_lbs, max_step=reference_step)
        if auc_from_file is not None:
            auc_loss_run = auc_from_file

    # Fallback: try summary.log_loss_series
    if auc_loss_run is None:
        summary = analytics.get("summary") if isinstance(analytics.get("summary"), dict) else {}
        lls = summary.get("log_loss_series")
        if isinstance(lls, list) and lls:
            lbs_from_series: dict[str, Any] = {str(row["step"]): row["loss"] for row in lls if isinstance(row, dict) and "step" in row and "loss" in row}
            auc_loss_run = _auc_trapezoid(lbs_from_series, max_step=reference_step)

    # Fall back score values
    if r_quality is None:
        r_quality = relative_score if relative_score is not None else 0.0
    if r_convergence is None:
        if loss_at_reference_step_run is not None and loss_at_reference_step_base is not None:
            r_convergence = loss_at_reference_step_base - loss_at_reference_step_run
        elif auc_loss_run is not None and auc_loss_base is not None:
            r_convergence = auc_loss_base - auc_loss_run
        else:
            r_convergence = 0.0

    # â”€â”€ is_baseline_run detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Phase 1 runs are baseline runs (jitter-only, no multiplier learning).
    # learning_param_rows lives under input_mode_learning in normal analytics payloads.
    # Fall back to legacy top-level ai.learning_param_rows if present.
    learning_param_rows = iml.get("learning_param_rows")
    if not isinstance(learning_param_rows, list) or not learning_param_rows:
        learning_param_rows = ai.get("learning_param_rows")
    phase = None
    is_baseline_run = False
    
    if isinstance(learning_param_rows, list) and learning_param_rows:
        first_row = learning_param_rows[0] if isinstance(learning_param_rows[0], dict) else {}
        phase = first_row.get("phase")
        # Phase 1 is baseline (jitter-only, no multiplier changes)
        is_baseline_run = phase == 1 or first_row.get("run_jitter_only", False)
    else:
        # Fallback: check for run_jitter_only flag
        is_baseline_run = bool(iml.get("run_jitter_only", False))

    # â”€â”€ max_steps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    return {
        "project_id": project_id,
        "project_name": project_name,
        "run_id": run_id,
        "phase": phase,
        "x_features": x_features,
        "selected_multipliers": selected_multipliers,
        "selected_parameter_multipliers": selected_parameter_multipliers,
        "selected_log_multipliers": selected_log_multipliers,
        "relative_score": float(relative_score) if relative_score is not None else 0.0,
        "relative_quality_score": float(r_quality),
        "convergence_score": float(r_convergence),
        "auc_loss_run": float(auc_loss_run) if auc_loss_run is not None else None,
        "auc_loss_base": float(auc_loss_base) if auc_loss_base is not None else None,
        "score_reference_step": int(reference_step),
        "loss_at_reference_step_run": float(loss_at_reference_step_run) if loss_at_reference_step_run is not None else None,
        "loss_at_reference_step_base": float(loss_at_reference_step_base) if loss_at_reference_step_base is not None else None,
        "s_best": float(s_best) if s_best is not None else None,
        "s_end": float(s_end) if s_end is not None else None,
        "s_run": float(s_run) if s_run is not None else None,
        "max_steps": int(max_steps) if max_steps is not None else None,
        "is_baseline_run": is_baseline_run,
    }


# â”€â”€ directory scanning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def iter_analytics_files(train_dir: Path):
    """Yield (project_name, run_dir, analytics_path) for every run in train_dir.

    Supports two layouts:
      Layout A (pipeline):  <train_dir>/<ProjectName>/runs/<run_id>/analytics/run_analytics_v1.json
      Layout B (flat runs): <train_dir>/runs/<run_id>/analytics/run_analytics_v1.json
    """
    # Layout A
    for project_dir in sorted(train_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        runs_dir = project_dir / "runs"
        if not runs_dir.exists():
            continue
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            analytics_path = run_dir / "analytics" / "run_analytics_v1.json"
            if analytics_path.exists():
                yield project_dir.name, run_dir, analytics_path

    # Layout B (fallback: train_dir itself has a runs/ subfolder)
    runs_direct = train_dir / "runs"
    if runs_direct.exists():
        for run_dir in sorted(runs_direct.iterdir()):
            if not run_dir.is_dir():
                continue
            analytics_path = run_dir / "analytics" / "run_analytics_v1.json"
            if analytics_path.exists():
                # Try to read project_name from the analytics JSON itself
                yield train_dir.name, run_dir, analytics_path


def build_dataset(train_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()

    for train_dir in train_dirs:
        if not train_dir.exists():
            logger.warning("train_dir not found: %s", train_dir)
            continue
        for project_name, run_dir, analytics_path in iter_analytics_files(train_dir):
            analytics = _read_json(analytics_path)
            if not isinstance(analytics, dict):
                continue
            row = _extract_row(analytics, project_name, run_dir)
            if row is None:
                continue
            dedup_key = f"{row['project_id']}::{row['run_id']}"
            if dedup_key in seen_run_ids:
                continue
            seen_run_ids.add(dedup_key)

            rows.append(row)

    return rows


# â”€â”€ main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Build offline training dataset for contextual-continuous learner",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more pipeline project root directories to scan for run analytics",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("bimba3d_backend/data/_offline_training/offline_dataset_v1.json"),
        help="Output path for the dataset JSON",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    rows = build_dataset(args.train_dir)

    if not rows:
        logger.error("No valid training rows found â€” check --train-dir paths")
        return 1

    # Summary statistics
    projects = sorted({r["project_name"] for r in rows})
    baseline_count = sum(1 for r in rows if r["is_baseline_run"])
    non_baseline_count = len(rows) - baseline_count

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "offline_dataset_v1",
        "version": 1,
        "total_rows": len(rows),
        "project_count": len(projects),
        "projects": projects,
        "baseline_rows": baseline_count,
        "non_baseline_rows": non_baseline_count,
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info("offline_dataset: rows=%d projects=%d", len(rows), len(projects))
    logger.info("  baseline_rows=%d  non_baseline_rows=%d", baseline_count, non_baseline_count)
    logger.info("  projects: %s", projects)
    logger.info("  written to: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

