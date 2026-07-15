"""Log-space multiplier schedules for workflow pipelines."""
from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from typing import Any

from bimba3d_backend.app.services.training_pipeline_storage import phase_run_count


def build_fixed_log_space_config(config: dict[str, Any], *, seed: int | None = None) -> dict[str, Any]:
    phases = config.get("phases") if isinstance(config.get("phases"), list) else []
    shared_config = config.get("shared_config") if isinstance(config.get("shared_config"), dict) else {}
    schedule_phase = _first_multiplier_phase(phases)
    count = phase_run_count(schedule_phase) if schedule_phase else 1
    mode = str((schedule_phase or {}).get("context_jitter_mode") or "uniform").strip().lower()
    seed_value = int(seed if seed is not None else random.SystemRandom().randint(1, 2_147_483_647))
    rng = random.Random(seed_value)

    bounds, bounds_source = _bounds_for_config(config, shared_config, mode)
    values = {
        "geometry_lr": _log_interval_values(rng, count, *bounds["geometry_lr"]),
        "appearance_lr": _log_interval_values(rng, count, *bounds["appearance_lr"]),
        "scale_lr": _log_interval_values(rng, count, *bounds["scale_lr"]),
    }
    pipeline_type = str(config.get("pipeline_type") or "").strip().lower()
    candidate_count = int(_safe_positive_float(shared_config.get("candidate_points"), 30))
    test_candidates = (
        {
            "geometry_lr_mult": _shuffled_grid_log_values(rng, candidate_count, *bounds["geometry_lr"]),
            "appearance_lr_mult": _shuffled_grid_log_values(rng, candidate_count, *bounds["appearance_lr"]),
            "densification_mult": _shuffled_grid_log_values(rng, candidate_count, *bounds["scale_lr"]),
        }
        if pipeline_type == "test"
        else {}
    )
    return {
        "pre_generated_log_multipliers": values,
        "test_candidate_log_multipliers": test_candidates,
        "multiplier_current_index": 0,
        "fixed_log_space_seed": seed_value,
        "test_candidate_seed": None,
        "test_candidate_count": candidate_count if pipeline_type == "test" else None,
        "test_candidate_generation": "grid_log_space" if pipeline_type == "test" else None,
        "fixed_log_space_generated_at": _utc_now(),
        "fixed_log_space_mode": mode,
        "fixed_log_space_method": "bounded_log_space_interval_sampling",
        "fixed_log_space_interval_count": count,
        "fixed_log_space_bounds": bounds,
        "fixed_log_space_bounds_source": bounds_source,
        "fixed_log_space_phase_number": (schedule_phase or {}).get("phase_number"),
    }


def _first_multiplier_phase(phases: list[Any]) -> dict[str, Any] | None:
    for phase in phases:
        if isinstance(phase, dict) and int(phase.get("phase_number") or 1) > 1 and bool(phase.get("context_jitter")):
            return phase
    for phase in phases:
        if isinstance(phase, dict) and int(phase.get("phase_number") or 1) > 1:
            return phase
    return phases[0] if phases and isinstance(phases[0], dict) else None


def _bounds_for_config(config: dict[str, Any], shared_config: dict[str, Any], mode: str) -> tuple[dict[str, tuple[float, float]], str]:
    model_bounds = _model_bounds_for_test_pipeline(config)
    if model_bounds is not None:
        return model_bounds, "selected_model"

    preset = _bounds_for_mode(mode)
    explicit = any(
        key in shared_config
        for key in (
            "geometry_log_multiplier_min",
            "geometry_log_multiplier_max",
            "appearance_log_multiplier_min",
            "appearance_log_multiplier_max",
            "densification_log_multiplier_min",
            "densification_log_multiplier_max",
        )
    )
    return {
        "geometry_lr": _read_bound_pair(
            shared_config,
            "geometry_log_multiplier_min",
            "geometry_log_multiplier_max",
            preset["geometry_lr"],
        ),
        "appearance_lr": _read_bound_pair(
            shared_config,
            "appearance_log_multiplier_min",
            "appearance_log_multiplier_max",
            preset["appearance_lr"],
        ),
        "scale_lr": _read_bound_pair(
            shared_config,
            "densification_log_multiplier_min",
            "densification_log_multiplier_max",
            preset["scale_lr"],
        ),
    }, "frontend_config" if explicit else "default_bounds_fallback"


def _model_bounds_for_test_pipeline(config: dict[str, Any]) -> dict[str, tuple[float, float]] | None:
    pipeline_type = str(config.get("pipeline_type") or "").strip().lower()
    if pipeline_type != "test":
        return None

    model_ids: list[str] = []
    raw_ids = config.get("source_model_ids")
    if isinstance(raw_ids, list):
        model_ids.extend(str(item).strip() for item in raw_ids if str(item or "").strip())
    if not model_ids and config.get("source_model_id"):
        model_ids.append(str(config.get("source_model_id")).strip())
    if not model_ids:
        return None

    try:
        from bimba3d_backend.app.services import workflow_model_registry

        model = workflow_model_registry.read_model(model_ids[0])
    except Exception:
        model = None
    if model is None or not isinstance(model.config, dict):
        return None

    raw_bounds = model.config.get("log_multiplier_bounds")
    if not isinstance(raw_bounds, dict) or not raw_bounds:
        return None
    return _normalise_bounds(raw_bounds)


def _normalise_bounds(bounds: dict[str, Any]) -> dict[str, tuple[float, float]]:
    defaults = _bounds_for_mode("uniform")
    out: dict[str, tuple[float, float]] = {}
    for group, default in defaults.items():
        raw = bounds.get(group)
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            lo = _safe_positive_float(raw[0], default[0])
            hi = _safe_positive_float(raw[1], default[1])
        else:
            lo, hi = default
        if hi < lo:
            lo, hi = hi, lo
        out[group] = (lo, hi)
    return out


def _bounds_for_mode(mode: str) -> dict[str, tuple[float, float]]:
    if mode == "mild":
        return {
            "geometry_lr": (0.8, 1.25),
            "appearance_lr": (0.8, 1.25),
            "scale_lr": (0.85, 1.18),
        }
    elif mode == "gaussian":
        return {
            "geometry_lr": (0.95, 1.05),
            "appearance_lr": (0.95, 1.05),
            "scale_lr": (0.95, 1.05),
        }
    return {
        "geometry_lr": (0.5, 2.0),
        "appearance_lr": (0.5, 2.0),
        "scale_lr": (0.7, 1.42),
    }


def _read_bound_pair(config: dict[str, Any], min_key: str, max_key: str, default: tuple[float, float]) -> tuple[float, float]:
    low = _safe_positive_float(config.get(min_key), default[0])
    high = _safe_positive_float(config.get(max_key), default[1])
    if high < low:
        low, high = high, low
    return low, high


def _safe_positive_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if parsed > 0 else default
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return default
        return parsed if parsed > 0 else default
    return default


def _log_interval_values(rng: random.Random, count: int, low: float, high: float) -> list[float]:
    low = max(float(low), 1e-9)
    high = max(float(high), low)
    count = max(1, int(count or 1))
    log_low = math.log(low)
    log_high = math.log(high)
    if math.isclose(log_low, log_high):
        return [round(math.exp(log_low), 8) for _ in range(count)]

    interval_width = (log_high - log_low) / float(count)
    logs = [
        rng.uniform(log_low + interval_width * index, log_low + interval_width * (index + 1))
        for index in range(count)
    ]
    rng.shuffle(logs)
    return [round(math.exp(value), 8) for value in logs]


def _balanced_log_values(count: int, low: float, high: float) -> list[float]:
    low = max(float(low), 1e-9)
    high = max(float(high), low)
    count = max(2, int(count or 30))
    log_low = math.log(low)
    log_high = math.log(high)
    if count == 1:
        return [round((log_low + log_high) / 2.0, 8)]
    step = (log_high - log_low) / float(count - 1)
    return [round(log_low + step * index, 8) for index in range(count)]


def _shuffled_grid_log_values(rng: random.Random, count: int, low: float, high: float) -> list[float]:
    values = _balanced_log_values(count, low, high)
    rng.shuffle(values)
    return values


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
