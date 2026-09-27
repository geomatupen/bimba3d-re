"""Optional baseline controls for test pipelines."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


TIME = "time_constrained_test"
GAUSSIAN = "gaussian_constrained_test"
KINDS = (TIME, GAUSSIAN)
PHASES = {TIME: 90, GAUSSIAN: 91}


def settings_signature(options: Any, kind: str = TIME) -> tuple[Any, ...]:
    if not isinstance(options, dict) or options.get("enabled") is not True:
        return (False,)
    return (
        True,
        str(options.get("model_id") or "").strip(),
        int(options.get("max_steps_ceiling", 15_000 if kind == GAUSSIAN else 12_000)),
    )


def enabled(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if str(config.get("pipeline_type") or "").lower() != "test":
        return []
    controls = config.get("additional_experiments") or {}
    return [
        (kind, controls[kind])
        for kind in KINDS
        if isinstance(controls.get(kind), dict) and controls[kind].get("enabled") is True
    ]


def validate(config: dict[str, Any]) -> None:
    selected = set(config.get("source_model_ids") or [])
    if config.get("source_model_id"):
        selected.add(config["source_model_id"])
    for kind, options in enabled(config):
        model_id = str(options.get("model_id") or "").strip()
        if model_id not in selected:
            raise ValueError(f"{kind} requires a selected test model.")
        if int(options.get("source_run_number", 1)) < 1:
            raise ValueError(f"{kind} source run number must be positive.")
        ceiling = int(options.get("max_steps_ceiling", 15_000 if kind == GAUSSIAN else 12_000))
        base_steps = int((config.get("shared_config") or {}).get("max_steps") or 5_000)
        minimum = base_steps + (1_000 if kind == GAUSSIAN else 1)
        if ceiling < minimum or ceiling > 100_000:
            raise ValueError(f"{kind} maximum steps must be between {minimum} and 100000.")


def source_summary(project_dir: Path, run_id: str) -> dict[str, Any]:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("Source run ID must be a run folder name.")
    path = project_dir / "runs" / run_id / "analytics" / "run_analytics_v1.json"
    if not path.is_file():
        raise ValueError(f"Source run {run_id} has no analytics.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}
    metrics = summary.get("metrics") or {}
    status = str(summary.get("status") or "").lower()
    hard_cap = bool(summary.get("gaussian_cap_reached")) or str(summary.get("reason") or "").lower() == "gaussian_hard_cap_reached"
    if status not in {"completed", "success", "done"} and not hard_cap:
        raise ValueError(f"Source run {run_id} did not complete or reach the Gaussian hard cap.")
    metadata_path = project_dir / "runs" / run_id / "outputs" / "engines" / "gsplat" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    eval_path = project_dir / "runs" / run_id / "outputs" / "engines" / "gsplat" / "eval_history.json"
    eval_history = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.is_file() else []
    final_eval = max(
        (row for row in eval_history if isinstance(row, dict)),
        key=lambda row: int(row.get("step") or -1),
        default={},
    ) if isinstance(eval_history, list) else {}
    training_time = (metadata.get("training_time") or {}).get("total_elapsed_seconds")
    total_seconds = training_time if isinstance(training_time, (int, float)) else metrics.get("total_time_seconds")
    stats_dir = project_dir / "runs" / run_id / "outputs" / "engines" / "gsplat" / "stats"
    eval_stats = sorted(
        (item for item in stats_dir.glob("val_step*.json") if item.stem.removeprefix("val_step").isdigit()),
        key=lambda item: int(item.stem.removeprefix("val_step")),
    )
    if eval_stats:
        final_step = int(eval_stats[-1].stem.removeprefix("val_step")) + 1
        if final_step > int(final_eval.get("step") or -1):
            try:
                latest = json.loads(eval_stats[-1].read_text(encoding="utf-8"))
                final_eval = {
                    "step": final_step,
                    "convergence_speed": latest.get("psnr"),
                    "sharpness_mean": latest.get("ssim"),
                    "lpips_mean": latest.get("lpips"),
                    "num_gaussians": latest.get("num_GS"),
                }
            except (OSError, ValueError, AttributeError):
                pass
    train_stats = sorted(
        stats_dir.glob("train_step*_rank0.json"),
        key=lambda item: int(item.stem.removeprefix("train_step").removesuffix("_rank0"))
        if item.stem.removeprefix("train_step").removesuffix("_rank0").isdigit() else -1,
    )
    loop_seconds = None
    if train_stats:
        try:
            value = json.loads(train_stats[-1].read_text(encoding="utf-8")).get("ellipse_time")
            if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
                loop_seconds = float(value)
        except (OSError, ValueError, AttributeError):
            pass
    return {
        "hard_cap": hard_cap,
        "time_seconds": total_seconds,
        "reported_total_seconds": metrics.get("total_time_seconds"),
        "training_loop_seconds": loop_seconds if loop_seconds is not None else total_seconds,
        "training_loop_time_source": "train_step_stats" if loop_seconds is not None else "total_time_fallback",
        "gaussians": (metadata.get("final_gaussian_count") or summary.get("gaussian_cap_count")) if hard_cap else (final_eval.get("num_gaussians") or metadata.get("final_gaussian_count") or summary.get("final_gaussian_count") or metrics.get("num_gaussians")),
        "gaussian_cap_freeze_applied": bool(metadata.get("gaussian_cap_freeze_applied") or summary.get("gaussian_cap_freeze_applied")),
        "gaussian_cap_step": metadata.get("gaussian_cap_step") or summary.get("gaussian_cap_step"),
        "step": final_eval.get("step"),
        "psnr": final_eval.get("convergence_speed"),
        "ssim": final_eval.get("sharpness_mean"),
        "lpips": final_eval.get("lpips_mean"),
    }
