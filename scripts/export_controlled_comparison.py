from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bimba3d_backend.app.services import training_pipeline_storage, workflow_pipeline_service


RUN_TYPES = (
    "baseline",
    "model_selected",
    "time_control",
    "gaussian_control",
)
METRICS = ("psnr", "ssim", "lpips")
MATCH_CRITERIA = (*METRICS, "all")


def _number(value: Any) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _project_dir(root: Path, project_name: str) -> Path:
    for candidate in (root / project_name, root / project_name.replace(" ", "_")):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Project directory not found for {project_name}")


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _step_map(stats_dir: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in stats_dir.glob("train_step*_rank0.json"):
        raw = path.stem.removeprefix("train_step").removesuffix("_rank0")
        if raw.isdigit():
            result[int(raw) + 1] = _read_json(path, {})
    return result


def _nearest_at_or_before(values: dict[int, Any], step: int) -> tuple[Any, int | None]:
    available = [candidate for candidate in values if candidate <= step]
    if not available:
        return None, None
    matched_step = max(available)
    return values[matched_step], matched_step


def _load_run(project_dir: Path, run_id: str, run_type: str, status: str | None) -> dict[str, Any]:
    run_dir = project_dir / "runs" / run_id
    engine_dir = run_dir / "outputs" / "engines" / "gsplat"
    analytics = _read_json(run_dir / "analytics" / "run_analytics_v1.json", {})
    history = _read_json(engine_dir / "eval_history.json", [])
    if not isinstance(history, list) or not history:
        raise ValueError(f"Run {run_id} has no evaluation history")

    loss_by_step: dict[int, Any] = {}
    for key, value in (analytics.get("loss_by_step") or {}).items():
        try:
            loss_by_step[int(key)] = _number(value)
        except (TypeError, ValueError):
            continue
    train_stats = _step_map(engine_dir / "stats")
    summary_metrics = ((analytics.get("summary") or {}).get("metrics") or {})
    training_end_step = max(train_stats, default=None)
    training_loop_seconds = _number(train_stats.get(training_end_step, {}).get("ellipse_time")) if training_end_step is not None else None

    checkpoints: list[dict[str, Any]] = []
    for source in sorted((row for row in history if isinstance(row, dict)), key=lambda row: int(row.get("step") or -1)):
        step = int(source.get("step") or 0)
        loss = _number(source.get("final_loss"))
        loss_step: int | None = step if loss is not None else None
        loss_source = "eval_history_exact" if loss is not None else None
        if loss is None:
            loss, loss_step = _nearest_at_or_before(loss_by_step, step)
            loss_source = "analytics_nearest_at_or_before" if loss is not None else None

        elapsed = _number(train_stats.get(step, {}).get("ellipse_time"))
        elapsed_source = "train_stats_exact" if elapsed is not None else None
        if elapsed is None:
            elapsed = _number(source.get("elapsed_seconds"))
            elapsed_source = "eval_history_exact" if elapsed is not None else None

        checkpoints.append({
            "run_type": run_type,
            "run_id": run_id,
            "run_status": status,
            "step": step,
            "training_seconds": elapsed,
            "training_seconds_source": elapsed_source,
            "loss": loss,
            "loss_step": loss_step,
            "loss_source": loss_source,
            "psnr": _number(source.get("convergence_speed")),
            "ssim": _number(source.get("sharpness_mean")),
            "lpips": _number(source.get("lpips_mean")),
            "gaussian_count": _number(source.get("num_gaussians")),
        })
    for checkpoint in checkpoints:
        checkpoint["is_final"] = False
        checkpoint["is_last_observed"] = False
    checkpoints[-1]["is_final"] = status != "partial"
    checkpoints[-1]["is_last_observed"] = True

    return {
        "run_id": run_id,
        "status": status,
        "reported_total_seconds": _number(summary_metrics.get("total_time_seconds")),
        "training_end_step": training_end_step,
        "training_loop_seconds": training_loop_seconds,
        "checkpoints": checkpoints,
        "final": checkpoints[-1],
    }


def _quality_delta(metric: str, actual: Any, reference: Any) -> float | None:
    actual_number = _number(actual)
    reference_number = _number(reference)
    if actual_number is None or reference_number is None:
        return None
    return float(reference_number - actual_number if metric == "lpips" else actual_number - reference_number)


def _meets(metric: str, actual: Any, reference: Any) -> bool | None:
    delta = _quality_delta(metric, actual, reference)
    return None if delta is None else delta >= 0


def _match_summary(checkpoints: list[dict[str, Any]], criterion: str) -> dict[str, Any]:
    key = f"meets_model_{criterion}"
    matching = [index for index, row in enumerate(checkpoints) if row.get(key) is True]
    first = checkpoints[matching[0]] if matching else None
    sustained = next(
        (
            row
            for index, row in enumerate(checkpoints)
            if row.get(key) is True and all(later.get(key) is True for later in checkpoints[index:])
        ),
        None,
    )
    return {
        "first_match_step": first.get("step") if first else None,
        "first_match_seconds": first.get("training_seconds") if first else None,
        "sustained_match_step": sustained.get("step") if sustained else None,
        "sustained_match_seconds": sustained.get("training_seconds") if sustained else None,
    }


def _summary_fields(prefix: str, run: dict[str, Any]) -> dict[str, Any]:
    final = run["final"]
    loss = final.get("loss")
    loss_step = final.get("loss_step")
    return {
        f"{prefix}_run_id": run["run_id"],
        f"{prefix}_status": run.get("status"),
        f"{prefix}_record_kind": "partial_last_observed" if run.get("status") == "partial" else "final",
        f"{prefix}_final_step": final.get("step") if run.get("status") != "partial" else None,
        f"{prefix}_last_observed_step": final.get("step"),
        f"{prefix}_training_end_step": run.get("training_end_step"),
        f"{prefix}_loss": loss,
        f"{prefix}_loss_step": loss_step,
        f"{prefix}_loss_at_step": f"{loss:.9g}@{loss_step}" if loss is not None and loss_step is not None else None,
        f"{prefix}_training_seconds": run.get("training_loop_seconds"),
        f"{prefix}_metric_training_seconds": final.get("training_seconds"),
        f"{prefix}_reported_total_seconds": run.get("reported_total_seconds"),
        f"{prefix}_psnr": final.get("psnr"),
        f"{prefix}_ssim": final.get("ssim"),
        f"{prefix}_lpips": final.get("lpips"),
        f"{prefix}_gaussian_count": final.get("gaussian_count"),
    }


def export(pipeline_id: str, output_dir: Path) -> tuple[Path, Path]:
    stored = training_pipeline_storage.get_pipeline(pipeline_id)
    if not stored:
        raise FileNotFoundError(f"Pipeline {pipeline_id} was not found")
    pipeline = workflow_pipeline_service.normalise_pipeline_detail(stored)
    config = pipeline.get("config") or {}
    root = Path(str(config.get("pipeline_folder") or pipeline.get("pipeline_folder") or "")).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Pipeline folder was not found: {root}")

    all_runs = [row for row in pipeline.get("runs", []) if isinstance(row, dict)]
    status_by_id = {str(row.get("run_id")): row.get("status") for row in all_runs if row.get("run_id")}
    controls_by_project: dict[str, dict[str, dict[str, Any]]] = {}
    for row in all_runs:
        kind = row.get("controlled_experiment")
        if kind not in {"time_constrained_test", "gaussian_constrained_test"} or row.get("status") != "success":
            continue
        controls_by_project.setdefault(str(row.get("project_name")), {})[str(kind)] = row

    configured_controls = config.get("additional_experiments") or {}
    for project in config.get("projects") or []:
        project_name = str(project.get("name") or "")
        controls = controls_by_project.get(project_name, {})
        if not project_name or not controls or set(controls) == {"time_constrained_test", "gaussian_constrained_test"}:
            continue
        missing_kind = "gaussian_constrained_test" if "gaussian_constrained_test" not in controls else "time_constrained_test"
        phase_number = 91 if missing_kind == "gaussian_constrained_test" else 90
        project_dir = _project_dir(root, project_name)
        candidates = [
            path for path in (project_dir / "runs").glob(f"*phase{phase_number}_run*")
            if (path / "outputs" / "engines" / "gsplat" / "eval_history.json").is_file()
        ]
        if not candidates:
            continue
        partial_dir = max(candidates, key=lambda path: path.stat().st_mtime)
        template = next(iter(controls.values()))
        model_metrics = template.get("reference_model_metrics") or {}
        partial = {
            "project_name": project_name,
            "controlled_experiment": missing_kind,
            "status": "partial",
            "run_id": partial_dir.name,
            "test_model_id": template.get("test_model_id"),
            "reference_model_run_id": template.get("reference_model_run_id"),
            "baseline_run_id": template.get("baseline_run_id") or project.get("baseline_run_id"),
            "target_time_seconds": model_metrics.get("training_loop_seconds") if missing_kind == "time_constrained_test" else None,
            "target_gaussians": model_metrics.get("gaussians") if missing_kind == "gaussian_constrained_test" else None,
            "max_steps_ceiling": (configured_controls.get(missing_kind) or {}).get("max_steps_ceiling"),
        }
        controls_by_project.setdefault(project_name, {})[missing_kind] = partial
        status_by_id[partial_dir.name] = "partial"

    summary_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for project_name in sorted(controls_by_project):
        controls = controls_by_project[project_name]
        if set(controls) != {"time_constrained_test", "gaussian_constrained_test"}:
            continue
        time_control = controls["time_constrained_test"]
        gaussian_control = controls["gaussian_constrained_test"]
        model_ids = {str(time_control.get("reference_model_run_id") or ""), str(gaussian_control.get("reference_model_run_id") or "")}
        baseline_ids = {str(time_control.get("baseline_run_id") or ""), str(gaussian_control.get("baseline_run_id") or "")}
        if len(model_ids) != 1 or "" in model_ids or len(baseline_ids) != 1 or "" in baseline_ids:
            raise ValueError(f"Controls for {project_name} do not share one model and baseline reference")

        run_ids = {
            "baseline": baseline_ids.pop(),
            "model_selected": model_ids.pop(),
            "time_control": str(time_control["run_id"]),
            "gaussian_control": str(gaussian_control["run_id"]),
        }
        project_dir = _project_dir(root, project_name)
        loaded = {
            run_type: _load_run(project_dir, run_id, run_type, status_by_id.get(run_id))
            for run_type, run_id in run_ids.items()
        }
        model_final = loaded["model_selected"]["final"]
        baseline_final = loaded["baseline"]["final"]

        for run_type, run in loaded.items():
            for checkpoint in run["checkpoints"]:
                row = {
                    "pipeline_id": pipeline_id,
                    "project": project_name,
                    "selected_model_id": time_control.get("test_model_id"),
                    **checkpoint,
                }
                model_flags: list[bool | None] = []
                for metric in METRICS:
                    row[f"{metric}_vs_model"] = _quality_delta(metric, row.get(metric), model_final.get(metric))
                    row[f"{metric}_vs_baseline"] = _quality_delta(metric, row.get(metric), baseline_final.get(metric))
                    row[f"meets_model_{metric}"] = _meets(metric, row.get(metric), model_final.get(metric)) if run_type in {"time_control", "gaussian_control"} else None
                    model_flags.append(row[f"meets_model_{metric}"])
                row["meets_model_all"] = all(model_flags) if all(value is not None for value in model_flags) else None
                checkpoint.update({key: row[key] for key in row if key.startswith("meets_model_")})
                checkpoint_rows.append(row)

        summary: dict[str, Any] = {
            "pipeline_id": pipeline_id,
            "project": project_name,
            "selected_model_id": time_control.get("test_model_id"),
            "time_control_target_seconds": time_control.get("target_time_seconds"),
            "gaussian_control_target_count": gaussian_control.get("target_gaussians"),
            "gaussian_control_step_limit": gaussian_control.get("max_steps_ceiling"),
        }
        for run_type in RUN_TYPES:
            summary.update(_summary_fields(run_type, loaded[run_type]))
        for control_type in ("time_control", "gaussian_control"):
            for criterion in MATCH_CRITERIA:
                for key, value in _match_summary(loaded[control_type]["checkpoints"], criterion).items():
                    summary[f"{control_type}_{criterion}_{key}"] = value
        summary_rows.append(summary)

    if not summary_rows:
        raise ValueError("No projects have both completed Time and Gaussian controlled runs")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "compare.csv"
    checkpoint_path = output_dir / "compare_checkpoints.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with checkpoint_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checkpoint_rows[0]))
        writer.writeheader()
        writer.writerows(checkpoint_rows)
    return summary_path, checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export baseline, selected-model, and controlled-run comparisons.")
    parser.add_argument("pipeline_id")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs" / "comparison")
    args = parser.parse_args()
    for path in export(args.pipeline_id, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
