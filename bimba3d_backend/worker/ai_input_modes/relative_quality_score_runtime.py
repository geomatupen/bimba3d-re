from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bimba3d_backend.app.services.relative_quality_scoring import raw_quality_score

from .common import apply_preset_updates, clamp_float

PRESETS = ["conservative", "balanced", "geometry_fast", "appearance_fast"]
HEURISTIC_PRESET_BONUS = 0.002


def _selector_dir(project_dir: Path) -> Path:
    return project_dir / "models" / "input_mode_selector"


def _selector_path(project_dir: Path) -> Path:
    return _selector_dir(project_dir) / "selector_model.json"


def _load_model(project_dir: Path) -> dict[str, Any]:
    path = _selector_path(project_dir)
    if not path.exists():
        raise FileNotFoundError(f"Selector model not found: {path}. Cannot fall back to default - model must be trained first.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception as e:
        raise ValueError(f"Selector model at {path} is invalid or corrupted: {e}")


def _save_model(project_dir: Path, model: dict[str, Any]) -> None:
    out_dir = _selector_dir(project_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _selector_path(project_dir)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(model, indent=2), encoding="utf-8")
    tmp.replace(path)


def _mode_entry(model: dict[str, Any], mode: str) -> dict[str, Any]:
    modes = model.setdefault("modes", {})
    entry = modes.setdefault(
        mode,
        {
            "bias": {k: 0.0 for k in PRESETS},
            "runs": 0,
            "score_mean": 0.0,
            "last": {},
        },
    )
    if not isinstance(entry.get("bias"), dict):
        entry["bias"] = {k: 0.0 for k in PRESETS}
    for p in PRESETS:
        if p not in entry["bias"]:
            entry["bias"][p] = 0.0
    return entry


def select_preset(
    *,
    project_dir: Path,
    mode: str,
    heuristic_preset: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    model = _load_model(project_dir)
    entry = _mode_entry(model, mode)
    bias = entry.get("bias", {})

    scores: dict[str, float] = {}
    for preset in PRESETS:
        base = float(bias.get(preset, 0.0) or 0.0)
        if preset == heuristic_preset:
            base += HEURISTIC_PRESET_BONUS
        scores[preset] = base

    selected_preset = max(PRESETS, key=lambda p: scores.get(p, 0.0))

    updates = apply_preset_updates(params, selected_preset)

    return {
        "selected_preset": selected_preset,
        "yhat_scores": scores,
        "updates": updates,
    }


def _normalize_series(values: list[float], invert: bool = False) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        out = [0.5 for _ in values]
    else:
        out = [(v - lo) / (hi - lo) for v in values]
    if invert:
        out = [1.0 - v for v in out]
    return out


def _step_value_with_neighbors(values: dict[int, float], step: int) -> float | None:
    for candidate in (step, step + 1, step - 1):
        value = values.get(int(candidate))
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _rows_by_step(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {
        int(row["step"]): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("step"), (int, float))
    }


def compute_score_summary(
    *,
    eval_rows: list[dict[str, Any]],
    baseline_eval_history: list[dict[str, Any]] | None,
    loss_by_step: dict[int, float] | None,
    elapsed_by_step: dict[int, float] | None,
    t_eval_best: int,
    t_end: int,
    prefer_quality_best: bool = False,
    include_breakdown: bool = False,
    baseline_loss_by_step_override: dict[int, float] | None = None,
    score_reference_step: int | None = None,
) -> dict[str, Any]:
    def _auc_trapezoid(series: dict[int, float], max_step: int) -> float | None:
        points: list[tuple[int, float]] = []
        for raw_step, raw_loss in (series or {}).items():
            if isinstance(raw_step, (int, float)) and isinstance(raw_loss, (int, float)):
                step_int = int(raw_step)
                if 0 <= step_int <= max_step:
                    points.append((step_int, float(raw_loss)))

        if not points:
            return None

        points.sort(key=lambda p: p[0])

        # Ensure we cover the full [0, max_step] window.
        if points[0][0] > 0:
            points.insert(0, (0, points[0][1]))
        if points[-1][0] < max_step:
            points.append((max_step, points[-1][1]))

        auc = 0.0
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x1 == x0:
                continue
            auc += (x1 - x0) * (y0 + y1) / 2.0
        return float(auc)

    loss_by_step_num = {
        int(k): float(v)
        for k, v in (loss_by_step or {}).items()
        if isinstance(k, int) and isinstance(v, (int, float))
    }
    elapsed_by_step_num = {
        int(k): float(v)
        for k, v in (elapsed_by_step or {}).items()
        if isinstance(k, int) and isinstance(v, (int, float))
    }
    eval_loss_by_step = {
        int(r.get("step")): float(r.get("final_loss"))
        for r in eval_rows
        if isinstance(r, dict)
        and isinstance(r.get("step"), (int, float))
        and isinstance(r.get("final_loss"), (int, float))
    }
    eval_elapsed_by_step = {
        int(r.get("step")): float(r.get("elapsed_seconds"))
        for r in eval_rows
        if isinstance(r, dict)
        and isinstance(r.get("step"), (int, float))
        and isinstance(r.get("elapsed_seconds"), (int, float))
    }

    loss_vals: list[float] = []
    elapsed_vals: list[float] = []
    for row in eval_rows:
        step = int(row.get("step", 0) or 0)
        loss_value = _step_value_with_neighbors(loss_by_step_num, step)
        if loss_value is None:
            loss_value = _step_value_with_neighbors(eval_loss_by_step, step)
        elapsed_value = _step_value_with_neighbors(elapsed_by_step_num, step)
        if elapsed_value is None:
            elapsed_value = _step_value_with_neighbors(eval_elapsed_by_step, step)
        loss_vals.append(float(loss_value) if isinstance(loss_value, (int, float)) else 0.0)
        elapsed_vals.append(float(elapsed_value) if isinstance(elapsed_value, (int, float)) else 0.0)

    baseline_rows: list[dict[str, Any]] = []
    if isinstance(baseline_eval_history, list):
        baseline_rows = [
            row
            for row in baseline_eval_history
            if isinstance(row, dict) and isinstance(row.get("step"), (int, float))
        ]
        baseline_rows.sort(key=lambda r: int(r.get("step", 0)))

    baseline_elapsed_vals = [float(r.get("elapsed_seconds") or 0.0) for r in baseline_rows] if baseline_rows else []

    b_steps: list[int] = []
    b_loss_vals: list[float] = []
    b_elapsed_vals: list[float] = []
    baseline_loss_by_step: dict[int, float] = {}
    if baseline_rows:
        b_steps = [int(r["step"]) for r in baseline_rows]
        # Use dense loss_by_step override when available (persisted from baseline run analytics).
        # This gives AUC_baseline the full training trajectory instead of just eval-step losses.
        if baseline_loss_by_step_override:
            baseline_loss_by_step = {
                int(k): float(v)
                for k, v in baseline_loss_by_step_override.items()
                if isinstance(k, (int, float)) and isinstance(v, (int, float))
            }
        else:
            baseline_loss_by_step = {
                int(r.get("step")): float(r.get("final_loss"))
                for r in baseline_rows
                if isinstance(r.get("step"), (int, float)) and isinstance(r.get("final_loss"), (int, float))
            }
        baseline_elapsed_by_step = {
            int(r.get("step")): float(r.get("elapsed_seconds"))
            for r in baseline_rows
            if isinstance(r.get("step"), (int, float)) and isinstance(r.get("elapsed_seconds"), (int, float))
        }
        b_loss_vals = [
            float(_step_value_with_neighbors(baseline_loss_by_step, int(r.get("step", 0) or 0)) or 0.0)
            for r in baseline_rows
        ]
        b_elapsed_vals = [
            float(_step_value_with_neighbors(baseline_elapsed_by_step, int(r.get("step", 0) or 0)) or 0.0)
            for r in baseline_rows
        ]

    if baseline_rows:
        joint_loss_norm = _normalize_series(loss_vals + b_loss_vals, invert=True)
        split = len(loss_vals)
        loss_norm = joint_loss_norm[:split]
        b_loss_norm = joint_loss_norm[split:]
    else:
        loss_norm = _normalize_series(loss_vals, invert=True)
        b_loss_norm = []

    baseline_time_ref = max(baseline_elapsed_vals) if baseline_elapsed_vals else 0.0
    time_ref = baseline_time_ref if baseline_time_ref > 0.0 else (max(elapsed_vals) if elapsed_vals else 1.0)
    time_ref = max(time_ref, 1e-6)

    by_step: dict[int, dict[str, float]] = {}
    for idx, row in enumerate(eval_rows):
        step = int(row["step"])
        q = raw_quality_score(row)
        l_score = loss_norm[idx]
        entry = {
            "l": l_score,
            "q": q,
            "t": 0.0,
            "s": q,
        }
        if include_breakdown:
            entry["elapsed"] = elapsed_vals[idx]
        by_step[step] = entry

    if prefer_quality_best and by_step:
        quality_scores_run = {step: data["q"] for step, data in by_step.items()}
        run_best_step = int(max(quality_scores_run.keys(), key=lambda s: quality_scores_run[s]))
    else:
        run_best_step = int(t_eval_best)

    baseline_comparison: dict[str, Any] | None = None
    relative_score = 0.0
    explicit_reference_step = score_reference_step is not None
    reference_step = int(score_reference_step or t_end)
    reference_step = max(1, reference_step)
    loss_run_key = f"loss_at_{reference_step}_run"
    loss_base_key = f"loss_at_{reference_step}_base"

    # AUC is still computed for display in the learning table (kept as informational metric).
    auc_loss_run = _auc_trapezoid(loss_by_step_num, max_step=reference_step)
    auc_loss_base = _auc_trapezoid(baseline_loss_by_step, max_step=reference_step) if baseline_rows else None

    loss_at_reference_run = _step_value_with_neighbors(loss_by_step_num, reference_step)
    loss_at_reference_base = _step_value_with_neighbors(baseline_loss_by_step, reference_step) if baseline_rows else None

    if loss_at_reference_run is not None and loss_at_reference_base is not None:
        r_convergence = float(loss_at_reference_base) - float(loss_at_reference_run)
    else:
        r_convergence = 0.0

    if baseline_rows:
        b_by_step: dict[int, dict[str, float]] = {}
        baseline_quality_scores: dict[int, float] = {}
        for idx, row in enumerate(baseline_rows):
            step = int(row["step"])
            q = raw_quality_score(row)
            l_score = b_loss_norm[idx]
            entry = {
                "l": l_score,
                "q": q,
                "t": 0.0,
                "s": q,
            }
            if include_breakdown:
                entry["elapsed"] = b_elapsed_vals[idx]
            b_by_step[step] = entry
            baseline_quality_scores[step] = q

        t_base_best = int(max(baseline_quality_scores.keys(), key=lambda s: baseline_quality_scores[s]))
        t_base_end = int(max(b_steps))
        t_run_score = reference_step if explicit_reference_step else t_end
        t_base_score = reference_step if explicit_reference_step else t_base_end
        if explicit_reference_step and t_run_score not in by_step:
            raise ValueError(f"Run evaluation history does not contain score_reference_step={reference_step}.")
        if explicit_reference_step and t_base_score not in b_by_step:
            raise ValueError(f"Baseline evaluation history does not contain score_reference_step={reference_step}.")

        s_best = float(by_step[run_best_step]["s"])
        s_end = float(by_step[t_end]["s"])
        s_run = float(by_step[t_run_score]["s"])

        s_base_best = float(b_by_step[t_base_best]["s"])
        s_base_end = float(b_by_step[t_base_end]["s"])
        s_base = float(b_by_step[t_base_score]["s"])

        r_quality = s_run - s_base
        relative_score = r_quality + r_convergence

        baseline_comparison = {
            "run_best_step": run_best_step,
            "run_end_step": t_end,
            "baseline_best_step": t_base_best,
            "baseline_end_step": t_base_end,
            "s_run_best": s_best,
            "s_run_end": s_end,
            "s_base_best": s_base_best,
            "s_base_end": s_base_end,
            "s_run": s_run,
            "s_base": s_base,
            "q_run": s_run,
            "q_base": s_base,
            "r_quality": r_quality,
            "auc_loss_run": auc_loss_run,
            "auc_loss_base": auc_loss_base,
            loss_run_key: float(loss_at_reference_run) if loss_at_reference_run is not None else None,
            loss_base_key: float(loss_at_reference_base) if loss_at_reference_base is not None else None,
            "r_convergence": r_convergence,
            "score_reference_step": reference_step,
            "auc_max_step": reference_step,
            "s_run_relative": relative_score,
            "score_weights": {
                "quality": 1.0,
                "convergence": 1.0,
            },
        }

        if include_breakdown:
            run_best_breakdown = by_step.get(run_best_step, {})
            run_end_breakdown = by_step.get(t_end, {})
            base_best_breakdown = b_by_step.get(t_base_best, {})
            base_end_breakdown = b_by_step.get(t_base_end, {})
            baseline_comparison.update(
                {
                    "run_best_l": run_best_breakdown.get("l"),
                    "run_best_q": run_best_breakdown.get("q"),
                    "run_best_t": run_best_breakdown.get("t"),
                    "run_best_elapsed": run_best_breakdown.get("elapsed"),
                    "run_end_l": run_end_breakdown.get("l"),
                    "run_end_q": run_end_breakdown.get("q"),
                    "run_end_t": run_end_breakdown.get("t"),
                    "run_end_elapsed": run_end_breakdown.get("elapsed"),
                    "base_best_l": base_best_breakdown.get("l"),
                    "base_best_q": base_best_breakdown.get("q"),
                    "base_best_t": base_best_breakdown.get("t"),
                    "base_best_elapsed": base_best_breakdown.get("elapsed"),
                    "base_end_l": base_end_breakdown.get("l"),
                    "base_end_q": base_end_breakdown.get("q"),
                    "base_end_t": base_end_breakdown.get("t"),
                    "base_end_elapsed": base_end_breakdown.get("elapsed"),
                    "time_ref": time_ref,
                }
            )
    else:
        t_run_score = reference_step if explicit_reference_step else t_end
        if explicit_reference_step and t_run_score not in by_step:
            raise ValueError(f"Run evaluation history does not contain score_reference_step={reference_step}.")
        s_best = float(by_step.get(run_best_step, {}).get("s", 0.0))
        s_end = float(by_step.get(t_end, {}).get("s", 0.0))
        s_run = float(by_step.get(t_run_score, {}).get("s", 0.0))
        r_quality = s_run
        relative_score = r_quality + r_convergence

        if include_breakdown:
            run_best_breakdown = by_step.get(run_best_step, {})
            run_end_breakdown = by_step.get(t_end, {})
            baseline_comparison = {
                "run_best_step": run_best_step,
                "run_end_step": t_end,
                "s_run_best": s_best,
                "s_run_end": s_end,
                "s_run": s_run,
                "q_run": s_run,
                "r_quality": r_quality,
                "auc_loss_run": auc_loss_run,
                "auc_loss_base": auc_loss_base,
                loss_run_key: float(loss_at_reference_run) if loss_at_reference_run is not None else None,
                loss_base_key: None,
                "r_convergence": r_convergence,
                "score_reference_step": reference_step,
                "auc_max_step": reference_step,
                "run_best_l": run_best_breakdown.get("l"),
                "run_best_q": run_best_breakdown.get("q"),
                "run_best_t": run_best_breakdown.get("t"),
                "run_best_elapsed": run_best_breakdown.get("elapsed"),
                "run_end_l": run_end_breakdown.get("l"),
                "run_end_q": run_end_breakdown.get("q"),
                "run_end_t": run_end_breakdown.get("t"),
                "run_end_elapsed": run_end_breakdown.get("elapsed"),
                "time_ref": time_ref,
                "score_weights": {
                    "quality": 1.0,
                    "convergence": 1.0,
                },
            }

    return {
        "s_best": s_best,
        "s_end": s_end,
        "s_run": s_run,
        "relative_score": relative_score,
        "baseline_comparison": baseline_comparison,
    }


def update_from_run(
    *,
    project_dir: Path,
    mode: str,
    selected_preset: str,
    yhat_scores: dict[str, float],
    eval_history: list[dict[str, Any]],
    baseline_eval_history: list[dict[str, Any]] | None,
    loss_by_step: dict[int, float],
    elapsed_by_step: dict[int, float],
    x_features: dict[str, Any] | None,
    run_id: str,
    logger,
    apply_update: bool = True,
    baseline_loss_by_step_override: dict[int, float] | None = None,
    score_reference_step: int | None = None,
) -> dict[str, Any]:
    if not eval_history:
        return {"updated": False, "reason": "no_eval_history"}

    eval_rows = [row for row in eval_history if isinstance(row, dict) and isinstance(row.get("step"), (int, float))]
    if not eval_rows:
        return {"updated": False, "reason": "no_eval_steps"}

    eval_rows.sort(key=lambda r: int(r.get("step", 0)))
    eval_steps = [int(r["step"]) for r in eval_rows]

    # Find best step by quality (PSNR + SSIM + LPIPS), not by loss
    # Quality metrics available only at eval steps
    quality_scores_by_step: dict[int, float] = {}
    for row in eval_rows:
        step = int(row["step"])
        psnr = float(row.get("convergence_speed", 0.0) or 0.0)
        ssim = float(row.get("sharpness_mean", 0.0) or 0.0)
        lpips = float(row.get("lpips_mean", 0.0) or 0.0)
        # Composite quality: 40% PSNR, 30% SSIM, 30% LPIPS (lower is better, so invert)
        # Note: using raw values here, normalization happens later
        quality_scores_by_step[step] = psnr + ssim + (1.0 - lpips) if lpips > 0 else psnr + ssim

    # Best step = highest quality score
    if quality_scores_by_step:
        t_best = int(max(quality_scores_by_step.keys(), key=lambda s: quality_scores_by_step[s]))
    else:
        t_best = int(eval_steps[-1])

    t_eval_best = t_best  # Already an evaluated step, no need to find nearest
    t_end = int(max(eval_steps))

    psnr_vals = [float(r.get("convergence_speed", 0.0) or 0.0) for r in eval_rows]
    ssim_vals = [float(r.get("sharpness_mean", 0.0) or 0.0) for r in eval_rows]
    lpips_vals = [float(r.get("lpips_mean", 0.0) or 0.0) for r in eval_rows]

    score_summary = compute_score_summary(
        eval_rows=eval_rows,
        baseline_eval_history=baseline_eval_history,
        loss_by_step=loss_by_step,
        elapsed_by_step=elapsed_by_step,
        t_eval_best=t_eval_best,
        t_end=t_end,
        prefer_quality_best=False,
        include_breakdown=False,
        baseline_loss_by_step_override=baseline_loss_by_step_override,
        score_reference_step=score_reference_step,
    )
    s_best = float(score_summary["s_best"])
    s_end = float(score_summary["s_end"])
    s_run = float(score_summary["s_run"])
    relative_score = float(score_summary["relative_score"])
    baseline_comparison = score_summary.get("baseline_comparison")

    eval_row_by_step = {int(row["step"]): dict(row) for row in eval_rows if isinstance(row.get("step"), (int, float))}
    outcomes = {
        "t_best": t_best,
        "t_eval_best": t_eval_best,
        "t_end": t_end,
        "best_anchor": dict(eval_row_by_step.get(t_eval_best, {})),
        "end_anchor": dict(eval_row_by_step.get(t_end, {})),
    }
    transition = {
        "x": dict(x_features or {}),
        "yhat": dict(yhat_scores),
        "k_star": selected_preset,
        "outcomes": outcomes,
        "s_run": s_run,
        "baseline_comparison": baseline_comparison,
        "relative_score": relative_score,
    }

    if apply_update:
        model = _load_model(project_dir)
        entry = _mode_entry(model, mode)
        runs = int(entry.get("runs", 0) or 0)
        score_mean = float(entry.get("score_mean", 0.0) or 0.0)
        delta = relative_score - score_mean

        alpha = 1.0 / float(runs + 1)
        score_mean = score_mean + alpha * delta
        entry["score_mean"] = score_mean
        entry["runs"] = runs + 1

        bias = entry.get("bias", {})
        lr = 0.10

        for preset in PRESETS:
            cur = float(bias.get(preset, 0.0) or 0.0)
            if preset == selected_preset:
                cur += lr * delta
            else:
                cur -= lr * 0.15 * delta
            bias[preset] = float(clamp_float(cur, -3.0, 3.0))
        entry["bias"] = bias
        entry["last"] = {
            "run_id": run_id,
            "selected_preset": selected_preset,
            "t_best": t_best,
            "t_eval_best": t_eval_best,
            "t_end": t_end,
            "s_best": s_best,
            "s_end": s_end,
            "s_run": s_run,
            "yhat_scores": yhat_scores,
            "transition": transition,
            "baseline_comparison": baseline_comparison,
            "relative_score": relative_score,
        }

        _save_model(project_dir, model)

        logger.info(
            "AI_INPUT_MODE_LEARN mode=%s preset=%s s_best=%.4f s_end=%.4f s_run=%.4f score=%.4f",
            mode,
            selected_preset,
            s_best,
            s_end,
            s_run,
            relative_score,
        )
    else:
        logger.info(
            "AI_INPUT_MODE_COMPARE_ONLY mode=%s preset=%s s_best=%.4f s_end=%.4f s_run=%.4f score=%.4f",
            mode,
            selected_preset,
            s_best,
            s_end,
            s_run,
            relative_score,
        )
    logger.info(
        "AI_INPUT_MODE_SCORE_OUTCOME mode=%s preset=%s score=%.4f score_positive=%s",
        mode,
        selected_preset,
        relative_score,
        str(relative_score > 0.0).lower(),
    )
    logger.info(
        "Input-mode selector updated run_id=%s mode=%s preset=%s s_best=%.4f s_end=%.4f s_run=%.4f",
        run_id,
        mode,
        selected_preset,
        s_best,
        s_end,
        s_run,
    )

    return {
        "updated": bool(apply_update),
        "mode": mode,
        "selected_preset": selected_preset,
        "t_best": t_best,
        "t_eval_best": t_eval_best,
        "t_end": t_end,
        "s_best": s_best,
        "s_end": s_end,
        "s_run": s_run,
        "yhat_scores": yhat_scores,
        "transition": transition,
        "baseline_comparison": baseline_comparison,
        "relative_score": relative_score,
        "compare_only": not bool(apply_update),
    }


def record_run_penalty(
    *,
    project_dir: Path,
    mode: str,
    selected_preset: str,
    yhat_scores: dict[str, float],
    penalty_score: float,
    reason: str,
    run_id: str,
    logger,
) -> dict[str, Any]:
    model = _load_model(project_dir)
    entry = _mode_entry(model, mode)
    runs = int(entry.get("runs", 0) or 0)
    score_mean = float(entry.get("score_mean", 0.0) or 0.0)
    relative_score = float(penalty_score)
    delta = relative_score - score_mean

    alpha = 1.0 / float(runs + 1)
    score_mean = score_mean + alpha * delta
    entry["score_mean"] = score_mean
    entry["runs"] = runs + 1

    bias = entry.get("bias", {})
    lr = 0.10
    for preset in PRESETS:
        cur = float(bias.get(preset, 0.0) or 0.0)
        if preset == selected_preset:
            cur += lr * delta
        else:
            cur -= lr * 0.15 * delta
        bias[preset] = float(clamp_float(cur, -3.0, 3.0))

    entry["bias"] = bias
    entry["last"] = {
        "run_id": run_id,
        "selected_preset": selected_preset,
        "yhat_scores": yhat_scores,
        "relative_score": relative_score,
        "reason": reason,
        "penalty": True,
    }

    _save_model(project_dir, model)

    logger.info(
        "AI_INPUT_MODE_PENALTY mode=%s preset=%s score=%.4f reason=%s",
        mode,
        selected_preset,
        relative_score,
        reason,
    )

    return {
        "updated": True,
        "mode": mode,
        "selected_preset": selected_preset,
        "relative_score": relative_score,
        "reason": reason,
        "penalty": True,
        "yhat_scores": yhat_scores,
    }

