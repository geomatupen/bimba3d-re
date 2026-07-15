from __future__ import annotations

from typing import Any


MetricRanges = dict[str, tuple[float, float]]


def quality_score(*, psnr_norm: float, ssim_norm: float, lpips_norm: float) -> float:
    return float((float(psnr_norm) + float(ssim_norm) + (1.0 - float(lpips_norm))) / 3.0)


def raw_quality_score(row: dict[str, Any]) -> float:
    missing = [
        key
        for key in ("convergence_speed", "sharpness_mean", "lpips_mean")
        if not isinstance(row.get(key), (int, float))
    ]
    if missing:
        step = row.get("step")
        raise ValueError(f"Evaluation row at step {step} is missing metric(s): {', '.join(missing)}")
    return quality_score(
        psnr_norm=float(row["convergence_speed"]),
        ssim_norm=float(row["sharpness_mean"]),
        lpips_norm=float(row["lpips_mean"]),
    )


def apply_pipeline_normalized_quality(rows: list[dict[str, Any]]) -> MetricRanges:
    eligible_rows = [
        row
        for row in rows
        if not bool(row.get("exclude_from_normalization"))
        and str(row.get("quality_score_source") or "") != "gaussian_cap_penalty"
    ]
    ranges = metric_ranges(eligible_rows)
    for row in eligible_rows:
        psnr_norm = normalize(row.get("final_psnr"), ranges["psnr"])
        ssim_norm = normalize(row.get("final_ssim"), ranges["ssim"])
        lpips_norm = normalize(row.get("final_lpips"), ranges["lpips"])
        q_quality = quality_score(psnr_norm=psnr_norm, ssim_norm=ssim_norm, lpips_norm=lpips_norm)
        row["psnr_norm"] = psnr_norm
        row["ssim_norm"] = ssim_norm
        row["lpips_norm"] = lpips_norm
        row["q_quality"] = q_quality
        row["normalization_ranges"] = {
            "psnr": list(ranges["psnr"]),
            "ssim": list(ranges["ssim"]),
            "lpips": list(ranges["lpips"]),
        }
    return ranges


def metric_ranges(rows: list[dict[str, Any]]) -> MetricRanges:
    return {
        "psnr": _range_for(rows, "final_psnr"),
        "ssim": _range_for(rows, "final_ssim"),
        "lpips": _range_for(rows, "final_lpips"),
    }


def normalize(value: Any, bounds: tuple[float, float]) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError("Cannot normalize a missing metric value.")
    lo, hi = bounds
    if abs(hi - lo) < 1e-12:
        raise ValueError(f"Cannot normalize with zero-width range: {lo}..{hi}")
    normalized = (float(value) - lo) / (hi - lo)
    return float(max(0.0, min(1.0, normalized)))


def _range_for(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        raise ValueError(f"Cannot normalize without metric values for {key}.")
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        raise ValueError(f"Cannot normalize {key}; all values are identical.")
    return lo, hi
