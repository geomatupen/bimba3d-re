from __future__ import annotations

import math
import struct
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from PIL import Image

from .common import ModeContext, PresetResult, apply_preset_updates, keep_only_feature_keys
from .exif_extractors import extract_camera_exif
from .feature_utils import calculate_gsd, calculate_terrain_roughness, infer_sensor_width, read_colmap_points3d


COMPACT_SCENE_DESCRIPTOR_MODE = "exif_compact_featurewise"
COMPACT_SCENE_DESCRIPTOR_KEYS: set[str] = {
    "gsd_median",
    "overlap_proxy",
    "coverage_spread",
    "camera_angle_bucket",
    "heading_consistency",
    "vegetation_cover_percentage",
    "vegetation_complexity_score",
    "terrain_roughness_proxy",
    "texture_density",
    "blur_motion_risk",
}


def _iter_images(image_dir: Path, *, limit: int | None = None) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    files = [path for path in Path(image_dir).glob("*") if path.is_file() and path.suffix.lower() in exts]
    files.sort()
    return files[:limit] if limit is not None else files


def _as_float(value: Any) -> float | None:
    try:
        if isinstance(value, tuple) and len(value) == 2:
            num, den = value
            den_f = float(den)
            return None if den_f == 0.0 else float(num) / den_f
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if "/" in cleaned:
                num_s, den_s = cleaned.split("/", 1)
                den = float(den_s.strip())
                return None if den == 0.0 else float(num_s.strip()) / den
            if cleaned.startswith("+"):
                cleaned = cleaned[1:]
            for unit in [" mm", "mm", " cm", "cm", " m", " s", "s", " ms", "ms", " deg", "deg", "°"]:
                cleaned = cleaned.replace(unit, "")
            return float(cleaned.strip())
        return float(value)
    except Exception:
        return None


def _extract_gps(exif: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    gps = exif.get("GPSInfo")
    if not isinstance(gps, dict):
        return None, None, None

    lat_direct = _as_float(gps.get("lat"))
    lon_direct = _as_float(gps.get("lon"))
    alt_direct = _as_float(gps.get("alt"))
    if lat_direct is not None or lon_direct is not None or alt_direct is not None:
        return lat_direct, lon_direct, alt_direct

    def _to_deg(ref_key: int, val_key: int) -> float | None:
        ref = gps.get(ref_key)
        val = gps.get(val_key)
        if not isinstance(val, (tuple, list)) or len(val) != 3:
            return None
        d = _as_float(val[0])
        m = _as_float(val[1])
        s = _as_float(val[2])
        if d is None or m is None or s is None:
            return None
        out = d + (m / 60.0) + (s / 3600.0)
        return -out if str(ref).upper() in {"S", "W"} else out

    return _to_deg(1, 2), _to_deg(3, 4), _as_float(gps.get(6))


def _extract_timestamp(exif: dict[str, Any]) -> datetime | None:
    raw = str(exif.get("DateTimeOriginal") or exif.get("DateTime") or "").strip()
    if not raw:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            pass
    return None


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))


def _angle_bucket_from_pitch(angle: float) -> str:
    if angle <= -80:
        return "nadir"
    if angle <= -60:
        return "oblique"
    return "high_oblique"


def _qvec_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return (
        (1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)),
        (2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)),
        (2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)),
    )


def _collect_colmap_pitch_angles(colmap_dir: Path, limit: int = 48) -> list[float]:
    candidates = [
        Path(colmap_dir) / "images.bin",
        Path(colmap_dir) / "0" / "images.bin",
        Path(colmap_dir) / "sparse" / "images.bin",
        Path(colmap_dir) / "sparse" / "0" / "images.bin",
    ]
    path = next((candidate for candidate in candidates if candidate.exists() and candidate.is_file()), None)
    if path is None:
        return []

    entries: list[tuple[str, float]] = []
    try:
        with path.open("rb") as handle:
            num_images = struct.unpack("Q", handle.read(8))[0]
            for _ in range(num_images):
                handle.read(4)
                qw, qx, qy, qz = struct.unpack("dddd", handle.read(32))
                handle.read(24)
                handle.read(4)
                name_bytes = bytearray()
                while True:
                    ch = handle.read(1)
                    if ch == b"" or ch == b"\x00":
                        break
                    name_bytes.extend(ch)
                name = name_bytes.decode("utf-8", errors="ignore")
                num_points = struct.unpack("Q", handle.read(8))[0]
                handle.read(24 * num_points)
                rotation = _qvec_to_rotmat(qw, qx, qy, qz)
                fz = max(-1.0, min(1.0, rotation[2][2]))
                entries.append((name, math.degrees(math.asin(fz))))
    except Exception:
        return []

    entries.sort(key=lambda item: item[0])
    return [pitch for _, pitch in entries[: min(limit, len(entries))]]


def _collect_processing_sizes(image_dir: Path, limit: int) -> tuple[list[int], list[int]]:
    widths: list[int] = []
    heights: list[int] = []
    for path in _iter_images(image_dir, limit=limit):
        try:
            with Image.open(path) as img:
                width, height = img.size
        except Exception:
            continue
        widths.append(int(width))
        heights.append(int(height))
    return widths, heights


def _image_metrics(path: Path) -> tuple[float, float, float, float, float]:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail((128, 128))
        width, height = rgb.size
        px = rgb.load()
        total = max(1, width * height)

        green_hits = 0
        luma_vals: list[float] = []
        edge_sum = 0.0
        lap_sum = 0.0

        def _luma(r: int, g: int, b: int) -> float:
            return 0.299 * r + 0.587 * g + 0.114 * b

        for y in range(height):
            for x in range(width):
                r, g, b = px[x, y]
                if g > r * 1.05 and g > b * 1.05:
                    green_hits += 1
                luma_vals.append(_luma(r, g, b))

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                center = _luma(*px[x, y])
                gx = _luma(*px[x + 1, y]) - _luma(*px[x - 1, y])
                gy = _luma(*px[x, y + 1]) - _luma(*px[x, y - 1])
                edge_sum += abs(gx) + abs(gy)
                lap_sum += abs(_luma(*px[x - 1, y]) + _luma(*px[x + 1, y]) + _luma(*px[x, y - 1]) + _luma(*px[x, y + 1]) - 4.0 * center)

        green_cover = float(green_hits) / float(total)
        mean_luma = sum(luma_vals) / float(len(luma_vals)) if luma_vals else 0.0
        var_luma = sum((value - mean_luma) ** 2 for value in luma_vals) / float(len(luma_vals)) if luma_vals else 0.0
        texture_density = min(1.0, edge_sum / float(max(1.0, (width - 2) * (height - 2) * 255.0)))
        roughness_proxy = min(1.0, (var_luma ** 0.5) / 64.0)
        blur_sharpness = min(1.0, lap_sum / float(max(1.0, (width - 2) * (height - 2) * 255.0)))
        blur_risk = 1.0 - blur_sharpness
        veg_complexity = min(1.0, 0.55 * green_cover + 0.45 * texture_density)
        return green_cover, veg_complexity, roughness_proxy, texture_density, blur_risk


def build_preset(ctx: ModeContext) -> PresetResult:
    images = _iter_images(ctx.metadata_image_dir, limit=48)
    sample = images[: min(24, len(images))]
    focal_lengths: list[float] = []
    gps_points: list[tuple[float, float]] = []
    altitudes: list[float] = []
    relative_altitudes: list[float] = []
    timestamps: list[datetime] = []

    first_camera_model = ""
    for path in sample:
        try:
            exif, _, _ = extract_camera_exif(path)
        except Exception:
            continue
        if not first_camera_model:
            first_camera_model = str(exif.get("Model") or "")
        focal = _as_float(exif.get("FocalLength"))
        if focal is not None:
            focal_lengths.append(focal)
        lat, lon, alt = _extract_gps(exif)
        if lat is not None and lon is not None:
            gps_points.append((lat, lon))
        if alt is not None:
            altitudes.append(alt)
        rel_alt = _as_float(exif.get("RelativeAltitude"))
        if rel_alt is not None:
            relative_altitudes.append(rel_alt)
        ts = _extract_timestamp(exif)
        if ts is not None:
            timestamps.append(ts)

    med_focal = max(2.0, min(300.0, float(median(focal_lengths)) if focal_lengths else 24.0))
    widths, _ = _collect_processing_sizes(ctx.processing_image_dir, limit=24)
    img_width_med = max(640, min(8000, int(median(widths)) if widths else 4000))

    gps_data_available = len(gps_points) >= 3 and bool(timestamps)
    avg_altitude = mean(relative_altitudes) if relative_altitudes else mean(altitudes) if altitudes else 120.0
    gsd_median = calculate_gsd(avg_altitude, med_focal, infer_sensor_width(first_camera_model), img_width_med) if gps_data_available and (relative_altitudes or altitudes) else 0.0

    if gps_data_available:
        steps_m = [_haversine_m(gps_points[idx - 1][0], gps_points[idx - 1][1], gps_points[idx][0], gps_points[idx][1]) for idx in range(1, len(gps_points))]
        bearings = [_bearing_deg(gps_points[idx - 1][0], gps_points[idx - 1][1], gps_points[idx][0], gps_points[idx][1]) for idx in range(1, len(gps_points))]
        lats = [point[0] for point in gps_points]
        lons = [point[1] for point in gps_points]
        lat_span_m = _haversine_m(min(lats), mean(lons), max(lats), mean(lons)) if lats else 0.0
        lon_span_m = _haversine_m(mean(lats), min(lons), mean(lats), max(lons)) if lons else 0.0
        coverage_spread = max(0.0, min(1.0, math.sqrt((lat_span_m ** 2) + (lon_span_m ** 2)) / 600.0))
        overlap_proxy = max(0.0, min(1.0, 1.0 - ((mean(steps_m) if steps_m else 0.0) / max(10.0, 1.2 * avg_altitude))))
        if len(bearings) >= 2:
            deltas = [min(abs(bearings[idx] - bearings[idx - 1]), 360.0 - abs(bearings[idx] - bearings[idx - 1])) for idx in range(1, len(bearings))]
            turn_std = pstdev(deltas) if len(deltas) >= 2 else (deltas[0] if deltas else 0.0)
            heading_consistency = max(0.0, min(1.0, 1.0 - (turn_std / 90.0)))
        else:
            heading_consistency = 0.5
    else:
        overlap_proxy = 0.5
        coverage_spread = 0.0
        heading_consistency = 0.5

    angle_samples = _collect_colmap_pitch_angles(ctx.colmap_dir, limit=48)
    coarse_buckets = ["nadir" if _angle_bucket_from_pitch(angle) == "nadir" else "oblique" for angle in angle_samples]
    if not coarse_buckets:
        camera_angle_bucket = 0
    elif len(set(coarse_buckets)) > 1:
        camera_angle_bucket = 3
    elif coarse_buckets[0] == "nadir":
        camera_angle_bucket = 1
    else:
        camera_angle_bucket = 2

    metrics = []
    for path in _iter_images(ctx.processing_image_dir, limit=20):
        try:
            metrics.append(_image_metrics(path))
        except Exception:
            continue
    green_cover = float(median([item[0] for item in metrics])) if metrics else 0.0
    veg_complexity = float(median([item[1] for item in metrics])) if metrics else 0.5
    texture_density = float(median([item[3] for item in metrics])) if metrics else 0.5
    blur_risk = float(median([item[4] for item in metrics])) if metrics else 0.5

    colmap_points = read_colmap_points3d(ctx.colmap_dir)
    if colmap_points is not None and len(colmap_points) > 0:
        terrain_roughness_proxy = calculate_terrain_roughness(colmap_points, grid_size=20, min_points_per_cell=3)
    elif metrics:
        terrain_roughness_proxy = float(median([item[2] for item in metrics]))
    else:
        terrain_roughness_proxy = 0.0

    if blur_risk >= 0.55 or (green_cover >= 0.60 and veg_complexity >= 0.60):
        preset = "conservative"
    elif terrain_roughness_proxy <= 0.35 and texture_density >= 0.60:
        preset = "geometry_fast"
    elif texture_density >= 0.68 and blur_risk <= 0.35:
        preset = "appearance_fast"
    else:
        preset = "balanced"

    features = keep_only_feature_keys(
        {
            "gsd_median": gsd_median,
            "overlap_proxy": overlap_proxy,
            "coverage_spread": coverage_spread,
            "camera_angle_bucket": camera_angle_bucket,
            "heading_consistency": heading_consistency,
            "vegetation_cover_percentage": green_cover,
            "vegetation_complexity_score": veg_complexity,
            "terrain_roughness_proxy": terrain_roughness_proxy,
            "texture_density": texture_density,
            "blur_motion_risk": blur_risk,
        },
        COMPACT_SCENE_DESCRIPTOR_KEYS,
    )
    return PresetResult(
        mode=COMPACT_SCENE_DESCRIPTOR_MODE,
        updates=apply_preset_updates(ctx.params, preset),
        features=features,
        notes=[
            "Compact scene descriptors for featurewise models.",
            "Features combine EXIF scale, flight geometry, image texture, vegetation, and sparse-scene roughness.",
        ],
    )
