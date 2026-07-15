from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _build_dataset(path: Path) -> None:
    rows = []
    projects = ["p1", "p2", "p3", "p4"]
    multipliers = [0.7, 0.9, 1.1, 1.3]

    for p in projects:
        for i, m in enumerate(multipliers):
            rows.append(
                {
                    "project_name": p,
                    "run_id": f"{p}_run_{i}",
                    "x_features": {
                        "focal_length_mm": 24.0 + i,
                        "iso": 200.0 + 100.0 * i,
                        "img_width_median": 4000.0,
                        "img_height_median": 3000.0,
                        "gsd_median": 0.03 + 0.01 * i,
                        "overlap_proxy": 0.55 + 0.05 * i,
                        "coverage_spread": 0.45 + 0.05 * i,
                        "camera_angle_bucket": i % 3,
                        "heading_consistency": 0.6,
                        "texture_density": 0.5,
                        "blur_motion_risk": 0.3,
                        "terrain_roughness_proxy": 0.4,
                        "vegetation_cover_percentage": 0.5,
                        "vegetation_complexity_score": 0.6,
                    },
                    "selected_multipliers": {
                        "position_lr_init_mult": m,
                        "scaling_lr_mult": m,
                        "rotation_lr_mult": m,
                        "feature_lr_mult": m,
                        "opacity_lr_mult": m,
                        "lambda_dssim_mult": m,
                        "densify_grad_threshold_mult": min(max(m, 0.7), 1.3),
                        "opacity_threshold_mult": min(max(m, 0.7), 1.3),
                    },
                    "relative_quality_score": 1.2 - abs(m - 1.1),
                    "convergence_score": 1.0 - abs(m - 0.9),
                    "is_baseline_run": False,
                }
            )

    path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")


def test_train_offline_models_writes_score_optimizer_schema(tmp_path):
    dataset = tmp_path / "offline_dataset_v1.json"
    out_dir = tmp_path / "models"
    _build_dataset(dataset)

    cmd = [
        sys.executable,
        "-m",
        "bimba3d_backend.scripts.train_offline_models",
        "--dataset",
        str(dataset),
        "--out-dir",
        str(out_dir),
        "--cv-folds",
        "2",
        "--cv-repeats",
        "1",
        "--lambda-candidates",
        "1.0",
        "2.0",
        "--candidate-points",
        "11",
    ]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr or res.stdout

    quality = json.loads((out_dir / "quality_model.json").read_text(encoding="utf-8"))
    convergence = json.loads((out_dir / "convergence_model.json").read_text(encoding="utf-8"))

    assert quality["schema"] == "offline_model_v3"
    assert convergence["schema"] == "offline_model_v3"
    assert quality["model"]["model_family"] == "ridge_score_optimizer"
    assert convergence["model"]["model_family"] == "ridge_score_optimizer"
    assert quality["model"]["candidate_points"] == 11
    assert "cv_report" in quality["metrics"]
    assert "cv_report" in convergence["metrics"]

