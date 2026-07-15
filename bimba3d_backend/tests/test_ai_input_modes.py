import tempfile
import unittest
from pathlib import Path

from PIL import Image

from bimba3d_backend.worker.ai_input_modes import apply_initial_preset
from bimba3d_backend.worker.ai_input_modes.compact_scene_descriptors import COMPACT_SCENE_DESCRIPTOR_KEYS
from bimba3d_backend.worker.ai_input_modes.resolver import normalize_ai_input_mode


class _DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


def _write_sample_image(path: Path) -> None:
    img = Image.new("RGB", (1600, 900), color=(60, 140, 70))
    img.save(path, format="JPEG")


class AiInputModesTests(unittest.TestCase):
    def test_legacy_mode_when_not_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            colmap_dir = root / "sparse"
            image_dir.mkdir(parents=True, exist_ok=True)
            colmap_dir.mkdir(parents=True, exist_ok=True)
            _write_sample_image(image_dir / "img_001.jpg")

            params = {"feature_lr": 0.0025}
            summary = apply_initial_preset(params, image_dir=image_dir, colmap_dir=colmap_dir, logger=_DummyLogger())

            self.assertEqual(summary["mode"], "not_configured")
            self.assertFalse(summary["applied"])
            self.assertEqual(params["feature_lr"], 0.0025)

    def test_legacy_mode_names_normalize_to_compact(self):
        for mode in ("exif_only", "exif_plus_flight_plan", "exif_plus_flight_plan_plus_external"):
            self.assertEqual(normalize_ai_input_mode(mode), "exif_compact_featurewise")

    def test_baseline_mode_uses_preset_override_without_selector_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            colmap_dir = root / "sparse"
            image_dir.mkdir(parents=True, exist_ok=True)
            colmap_dir.mkdir(parents=True, exist_ok=True)
            _write_sample_image(image_dir / "img_001.jpg")

            params = {
                "mode": "baseline",
                "ai_input_mode": "exif_compact_featurewise",
                "ai_selector_strategy": "featurewise_ridge_regression",
                "preset_override": "geometry_fast",
                "feature_lr": 0.0025,
                "position_lr_init": 1.6e-4,
                "scaling_lr": 5.0e-3,
                "opacity_lr": 5.0e-2,
                "rotation_lr": 1.0e-3,
                "densify_grad_threshold": 2.0e-4,
                "opacity_threshold": 0.005,
                "lambda_dssim": 0.2,
            }

            summary = apply_initial_preset(params, image_dir=image_dir, colmap_dir=colmap_dir, logger=_DummyLogger())

            self.assertEqual(summary["mode"], "baseline")
            self.assertTrue(summary["applied"])
            self.assertEqual(summary["selected_preset"], "geometry_fast")
            self.assertEqual(summary["yhat_scores"], {})
            self.assertEqual(summary["features"], {})
            self.assertAlmostEqual(params["feature_lr"], 0.00225)
            self.assertAlmostEqual(params["position_lr_init"], 1.856e-4)

    def test_compact_scene_descriptor_mode_applies_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            colmap_dir = root / "sparse"
            image_dir.mkdir(parents=True, exist_ok=True)
            colmap_dir.mkdir(parents=True, exist_ok=True)
            _write_sample_image(image_dir / "img_001.jpg")

            params = {
                "ai_input_mode": "exif_compact_featurewise",
                "ai_selector_strategy": "compact_featurewise_ridge_regression",
                "feature_lr": 0.0025,
                "position_lr_init": 1.6e-4,
                "position_lr_final": 1.6e-6,
                "tune_interval": 100,
                "tune_min_improvement": 0.005,
                "densify_grad_threshold": 0.0002,
                "run_jitter_only": True,
            }
            summary = apply_initial_preset(params, image_dir=image_dir, colmap_dir=colmap_dir, logger=_DummyLogger())

            self.assertEqual(summary["mode"], "exif_compact_featurewise")
            self.assertTrue(summary["applied"])
            self.assertIn("feature_lr", params)
            self.assertGreater(params["feature_lr"], 0.0)
            self.assertGreaterEqual(params["tune_interval"], 50)
            self.assertLessEqual(params["tune_interval"], 400)
            self.assertEqual(set(summary["features"].keys()), COMPACT_SCENE_DESCRIPTOR_KEYS)

    def test_legacy_groupwise_selector_gets_legacy_feature_extras(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            colmap_dir = root / "sparse"
            image_dir.mkdir(parents=True, exist_ok=True)
            colmap_dir.mkdir(parents=True, exist_ok=True)
            _write_sample_image(image_dir / "img_001.jpg")

            params = {
                "ai_input_mode": "exif_compact_featurewise",
                "ai_selector_strategy": "featurewise_ridge_regression",
                "feature_lr": 0.0025,
                "position_lr_init": 1.6e-4,
                "run_jitter_only": True,
            }
            summary = apply_initial_preset(params, image_dir=image_dir, colmap_dir=colmap_dir, logger=_DummyLogger())

            self.assertEqual(summary["mode"], "exif_compact_featurewise")
            self.assertTrue(COMPACT_SCENE_DESCRIPTOR_KEYS.issubset(set(summary["features"].keys())))
            for key in ("focal_length_mm", "shutter_s", "iso", "img_width_median", "img_height_median"):
                self.assertIn(key, summary["features"])

    def test_feature_summary_is_cached_per_project_and_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            colmap_dir = root / "sparse"
            image_dir.mkdir(parents=True, exist_ok=True)
            colmap_dir.mkdir(parents=True, exist_ok=True)
            _write_sample_image(image_dir / "img_001.jpg")

            params = {
                "ai_input_mode": "exif_compact_featurewise",
                "ai_selector_strategy": "compact_featurewise_ridge_regression",
                "feature_lr": 0.0025,
                "position_lr_init": 1.6e-4,
                "run_jitter_only": True,
            }
            first = apply_initial_preset(params, image_dir=image_dir, colmap_dir=colmap_dir, logger=_DummyLogger())
            second = apply_initial_preset(params, image_dir=image_dir, colmap_dir=colmap_dir, logger=_DummyLogger())

            self.assertFalse(first["cache_used"])
            self.assertTrue(second["cache_used"])
            self.assertEqual(first["features"], second["features"])
            self.assertEqual(first["heuristic_preset"], second["heuristic_preset"])


if __name__ == "__main__":
    unittest.main()
