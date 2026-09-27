import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bimba3d_backend.app.services import controlled_experiments
from bimba3d_backend.app.services import workflow_pipeline_service
from bimba3d_backend.app.services.training_pipeline_orchestrator import PipelineOrchestrator
from bimba3d_backend.app.services.workflow_pipeline_service import _calculate_total_runs


class ControlledExperimentTests(unittest.TestCase):
    def test_remove_action_rejects_mislabeled_baseline_before_deleting_anything(self):
        with tempfile.TemporaryDirectory() as directory:
            project_dir = Path(directory) / "Project" / "runs"
            for run_id in ("base", "time_run"):
                (project_dir / run_id).mkdir(parents=True)
            pipeline = {
                "id": "test", "status": "completed",
                "config": {"pipeline_type": "test", "pipeline_folder": directory, "additional_experiments": {
                    controlled_experiments.TIME: {"enabled": True, "model_id": "ridge"},
                }},
                "runs": [
                    {"project_name": "Project", "run_id": "time_run", "phase": 90, "controlled_experiment": controlled_experiments.TIME},
                    {"project_name": "Project", "run_id": "base", "phase": 1, "controlled_experiment": controlled_experiments.TIME},
                ],
            }
            with patch.object(workflow_pipeline_service, "_require_pipeline", return_value=pipeline):
                with self.assertRaisesRegex(ValueError, "Only Time and Gaussian controlled runs"):
                    workflow_pipeline_service.remove_additional_experiment_runs("test", controlled_experiments.TIME)
            self.assertTrue((project_dir / "base").is_dir())
            self.assertTrue((project_dir / "time_run").is_dir())

    def test_remove_additional_runs_keeps_other_run_types_and_config(self):
        for kind, removed_id, preserved_id in (
            (controlled_experiments.TIME, "time_run", "gaussian_run"),
            (controlled_experiments.GAUSSIAN, "gaussian_run", "time_run"),
        ):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                project_dir = Path(directory) / "Project"
                for run_id in ("base", "model", "time_run", "gaussian_run"):
                    (project_dir / "runs" / run_id).mkdir(parents=True)
                config = {"pipeline_type": "test", "pipeline_folder": directory, "additional_experiments": {
                    controlled_experiments.TIME: {"enabled": True, "model_id": "ridge"},
                    controlled_experiments.GAUSSIAN: {"enabled": True, "model_id": "ridge"},
                }}
                runs = [
                    {"project_name": "Project", "run_id": "base", "phase": 1},
                    {"project_name": "Project", "run_id": "model", "phase": 2},
                    {"project_name": "Project", "run_id": "time_run", "phase": 90, "controlled_experiment": controlled_experiments.TIME},
                    {"project_name": "Project", "run_id": "gaussian_run", "phase": 91, "controlled_experiment": controlled_experiments.GAUSSIAN},
                ]
                pipeline = {"id": "test", "status": "completed", "config": config, "runs": runs, "total_runs": 4}

                def update(_pipeline_id, values):
                    pipeline.update(values)
                    return pipeline

                def refresh(_pipeline_id):
                    pipeline["pending_runs"] = 2
                    return pipeline

                with patch.object(workflow_pipeline_service, "_require_pipeline", return_value=pipeline), \
                     patch.object(workflow_pipeline_service, "_calculate_total_runs", return_value=4), \
                     patch.object(workflow_pipeline_service, "normalise_pipeline_detail", return_value={}), \
                     patch.object(workflow_pipeline_service.training_pipeline_storage, "update_pipeline", side_effect=update), \
                     patch.object(workflow_pipeline_service.training_pipeline_storage, "refresh_pipeline_counters", side_effect=refresh):
                    result = workflow_pipeline_service.remove_additional_experiment_runs("test", kind)

                self.assertEqual(result["removed_run_count"], 1)
                self.assertTrue(result["resumable"])
                self.assertEqual(pipeline["status"], "stopped")
                self.assertIs(pipeline["config"], config)
                self.assertEqual({run["run_id"] for run in pipeline["runs"]}, {"base", "model", preserved_id})
                for run_id in ("base", "model", preserved_id):
                    self.assertTrue((project_dir / "runs" / run_id).is_dir())
                self.assertFalse((project_dir / "runs" / removed_id).exists())

    def test_config_override_removes_only_changed_control_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            project_dir = Path(directory) / "Project"
            for run_id in ("time_run", "gaussian_run"):
                (project_dir / "runs" / run_id).mkdir(parents=True)
            old_config = {
                "pipeline_type": "test", "pipeline_folder": directory,
                "additional_experiments": {
                    controlled_experiments.TIME: {"enabled": True, "model_id": "ridge", "max_steps_ceiling": 12_000},
                    controlled_experiments.GAUSSIAN: {"enabled": True, "model_id": "ridge", "max_steps_ceiling": 12_000},
                },
            }
            runs = [
                {"project_name": "Project", "run_id": "time_run", "phase": 90, "controlled_experiment": controlled_experiments.TIME},
                {"project_name": "Project", "run_id": "gaussian_run", "phase": 91, "controlled_experiment": controlled_experiments.GAUSSIAN},
                {"project_name": "Project", "run_id": "base", "phase": 1},
            ]
            pipeline = {"id": "test", "name": "Test", "status": "completed", "config": old_config, "runs": runs, "total_runs": 3}

            def update(_pipeline_id, values):
                pipeline.update(values)
                return pipeline

            def refresh(_pipeline_id):
                pipeline["pending_runs"] = 1 if len(pipeline["runs"]) < 3 else 0
                return pipeline

            with patch.object(workflow_pipeline_service, "_require_pipeline", return_value=pipeline), \
                 patch.object(workflow_pipeline_service, "_prepare_pipeline_config", side_effect=lambda value, **_: dict(value)), \
                 patch.object(workflow_pipeline_service, "_preserve_existing_pipeline_state"), \
                 patch.object(workflow_pipeline_service, "_calculate_total_runs", return_value=3), \
                 patch.object(workflow_pipeline_service, "normalise_pipeline_detail", return_value={}), \
                 patch.object(workflow_pipeline_service.training_pipeline_storage, "update_pipeline", side_effect=update), \
                 patch.object(workflow_pipeline_service.training_pipeline_storage, "refresh_pipeline_counters", side_effect=refresh):
                unrelated = {**old_config, "name": "Renamed"}
                workflow_pipeline_service.update_workflow_pipeline_config("test", unrelated)
                self.assertTrue((project_dir / "runs" / "time_run").is_dir())
                self.assertTrue((project_dir / "runs" / "gaussian_run").is_dir())

                changed = {**old_config, "additional_experiments": {
                    **old_config["additional_experiments"],
                    controlled_experiments.TIME: {"enabled": True, "model_id": "ridge", "max_steps_ceiling": 16_000},
                }}
                with self.assertRaisesRegex(ValueError, "Confirm override"):
                    workflow_pipeline_service.update_workflow_pipeline_config("test", changed)
                self.assertTrue((project_dir / "runs" / "time_run").is_dir())

                result = workflow_pipeline_service.update_workflow_pipeline_config(
                    "test", changed, override_controlled_kinds=[controlled_experiments.TIME]
                )
                self.assertEqual(result["overridden_run_count"], 1)
                self.assertFalse((project_dir / "runs" / "time_run").exists())
                self.assertTrue((project_dir / "runs" / "gaussian_run").is_dir())
                self.assertEqual(len(pipeline["runs"]), 2)
                self.assertEqual(pipeline["status"], "stopped")

    def test_automatic_reference_uses_latest_completed_selected_model_run(self):
        with tempfile.TemporaryDirectory() as directory:
            project_dir = Path(directory)
            sparse_dir = project_dir / "outputs" / "sparse" / "0"
            sparse_dir.mkdir(parents=True)
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                (sparse_dir / name).write_bytes(b"test")
            for run_id, seconds, gaussians in (("base", 100, 500), ("older", 120, 700), ("latest", 140, 900)):
                analytics = project_dir / "runs" / run_id / "analytics" / "run_analytics_v1.json"
                analytics.parent.mkdir(parents=True)
                analytics.write_text(json.dumps({"summary": {"status": "completed", "metrics": {
                    "total_time_seconds": seconds, "num_gaussians": gaussians,
                }}}), encoding="utf-8")
            pipeline = {"id": "test", "config": {
                "pipeline_type": "test", "source_model_ids": ["ridge"],
                "shared_config": {"max_steps": 5000, "gaussian_hard_cap": 6000000},
                "additional_experiments": {controlled_experiments.TIME: {"enabled": True, "model_id": "ridge"}},
            }, "runs": [
                {"project_name": "Project", "phase": 2, "run": 1, "test_model_id": "ridge", "run_id": "older", "status": "success"},
                {"project_name": "Project", "phase": 2, "run": 2, "test_model_id": "ridge", "run_id": "latest", "status": "success"},
            ]}
            orchestrator = PipelineOrchestrator("test")
            phase = {"phase_number": 90, "controlled_experiment": controlled_experiments.TIME, "name": controlled_experiments.TIME}
            with patch.object(orchestrator, "_get_or_create_project_dir", return_value=project_dir):
                config = orchestrator._build_run_config(pipeline, {"name": "Project", "baseline_run_id": "base"}, phase, 1, test_model_id="ridge")
            self.assertEqual(config["reference_model_run_id"], "latest")
            self.assertEqual(config["training_time_limit_seconds"], 140)
            self.assertEqual(config["max_steps"], 12_000)

    def test_baseline_control_uses_selected_reference_run(self):
        with tempfile.TemporaryDirectory() as directory:
            project_dir = Path(directory)
            sparse_dir = project_dir / "outputs" / "sparse" / "0"
            sparse_dir.mkdir(parents=True)
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                (sparse_dir / name).write_bytes(b"test")
            for run_id, seconds, gaussians in (("base", 100, 500), ("model_run_2", 140, 900)):
                analytics = project_dir / "runs" / run_id / "analytics" / "run_analytics_v1.json"
                analytics.parent.mkdir(parents=True)
                analytics.write_text(json.dumps({"summary": {"status": "completed", "metrics": {
                    "total_time_seconds": seconds, "num_gaussians": gaussians,
                }}}), encoding="utf-8")
            pipeline = {
                "id": "test", "config": {
                    "pipeline_type": "test", "source_model_id": "ridge", "source_model_ids": ["ridge"],
                    "shared_config": {"max_steps": 5000, "gaussian_hard_cap": 6000000},
                    "additional_experiments": {
                        controlled_experiments.TIME: {"enabled": True, "model_id": "ridge", "source_run_number": 2},
                        controlled_experiments.GAUSSIAN: {"enabled": True, "model_id": "ridge", "source_run_number": 2},
                    },
                },
                "runs": [{"project_name": "Project", "phase": 2, "run": 2, "test_model_id": "ridge", "run_id": "model_run_2", "status": "success"}],
            }
            project = {"name": "Project", "baseline_run_id": "base"}
            orchestrator = PipelineOrchestrator("test")
            with patch.object(orchestrator, "_get_or_create_project_dir", return_value=project_dir):
                for kind, phase_number in controlled_experiments.PHASES.items():
                    phase = {"phase_number": phase_number, "controlled_experiment": kind, "name": kind}
                    config = orchestrator._build_run_config(pipeline, project, phase, 1, test_model_id="ridge")
                    self.assertIsNone(config["ai_input_mode"])
                    self.assertFalse(config["update_model"])
                    self.assertEqual(config["reference_model_run_id"], "model_run_2")
                    if kind == controlled_experiments.TIME:
                        self.assertEqual(config["training_time_limit_seconds"], 140)
                        self.assertGreater(config["max_steps"], 5000)
                    else:
                        self.assertEqual(config["target_gaussians"], 900)
                        self.assertNotIn("densification_multiplier", config)
                        self.assertEqual(config["max_steps"], 15_000)
                        self.assertEqual(config["densify_until_iter"], 15_000)
                        self.assertEqual(config["gaussian_post_cap_steps"], 1000)

    def test_control_slots_add_one_run_per_project_per_enabled_control(self):
        config = {
            "pipeline_type": "test",
            "source_model_ids": ["ridge", "mlp"],
            "projects": [{"name": "a"}, {"name": "b"}],
            "phases": [
                {"phase_number": 1, "exploration_runs_per_project": 1},
                {"phase_number": 2, "exploration_runs_per_project": 1},
            ],
            "additional_experiments": {
                controlled_experiments.TIME: {"enabled": True, "model_id": "ridge"},
                controlled_experiments.GAUSSIAN: {"enabled": True, "model_id": "ridge"},
            },
        }
        controlled_experiments.validate(config)
        self.assertEqual(_calculate_total_runs(config), 10)

    def test_configurable_step_ceiling_requires_post_cap_room(self):
        config = {
            "pipeline_type": "test", "source_model_ids": ["ridge"],
            "shared_config": {"max_steps": 11_500},
            "additional_experiments": {
                controlled_experiments.GAUSSIAN: {"enabled": True, "model_id": "ridge", "max_steps_ceiling": 12_000},
            },
        }
        with self.assertRaisesRegex(ValueError, "maximum steps"):
            controlled_experiments.validate(config)
        config["additional_experiments"][controlled_experiments.GAUSSIAN]["max_steps_ceiling"] = 14_000
        controlled_experiments.validate(config)

    def test_hard_cap_reference_exposes_budget_without_final_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "runs" / "selected"
            analytics = run_root / "analytics" / "run_analytics_v1.json"
            analytics.parent.mkdir(parents=True)
            analytics.write_text(json.dumps({"summary": {
                "status": "partial_completed",
                "gaussian_cap_reached": True,
                "gaussian_cap_count": 6_100_000,
                "metrics": {"total_time_seconds": 1200},
            }}), encoding="utf-8")
            result = controlled_experiments.source_summary(Path(directory), "selected")
            self.assertTrue(result["hard_cap"])
            self.assertEqual(result["gaussians"], 6_100_000)
            self.assertEqual(result["time_seconds"], 1200)
            self.assertIsNone(result["psnr"])

    def test_time_reference_excludes_final_evaluation_when_step_stats_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "runs" / "selected"
            analytics = run_root / "analytics" / "run_analytics_v1.json"
            analytics.parent.mkdir(parents=True)
            analytics.write_text(json.dumps({"summary": {"status": "completed", "metrics": {"total_time_seconds": 140}}}), encoding="utf-8")
            stats = run_root / "outputs" / "engines" / "gsplat" / "stats"
            stats.mkdir(parents=True)
            (stats / "train_step4999_rank0.json").write_text(json.dumps({"ellipse_time": 117}), encoding="utf-8")
            result = controlled_experiments.source_summary(Path(directory), "selected")
            self.assertEqual(result["time_seconds"], 140)
            self.assertEqual(result["training_loop_seconds"], 117)
            self.assertEqual(result["training_loop_time_source"], "train_step_stats")

    def test_source_summary_uses_latest_numeric_eval_step(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "runs" / "selected"
            analytics = run_root / "analytics" / "run_analytics_v1.json"
            analytics.parent.mkdir(parents=True)
            analytics.write_text(json.dumps({"summary": {"status": "completed", "metrics": {"total_time_seconds": 1000}}}), encoding="utf-8")
            engine = run_root / "outputs" / "engines" / "gsplat"
            stats = engine / "stats"
            stats.mkdir(parents=True)
            (engine / "eval_history.json").write_text(json.dumps([
                {"step": 9000, "num_gaussians": 818856, "convergence_speed": 25.5},
                {"step": 10000, "num_gaussians": 850072, "convergence_speed": 25.4},
            ]), encoding="utf-8")
            (engine / "metadata.json").write_text(json.dumps({"final_gaussian_count": 850072}), encoding="utf-8")
            (stats / "val_step11999.json").write_text(json.dumps({
                "psnr": 25.718, "ssim": 0.8022, "lpips": 0.2996, "num_GS": 877406,
            }), encoding="utf-8")
            result = controlled_experiments.source_summary(Path(directory), "selected")
            self.assertEqual(result["step"], 12000)
            self.assertEqual(result["gaussians"], 877406)
            self.assertAlmostEqual(result["psnr"], 25.718)

    def test_eval_collector_orders_steps_numerically(self):
        from bimba3d_backend.worker.entrypoint import _collect_eval_history

        with tempfile.TemporaryDirectory() as directory:
            stats = Path(directory) / "stats"
            stats.mkdir()
            (stats / "val_step9999.json").write_text(json.dumps({"psnr": 25.4, "num_GS": 850072}), encoding="utf-8")
            (stats / "val_step11999.json").write_text(json.dumps({"psnr": 25.7, "num_GS": 877406}), encoding="utf-8")
            history = _collect_eval_history(Path(directory), {}, "baseline")
            self.assertEqual([row["step"] for row in history], [10000, 12000])


if __name__ == "__main__":
    unittest.main()
