import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bimba3d_backend.app.services.training_pipeline_orchestrator import PipelineOrchestrator


class TrainingPipelineOrchestratorTests(unittest.TestCase):
    def test_runs_per_project_executes_all_run_slots(self):
        pipeline_id = "pipeline_test"
        pipeline_state = {
            "id": pipeline_id,
            "status": "running",
            "current_phase": 1,
            "current_run": 1,
            "current_project_index": 0,
            "total_runs": 12,
            "completed_runs": 0,
            "failed_runs": 0,
            "runs": [],
            "config": {
                "phases": [
                    {
                        "phase_number": 1,
                        "name": "phase_one",
                        "exploration_runs_per_project": 6,
                        "shuffle_order": False,
                    }
                ],
                "projects": [
                    {"name": "proj_a", "dataset_path": "D:/datasets/a"},
                    {"name": "proj_b", "dataset_path": "D:/datasets/b"},
                ],
                "thermal_management": {"enabled": False},
                "shared_config": {},
            },
        }

        def _fake_get_pipeline(pid: str):
            self.assertEqual(pid, pipeline_id)
            return dict(pipeline_state)

        def _fake_update_pipeline(pid: str, updates: dict):
            self.assertEqual(pid, pipeline_id)
            pipeline_state.update(updates)
            return dict(pipeline_state)

        orchestrator = PipelineOrchestrator(pipeline_id)

        with (
            patch(
                "bimba3d_backend.app.services.training_pipeline_storage.get_pipeline",
                side_effect=_fake_get_pipeline,
            ),
            patch(
                "bimba3d_backend.app.services.training_pipeline_storage.update_pipeline",
                side_effect=_fake_update_pipeline,
            ),
            patch.object(orchestrator, "_execute_run", return_value=None) as execute_run_mock,
        ):
            orchestrator._run()

        self.assertEqual(execute_run_mock.call_count, 12)

        observed_runs = {call.args[3] for call in execute_run_mock.call_args_list}
        self.assertEqual(observed_runs, {1, 2, 3, 4, 5, 6})

    def test_execute_training_run_allows_missing_learning_file_for_completed_status(self):
        orchestrator = PipelineOrchestrator("pipeline_test")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_images = temp_root / "source_images"
            source_images.mkdir(parents=True, exist_ok=True)
            (source_images / "img_0001.jpg").write_bytes(b"test")

            project_dir = temp_root / "pipeline_project"
            project_dir.mkdir(parents=True, exist_ok=True)
            (project_dir / "config.json").write_text(
                json.dumps({"id": "project_123", "source_dir": str(source_images)}),
                encoding="utf-8",
            )

            data_dir = temp_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            with (
                patch("bimba3d_backend.app.config.DATA_DIR", data_dir),
                patch("bimba3d_backend.worker.pipeline.run_full_pipeline", return_value=None) as run_full_pipeline_mock,
                patch(
                    "bimba3d_backend.app.services.training_pipeline_orchestrator.project_status.get_status",
                    return_value={"status": "completed"},
                ),
            ):
                success, score = orchestrator._execute_training_run(
                    run_config={"phase_number": 1, "max_steps": 7000},
                    project_dir=project_dir,
                    run_id="phase1_pass1_run1_20260424_000000",
                )

        self.assertTrue(success)
        self.assertIsNone(score)
        self.assertEqual(run_full_pipeline_mock.call_args.args[1]["max_steps"], 7000)


if __name__ == "__main__":
    unittest.main()

