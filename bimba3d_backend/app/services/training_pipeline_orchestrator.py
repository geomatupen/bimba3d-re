"""Training pipeline orchestrator - executes cross-project training with thermal management."""
from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from bimba3d_backend.app.config import DATA_DIR
from bimba3d_backend.app.services import status as project_status
from bimba3d_backend.app.services import training_pipeline_storage
from bimba3d_backend.app.services.training_pipeline_storage import phase_run_count

logger = logging.getLogger(__name__)

SELECTOR_OWNED_LEARNED_KEYS = [
    "feature_lr",
    "position_lr_init",
    "scaling_lr",
    "opacity_lr",
    "rotation_lr",
    "densify_grad_threshold",
    "opacity_threshold",
    "lambda_dssim",
]

# Global registry of running orchestrators
_running_orchestrators: dict[str, "PipelineOrchestrator"] = {}


class PipelineOrchestrator:
    """Orchestrates multi-project training pipeline execution with thermal management."""

    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.session_id = uuid.uuid4().hex
        self.should_stop = False
        self.thread: Optional[threading.Thread] = None
        self.current_run_project_name: Optional[str] = None

    def start(self):
        """Start orchestrator in background thread."""
        if self.thread and self.thread.is_alive():
            logger.warning(f"Pipeline {self.pipeline_id} orchestrator already running")
            return

        self.should_stop = False
        training_pipeline_storage.update_pipeline(
            self.pipeline_id,
            {
                "orchestrator_session_id": self.session_id,
                "cooldown_active": False,
                "next_run_scheduled_at": None,
                "cooldown_session_id": None,
            },
        )
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        _running_orchestrators[self.pipeline_id] = self
        logger.info(f"Started orchestrator for pipeline {self.pipeline_id}")

    def stop(self):
        """Signal orchestrator to stop."""
        self.should_stop = True
        logger.info(f"Stopping orchestrator for pipeline {self.pipeline_id}")

    def pause(self):
        """Pause execution (will wait for current run to complete)."""
        pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)
        if pipeline:
            training_pipeline_storage.update_pipeline(self.pipeline_id, {"status": "paused"})

    def resume(self):
        """Resume execution."""
        pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)
        if pipeline:
            training_pipeline_storage.update_pipeline(self.pipeline_id, {"status": "running"})

    def _run(self):
        """Main orchestrator loop."""
        try:
            pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)
            if not pipeline:
                logger.error(f"Pipeline {self.pipeline_id} not found")
                return

            config = pipeline["config"]
            phases = config["phases"]
            projects = config["projects"]
            thermal = config.get("thermal_management", {})

            # Execute each phase
            for phase_idx, phase in enumerate(phases):
                if self.should_stop:
                    break

                phase_num = phase["phase_number"]
                training_pipeline_storage.update_pipeline(self.pipeline_id, {"current_phase": phase_num})

                logger.info(f"Pipeline {self.pipeline_id}: Starting Phase {phase_num} - {phase['name']}")

                # Execute configured runs within the phase.
                phase_run_total = phase_run_count(phase)
                for phase_run_idx in range(phase_run_total):
                    if self.should_stop:
                        break

                    current_run = phase_run_idx + 1
                    training_pipeline_storage.update_pipeline(
                        self.pipeline_id,
                        {
                            "current_run": current_run,
                        },
                    )

                    # Shuffle project order if requested
                    project_order = list(range(len(projects)))
                    if phase.get("shuffle_order", False):
                        random.shuffle(project_order)

                    # Execute each project for this phase run.
                    for proj_idx in project_order:
                        if self.should_stop:
                            break

                        project = projects[proj_idx]
                        training_pipeline_storage.update_pipeline(self.pipeline_id, {"current_project_index": proj_idx})

                        # One project execution per phase run.
                        if self.should_stop:
                            break

                        # Reload pipeline to get latest config (e.g., baseline_run_id from previous phase)
                        pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)
                        if pipeline:
                            updated_projects = pipeline.get("config", {}).get("projects", [])
                            if proj_idx < len(updated_projects):
                                project = updated_projects[proj_idx]

                        # Check if paused between runs.
                        while pipeline and pipeline["status"] == "paused" and not self.should_stop:
                            logger.info(f"Pipeline {self.pipeline_id} paused, waiting...")
                            time.sleep(5)
                            pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)

                        if self.should_stop:
                            break
                        if not pipeline:
                            logger.error(f"Pipeline {self.pipeline_id} disappeared during execution")
                            self.should_stop = True
                            break
                        if pipeline.get("status") in {"stopped", "failed"}:
                            self.should_stop = True
                            break

                        # Test pipeline guard: before any non-baseline test run,
                        # ensure this project has a successful baseline run.
                        config = pipeline.get("config", {})
                        pipeline_type = config.get("pipeline_type", "offline_data")
                        if pipeline_type == "test" and phase.get("phase_number") != 1:
                            baseline_ready = self._ensure_baseline_before_test_run(pipeline, project)
                            if not baseline_ready:
                                logger.error(
                                    f"Pipeline {self.pipeline_id}: Baseline is missing/failed for {project.get('name')}; "
                                    "skipping test run"
                                )
                                continue
                            # Reload pipeline/project after potential baseline creation.
                            pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)
                            if not pipeline:
                                logger.error(f"Pipeline {self.pipeline_id} disappeared after baseline guard")
                                self.should_stop = True
                                break
                            updated_projects = pipeline.get("config", {}).get("projects", [])
                            if proj_idx < len(updated_projects):
                                project = updated_projects[proj_idx]

                        # Execute training run â€” for test pipelines with multiple models: run once per model
                        config = pipeline.get("config", {})
                        pipeline_type = config.get("pipeline_type", "offline_data")
                        source_model_ids = config.get("source_model_ids") or []
                        if not source_model_ids and config.get("source_model_id"):
                            source_model_ids = [config["source_model_id"]]

                        run_outcome = "skipped"
                        cooldown_accounted_for_model_loop = False
                        if pipeline_type == "test" and phase.get("phase_number") != 1 and len(source_model_ids) > 1:
                            for model_id in source_model_ids:
                                if self.should_stop:
                                    break
                                training_pipeline_storage.update_pipeline(self.pipeline_id, {"current_test_model_id": model_id})
                                run_outcome = self._execute_run(
                                    pipeline, project, phase,
                                    phase_run_idx + 1,
                                    test_model_id=model_id,
                                )
                                if (
                                    run_outcome != "skipped"
                                    and thermal.get("enabled", False)
                                    and not self.should_stop
                                ):
                                    cooldown_accounted_for_model_loop = True
                                    self._apply_thermal_management_if_due(thermal)
                        else:
                            single_test_model_id = None
                            if pipeline_type == "test" and phase.get("phase_number") != 1:
                                single_test_model_id = source_model_ids[0] if source_model_ids else config.get("source_model_id")
                            training_pipeline_storage.update_pipeline(self.pipeline_id, {"current_test_model_id": single_test_model_id})

                            run_outcome = self._execute_run(
                                pipeline, project, phase,
                                phase_run_idx + 1,
                                test_model_id=single_test_model_id,
                            )

                        # Skipped runs should not leave cooldown UI state active.
                        if run_outcome == "skipped":
                            training_pipeline_storage.update_pipeline(self.pipeline_id, {
                                "cooldown_active": False,
                                "next_run_scheduled_at": None,
                                "cooldown_session_id": None,
                            })

                        # Apply thermal management (cooldown) between individual runs.
                        if (
                            thermal.get("enabled", False)
                            and not self.should_stop
                            and run_outcome != "skipped"
                            and not cooldown_accounted_for_model_loop
                        ):
                            self._apply_thermal_management_if_due(thermal)

            # Determine final status based on run results
            pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)
            stored_status = str((pipeline or {}).get("status") or "").lower()
            failed_runs = int((pipeline or {}).get("failed_runs", 0) or 0)
            completed_runs = int((pipeline or {}).get("completed_runs", 0) or 0)
            hard_cap_runs = int((pipeline or {}).get("hard_cap_runs", 0) or 0)
            total_runs = int((pipeline or {}).get("total_runs", 0) or 0)

            # Determine status: completed only if no failures and all runs succeeded
            if self.should_stop or stored_status == "stopped":
                final_status = "stopped"
            elif failed_runs > 0 and completed_runs == 0:
                # All runs failed
                final_status = "failed"
            elif failed_runs > 0:
                # Some runs failed but some succeeded
                final_status = "completed_with_failures"
            elif hard_cap_runs > 0:
                # No failures, but one or more runs hit gaussian hard cap.
                final_status = "completed_with_hard_caps"
            else:
                # All runs succeeded
                final_status = "completed"

            training_pipeline_storage.update_pipeline(self.pipeline_id, {
                "status": final_status,
                "current_test_model_id": None,
                "completed_at": datetime.utcnow().isoformat() + "Z"
            })

            logger.info(
                "Pipeline %s finished with status=%s (completed=%s, hard_cap=%s, failed=%s, total=%s)",
                self.pipeline_id,
                final_status,
                completed_runs,
                hard_cap_runs,
                failed_runs,
                total_runs,
            )

        except Exception as e:
            logger.exception(f"Pipeline {self.pipeline_id} failed: {e}")
            training_pipeline_storage.update_pipeline(self.pipeline_id, {
                "status": "failed",
                "last_error": str(e),
                "current_test_model_id": None,
                "cooldown_active": False,
                "next_run_scheduled_at": None,
                "cooldown_session_id": None,
                "completed_at": datetime.utcnow().isoformat() + "Z"
            })

        finally:
            if _running_orchestrators.get(self.pipeline_id) is self:
                _running_orchestrators.pop(self.pipeline_id, None)

    def _get_or_create_project_dir(self, pipeline: dict, project: dict) -> Path:
        """Get or create project directory within pipeline folder.

        Structure:
          {pipeline_folder}/
            â”œâ”€â”€ shared_models/  â† Shared models for cross-project learning
            â”‚   â””â”€â”€ featurewise_ridge_regression/
            â”œâ”€â”€ {project1_name}/  â† Project directory (COLMAP, runs)
            â”œâ”€â”€ {project2_name}/
            â””â”€â”€ ...

        Projects reference original data via config.json source_dir
        Shared model directory enables cross-project knowledge accumulation

        If colmap_source_project_id is specified, copies COLMAP from that project
        """
        config = pipeline["config"]
        pipeline_folder = Path(config["pipeline_folder"])
        project_name = project["name"]
        # Sanitize project name for folder (replace spaces with underscores)
        sanitized_project_name = project_name.replace(" ", "_")
        project_dir = pipeline_folder / sanitized_project_name
        colmap_source_project_id = project.get("colmap_source_project_id")

        # Create shared model directory at pipeline level
        shared_model_dir = pipeline_folder / "shared_models"
        shared_model_dir.mkdir(parents=True, exist_ok=True)

        # Create project directory if it doesn't exist
        project_created = False
        if not project_dir.exists():
            project_dir.mkdir(parents=True, exist_ok=True)
            project_created = True

            # Create symlink to source images directory
            source_path = Path(project["dataset_path"])
            images_link = project_dir / "images"

            if not images_link.exists():
                symlink_success = False
                try:
                    # Create symlink (Windows requires admin or developer mode)
                    import os
                    if os.name == 'nt':
                        # On Windows, try junction first (doesn't require admin)
                        try:
                            import subprocess
                            result = subprocess.run(['mklink', '/J', str(images_link), str(source_path)],
                                         shell=True, check=True, capture_output=True, text=True)
                            symlink_success = True
                            logger.info(f"Created images junction: {images_link} -> {source_path}")
                        except Exception as junction_err:
                            logger.warning(f"Junction creation failed: {junction_err}, trying symlink")
                            try:
                                images_link.symlink_to(source_path, target_is_directory=True)
                                symlink_success = True
                                logger.info(f"Created images symlink: {images_link} -> {source_path}")
                            except OSError as symlink_err:
                                logger.warning(f"Symlink creation also failed: {symlink_err}")
                    else:
                        # Unix/Linux: standard symlink
                        images_link.symlink_to(source_path, target_is_directory=True)
                        symlink_success = True
                        logger.info(f"Created images symlink: {images_link} -> {source_path}")

                except Exception as e:
                    logger.warning(f"Failed to create images symlink/junction: {e}")
                
                # If symlink/junction failed, copy images instead
                if not symlink_success:
                    logger.info(f"Falling back to copying images from {source_path} to {images_link}")
                    try:
                        import shutil
                        images_link.mkdir(parents=True, exist_ok=True)
                        
                        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                        copied_count = 0
                        
                        for img_file in source_path.iterdir():
                            if img_file.is_file() and img_file.suffix in image_extensions:
                                dest_file = images_link / img_file.name
                                if not dest_file.exists():
                                    shutil.copy2(img_file, dest_file)
                                    copied_count += 1
                        
                        logger.info(f"âœ“ Copied {copied_count} images to project folder")
                        
                        if copied_count == 0:
                            raise RuntimeError(f"No images found in source directory: {source_path}")
                    
                    except Exception as copy_err:
                        logger.error(f"Failed to copy images: {copy_err}")
                        # Create a note file for debugging
                        (project_dir / "images_source.txt").write_text(str(source_path))
                        raise RuntimeError(f"Failed to set up images directory: {copy_err}") from copy_err

            # Create config.json with reference to source data
            config_data = {
                "id": str(uuid.uuid4()),
                "name": project_name,
                "source_dir": project["dataset_path"],  # Points to read-only data folder
                "shared_model_dir": str(shared_model_dir),  # Shared models for cross-project learning
                "created_at": datetime.utcnow().isoformat() + "Z",
                "created_by": "training_pipeline",
                "pipeline_id": pipeline["id"],
                "pipeline_name": config.get("name"),
                "pipeline_path": str(pipeline_folder),  # Direct path to pipeline folder for efficient lookup
                **config.get("shared_config", {})
            }

            config_path = project_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            # Store project_id in pipeline config for UI navigation
            self._update_project_id_in_pipeline(pipeline, project["name"], config_data["id"])

            logger.info(f"Created project directory: {project_dir} with shared models at {shared_model_dir}")

        # A previous interrupted setup can leave images/COLMAP folders without config.json.
        # The worker and AI runtime require this file, so repair it before any run starts.
        config_path = project_dir / "config.json"
        if not config_path.exists():
            config_data = {
                "id": str(uuid.uuid4()),
                "name": project_name,
                "source_dir": project["dataset_path"],
                "shared_model_dir": str(shared_model_dir),
                "created_at": datetime.utcnow().isoformat() + "Z",
                "created_by": "training_pipeline",
                "pipeline_id": pipeline["id"],
                "pipeline_name": config.get("name"),
                "pipeline_path": str(pipeline_folder),
                **config.get("shared_config", {}),
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)
            self._update_project_id_in_pipeline(pipeline, project["name"], config_data["id"])
            logger.info(f"Repaired missing project config: {config_path}")

        # A partial copy can leave the project image folder missing files that
        # the copied COLMAP sparse model references. Copy only missing images;
        # do not delete or overwrite existing files.
        source_path = Path(project["dataset_path"])
        images_dir = project_dir / "images"
        if source_path.exists() and source_path.is_dir() and images_dir.exists() and images_dir.is_dir() and not images_dir.is_symlink():
            try:
                import shutil

                image_extensions = {".jpg", ".jpeg", ".png"}
                copied_missing = 0
                for image_file in source_path.iterdir():
                    if not image_file.is_file() or image_file.suffix.lower() not in image_extensions:
                        continue
                    target_file = images_dir / image_file.name
                    if not target_file.exists():
                        shutil.copy2(image_file, target_file)
                        copied_missing += 1
                if copied_missing:
                    logger.info("Repaired image folder for %s by copying %s missing image(s)", project_name, copied_missing)
            except Exception as exc:
                logger.warning("Failed to repair image folder for %s: %s", project_name, exc)

        # Copy COLMAP from source project if specified (do this even if project already exists)
        # This allows restarting pipelines with COLMAP copy enabled
        if colmap_source_project_id:
            target_colmap = project_dir / "outputs" / "sparse"
            if not target_colmap.exists():
                self._copy_colmap_from_source(project_dir, colmap_source_project_id, project_name)
            else:
                logger.info(f"COLMAP already exists for {project_name}, skipping copy")

        return project_dir

    def _update_project_id_in_pipeline(self, pipeline: dict, project_name: str, project_id: str):
        """Update project_id in pipeline config for UI navigation."""
        try:
            config = pipeline.get("config", {})
            projects = config.get("projects", [])
            for proj in projects:
                if proj.get("name") == project_name and not proj.get("project_id"):
                    proj["project_id"] = project_id
                    training_pipeline_storage.update_pipeline(pipeline["id"], {"config": config})
                    break
        except Exception as e:
            logger.warning(f"Failed to update project_id in pipeline config: {e}")

    def _copy_colmap_from_source(self, target_project_dir: Path, source_project_id: str, target_project_name: str):
        """Copy COLMAP sparse reconstruction and config from source project to target project.
        
        Uses the same project lookup logic as the rest of the system to find projects
        in DATA_DIR, pipeline folders, etc.
        """
        try:
            import shutil
            from bimba3d_backend.app.api.projects import _find_project_dir

            # Use the standard project finder which searches DATA_DIR and pipeline folders
            source_project_dir = _find_project_dir(source_project_id)
            
            if not source_project_dir:
                logger.warning(f"Source project {source_project_id} not found, skipping COLMAP copy")
                return
            
            logger.info(f"Found source project for COLMAP copy: {source_project_dir}")

            # Check if source has COLMAP outputs
            source_colmap = source_project_dir / "outputs" / "sparse"
            if not source_colmap.exists() or not (source_colmap / "0").exists():
                logger.warning(f"Source project {source_project_id} has no COLMAP outputs, skipping copy")
                return

            # Create target outputs directory
            target_outputs = target_project_dir / "outputs"
            target_outputs.mkdir(parents=True, exist_ok=True)

            # Copy COLMAP sparse directory
            target_colmap = target_outputs / "sparse"
            if target_colmap.exists():
                logger.info(f"Target already has COLMAP, skipping copy for {target_project_name}")
            else:
                shutil.copytree(source_colmap, target_colmap)
                logger.info(f"âœ“ Copied COLMAP from project {source_project_id} to {target_project_name} (saved ~15-30 minutes)")

            # Copy COLMAP-related config settings from source project
            source_config_path = source_project_dir / "config.json"
            target_config_path = target_project_dir / "config.json"
            
            if source_config_path.exists() and target_config_path.exists():
                try:
                    with open(source_config_path, "r") as f:
                        source_config = json.load(f)
                    
                    with open(target_config_path, "r") as f:
                        target_config = json.load(f)
                    
                    # Copy COLMAP-related settings
                    colmap_keys = [
                        "colmap_camera_model",
                        "colmap_camera_params",
                        "colmap_matcher",
                        "colmap_vocab_tree_path",
                        "colmap_gpu_index",
                        "colmap_num_threads",
                        "colmap_use_gpu",
                        "image_width",
                        "image_height",
                        "focal_length",
                        "camera_model",
                    ]
                    
                    updated = False
                    for key in colmap_keys:
                        if key in source_config and key not in target_config:
                            target_config[key] = source_config[key]
                            updated = True
                    
                    if updated:
                        with open(target_config_path, "w") as f:
                            json.dump(target_config, f, indent=2)
                        logger.info(f"âœ“ Copied COLMAP config settings to {target_project_name}")
                
                except Exception as e:
                    logger.warning(f"Failed to copy COLMAP config settings: {e}")

        except Exception as e:
            logger.error(f"Failed to copy COLMAP from source project: {e}", exc_info=True)
            # Non-fatal: pipeline will run COLMAP if copy fails

    def _execute_run(
        self,
        pipeline: dict,
        project: dict,
        phase: dict,
        run_number: int,
        test_model_id: Optional[str] = None,
    ) -> str:
        """Execute a single training run.

        Args:
            test_model_id: If provided, seed this specific model for this run (multi-model test)

        Returns:
            "skipped" when no run was executed (already completed)
            "completed" when a run was attempted (success/failure)
        """
        project_name = project["name"]
        self.current_run_project_name = project_name
        run_id: Optional[str] = None

        logger.info(
            f"Pipeline {self.pipeline_id}: Running {project_name}, "
            f"phase {phase['phase_number']}, run {run_number}"
        )

        try:
            # Get/create project directory in pipeline folder
            project_dir = self._get_or_create_project_dir(pipeline, project)

            # For test pipelines: seed the model weights into the project before running
            config = pipeline.get("config", {})
            pipeline_type = config.get("pipeline_type", "offline_data")
            slot_key = self._retry_slot_key(project_name, phase.get("phase_number"), run_number, test_model_id)
            legacy_slot_key = self._retry_slot_key(project_name, phase.get("phase_number"), run_number, None)
            retry_fixed = config.get("retry_fixed_params")
            retry_targets = config.get("retry_target_slots")
            retry_mode_active = bool(config.get("retry_mode_active"))
            retry_slot_targeted = (
                retry_mode_active
                and isinstance(retry_targets, list)
                and ({slot_key, legacy_slot_key} & {str(target) for target in retry_targets})
            )
            retry_slot_active = (
                retry_mode_active
                and isinstance(retry_fixed, dict)
                and (
                    (isinstance(retry_fixed.get(slot_key), dict) and bool(retry_fixed.get(slot_key)))
                    or (isinstance(retry_fixed.get(legacy_slot_key), dict) and bool(retry_fixed.get(legacy_slot_key)))
                )
            )
            if pipeline_type == "test" and phase.get("phase_number") != 1:
                # Use specific model if provided (multi-model test), otherwise fall back to config
                model_to_seed = test_model_id or config.get("source_model_id")
                if model_to_seed:
                    from bimba3d_backend.app.services import model_registry, workflow_model_seeding
                    model_record = model_registry.resolve_reusable_model(model_to_seed)
                    workflow_model = workflow_model_seeding.read_workflow_model(str(model_to_seed)) if not model_record else None
                    if workflow_model:
                        seeded = workflow_model_seeding.seed_workflow_model_into_project(workflow_model, project_dir)
                        logger.info(f"Seeded workflow model '{model_to_seed}' into {project_name} at {seeded}")
                    elif model_record:
                        seeded = model_registry.seed_learner_weights_into_project(model_record, project_dir)
                        if seeded:
                            logger.info(f"Seeded model '{model_to_seed}' into {project_name}")
                        else:
                            logger.warning(f"Failed to seed model '{model_to_seed}' into {project_name}")
                    else:
                        raise FileNotFoundError(f"Selected test model was not found: {model_to_seed}")

            # Check if this run already completed successfully in pipeline history.
            # For multi-model test pipelines, match test_model_id so that a success
            # by one model does not suppress another model's (re)run in the same slot.
            def _model_matches(ex: dict[str, Any]) -> bool:
                if not test_model_id:
                    return True
                existing_model_id = ex.get("test_model_id") or ex.get("source_model_id") or ex.get("model_id")
                return str(existing_model_id or "") == str(test_model_id)
            for existing in pipeline.get("runs", []):
                if (
                    str(existing.get("project_name") or "") == project_name
                    and str(existing.get("phase")) == str(phase["phase_number"])
                    and str(existing.get("run")) == str(run_number)
                    and str(existing.get("status") or "") in {"success", "partial_completed"}
                    and _model_matches(existing)
                ):
                    logger.info(
                        f"Pipeline {self.pipeline_id}: Run already completed for {project_name} "
                        f"phase {phase['phase_number']} run {run_number}, skipping"
                    )
                    return "skipped"
                if (
                    str(existing.get("project_name") or "") == project_name
                    and str(existing.get("phase")) == str(phase["phase_number"])
                    and str(existing.get("run")) == str(run_number)
                    and str(existing.get("status") or "").lower() == "failed"
                    and _model_matches(existing)
                    # retry_slot_active: offline_data pipelines with fixed params
                    # retry_slot_targeted: test pipelines where slot is in retry_target_slots
                    and not retry_slot_active
                    and not retry_slot_targeted
                ):
                    logger.info(
                        f"Pipeline {self.pipeline_id}: Run previously failed for {project_name} "
                        f"phase {phase['phase_number']} run {run_number}; skipping during normal resume"
                    )
                    return "skipped"
                if (
                    str(existing.get("project_name") or "") == project_name
                    and str(existing.get("phase")) == str(phase["phase_number"])
                    and str(existing.get("run")) == str(run_number)
                    and str(existing.get("status") or "").lower() == "hard_cap_reached"
                    and _model_matches(existing)
                    and not retry_slot_targeted
                ):
                    logger.info(
                        f"Pipeline {self.pipeline_id}: Run reached gaussian hard cap for {project_name} "
                        f"phase {phase['phase_number']} run {run_number}; skipping unless hard-cap retry is selected"
                    )
                    return "skipped"

            # For phase 1 (baseline), also check if baseline run directory exists and succeeded
            if phase["phase_number"] == 1:
                runs_root = project_dir / "runs"
                if runs_root.exists() and runs_root.is_dir():
                    # Look for existing baseline run (first run alphabetically)
                    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.name)
                    if run_dirs:
                        existing_baseline_dir = run_dirs[0]
                        # Check if it completed successfully
                        analytics_file = existing_baseline_dir / "analytics" / "run_analytics_v1.json"
                        if analytics_file.exists():
                            try:
                                with open(analytics_file, "r", encoding="utf-8") as fh:
                                    analytics = json.load(fh)
                                    summary = analytics.get("summary", {})
                                    status = str(summary.get("status", "")).lower()
                                    if status in {"completed", "success", "done"}:
                                        logger.info(
                                            f"Pipeline {self.pipeline_id}: Baseline run {existing_baseline_dir.name} "
                                            f"already completed successfully for {project_name}, skipping"
                                        )
                                        # Store baseline_run_id if not already stored
                                        if not project.get("baseline_run_id"):
                                            config = pipeline.get("config", {})
                                            projects = config.get("projects", [])
                                            for proj in projects:
                                                if proj.get("name") == project_name:
                                                    proj["baseline_run_id"] = existing_baseline_dir.name
                                                    training_pipeline_storage.update_pipeline(self.pipeline_id, {"config": config})
                                                    logger.info(f"Stored existing baseline_run_id={existing_baseline_dir.name} for project {project_name}")
                                                    break
                                        return "skipped"
                                    else:
                                        logger.info(
                                            f"Pipeline {self.pipeline_id}: Existing baseline run {existing_baseline_dir.name} "
                                            f"for {project_name} has status '{status}' - will re-run"
                                        )
                            except Exception as e:
                                logger.warning(f"Failed to read baseline analytics for {project_name}: {e}")

            # Build run configuration
            run_config = self._build_run_config(
                pipeline,
                project,
                phase,
                run_number,
                test_model_id=test_model_id,
            )

            # Generate unique run ID with project name prefix
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            # Sanitize project name for use in run_id
            project_name_safe = project_name.lower().replace(" ", "_").replace("-", "_")
            # Include model name in run_id for multi-model tests
            if test_model_id:
                model_short = test_model_id.split("_")[-1][:12] if "_" in test_model_id else test_model_id[:12]
                run_id = f"{project_name_safe}_phase{phase['phase_number']}_run{run_number}_{model_short}_{timestamp}"
            else:
                run_id = f"{project_name_safe}_phase{phase['phase_number']}_run{run_number}_{timestamp}"

            training_pipeline_storage.update_pipeline(
                self.pipeline_id,
                {
                    "active_run": {
                        "project_name": project_name,
                        "run_id": run_id,
                        "phase": phase["phase_number"],
                        "run": run_number,
                        "test_model_id": test_model_id,
                        "status": "running",
                        "started_at": datetime.utcnow().isoformat() + "Z",
                    }
                },
            )

            # Execute actual training
            logger.info(
                f"Pipeline {self.pipeline_id}: Starting training run {run_id} for {project_name} "
                f"phase {phase['phase_number']} run {run_number}"
            )
            success, score, result_status = self._execute_training_run(run_config, project_dir, run_id, test_model_id=test_model_id)

            # Generate run name
            phase_name = phase.get("name", f"Phase {phase['phase_number']}")
            run_name = f"{phase_name} - Phase {phase['phase_number']} Run {run_number}"

            # Record result â€” include per-group multipliers for overview display
            # Read from run analytics if available, otherwise use what was injected into params
            group_mults_for_log: dict = {}
            analytics_path = project_dir / "runs" / run_id / "analytics" / "run_analytics_v1.json"
            if analytics_path.exists():
                try:
                    _analytics = json.loads(analytics_path.read_text(encoding="utf-8"))
                    _insights = (_analytics.get("ai") or {}).get("input_mode_insights") or {}
                    group_mults_for_log = dict(_insights.get("group_multipliers") or {})
                except Exception:
                    pass
            # Fallback: use what was sent to the worker
            if not group_mults_for_log:
                for gk, pk in (("geometry_lr", "geometry_lr_multiplier"),
                                ("appearance_lr", "appearance_lr_multiplier"),
                                ("densification", "scale_lr_multiplier")):
                    v = run_config.get(pk)
                    if isinstance(v, (int, float)):
                        group_mults_for_log[gk] = {"multiplier": float(v)}

            run_result = {
                "project_name": project_name,
                "phase": phase["phase_number"],
                "run": run_number,
                "run_id": run_id,
                "run_name": run_name,
                "status": result_status,
                "score": score,
                "test_model_id": test_model_id,
                "group_multipliers": group_mults_for_log,
                "completed_at": datetime.utcnow().isoformat() + "Z",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            training_pipeline_storage.add_run_result(self.pipeline_id, run_result)
            self._clear_retry_slot_entry(
                pipeline,
                project_name=project_name,
                phase_num=phase.get("phase_number"),
                run_number=run_number,
                test_model_id=test_model_id,
            )

            if success:
                logger.info(f"Pipeline {self.pipeline_id}: Run completed with status={result_status}, score={score}")

                # If this was a baseline run (phase 1), store run_id as baseline for future phases
                if phase["phase_number"] == 1:
                    # Update the project dict in pipeline config to include baseline_run_id
                    pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)
                    if pipeline:
                        config = pipeline.get("config", {})
                        projects = config.get("projects", [])
                        for proj in projects:
                            if proj.get("name") == project_name:
                                proj["baseline_run_id"] = run_id
                                training_pipeline_storage.update_pipeline(self.pipeline_id, {"config": config})
                                try:
                                    project_status.update_base_session_id(str(project.get("id") or ""), run_id)
                                except Exception as exc:
                                    logger.warning(
                                        "Failed to store project base_session_id=%s for %s: %s",
                                        run_id,
                                        project_name,
                                        exc,
                                    )
                                logger.info(f"Stored baseline_run_id={run_id} for project {project_name}")
                                break
            else:
                logger.warning(f"Pipeline {self.pipeline_id}: Run failed")

            return "completed"

        except Exception as e:
            logger.exception(f"Pipeline {self.pipeline_id}: Run execution failed: {e}")
            training_pipeline_storage.add_run_result(
                self.pipeline_id,
                {
                    "project_name": project_name,
                    "phase": phase.get("phase_number"),
                    "run": run_number,
                    "run_id": run_id,
                    "status": "failed",
                    "score": None,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                },
            )
            self._clear_retry_slot_entry(
                pipeline,
                project_name=project_name,
                phase_num=phase.get("phase_number"),
                run_number=run_number,
                test_model_id=test_model_id,
            )
            training_pipeline_storage.update_pipeline(self.pipeline_id, {"last_error": str(e)})
            return "completed"

        finally:
            training_pipeline_storage.update_pipeline(self.pipeline_id, {"active_run": None})
            self.current_run_project_name = None

    def _retry_slot_key(self, project_name: str, phase_num: Any, run_number: Any, model_id: Any = None) -> str:
        model_part = str(model_id or "").strip()
        return f"{project_name}|{phase_num}|{run_number}|{model_part}"

    def _clear_retry_slot_entry(
        self,
        pipeline: dict,
        *,
        project_name: str,
        phase_num: Any,
        run_number: Any,
        test_model_id: Any = None,
    ) -> None:
        """Remove a retry queue slot after the retry attempt is consumed."""
        try:
            config = pipeline.get("config", {})
            retry_fixed = config.get("retry_fixed_params")
            retry_targets = config.get("retry_target_slots")
            key = self._retry_slot_key(project_name, phase_num, run_number, test_model_id)
            legacy_key = self._retry_slot_key(project_name, phase_num, run_number, None)

            changed = False
            if isinstance(retry_fixed, dict):
                popped = retry_fixed.pop(key, None)
                popped_legacy = retry_fixed.pop(legacy_key, None)
                if popped is not None or popped_legacy is not None:
                    config["retry_fixed_params"] = retry_fixed
                    changed = True

            if isinstance(retry_targets, list):
                target_values = {str(target) for target in retry_targets}
                if key in target_values or legacy_key in target_values:
                    config["retry_target_slots"] = [
                        target
                        for target in retry_targets
                        if str(target) not in {key, legacy_key}
                    ]
                    changed = True

            remaining_targets = config.get("retry_target_slots")
            remaining_count = len(remaining_targets) if isinstance(remaining_targets, list) else 0
            if changed and remaining_count <= 0:
                config["retry_mode_active"] = False
                config["retry_include_hard_cap"] = False
            if changed:
                training_pipeline_storage.update_pipeline(self.pipeline_id, {"config": config})
        except Exception:
            logger.debug("Failed to clear retry queue slot", exc_info=True)

    def _is_successful_run_dir(self, run_dir: Path) -> bool:
        """Return True when run analytics exist and status indicates success/completion."""
        analytics_file = run_dir / "analytics" / "run_analytics_v1.json"
        if not analytics_file.exists():
            return False
        try:
            with open(analytics_file, "r", encoding="utf-8") as fh:
                analytics = json.load(fh)
            summary = analytics.get("summary", {}) if isinstance(analytics, dict) else {}
            status = str(summary.get("status", "")).lower()
            return status in {"completed", "success", "done"}
        except Exception:
            return False

    def _find_existing_successful_baseline_run_id(self, project_dir: Path) -> Optional[str]:
        """Find a successful baseline run in project runs directory, if any."""
        runs_root = project_dir / "runs"
        if not runs_root.exists() or not runs_root.is_dir():
            return None

        run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.name)
        for run_dir in run_dirs:
            if not self._is_successful_run_dir(run_dir):
                continue

            # Prefer explicit mode from run config.
            run_cfg_file = run_dir / "run_config.json"
            run_mode = None
            if run_cfg_file.exists():
                try:
                    with open(run_cfg_file, "r", encoding="utf-8") as fh:
                        run_cfg = json.load(fh)
                    if isinstance(run_cfg, dict):
                        run_mode = str(run_cfg.get("mode", "") or "").strip().lower()
                except Exception:
                    run_mode = None

            # Fallback to analytics summary mode.
            if not run_mode:
                analytics_file = run_dir / "analytics" / "run_analytics_v1.json"
                try:
                    with open(analytics_file, "r", encoding="utf-8") as fh:
                        analytics = json.load(fh)
                    summary = analytics.get("summary", {}) if isinstance(analytics, dict) else {}
                    run_mode = str(summary.get("mode", "") or "").strip().lower()
                except Exception:
                    run_mode = None

            if run_mode == "baseline":
                return run_dir.name

        return None

    def _persist_project_baseline_run_id(self, pipeline: dict, project_name: str, baseline_run_id: str) -> None:
        """Persist baseline_run_id into pipeline config project entry."""
        config = pipeline.get("config", {})
        projects = config.get("projects", [])
        for proj in projects:
            if proj.get("name") == project_name:
                proj["baseline_run_id"] = baseline_run_id
                training_pipeline_storage.update_pipeline(self.pipeline_id, {"config": config})
                return

    def _ensure_baseline_before_test_run(self, pipeline: dict, project: dict) -> bool:
        """For test pipelines, ensure a successful baseline exists before non-baseline runs."""
        try:
            project_name = str(project.get("name") or "")
            project_dir = self._get_or_create_project_dir(pipeline, project)

            # 1) baseline_run_id already set and successful.
            baseline_run_id = str(project.get("baseline_run_id") or "").strip()
            if baseline_run_id:
                run_dir = project_dir / "runs" / baseline_run_id
                if run_dir.exists() and self._is_successful_run_dir(run_dir):
                    return True

            # 2) discover any existing successful baseline run and persist it.
            discovered = self._find_existing_successful_baseline_run_id(project_dir)
            if discovered:
                self._persist_project_baseline_run_id(pipeline, project_name, discovered)
                logger.info(
                    f"Pipeline {self.pipeline_id}: Using existing baseline {discovered} for {project_name} before test"
                )
                return True

            # 3) no baseline found -> run baseline first.
            config = pipeline.get("config", {})
            phases = config.get("phases", [])
            baseline_phase = next((p for p in phases if int(p.get("phase_number", 0)) == 1), None)
            if not isinstance(baseline_phase, dict):
                baseline_phase = {
                    "phase_number": 1,
                    "name": "Baseline",
                    "update_model": False,
                    "context_jitter": False,
                    "session_execution_mode": "test",
                }

            logger.info(
                f"Pipeline {self.pipeline_id}: Baseline missing for {project_name}; running baseline before test"
            )
            self._execute_run(
                pipeline,
                project,
                baseline_phase,
                run_number=1,
                test_model_id=None,
            )

            # Verify baseline exists after attempt.
            pipeline_after = training_pipeline_storage.get_pipeline(self.pipeline_id) or pipeline
            cfg_after = pipeline_after.get("config", {})
            projects_after = cfg_after.get("projects", [])
            project_after = next((p for p in projects_after if p.get("name") == project_name), project)
            baseline_after = str(project_after.get("baseline_run_id") or "").strip()
            if baseline_after:
                run_dir_after = project_dir / "runs" / baseline_after
                if run_dir_after.exists() and self._is_successful_run_dir(run_dir_after):
                    return True

            discovered_after = self._find_existing_successful_baseline_run_id(project_dir)
            if discovered_after:
                self._persist_project_baseline_run_id(pipeline_after, project_name, discovered_after)
                return True

            logger.error(
                f"Pipeline {self.pipeline_id}: Could not establish successful baseline for {project_name}"
            )
            return False
        except Exception as exc:
            logger.exception(
                f"Pipeline {self.pipeline_id}: Baseline guard failed for {project.get('name')}: {exc}"
            )
            return False

    def _build_run_config(
        self,
        pipeline: dict,
        project: dict,
        phase: dict,
        run_number: int,
        test_model_id: Optional[str] = None,
    ) -> dict:
        """Build configuration for a single training run."""
        config = pipeline["config"]
        shared_config = config["shared_config"]

        # Start with shared config
        run_config = shared_config.copy()

        # Apply phase-specific overrides
        if phase.get("strategy_override"):
            run_config["ai_selector_strategy"] = phase["strategy_override"]

        # For non-baseline phases, ensure AI input mode is preserved
        # unless explicitly overridden by phase config
        if not phase.get("preset_override") and "ai_input_mode" in shared_config:
            run_config["ai_input_mode"] = shared_config["ai_input_mode"]

        # Pipeline AI phases must use the same Core AI optimization path as manual
        # project runs. Without this, the worker applies an initial AI preset but
        # never writes local learner state or updates the model.
        if run_config.get("ai_input_mode") and phase.get("phase_number") != 1:
            run_config["tune_scope"] = "core_ai_optimization"

        if phase.get("preset_override"):
            run_config["preset_override"] = phase["preset_override"]

        run_jitter_only = phase.get("run_jitter_only")
        # For exploration phases with context_jitter enabled: always use pure jitter
        # (no model prediction). run_jitter_only is automatically true when context_jitter
        # is on â€” there's no point querying an untrained model.
        if run_jitter_only is None:
            run_jitter_only = bool(phase.get("context_jitter", False))
        # Override: if context_jitter is on, always force jitter_only regardless of config
        if phase.get("context_jitter", False):
            run_jitter_only = True
        run_config["run_jitter_only"] = bool(run_jitter_only)

        # Session execution mode
        run_config["session_execution_mode"] = phase.get("session_execution_mode", "train")

        # Update model flag
        run_config["update_model"] = phase.get("update_model", True)

        # Test pipeline overrides: force test mode, no model update, no jitter
        pipeline_type = config.get("pipeline_type", "offline_data")
        if pipeline_type == "test" and phase.get("phase_number") != 1:
            run_config["session_execution_mode"] = "test"
            # If "contribute_to_training" is enabled, allow model updates
            contribute = config.get("contribute_to_training", False)
            run_config["update_model"] = bool(contribute)
            run_config["run_jitter_mode"] = None
            run_config["run_jitter_only"] = False
            # For multi-model test pipelines, bind AI mode/strategy to the active test model.
            # Without this, a seeded neural checkpoint can still be executed through ridge mode
            # inherited from shared_config (e.g. exif_compact_featurewise).
            active_test_model_id = test_model_id or config.get("source_model_id")
            if active_test_model_id:
                try:
                    from bimba3d_backend.app.services import model_registry, workflow_model_seeding

                    model_record = model_registry.resolve_reusable_model(str(active_test_model_id))
                    workflow_model = workflow_model_seeding.read_workflow_model(str(active_test_model_id)) if not model_record else None
                    if workflow_model:
                        ai_profile = workflow_model_seeding.model_ai_profile(workflow_model)
                        evaluation_step = workflow_model_seeding.model_evaluation_step(workflow_model)
                        if not evaluation_step:
                            raise ValueError(
                                "Selected workflow model does not declare model_evaluation_step. "
                                "Retrain the model from valid Training Data."
                            )
                        run_config["source_workflow_model_id"] = workflow_model.model_id
                        run_config["source_model_name"] = workflow_model.model_name
                        run_config["source_model_family"] = workflow_model.model_family
                        run_config["source_training_data_id"] = workflow_model.source_training_data_id
                        run_config["model_evaluation_step"] = evaluation_step
                        run_config["score_reference_step"] = evaluation_step
                    else:
                        ai_profile = model_registry.resolve_model_ai_profile(model_record if isinstance(model_record, dict) else None)

                    model_mode = str(ai_profile.get("ai_input_mode") or "").strip().lower()
                    model_strategy = str(ai_profile.get("ai_selector_strategy") or "").strip().lower()

                    if model_mode == "exif_compact_featurewise":
                        run_config["ai_input_mode"] = model_mode
                    if model_strategy in {
                        "featurewise_ridge_regression",
                        "featurewise_mlp",
                        "compact_featurewise_ridge_regression",
                        "compact_featurewise_mlp",
                        "compact_descriptor_mlp",
                    }:
                        run_config["ai_selector_strategy"] = model_strategy

                    # Keep tune scope enabled for AI-learning payload persistence in test mode.
                    if run_config.get("ai_input_mode"):
                        run_config["tune_scope"] = "core_ai_optimization"

                    logger.info(
                        "Pipeline %s: Applied test model AI profile model_id=%s mode=%s strategy=%s",
                        self.pipeline_id,
                        active_test_model_id,
                        run_config.get("ai_input_mode"),
                        run_config.get("ai_selector_strategy"),
                    )
                except Exception as e:
                    logger.warning(
                        "Pipeline %s: Failed to apply AI profile from test model %s: %s",
                        self.pipeline_id,
                        active_test_model_id,
                        e,
                    )
        elif pipeline_type == "offline_data" and phase.get("phase_number") != 1:
            run_config["ai_input_mode"] = shared_config.get("ai_input_mode")
            run_config["ai_selector_strategy"] = shared_config.get("ai_selector_strategy")
            if run_config.get("ai_input_mode"):
                run_config["tune_scope"] = "core_ai_optimization"
            run_config["run_jitter_mode"] = None
            run_config["run_jitter_only"] = True

        # Baseline reference
        if project.get("baseline_run_id"):
            run_config["baseline_session_id"] = project["baseline_run_id"]

        # Retry-failed override (training pipelines only):
        # if captured params from the failed run exist, replay the same LR values
        # and disable fresh selector/jitter exploration for this slot.
        retry_fixed = config.get("retry_fixed_params")
        slot_key = self._retry_slot_key(project.get("name", ""), phase.get("phase_number"), run_number)
        if pipeline_type == "offline_data" and bool(config.get("retry_mode_active")) and isinstance(retry_fixed, dict):
            fixed_params = retry_fixed.get(slot_key)
            if isinstance(fixed_params, dict) and fixed_params:
                for k in SELECTOR_OWNED_LEARNED_KEYS:
                    if isinstance(fixed_params.get(k), (int, float)):
                        run_config[k] = float(fixed_params[k])

                # Retry the same offline-data multiplier slot. Keep the same AI
                # profile used by normal phase-2 exploration so the gsplat
                # engine follows the offline learning path, not removed legacy
                # online adaptation.
                run_config["ai_input_mode"] = shared_config.get("ai_input_mode")
                run_config["ai_selector_strategy"] = shared_config.get("ai_selector_strategy")
                if run_config.get("ai_input_mode"):
                    run_config["tune_scope"] = "core_ai_optimization"
                run_config["run_jitter_mode"] = None
                run_config["run_jitter_only"] = True

        # Run metadata
        run_config["pipeline_id"] = pipeline["id"]
        run_config["phase_number"] = phase["phase_number"]
        run_config["phase_run_number"] = run_number
        run_config["phase_run"] = run_number
        run_config["phase_runs_total"] = phase_run_count(phase)
        run_config["test_model_id"] = test_model_id

        return run_config

    def _execute_training_run(
        self,
        run_config: dict,
        project_dir: Path,
        run_id: str,
        test_model_id: Optional[str] = None,
    ) -> tuple[bool, Optional[float], str]:
        """Execute actual training run using the project pipeline system.

        This calls the same training system used by individual projects.
        """
        from bimba3d_backend.worker import pipeline

        phase_num = run_config.get("phase_number", 1)

        # Determine stage based on phase and whether COLMAP is complete and successful
        colmap_sparse_dir = project_dir / "outputs" / "sparse" / "0"
        colmap_complete = False

        if colmap_sparse_dir.exists():
            # Check if COLMAP completed successfully by looking for required files
            cameras_file = colmap_sparse_dir / "cameras.bin"
            images_file = colmap_sparse_dir / "images.bin"
            points_file = colmap_sparse_dir / "points3D.bin"

            if cameras_file.exists() and images_file.exists() and points_file.exists():
                # All required COLMAP files exist, consider it complete
                colmap_complete = True
                logger.info(f"âœ“ COLMAP outputs found and complete for {project_dir.name}")
            else:
                logger.warning(f"âš  COLMAP directory exists but incomplete for {project_dir.name} - will re-run COLMAP")
                # Clean up incomplete COLMAP outputs to avoid lock conflicts
                import shutil
                try:
                    shutil.rmtree(colmap_sparse_dir.parent, ignore_errors=True)
                    # Also remove database files that may be locked from a killed process
                    outputs_dir = project_dir / "outputs"
                    for db_file in outputs_dir.glob("database.db*"):
                        try:
                            db_file.unlink(missing_ok=True)
                        except Exception:
                            pass
                    logger.info(f"Cleaned up incomplete COLMAP outputs for {project_dir.name}")
                except Exception as e:
                    logger.warning(f"Failed to clean up incomplete COLMAP for {project_dir.name}: {e}")

        if phase_num == 1 and not colmap_complete:
            # Phase 1 and COLMAP not complete: Run full pipeline (COLMAP + training)
            stage = "full"
            logger.info(f"Phase 1: Running full pipeline (COLMAP + training) for {project_dir.name}")
        else:
            # Phase 2+ OR COLMAP already complete: Only run training
            stage = "train_only"
            if phase_num == 1:
                logger.info(f"Phase 1: Skipping COLMAP (already complete), running training only for {project_dir.name}")

        max_steps_value = run_config.get("max_steps")
        if max_steps_value is None:
            raise ValueError(
                "Pipeline run config must include max_steps. Set the training step count in the pipeline shared config."
            )

        # Build params for the worker
        params = {
            "run_id": run_id,
            "stage": stage,
            "mode": "baseline" if phase_num == 1 else "modified",
            "max_steps": int(max_steps_value),
            "gaussian_hard_cap": int(run_config.get("gaussian_hard_cap", 6_000_000)),
            "eval_interval": run_config.get("eval_interval", 1000),
            "log_interval": run_config.get("log_interval", 100),
            "densify_until_iter": run_config.get("densify_until_iter", 4000),
            "images_max_size": run_config.get("images_max_size"),
            "ai_input_mode": run_config.get("ai_input_mode"),
            "ai_selector_strategy": run_config.get("ai_selector_strategy"),
            "tune_scope": run_config.get("tune_scope"),
            "trend_scope": run_config.get("trend_scope", "run"),
            "session_execution_mode": run_config.get("session_execution_mode", "train"),
            "baseline_session_id": run_config.get("baseline_session_id"),
            "update_model": run_config.get("update_model", True),
            "context_jitter_enabled": run_config.get("context_jitter_enabled", False),
            "context_jitter_mode": "uniform",  # Always use uniform mode for consistent behavior
            "preset_override": run_config.get("preset_override"),
            "pipeline_id": run_config.get("pipeline_id"),
            "phase": phase_num,
            "run": run_config.get("phase_run_number", run_config.get("phase_run", 1)),
            "phase_run": run_config.get("phase_run", 1),
            "phase_runs_total": run_config.get("phase_runs_total", 1),
        }

        # Apply pre-generated log space multipliers from pipeline config.
        # Training pipelines no longer use legacy per-run jitter generation.
        current_pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id) or {}
        pipeline_config = current_pipeline.get("config", {}) if isinstance(current_pipeline, dict) else {}
        pre_gen_multipliers = pipeline_config.get("pre_generated_log_multipliers", {})
        phase_run_number = int(run_config.get("phase_run_number") or run_config.get("phase_run") or 1)
        current_index = max(0, phase_run_number - 1) if phase_num != 1 else 0

        geometry_multipliers = pre_gen_multipliers.get("geometry_lr", []) if isinstance(pre_gen_multipliers, dict) else []
        appearance_multipliers = pre_gen_multipliers.get("appearance_lr", []) if isinstance(pre_gen_multipliers, dict) else []
        scale_multipliers = pre_gen_multipliers.get("scale_lr", []) if isinstance(pre_gen_multipliers, dict) else []

        pipeline_type = str(pipeline_config.get("pipeline_type") or "").strip().lower()
        use_fixed_schedule = (
            pipeline_type != "test"
            and phase_num != 1
            and bool(geometry_multipliers or appearance_multipliers or scale_multipliers)
        )
        if use_fixed_schedule and (
            current_index >= len(geometry_multipliers)
            or current_index >= len(appearance_multipliers)
            or current_index >= len(scale_multipliers)
        ):
            raise RuntimeError(
                f"Workflow multiplier schedule is incomplete for phase run {phase_run_number}. "
                "Regenerate and save the log-space schedule before running exploration."
            )
        geom_mult = float(geometry_multipliers[current_index]) if use_fixed_schedule else 1.0
        app_mult = float(appearance_multipliers[current_index]) if use_fixed_schedule else 1.0
        scale_mult = float(scale_multipliers[current_index]) if use_fixed_schedule else 1.0

        if use_fixed_schedule:
            pipeline_config["multiplier_current_index"] = current_index
            training_pipeline_storage.update_pipeline(self.pipeline_id, {"config": pipeline_config})

        candidate_logs = pipeline_config.get("test_candidate_log_multipliers")
        if isinstance(candidate_logs, dict) and candidate_logs:
            params["candidate_log_multipliers_by_group"] = candidate_logs
            params["test_candidate_seed"] = pipeline_config.get("test_candidate_seed")
            params["test_candidate_count"] = pipeline_config.get("test_candidate_count")

        if use_fixed_schedule:
            params["geometry_lr_multiplier"] = geom_mult
            params["appearance_lr_multiplier"] = app_mult
            params["scale_lr_multiplier"] = scale_mult

        # Offline-data exploration uses pre-generated values. Test runs use
        # candidate_log_multipliers_by_group only for model scoring.
        has_pre_generated = use_fixed_schedule
        jitter_enabled = run_config.get("context_jitter", False)

        if has_pre_generated:
            # Use the pre-generated values, skip random sampling
            params["run_jitter_mode"] = None
        elif pipeline_type != "test" and jitter_enabled and phase_num != 1:
            raise RuntimeError(
                "Workflow multiplier schedule is missing. Regenerate and save the log-space schedule before running exploration."
            )
        else:
            params["run_jitter_mode"] = None
        
        params.update({
            "run_jitter_only": run_config.get("run_jitter_only", False),
            "test_model_id": test_model_id or run_config.get("test_model_id"),
            # Storage management
            "save_eval_images": run_config.get("save_eval_images", True),
            "replace_eval_images": run_config.get("replace_eval_images", False),
            "save_checkpoints": run_config.get("save_checkpoints", True),
            "replace_checkpoints": run_config.get("replace_checkpoints", False),
            "save_final_splat": run_config.get("save_final_splat", True),
            "save_best_splat": run_config.get("save_best_splat", run_config.get("save_final_splat", True)),
        })

        # Read project ID from config.json
        config_file = project_dir / "config.json"
        with open(config_file, "r") as f:
            project_config = json.load(f)

        project_id = project_config["id"]

        try:
            project_status.update_status(
                project_id,
                "processing",
                progress=0,
                stage="starting",
                stage_progress=0,
                message=(
                    f"Pipeline run {run_id} starting "
                    f"(phase {phase_num}, run {run_config.get('phase_run_number', 1)})."
                ),
                current_run_id=run_id,
            )

            # Pipeline projects work directly from the user-specified pipeline folder
            # Images are symlinked during project creation, verify they exist
            images_dir = project_dir / "images"
            
            if not images_dir.exists():
                source_dir = project_config.get("source_dir")
                if source_dir:
                    source_path = Path(source_dir)
                    if source_path.exists() and source_path.is_dir():
                        try:
                            images_dir.mkdir(parents=True, exist_ok=True)
                            import shutil
                            copied = 0
                            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                                for src in source_path.glob(ext):
                                    if src.is_file():
                                        dst = images_dir / src.name
                                        if not dst.exists():
                                            shutil.copy2(src, dst)
                                            copied += 1
                            logger.info(
                                "Recovered missing images directory for %s by copying %d files from source_dir",
                                project_id,
                                copied,
                            )
                        except Exception as recovery_err:
                            logger.warning("Failed recovering missing images directory: %s", recovery_err)

            if not images_dir.exists():
                error_msg = f"CRITICAL: Images directory not found at {images_dir}. Symlink may have failed during project creation."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Verify images are accessible (either through symlink or actual files)
            image_count = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png")))
            if image_count == 0:
                error_msg = f"CRITICAL: No images found in {images_dir}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            logger.info(f"âœ“ Found {image_count} images in project directory")

            # Run pipeline directly from the user-specified project directory
            # Provide the actual project directory as an override
            params["project_dir_override"] = str(project_dir)

            pipeline.run_full_pipeline(project_id, params)

            final_status = project_status.get_status(project_id)
            final_state = str(final_status.get("status") or "").strip().lower()
            if final_state == "partial_completed":
                score_value = final_status.get("relative_quality_score")
                score = float(score_value) if isinstance(score_value, (int, float)) else -1.0
                result_status = "hard_cap_reached" if bool(final_status.get("gaussian_cap_reached")) else "partial_completed"
                logger.warning(
                    "Training run %s ended as %s with score=%s",
                    run_id,
                    result_status,
                    score,
                )
                return True, score, result_status
            if final_state in {"failed", "stopped"}:
                logger.warning(
                    "Training run %s ended with non-success project status=%s",
                    run_id,
                    final_state,
                )
                return False, None, "failed"

            pipeline_run_dir = project_dir / "runs" / run_id
            analytics_file = pipeline_run_dir / "analytics" / "run_analytics_v1.json"
            score: Optional[float] = None
            if analytics_file.exists():
                with open(analytics_file, "r", encoding="utf-8") as f:
                    analytics = json.load(f)
                ai_block = analytics.get("ai") if isinstance(analytics, dict) else {}
                learning = ai_block.get("input_mode_learning") if isinstance(ai_block, dict) else {}
                if isinstance(learning, dict) and isinstance(learning.get("score"), (int, float)):
                    score = float(learning.get("score"))
            if score is None:
                logger.info(
                    "Run %s completed without relative score in canonical analytics; treating as success.",
                    run_id,
                )

            return True, score, "success"

        except Exception as e:
            logger.error(f"Training execution failed: {e}", exc_info=True)
            return False, None, "failed"

    def _old_simulate_training_run(self, run_config: dict) -> tuple[bool, Optional[float]]:
        """OLD SIMULATION - REPLACED BY REAL TRAINING.

        Kept for reference only. Remove after verification.
        """
        time.sleep(2)
        success = random.random() > 0.1

        # Baseline runs (phase 1) don't have scores - they're just reference runs
        # Scores only apply to AI-driven phases where we compare against baseline
        phase_num = run_config.get("phase_number", 1)
        if phase_num == 1:
            # Baseline phase - no score calculation
            score = None
        else:
            # AI learning phases - simulate score (quality improvement vs baseline)
            score = random.uniform(-0.2, 0.3) if success else None

        return success, score

    def _apply_thermal_management(self, thermal_config: dict):
        """Apply thermal management strategy (cooldown period)."""
        if self.should_stop or not self._owns_current_pipeline_session():
            return

        strategy = thermal_config.get("strategy", "fixed_interval")

        training_pipeline_storage.update_pipeline(
            self.pipeline_id,
            {
                "cooldown_active": True,
                "cooldown_session_id": self.session_id,
            },
        )

        if strategy == "fixed_interval":
            cooldown_minutes = thermal_config.get("cooldown_minutes", 10)
            logger.info(f"Pipeline {self.pipeline_id}: Cooling down for {cooldown_minutes} minutes")

            # Calculate next run time
            next_run = datetime.utcnow() + timedelta(minutes=cooldown_minutes)
            training_pipeline_storage.update_pipeline(self.pipeline_id, {
                "next_run_scheduled_at": next_run.isoformat() + "Z"
            })

            if not self._sleep_cooldown(cooldown_minutes * 60):
                self._clear_owned_cooldown()
                return

        elif strategy == "temperature_based":
            # TODO: Implement GPU temperature monitoring
            # For now, fall back to fixed interval
            logger.warning("Temperature-based cooldown not yet implemented, using fixed interval")
            cooldown_minutes = thermal_config.get("cooldown_minutes", 10)
            if not self._sleep_cooldown(cooldown_minutes * 60):
                self._clear_owned_cooldown()
                return

        elif strategy == "time_scheduled":
            # TODO: Implement time-of-day scheduling
            logger.warning("Time-scheduled cooldown not yet implemented, using fixed interval")
            cooldown_minutes = thermal_config.get("cooldown_minutes", 10)
            if not self._sleep_cooldown(cooldown_minutes * 60):
                self._clear_owned_cooldown()
                return

        self._clear_owned_cooldown(last_run_ended=True)

    def _owns_current_pipeline_session(self) -> bool:
        pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)
        if not pipeline:
            return False
        if pipeline.get("status") != "running":
            return False
        return str(pipeline.get("orchestrator_session_id") or "") == self.session_id

    def _sleep_cooldown(self, seconds: float) -> bool:
        end_time = time.time() + max(0.0, float(seconds))
        while time.time() < end_time:
            if self.should_stop or not self._owns_current_pipeline_session():
                return False
            remaining = end_time - time.time()
            if remaining <= 0:
                break
            time.sleep(min(1.0, remaining))
        return self._owns_current_pipeline_session()

    def _clear_owned_cooldown(self, *, last_run_ended: bool = False) -> None:
        pipeline = training_pipeline_storage.get_pipeline(self.pipeline_id)
        if not pipeline:
            return
        if str(pipeline.get("cooldown_session_id") or "") != self.session_id:
            return

        updates = {
            "cooldown_active": False,
            "next_run_scheduled_at": None,
            "cooldown_session_id": None,
        }
        if last_run_ended:
            updates["last_run_ended_at"] = datetime.utcnow().isoformat() + "Z"
        training_pipeline_storage.update_pipeline(self.pipeline_id, updates)

    def _apply_thermal_management_if_due(self, thermal_config: dict) -> None:
        """Apply cooldown after completed runs; start/resume clears stale cooldown before work begins."""
        self._apply_thermal_management(thermal_config)


# ========== Public API ==========

def start_pipeline_orchestrator(pipeline_id: str):
    """Start pipeline orchestrator in background."""
    # --- Step 1: Signal any existing orchestrator thread to stop ---
    existing = _running_orchestrators.get(pipeline_id)
    if existing and existing.thread and existing.thread.is_alive():
        logger.warning("Pipeline %s already has an active orchestrator; stopping it before starting a new one", pipeline_id)
        existing.stop()
    elif existing:
        _running_orchestrators.pop(pipeline_id, None)

    # --- Step 2: Kill worker subprocesses BEFORE joining the thread ---
    # Killing the worker unblocks any thread that is blocked on subprocess.run(),
    # which lets it exit quickly and prevents a second orchestrator from running
    # in parallel with the first one recovering from the kill.
    try:
        from bimba3d_backend.app.services.colmap import stop_all_local_workers
        killed = stop_all_local_workers()
        if killed:
            logger.info("Killed %d active worker process(es) before starting orchestrator %s", killed, pipeline_id)
    except Exception:
        logger.debug("Failed to stop active local workers before orchestrator start", exc_info=True)

    # --- Step 3: Now wait for the existing thread to actually exit ---
    if existing and existing.thread and existing.thread.is_alive():
        try:
            existing.thread.join(timeout=10)
        except Exception:
            logger.debug("Failed while waiting for existing orchestrator to stop", exc_info=True)
        if existing.thread.is_alive():
            logger.warning(
                "Pipeline %s previous orchestrator thread did not stop within timeout; "
                "starting new orchestrator anyway",
                pipeline_id,
            )

    orchestrator = PipelineOrchestrator(pipeline_id)
    orchestrator.start()
    return orchestrator


def stop_pipeline_orchestrator(pipeline_id: str):
    """Stop running orchestrator."""
    orchestrator = _running_orchestrators.get(pipeline_id)
    if orchestrator:
        orchestrator.stop()
        try:
            if orchestrator.thread and orchestrator.thread.is_alive():
                orchestrator.thread.join(timeout=5)
        except Exception:
            logger.debug("Failed while waiting for orchestrator to stop", exc_info=True)


def get_orchestrator_status(pipeline_id: str) -> Optional[dict]:
    """Get current orchestrator status."""
    orchestrator = _running_orchestrators.get(pipeline_id)
    if not orchestrator:
        return None

    return {
        "is_running": orchestrator.thread and orchestrator.thread.is_alive(),
        "current_project": orchestrator.current_run_project_name,
    }
