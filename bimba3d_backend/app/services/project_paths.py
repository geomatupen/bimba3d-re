import json
import logging
from pathlib import Path
from typing import Optional

from bimba3d_backend.app.config import DATA_DIR
from bimba3d_backend.app.services import training_pipeline_storage

logger = logging.getLogger(__name__)


def find_project_dir(project_id: str) -> Optional[Path]:
    """Resolve regular and workflow-owned project folders by project id."""
    data_dir_project = DATA_DIR / project_id
    if data_dir_project.exists():
        return data_dir_project

    pipeline_folders_to_scan: list[Path] = []
    for pipeline_meta in training_pipeline_storage.list_pipelines():
        pipeline_folder = Path(pipeline_meta.get("config", {}).get("pipeline_folder", ""))
        if pipeline_folder.exists():
            pipeline_folders_to_scan.append(pipeline_folder)

    common_roots = [
        Path("E:/Thesis/PipelineProjects"),
        Path("E:/Thesis/PipelineTests"),
        Path("E:/Thesis/Pipeline Projects"),
        Path("E:/Thesis/Pipeline Tests"),
    ]
    # TEMP LEGACY: keep only while old pipeline projects are used as COLMAP copy sources.
    explicit_pipeline_dirs = [
        Path("E:/Thesis/PipelineProjects/Training_June_2"),
        Path("E:/Thesis/PipelineTests/Test_2026-05-31"),
    ]
    for pipeline_folder in explicit_pipeline_dirs:
        if pipeline_folder.exists() and pipeline_folder not in pipeline_folders_to_scan:
            pipeline_folders_to_scan.append(pipeline_folder)
    for root in common_roots:
        if root.exists():
            if (root / "pipeline.json").exists() and root not in pipeline_folders_to_scan:
                pipeline_folders_to_scan.append(root)
            for child in root.iterdir():
                if child.is_dir() and (child / "pipeline.json").exists() and child not in pipeline_folders_to_scan:
                    pipeline_folders_to_scan.append(child)

    for pipeline_folder in pipeline_folders_to_scan:
        for potential_project_folder in pipeline_folder.iterdir():
            if not potential_project_folder.is_dir():
                continue
            if potential_project_folder.name in ["shared_models", "training_pipelines", "pipeline.json"]:
                continue
            if potential_project_folder.name == project_id:
                logger.debug("Found project %s at %s", project_id, potential_project_folder)
                return potential_project_folder

            config_file = potential_project_folder / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, "r", encoding="utf-8") as handle:
                        config = json.load(handle)
                    if config.get("id") == project_id:
                        logger.debug("Found project %s at %s", project_id, potential_project_folder)
                        return potential_project_folder
                except Exception as exc:
                    logger.warning("Error reading config from %s: %s", config_file, exc)

    logger.warning("Project %s not found in any location", project_id)
    return None
