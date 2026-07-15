"""Workflow EXIF extraction endpoints with real-time progress tracking."""
import json
import threading
import time
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from bimba3d_backend.app.config import DATA_DIR
from bimba3d_backend.app.services import training_pipeline_storage

logger = logging.getLogger(__name__)
router = APIRouter()

# Global state for tracking extraction progress
_exif_extraction_state: Dict[str, Any] = {}


def _get_exif_results_path(pipeline_id: str) -> Path:
    """Get path to persisted EXIF test results JSON file"""
    pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
    if not pipeline:
        return None
    config = pipeline.get("config", {})
    pipeline_folder = Path(config.get("pipeline_folder", ""))
    return pipeline_folder / "exif_test_results.json"


def _save_exif_results(pipeline_id: str, results: list) -> None:
    """Save EXIF test results to pipeline folder"""
    try:
        results_path = _get_exif_results_path(pipeline_id)
        if results_path:
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump({
                    "pipeline_id": pipeline_id,
                    "extracted_at": datetime.utcnow().isoformat() + "Z",
                    "results": results
                }, f, indent=2)
            logger.info(f"Saved EXIF test results to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save EXIF results: {e}")


def _load_exif_results(pipeline_id: str) -> Dict[str, Any] | None:
    """Load persisted EXIF test results from pipeline folder"""
    try:
        results_path = _get_exif_results_path(pipeline_id)
        if results_path and results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load EXIF results: {e}")
    return None


@router.post("/{pipeline_id}/test-exif/start")
async def start_exif_extraction_test(pipeline_id: str):
    """
    Start EXIF extraction test with progress tracking.
    Returns immediately with a task ID.
    """
    from bimba3d_backend.worker.ai_input_modes.common import ModeContext
    from bimba3d_backend.worker.ai_input_modes.compact_scene_descriptors import build_preset as build_compact_descriptors
    from bimba3d_backend.worker.ai_input_modes.exif_extractors import extract_camera_exif

    try:
        # Load pipeline
        pipeline = training_pipeline_storage.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        config = pipeline.get("config", {})
        pipeline_folder = Path(config.get("pipeline_folder", ""))
        if not pipeline_folder.exists():
            raise HTTPException(status_code=404, detail="Pipeline folder not found")

        projects_config = config.get("projects", [])

        # Clear old persisted results before starting new extraction
        results_path = _get_exif_results_path(pipeline_id)
        if results_path and results_path.exists():
            results_path.unlink()
            logger.info(f"Cleared old EXIF test results: {results_path}")

        # Initialize state
        task_id = str(uuid.uuid4())
        _exif_extraction_state[task_id] = {
            "pipeline_id": pipeline_id,
            "pipeline_name": config.get("name", ""),
            "status": "running",
            "total_projects": len(projects_config),
            "current_project": 0,
            "progress": [],
            "results": [],
            "should_stop": False,
            "started_at": datetime.utcnow().isoformat() + "Z",
        }

        # Background extraction task
        def run_extraction():
            state = _exif_extraction_state[task_id]

            try:
                for idx, proj_config in enumerate(projects_config):
                    # Check if stop requested
                    if state["should_stop"]:
                        state["status"] = "stopped"
                        state["progress"].append({
                            "project_name": "",
                            "mode": "",
                            "status": "stopped",
                            "message": "Extraction stopped by user",
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                        })
                        return

                    project_name = proj_config.get("name", "")
                    project_id = proj_config.get("project_id", "")
                    project_dir = pipeline_folder / project_name

                    state["current_project"] = idx + 1
                    state["progress"].append({
                        "project_name": project_name,
                        "mode": "detecting",
                        "status": "running",
                        "message": f"Processing project {idx + 1}/{len(projects_config)}",
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    })

                    if not project_dir.exists():
                        state["progress"].append({
                            "project_name": project_name,
                            "mode": "error",
                            "status": "error",
                            "message": "Project directory not found",
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                        })
                        continue

                    # Find sample image for camera detection
                    metadata_image_dir = Path(proj_config.get("dataset_path", ""))
                    if not metadata_image_dir.exists():
                        metadata_image_dir = project_dir / "images"

                    sample_image = None
                    camera_make = None
                    camera_model = None

                    if metadata_image_dir.exists():
                        image_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
                        images = [f for f in metadata_image_dir.glob("*") if f.suffix in image_exts]
                        if images:
                            sample_image = images[0]
                            try:
                                exif, _, _ = extract_camera_exif(sample_image)
                                camera_make = exif.get("Make", "Unknown")
                                camera_model = exif.get("Model", "Unknown")
                                state["progress"].append({
                                    "project_name": project_name,
                                    "mode": "camera_detected",
                                    "status": "running",
                                    "message": f"Camera: {camera_make} {camera_model}",
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                })
                            except Exception as e:
                                state["progress"].append({
                                    "project_name": project_name,
                                    "mode": "camera_detection",
                                    "status": "warning",
                                    "message": f"Camera detection failed: {str(e)}",
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                })

                    # Processing directories
                    processing_image_dir = project_dir / "images_resized"
                    colmap_dir = project_dir / "outputs"

                    # Create minimal context
                    class MockParams:
                        def get(self, key, default=None):
                            return default

                    ctx = ModeContext(
                        metadata_image_dir=metadata_image_dir,
                        processing_image_dir=processing_image_dir,
                        colmap_dir=colmap_dir,
                        params=MockParams()
                    )

                    # Test the single compact descriptor extraction mode used by model training/testing.
                    modes = [
                        ("exif_compact_featurewise", build_compact_descriptors, "Compact Scene Descriptors"),
                    ]

                    for mode_name, build_func, mode_label in modes:
                        if state["should_stop"]:
                            break

                        state["progress"].append({
                            "project_name": project_name,
                            "mode": mode_name,
                            "status": "running",
                            "message": f"Extracting {mode_label}...",
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                        })

                        start_time = time.time()
                        try:
                            preset_result = build_func(ctx)
                            extraction_time_ms = (time.time() - start_time) * 1000

                            features = preset_result.features

                            # Total model descriptor features. Diagnostic availability flags are not emitted.
                            missing_count = 0
                            feature_keys = list(features.keys())
                            total_count = len(feature_keys)
                            completeness_percent = 100.0 if total_count > 0 else 0.0

                            # Log progress
                            state["progress"].append({
                                "project_name": project_name,
                                "mode": mode_name,
                                "status": "complete",
                                "message": f"{mode_label}: extracted {total_count} descriptor{'' if total_count == 1 else 's'}",
                                "missing_count": missing_count,
                                "completeness_percent": round(completeness_percent, 1),
                                "extraction_time_ms": round(extraction_time_ms, 2),
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                            })

                            # Build dependency metadata
                            # Track which features are derived and what their dependencies are
                            dependencies = {}

                            # Compact scene descriptor: focal_length_mm depends on EXIF.
                            if "focal_length_mm" in features:
                                dependencies["focal_length_mm"] = {
                                    "type": "exif",
                                    "sources": ["EXIF:FocalLength", "XMP:drone-dji:CalibratedFocalLength", "LensModel"],
                                    "defaulted": False
                                }

                            # Compact scene descriptor: GSD depends on altitude, focal, sensor, and image dimensions.
                            if "gsd_median" in features:
                                dependencies["gsd_median"] = {
                                    "type": "calculated",
                                    "formula": "GSD = (altitude × sensor_width) / (focal_length × image_width)",
                                    "depends_on": ["RelativeAltitude", "focal_length_mm", "SensorWidth", "image_width"],
                                    "defaulted": False,
                                    "reason": None
                                }

                            # Compact scene descriptor: camera_angle_bucket depends on pitch.
                            if "camera_angle_bucket" in features:
                                dependencies["camera_angle_bucket"] = {
                                    "type": "derived",
                                    "depends_on": ["Pitch", "CameraElevationAngle", "GimbalPitchDegree"],
                                    "defaulted": False
                                }

                            # Compact scene descriptor: overlap_proxy depends on GPS.
                            if "overlap_proxy" in features:
                                dependencies["overlap_proxy"] = {
                                    "type": "calculated",
                                    "depends_on": ["GPS coordinates", "image_count"],
                                    "defaulted": False
                                }

                            # Compact scene descriptor: vegetation features depend on pixel analysis.
                            if "vegetation_cover_percentage" in features:
                                dependencies["vegetation_cover_percentage"] = {
                                    "type": "pixel_analysis",
                                    "depends_on": ["image_pixels (green channel analysis)"],
                                    "defaulted": False
                                }

                            if "texture_density" in features:
                                dependencies["texture_density"] = {
                                    "type": "pixel_analysis",
                                    "depends_on": ["image_pixels (edge detection)"],
                                    "defaulted": False
                                }

                            if "blur_motion_risk" in features:
                                dependencies["blur_motion_risk"] = {
                                    "type": "pixel_analysis",
                                    "depends_on": ["image_pixels (Laplacian variance)"],
                                    "defaulted": False
                                }

                            # Compact scene descriptor: terrain roughness depends on COLMAP.
                            if "terrain_roughness_proxy" in features:
                                dependencies["terrain_roughness_proxy"] = {
                                    "type": "colmap_geometry",
                                    "depends_on": ["COLMAP sparse points (elevation variance)"],
                                    "defaulted": False
                                }

                            # Store result with dependencies
                            state["results"].append({
                                "project_id": project_id,
                                "project_name": project_name,
                                "mode": mode_name,
                                "features": features,
                                "dependencies": dependencies,
                                "missing_count": missing_count,
                                "total_count": total_count,
                                "completeness_percent": round(completeness_percent, 1),
                                "extraction_time_ms": round(extraction_time_ms, 2),
                                "camera_make": camera_make,
                                "camera_model": camera_model,
                                "sample_image_path": str(sample_image) if sample_image else None,
                            })

                        except Exception as e:
                            state["progress"].append({
                                "project_name": project_name,
                                "mode": mode_name,
                                "status": "error",
                                "message": f"{mode_label}: {str(e)}",
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                            })

                # Completed
                state["status"] = "complete"
                state["progress"].append({
                    "project_name": "",
                    "mode": "",
                    "status": "complete",
                    "message": f"Extraction complete! Tested {len(projects_config)} projects.",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })

                # Persist results to disk
                _save_exif_results(pipeline_id, state["results"])

            except Exception as e:
                logger.error(f"Extraction task failed: {e}")
                state["status"] = "error"
                state["progress"].append({
                    "project_name": "",
                    "mode": "",
                    "status": "error",
                    "message": f"Extraction failed: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })

        # Start background thread
        thread = threading.Thread(target=run_extraction, daemon=True)
        thread.start()

        return {
            "task_id": task_id,
            "pipeline_id": pipeline_id,
            "status": "started",
            "total_projects": len(projects_config),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start EXIF extraction test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{pipeline_id}/test-exif/progress/{task_id}")
async def get_exif_extraction_progress(pipeline_id: str, task_id: str):
    """Get progress of running EXIF extraction test"""
    if task_id not in _exif_extraction_state:
        raise HTTPException(status_code=404, detail="Task not found")

    return _exif_extraction_state[task_id]


@router.get("/{pipeline_id}/test-exif/results")
async def get_exif_test_results(pipeline_id: str):
    """
    Get persisted EXIF test results from pipeline folder.
    Returns cached results if they exist, otherwise returns null.
    """
    try:
        results_data = _load_exif_results(pipeline_id)
        if results_data:
            return {
                "has_results": True,
                "extracted_at": results_data.get("extracted_at"),
                "results": results_data.get("results", [])
            }
        else:
            return {
                "has_results": False,
                "results": []
            }
    except Exception as e:
        logger.error(f"Failed to get EXIF test results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{pipeline_id}/test-exif/stop/{task_id}")
async def stop_exif_extraction_test(pipeline_id: str, task_id: str):
    """Stop running EXIF extraction test"""
    if task_id not in _exif_extraction_state:
        raise HTTPException(status_code=404, detail="Task not found")

    _exif_extraction_state[task_id]["should_stop"] = True

    return {
        "task_id": task_id,
        "status": "stopping",
        "message": "Stop signal sent"
    }
