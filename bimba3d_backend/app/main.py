import logging
import os
import tempfile
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from bimba3d_backend.app.api.model_training import router as model_training_router
from bimba3d_backend.app.api.models import router as models_router
from bimba3d_backend.app.api.projects import router as projects_router
from bimba3d_backend.app.api.training_data import router as training_data_router
from bimba3d_backend.app.api.workflow_exif import router as workflow_exif_router
from bimba3d_backend.app.api.workflow_pipelines import router as workflow_pipelines_router
from bimba3d_backend.app.config import ALLOWED_ORIGINS
from bimba3d_backend.app.config import DATA_DIR
import json
from pathlib import Path
import time

# Keep temporary files and ML caches off the Windows user profile drive.
_temp_path = Path(os.environ.get("BIMBA3D_TEMP_DIR") or "E:/Thesis/Temp").expanduser().resolve()
try:
    _temp_path.mkdir(parents=True, exist_ok=True)
    os.environ["BIMBA3D_TEMP_DIR"] = str(_temp_path)
    os.environ["TEMP"] = str(_temp_path)
    os.environ["TMP"] = str(_temp_path)
    os.environ["TMPDIR"] = str(_temp_path)
    os.environ.setdefault("XDG_CACHE_HOME", str(_temp_path / ".cache"))
    os.environ.setdefault("TORCH_HOME", str(_temp_path / ".cache" / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(_temp_path / ".cache" / "matplotlib"))
    os.environ.setdefault("HF_HOME", str(_temp_path / ".cache" / "huggingface"))
    tempfile.tempdir = str(_temp_path)
except Exception as _e:
    logging.warning(f"Could not use temp/cache dir {_temp_path}: {_e}. Using system default.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info(f"Configured temp directory: {tempfile.gettempdir()}")

app = FastAPI(title="Gaussian Splat Backend")


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects_router, prefix="/projects")
app.include_router(training_data_router, prefix="/api/workflow/training-data")
app.include_router(models_router, prefix="/api/models")
app.include_router(model_training_router, prefix="/api/workflow/model-training")
app.include_router(workflow_pipelines_router, prefix="/api/workflow/pipelines")
app.include_router(workflow_exif_router, prefix="/api/workflow/pipelines")


@app.on_event("startup")
def mark_interrupted_projects():
    """On backend start, mark any projects that were 'processing' as stopped/resumable.

    This ensures the frontend doesn't continue to show 'processing' for jobs
    that were interrupted by a backend restart or crash.
    """
    note = "Backend restarted - processing interrupted. Please resume when ready."
    from bimba3d_backend.app.services.colmap import stop_project_worker_containers
    from bimba3d_backend.app.services import training_pipeline_storage

    # Recover training pipelines that were marked running before backend restart.
    # Their orchestrator threads are gone after process restart, so they must be paused.
    try:
        for pipeline in training_pipeline_storage.list_pipelines(limit=1000):
            if str(pipeline.get("status") or "").lower() != "running":
                continue

            pipeline_id = str(pipeline.get("id") or "")
            if not pipeline_id:
                continue

            training_pipeline_storage.update_pipeline(
                pipeline_id,
                {
                    "status": "paused",
                    "cooldown_active": False,
                    "next_run_scheduled_at": None,
                    "active_run": None,
                    "last_error": note,
                },
            )
            logging.info("Paused interrupted pipeline after restart: %s", pipeline_id)
    except Exception:
        logging.exception("Failed to recover interrupted training pipelines")

    for proj_dir in DATA_DIR.iterdir():
        try:
            if not proj_dir.is_dir():
                continue
            stopped = stop_project_worker_containers(proj_dir.name)
            if stopped:
                logging.info("Stopped %d stale worker container(s) for %s", stopped, proj_dir.name)
            status_file = proj_dir / "status.json"
            if not status_file.exists():
                continue
            try:
                with open(status_file, 'r') as f:
                    data = json.load(f)
            except Exception:
                data = {}
            if data.get("status") == "processing":
                data["status"] = "stopped"
                data["progress"] = data.get("progress", 0)
                data["error"] = note
                data["stop_requested"] = True
                data["stopped_stage"] = data.get("stage", "unknown")
                data["resumable"] = True
                data["percentage"] = data.get("percentage", 0.0)
                # write atomically
                tmp = status_file.with_suffix('.tmp')
                with open(tmp, 'w') as f:
                    json.dump(data, f)
                tmp.replace(status_file)
        except Exception:
            logging.exception(f"Failed to mark interrupted project: {proj_dir}")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/health/gpu")
def gpu_health():
    """Report GPU availability and basic CUDA/device info."""
    try:
        import torch
        available = torch.cuda.is_available()
        count = torch.cuda.device_count() if available else 0
        devices = []
        for i in range(count):
            try:
                devices.append(torch.cuda.get_device_name(i))
            except Exception:
                devices.append(f"cuda:{i}")
        return {
            "gpu_available": available,
            "device_count": count,
            "devices": devices,
            "cuda_version": getattr(torch.version, "cuda", None),
        }
    except Exception:
        return {
            "gpu_available": False,
            "device_count": 0,
            "devices": [],
            "cuda_version": None,
        }


DEFAULT_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "bimba3d_frontend" / "dist"
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", str(DEFAULT_FRONTEND_DIST))).resolve()
if FRONTEND_DIST.exists():
    app.mount("/", SPAStaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    logging.info("Serving frontend build from %s", FRONTEND_DIST)
else:
    logging.info("Frontend dist not found at %s; API-only mode", FRONTEND_DIST)
