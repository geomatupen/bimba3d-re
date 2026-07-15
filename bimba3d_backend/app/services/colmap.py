import subprocess
import time
import logging
import json
import os
import re
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_ACTIVE_LOCAL_WORKERS: dict[str, subprocess.Popen] = {}
_ACTIVE_LOCAL_WORKERS_LOCK = threading.Lock()


def _register_local_worker_process(project_id: str, proc: subprocess.Popen) -> None:
    with _ACTIVE_LOCAL_WORKERS_LOCK:
        _ACTIVE_LOCAL_WORKERS[project_id] = proc


def _pop_local_worker_process(project_id: str) -> subprocess.Popen | None:
    with _ACTIVE_LOCAL_WORKERS_LOCK:
        return _ACTIVE_LOCAL_WORKERS.pop(project_id, None)


def stop_local_worker_process(project_id: str, timeout_seconds: float = 2.0) -> bool:
    """Terminate an active local worker process for the project, if present."""
    proc = _pop_local_worker_process(project_id)
    if proc is None:
        return False

    try:
        if proc.poll() is not None:
            return False
    except Exception:
        return False

    try:
        _kill_process_tree(proc.pid)
        proc.wait(timeout=timeout_seconds)
        return True
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=timeout_seconds)
            return True
        except Exception as exc:
            logger.warning("Failed to terminate local worker for project %s: %s", project_id, exc)
            return False


def stop_all_local_workers() -> int:
    """Kill all active local worker processes. Used on pipeline stop/restart."""
    stopped = 0
    with _ACTIVE_LOCAL_WORKERS_LOCK:
        pids = list(_ACTIVE_LOCAL_WORKERS.keys())
    for project_id in pids:
        if stop_local_worker_process(project_id):
            stopped += 1
    return stopped


def _kill_process_tree(pid: int) -> None:
    """Kill a process and all its children (Windows-compatible)."""
    import signal
    if os.name == "nt":
        # On Windows, use taskkill /T to kill the process tree
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=5,
            )
        except Exception:
            # Fallback: just kill the main process
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
    else:
        # On Unix, kill the process group
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass

_COLMAP_CAMERA_MODELS = {
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "OPENCV_FISHEYE",
    "FULL_OPENCV",
    "FOV",
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
    "THIN_PRISM_FISHEYE",
}


def _parse_boolish(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_colmap_camera_model(value: object) -> str:
    candidate = str(value or "OPENCV").strip().upper()
    if candidate in _COLMAP_CAMERA_MODELS:
        return candidate
    logger.warning("Unsupported COLMAP camera_model '%s'; falling back to OPENCV", value)
    return "OPENCV"

COLMAP_EXE = (os.getenv("COLMAP_EXE") or "colmap").strip() or "colmap"


def _prepare_subprocess_command(cmd: list[str]) -> tuple[list[str] | str, bool]:
    if os.name == "nt" and cmd:
        resolved = list(cmd)
        exe = str(resolved[0])
        exe_lower = exe.lower()
        if exe_lower.endswith("colmap.exe"):
            exe_path = Path(exe)
            candidates = [
                exe_path.with_name("COLMAP.bat"),
                exe_path.parent.parent / "COLMAP.bat",
            ]
            for candidate in candidates:
                if candidate.exists():
                    resolved[0] = str(candidate)
                    break
        if str(resolved[0]).lower().endswith((".bat", ".cmd")):
            return subprocess.list2cmdline(resolved), True
        return resolved, False
    return cmd, False

_PROGRESS_BAR_PATTERN = re.compile(r"\b\d{1,3}%\|.*\|\s*\d+(?:\.\d+)?[KMG]?/\d+(?:\.\d+)?[KMG]?")


def _is_noisy_progress_line(line: str) -> bool:
    """Return True for transient progress-bar style lines that flood API logs."""
    text = (line or "").strip()
    if not text:
        return True
    return bool(_PROGRESS_BAR_PATTERN.search(text))

def _worker_container_name(project_id: str) -> str:
    return f"bimba3d-worker-{project_id}"


def _find_project_worker_containers(project_id: str) -> list[str]:
    """Return running container IDs that belong to the given project."""
    container_ids: list[str] = []
    label = f"com.bimba3d.project_id={project_id}"
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"label={label}"],
            check=True,
            capture_output=True,
            text=True,
        )
        container_ids.extend([line.strip() for line in result.stdout.splitlines() if line.strip()])
    except Exception:
        pass

    return container_ids


def stop_project_worker_containers(project_id: str) -> int:
    """Stop any running docker worker containers associated with a project."""
    stopped = 0
    for cid in _find_project_worker_containers(project_id):
        try:
            subprocess.run(["docker", "stop", cid], check=True, capture_output=True, text=True)
            stopped += 1
        except Exception as exc:
            logger.warning("Failed to stop worker container %s for project %s: %s", cid, project_id, exc)
    return stopped


def get_project_worker_container_ids(project_id: str) -> list[str]:
    """List running worker container IDs associated with a project."""
    return _find_project_worker_containers(project_id)


def run_colmap_docker(project_id: str, params: dict = None) -> None:
    """
    Run COLMAP via Docker worker.
    """
    from bimba3d_backend.app.config import DATA_DIR
    
    worker_params = dict(params or {})

    params_json = json.dumps(worker_params)
    # DATA_DIR is now /path/to/websplat-backend/data/projects
    # Mount parent: /path/to/websplat-backend/data -> /data
    data_dir = DATA_DIR.parent

    # Ensure a writable cache directory exists on the host and will be mounted
    cache_dir = data_dir / ".cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create cache dir {cache_dir}: {e}")
    
    # Run container as the host backend user to avoid root-owned output files.
    get_uid = getattr(os, "getuid", None)
    get_gid = getattr(os, "getgid", None)
    uid = get_uid() if callable(get_uid) else None
    gid = get_gid() if callable(get_gid) else None
    user_flag = ["-u", f"{uid}:{gid}"] if uid is not None and gid is not None else []

    cmd = [
        "docker", "run", "--rm",
        *user_flag,
        "--name", _worker_container_name(project_id),
        "--label", "com.bimba3d.service=worker",
        "--label", f"com.bimba3d.project_id={project_id}",
        "--ipc", "host",
        "--shm-size", os.getenv("DOCKER_WORKER_SHM_SIZE", "8g"),
        "--gpus", "all",  # Enable GPU access
        # Mount project data and a writable cache into the container so PyTorch
        # can build extensions without trying to write to '/.cache' inside root.
        "-v", f"{data_dir}:/data",
        "-v", f"{cache_dir}:/data/.cache",
        "-e", "TORCH_EXTENSIONS_DIR=/data/.cache/torch_extensions",
        "-e", "XDG_CACHE_HOME=/data/.cache",
        "-e", "MPLCONFIGDIR=/data/.cache/matplotlib",
        "-e", "TORCH_HUB_DISABLE_PROGRESS_BARS=1",
        "-e", "BIMBA3D_DOCKER_WORKER=1",
        "bimba3d-worker:latest",
        project_id,
        "--data-dir", "/data/projects",
        "--params", params_json
    ]
    
    logger.info(f"Running Docker worker: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip('\n')
            if _is_noisy_progress_line(line):
                continue
            try:
                logger.info(line)
            except Exception:
                pass
        rc = proc.wait()
        if rc != 0:
            # If container was killed by OOM (137) provide a helpful status message
            try:
                from bimba3d_backend.app.services import status as _status
                if project_id and rc == 137:
                    msg = (
                        "Worker container exited with code 137 (out of memory). "
                        "This usually means the host ran out of RAM or the GPU ran out of memory. "
                        "Try reducing the number of initialized Gaussians (Trainer -> Init Gaussians), "
                        "or lower COLMAP `max_image_size` and `mapper_num_threads` in the Process configuration."
                    )
                    try:
                        _status.update_status(project_id, "failed", error=msg, message=msg)
                    except Exception:
                        pass
            except Exception:
                pass
            raise subprocess.CalledProcessError(rc, cmd)
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker worker failed: returncode={getattr(e, 'returncode', None)}")
        raise


def run_worker_local(project_id: str, params: dict = None) -> None:
    """Run the same worker.entrypoint pipeline locally (without Docker)."""
    from bimba3d_backend.app.config import DATA_DIR
    from pathlib import Path

    worker_params = dict(params or {})

    params_json = json.dumps(worker_params)

    # Use user-specified project directory if provided (for pipeline projects)
    # Otherwise use DATA_DIR (for regular projects)
    project_dir_override = worker_params.get("project_dir_override")
    if project_dir_override:
        # For pipeline projects, the override IS the full project directory
        # Pass parent as data-dir, and use project name (last part of path) as project_id
        project_path = Path(project_dir_override)
        data_dir = project_path.parent
        # Use the folder name as the "project_id" for the worker
        project_folder_name = project_path.name
        logger.info(f"Using project directory override: {project_path}")
        logger.info(f"  data_dir: {data_dir}")
        logger.info(f"  project_folder: {project_folder_name}")
    else:
        data_dir = DATA_DIR
        project_folder_name = project_id

    cmd = [
        sys.executable,
        "-m",
        "bimba3d_backend.worker.entrypoint",
        project_folder_name,  # Use folder name instead of UUID project_id
        "--data-dir",
        str(data_dir),
        "--params",
        params_json,
    ]

    logger.info("Running local worker: %s", " ".join(cmd))
    child_env = os.environ.copy()
    child_env["BIMBA3D_DOCKER_WORKER"] = "0"
    child_env["BIMBA3D_PARENT_LOGGING"] = "1"
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env,
            creationflags=creationflags,
        )
        _register_local_worker_process(project_id, proc)
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if _is_noisy_progress_line(line):
                continue
            try:
                logger.info(line)
            except Exception:
                pass
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
    except subprocess.CalledProcessError as e:
        logger.error("Local worker failed: returncode=%s", getattr(e, "returncode", None))
        raise
    finally:
        _pop_local_worker_process(project_id)


def _run_cmd_with_retry(cmd: list[str], retries: int = 3, delay_sec: float = 2.0):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            run_cmd, use_shell = _prepare_subprocess_command(cmd)
            res = subprocess.run(run_cmd, check=True, capture_output=True, text=True, shell=use_shell)
            if res.stdout:
                logger.info(res.stdout.strip())
            if res.stderr:
                logger.debug(res.stderr.strip())
            return
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").lower()
            last_err = e
            if os.name == "nt" and getattr(e, "returncode", None) in (3221225781, 3221225786):
                logger.error(
                    "COLMAP crashed on Windows (code=%s). If COLMAP_EXE points to colmap.exe, set it to COLMAP.bat instead; also ensure Microsoft Visual C++ 2015-2022 Redistributable is installed.",
                    getattr(e, "returncode", None),
                )
            if "database is locked" in stderr or "busy" in stderr:
                logger.warning(f"SQLite busy/locked (attempt {attempt}/{retries}). Retrying after {delay_sec}s...")
                time.sleep(delay_sec)
                continue
            logger.error(f"Command failed: {cmd}\nSTDERR: {e.stderr}")
            raise
    logger.error(f"Command failed after retries: {cmd}\nERR: {last_err}")
    raise last_err


def _cleanup_sqlite_sidecars(db_path: Path):
    for suffix in ("-wal", "-shm"):
        sidecar = db_path.with_name(db_path.name + suffix)
        if sidecar.exists():
            try:
                sidecar.unlink()
                logger.info(f"Removed stale SQLite sidecar: {sidecar}")
            except Exception as e:
                logger.warning(f"Failed to remove sidecar {sidecar}: {e}")


def _colmap_cmd(*args: str) -> list[str]:
    return [COLMAP_EXE, *args]


def run_colmap(image_dir: Path, output_dir: Path, params: dict | None = None):
    """
    Run COLMAP feature extraction, matching, and sparse reconstruction.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory for COLMAP outputs
        
    Returns:
        Path to sparse reconstruction directory
        
    Raises:
        subprocess.CalledProcessError: If COLMAP commands fail
        FileNotFoundError: If COLMAP is not installed
    """
    db_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove existing database and sidecars to prevent locking issues
    if db_path.exists():
        logger.info(f"Removing existing database: {db_path}")
        try:
            db_path.unlink()
        except Exception:
            db_path.write_bytes(b"")
    _cleanup_sqlite_sidecars(db_path)
    
    try:
        # Allow optional tuning via params.colmap
        p = params.get("colmap", {}) if isinstance(params, dict) else {}
        camera_model = _resolve_colmap_camera_model(p.get("camera_model"))
        single_camera = _parse_boolish(p.get("single_camera"), True)

        # 1️⃣ Feature extraction
        logger.info("Running COLMAP feature extraction...")
        feat_cmd = _colmap_cmd(
            "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(image_dir),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1" if single_camera else "0",
        )
        camera_params = p.get("camera_params")
        if isinstance(camera_params, str) and camera_params.strip():
            feat_cmd += ["--ImageReader.camera_params", camera_params.strip()]
        if p.get("max_image_size"):
            feat_cmd += ["--SiftExtraction.max_image_size", str(p.get("max_image_size"))]
        else:
            feat_cmd += ["--SiftExtraction.max_image_size", "1600"]
        if p.get("peak_threshold") is not None:
            feat_cmd += ["--SiftExtraction.peak_threshold", str(p.get("peak_threshold"))]
        else:
            feat_cmd += ["--SiftExtraction.peak_threshold", "0.01"]

        _run_cmd_with_retry(feat_cmd)
        logger.info("✓ Feature extraction completed")

        # 2️⃣ Feature matching
        logger.info("Running COLMAP feature matching...")
        guided = p.get("guided_matching")
        matching_type = p.get("matching_type", "exhaustive")
        if matching_type == "sequential":
            match_cmd = _colmap_cmd(
                "sequential_matcher",
                "--database_path", str(db_path),
            )
        else:
            match_cmd = _colmap_cmd(
                "exhaustive_matcher",
                "--database_path", str(db_path),
            )
        if guided is not None:
            match_cmd += ["--SiftMatching.guided_matching", "1" if guided else "0"]
        else:
            match_cmd += ["--SiftMatching.guided_matching", "1"]

        _run_cmd_with_retry(match_cmd)
        logger.info("✓ Feature matching completed")

        # 3️⃣ Sparse reconstruction
        logger.info("Running COLMAP sparse reconstruction (mapper)...")
        mapper_cmd = _colmap_cmd(
            "mapper",
            "--database_path", str(db_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.ba_refine_principal_point", "1",
            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_extra_params", "1",
        )
        if p.get("mapper_num_threads"):
            mapper_cmd += ["--Mapper.num_threads", str(p.get("mapper_num_threads"))]
        else:
            # Default to 2 mapper threads to reduce memory usage on typical hosts
            mapper_cmd += ["--Mapper.num_threads", "2"]

        _run_cmd_with_retry(mapper_cmd)
        logger.info("✓ Sparse reconstruction completed")
        
        # Verify outputs
        if not (sparse_dir / "0").exists():
            raise FileNotFoundError(f"COLMAP reconstruction failed - no output in {sparse_dir}")
        
        logger.info(f"COLMAP outputs saved to: {sparse_dir}")
        # Attempt to generate compact points.bin for easier web consumption
        try:
            from bimba3d_backend.app.services import pointsbin
            # Iterate recon dirs and convert any with COLMAP points
            for d in sorted([p for p in sparse_dir.iterdir() if p.is_dir()]):
                try:
                    cnt = pointsbin.convert_colmap_recon_to_pointsbin(d)
                    if cnt:
                        logger.info(f"Converted COLMAP recon {d} -> points.bin ({cnt} points)")
                except Exception as e:
                    logger.warning(f"Failed to convert recon {d} to points.bin: {e}")
        except Exception:
            # Non-fatal if converter missing or fails; continue returning sparse_dir
            logger.debug("pointsbin converter not available or failed")

        return sparse_dir
        
    except FileNotFoundError as e:
        logger.error("COLMAP not found. Ensure `%s` is executable or set COLMAP_EXE to full path.", COLMAP_EXE)
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"COLMAP command failed: {e.stderr}")
        raise
