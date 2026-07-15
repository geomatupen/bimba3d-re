import json
from pathlib import Path

from bimba3d_backend.app.services.project_json import read_json_if_exists, write_json_atomic

SHARED_CONFIG_FILE = "shared_config.json"


def get_project_shared_config_path(project_dir: Path) -> Path:
    return project_dir / SHARED_CONFIG_FILE


def extract_shared_config_from_params(params: dict | None) -> dict:
    data = params if isinstance(params, dict) else {}
    shared: dict = {}

    if "images_resize_enabled" in data:
        shared["images_resize_enabled"] = bool(data.get("images_resize_enabled"))

    image_size = data.get("images_max_size")
    if isinstance(image_size, (int, float)):
        shared["images_max_size"] = int(image_size)

    colmap_in = data.get("colmap")
    if isinstance(colmap_in, dict):
        shared["colmap"] = json.loads(json.dumps(colmap_in))

    return shared


def merge_shared_config_into_params(params: dict, shared: dict | None) -> dict:
    merged = dict(params)
    if not isinstance(shared, dict):
        return merged

    if "images_resize_enabled" in shared:
        merged["images_resize_enabled"] = bool(shared.get("images_resize_enabled"))

    if "images_max_size" in shared:
        merged["images_max_size"] = shared.get("images_max_size")

    shared_colmap = shared.get("colmap")
    if isinstance(shared_colmap, dict):
        current_colmap = merged.get("colmap") if isinstance(merged.get("colmap"), dict) else {}
        merged["colmap"] = {
            **current_colmap,
            **json.loads(json.dumps(shared_colmap)),
        }

    return merged


def normalize_shared_doc(raw: dict | None, base_run_id: str | None = None) -> dict:
    doc = raw if isinstance(raw, dict) else {}
    shared = doc.get("shared") if isinstance(doc.get("shared"), dict) else {}
    version = doc.get("version")
    if not isinstance(version, int) or version < 1:
        version = 1

    active_shared = doc.get("active_shared") if isinstance(doc.get("active_shared"), dict) else None
    normalized = {
        "version": version,
        "base_run_id": doc.get("base_run_id") if isinstance(doc.get("base_run_id"), str) else base_run_id,
        "updated_at": doc.get("updated_at") if isinstance(doc.get("updated_at"), str) else None,
        "active_sparse_version": doc.get("active_sparse_version") if isinstance(doc.get("active_sparse_version"), int) else None,
        "active_sparse_updated_at": doc.get("active_sparse_updated_at") if isinstance(doc.get("active_sparse_updated_at"), str) else None,
        "active_shared": active_shared,
        "shared": shared,
    }
    if not normalized["base_run_id"] and base_run_id:
        normalized["base_run_id"] = base_run_id
    return normalized


def read_project_shared_config(project_dir: Path, base_run_id: str | None = None) -> dict:
    path = get_project_shared_config_path(project_dir)
    raw = read_json_if_exists(path)
    normalized = normalize_shared_doc(raw, base_run_id=base_run_id)

    if isinstance(normalized.get("shared"), dict) and normalized.get("shared"):
        return normalized

    if base_run_id:
        base_run_cfg = read_json_if_exists(project_dir / "runs" / base_run_id / "run_config.json")
        if isinstance(base_run_cfg, dict):
            resolved = base_run_cfg.get("resolved_params") if isinstance(base_run_cfg.get("resolved_params"), dict) else {}
            inferred_shared = extract_shared_config_from_params(resolved)
            if inferred_shared:
                normalized["shared"] = inferred_shared
                if not isinstance(normalized.get("active_shared"), dict):
                    normalized["active_shared"] = json.loads(json.dumps(inferred_shared))
                if not isinstance(normalized.get("active_sparse_version"), int):
                    normalized["active_sparse_version"] = int(normalized.get("version") or 1)

    return normalized


def write_project_shared_config(project_dir: Path, doc: dict) -> None:
    path = get_project_shared_config_path(project_dir)
    write_json_atomic(path, normalize_shared_doc(doc))
