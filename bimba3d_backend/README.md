# Bimba3d Backend

FastAPI backend for project management, processing pipeline (COLMAP + training), and status/preview endpoints.

## Setup and Run
Canonical install/run instructions are maintained in the root README:
- [../README.md](../README.md)

Use that as the single source of truth to avoid duplicated instructions.

## Dependency Profiles
- `requirements.local.txt`
  - Host/local backend dependency set used by the general local runtime.
  - On Windows local GPU setups, do not treat this file as the standalone install recipe; use the root README Windows install flow instead.
- `requirements.docker-worker.txt`
  - Docker worker Python deps reference.
  - In Docker builds, `torch`/`torchvision`/`torchaudio` and `gsplat` are installed separately in `Dockerfile.worker` using compatibility resolution from `compatibility-matrix.json`.
- `requirements.windows.txt`
  - Remaining backend dependencies for the Windows local setup after manually installing the pinned PyTorch stack and building `gsplat`.
- `../compatibility-matrix.json`
  - Matrix for CUDA/Torch/COLMAP/gsplat compatibility used by Docker worker build resolver.
- `scripts/resolve_compatibility_profile.py`
  - Resolves build profile from detected CUDA and emits shell exports consumed by `Dockerfile.worker`.
- `requirements.txt`
  - Convenience pointer to `requirements.local.txt`.

## Key Endpoints
- `POST /projects` — create a project
- `POST /projects/{id}/images` — upload images
- `POST /projects/{id}/process` — start pipeline
  - Body params include:
    - `stage`: `full` | `colmap_only` | `train_only`
    - `max_steps`, `batch_size`
    - `splat_export_interval`, `png_export_interval`, `auto_early_stop`
- `POST /projects/{id}/stop` — request manual stop
- `GET /projects/{id}/status` — status with `stage`, `message`, `device`
- `GET /projects/{id}/preview` — latest preview PNG
- `GET /health/gpu` — GPU availability and device info

## Pipeline Stages
- `full`: COLMAP sparse reconstruction plus training
- `colmap_only`: sparse reconstruction only
- `train_only`: training only, using existing sparse outputs

## Runtime Notes
- The backend can serve the built frontend from `bimba3d_frontend/dist` when that directory exists.
- For runtime environment variables and startup guidance, use the root README.
- If CUDA is unavailable, training falls back to CPU and will be slower.
