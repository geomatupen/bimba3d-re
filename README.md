# Bimba3d Monorepo

This repository contains:
- `bimba3d_backend`: FastAPI API, project processing, model training, and pipeline workflow services.
- `bimba3d_frontend`: React + Vite frontend.

The current backend can serve the built frontend directly. If `bimba3d_frontend/dist` exists, `uvicorn` serves both the UI and the API from the same process. If it does not exist, the backend runs in API-only mode.

## Prerequisites
- Python 3.12+
- Node.js 18+
- COLMAP installed on the host OS and available on `PATH`, or configured via `COLMAP_EXE`
- Optional for GPU training: NVIDIA driver, CUDA-enabled PyTorch, and a working CUDA toolchain for `gsplat`

Important:
- `requirements.local.txt` is not a universal install recipe for every platform.
- On Windows, do not rely on `python -m pip install -r bimba3d_backend\requirements.local.txt` alone for a local training setup.
- Windows local setup uses a separate install order for PyTorch, `ninja`, and `gsplat`, then installs the remaining backend packages from `requirements.windows.txt`.

### System install checklist
Install these before creating the project environment:
- Python 3.12 or newer
- Node.js 18 or newer
- COLMAP on the host machine
- Optional on Windows for local GPU training: Visual Studio Build Tools, CUDA Toolkit, and NVIDIA drivers

Quick verification commands:

```bash
python --version
node --version
colmap -h
```

If `colmap` is not on `PATH`, set `COLMAP_EXE` to the full executable or `.bat` path before running the backend.

## Repository Layout
- `bimba3d_backend/app`: FastAPI app, API routes, services, schemas, and models
- `bimba3d_backend/data`: local project data, exports, models, and workflow artifacts
- `bimba3d_backend/scripts`: utility and offline-training scripts
- `bimba3d_frontend/src`: frontend application source
- `compatibility-matrix.json`: compatibility data used by build and training tooling

## Install

### 1. Clone and enter the repository

```bash
git clone <your-repo-url>
cd bimba3d-re
```

### 2. Backend Python environment

#### Linux

For Linux host installs, the backend Python dependencies are installed from `requirements.local.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r bimba3d_backend/requirements.local.txt
```

#### Windows PowerShell

For Windows local installs, use the Windows-specific sequence below instead of installing `requirements.local.txt` directly.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121
python -m pip install --force-reinstall ninja
```

Then set the CUDA/toolchain environment and build `gsplat` before installing the remaining backend packages:

```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
$env:DISTUTILS_USE_SDK = "1"
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:PATH"

python -m pip install --force-reinstall --no-deps --no-cache-dir --no-binary=gsplat gsplat==1.5.3 --no-build-isolation -v
python -m pip install -r bimba3d_backend\requirements.windows.txt
```

Notes for Windows:
- Use 64-bit Python.
- Use an x64 compiler environment when building `gsplat`.
- Prefer `COLMAP.bat` via `COLMAP_EXE` if `colmap.exe` is unstable.
- `windows_install.text` contains the longer verified Windows setup, including toolchain guidance.

### 3. Frontend dependencies

#### Linux

```bash
cd bimba3d_frontend
npm install
cd ..
```

#### Windows PowerShell

```powershell
cd bimba3d_frontend
npm install
cd ..
```

### 4. Optional environment variables
Set these only when you need to override defaults:

```powershell
$env:COLMAP_EXE = "D:\path\to\COLMAP.bat"
$env:BIMBA3D_DATA_DIR = "D:\path\to\projects"
$env:BIMBA3D_TEMP_DIR = "D:\path\to\temp"
$env:ALLOWED_ORIGINS = "http://localhost:5173,http://localhost:5174"
```

On macOS or Linux:

```bash
export COLMAP_EXE=/path/to/colmap
export BIMBA3D_DATA_DIR=/path/to/projects
export BIMBA3D_TEMP_DIR=/path/to/temp
export ALLOWED_ORIGINS=http://localhost:5173,http://localhost:5174
```

### 5. Sanity checks
Before starting the app, verify the main pieces are installed:

#### Linux

```bash
python -c "import torch; print(torch.__version__)"
python -c "import fastapi; print(fastapi.__version__)"
cd bimba3d_frontend
npm run build
cd ..
```

#### Windows PowerShell

```powershell
python -c "import torch; print(torch.__version__)"
python -c "import fastapi; print(fastapi.__version__)"
cd bimba3d_frontend
npm run build
cd ..
```

If you expect GPU training, also verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## Run

### Recommended: one backend process serving both UI and API
Build the frontend once, then start the backend on port `8005`:

#### Linux

```bash
cd bimba3d_frontend
npm run build
cd ..

source .venv/bin/activate
python -m uvicorn bimba3d_backend.app.main:app --reload --port 8005
```

#### Windows PowerShell

```powershell
cd bimba3d_frontend
npm run build
cd ..

.\.venv\Scripts\Activate.ps1
python -m uvicorn bimba3d_backend.app.main:app --reload --port 8005
```

Open `http://localhost:8005`.

### Frontend development mode
The frontend dev client detects Vite ports `5173` and `5174` and calls the backend at port `8005`.

#### Linux

Terminal 1:

```bash
source .venv/bin/activate
python -m uvicorn bimba3d_backend.app.main:app --reload --port 8005
```

Terminal 2:

```bash
cd bimba3d_frontend
npm run dev
```

#### Windows PowerShell

Terminal 1:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn bimba3d_backend.app.main:app --reload --port 8005
```

Terminal 2:

```powershell
cd bimba3d_frontend
npm run dev
```

Open `http://localhost:5173`.

## Runtime Notes

### Frontend serving behavior
- The backend serves static files from `bimba3d_frontend/dist` by default.
- Override that path with `FRONTEND_DIST` if needed.
- If the build output is missing, the backend stays available as API-only mode.

### Data and temp directories
- Project data defaults to `bimba3d_backend/data/projects`.
- Override the data location with `BIMBA3D_DATA_DIR`.
- Temporary files and ML caches default to `E:/Thesis/Temp` in the backend startup path.
- Override that location with `BIMBA3D_TEMP_DIR`.

### CORS and host access
- Local frontend development is allowed from `http://localhost:5173` and `http://localhost:5174`.
- Additional allowed origins can be supplied through `ALLOWED_ORIGINS` as a comma-separated list.

### COLMAP configuration
- The backend resolves COLMAP using `COLMAP_EXE`, otherwise it falls back to `colmap` from `PATH`.
- On Windows, `COLMAP.bat` is preferred over `colmap.exe` when the executable has runtime issues.

### Worker mode
- Project processing still supports `worker_mode` values `local` and `docker`.
- If a request omits `worker_mode`, the backend resolves it from the `WORKER_MODE` environment variable and otherwise defaults to `local`.
- This is a request/runtime detail, not part of the normal frontend startup flow.

## Useful Endpoints
- `GET /health`: basic health check
- `GET /health/gpu`: CUDA/device availability summary
- `POST /projects`: create a project
- `POST /projects/{id}/process`: start project processing
- `GET /api/models`: list registered models
- `GET /api/workflow/pipelines`: list workflow pipelines

## Windows GPU Setup Notes
- `requirements.local.txt` installs Python packages only. COLMAP still needs a native OS installation.
- If Windows installs a CPU-only PyTorch build, reinstall a CUDA-enabled wheel explicitly.
- If you build `gsplat` locally on Windows, you also need a CUDA toolkit and x64 build tools.
- See `windows_install.text` for the verified Windows CUDA and `gsplat` setup sequence used in this repository.

## Notes
- Use underscore-based Python imports such as `bimba3d_backend.app.main:app`.
- The frontend API client assumes backend port `8005` during Vite development.
