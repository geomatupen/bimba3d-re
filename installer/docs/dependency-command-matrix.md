# Bimba3D Windows Dependency Command Matrix

Pinned baseline (current draft):
- Visual Studio Build Tools 2022 (v17)
- VC++ Redistributable (latest v14 x64)
- CUDA Toolkit 12.5
- COLMAP 4.0.1 (Windows CUDA zip)

## 1) Visual Studio Build Tools (optional package in chain)
- Source: `https://aka.ms/vs/17/release/vs_BuildTools.exe`
- Detect (example): registry keys under `HKLM\SOFTWARE\Microsoft\VisualStudio\Setup\Instances`
- Silent install (bundle):
  - `vs_BuildTools.exe --quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended`
- Exit codes:
  - `0` success
  - `3010` success + reboot required

Notes:
- Controlled by bundle variable `InstallBuildTools` (default `0`).
- Set `InstallBuildTools=1` only if end-user machines need local compile toolchain.

## 2) VC++ Runtime (recommended for runtime-only machines)
- Source: `https://aka.ms/vc14/vc_redist.x64.exe`
- Detect:
  - `HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64`
  - `Installed=1`
- Silent install:
  - `vc_redist.x64.exe /install /quiet /norestart`
- Exit codes:
  - `0` success
  - `3010` reboot required

## 3) CUDA Toolkit or CUDA Runtime Strategy
### Option A (Toolkit install)
- Source (pinned draft):
  - `https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda_12.5.0_windows.exe`
- Detect:
  - file exists `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe`
- Silent install (bundle draft):
  - `cuda_installer.exe -s`

### Option B (Redistributable runtime-only)
- Source: package only redistributable CUDA DLL set required by your app
- Detect: file version checks in app runtime folder
- Install: copy/upgrade runtime DLL payload
- Notes: keep exactly to CUDA EULA distributable components list

## 4) COLMAP (Windows CUDA build)
- Source:
  - `https://github.com/colmap/colmap/releases/download/4.0.1/colmap-x64-windows-cuda.zip`
- Detect:
  - check for `%ProgramFiles%\Bimba3D\third_party\colmap\COLMAP.bat`
- Install:
  - run `payloads\install-colmap.cmd "<install-dir>" "<colmap-zip>"`
  - extract zip to `%ProgramFiles%\Bimba3D\third_party\colmap`
- Config:
  - set `COLMAP_EXE` to `%ProgramFiles%\Bimba3D\third_party\colmap\COLMAP.bat`

## 5) Bimba3D App Payload
- Source: your packaged backend/frontend installer payload
- Detect: product code / file version
- Silent install:
  - MSI: `msiexec /i Bimba3D.msi /qn /norestart`
  - EXE: app-specific silent args

## 6) Post-Install Validation
- Verify app files exist
- Verify COLMAP executable path exists
- Verify required Python/venv/runtime artifacts if included
- Smoke test command (optional):
  - `<app-cli> --healthcheck`

## Bundle Variables to Define
- `Bimba3DInstallDir`
- `ColmapInstallDir`
- `CudaInstallDir`
- `LogRoot`
- `InstallBuildTools`

## Log Locations
- Burn log: `%ProgramData%\Bimba3D\Logs\bundle.log`
- Package logs: `%ProgramData%\Bimba3D\Logs\*.log`
