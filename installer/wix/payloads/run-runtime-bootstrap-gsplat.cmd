@echo off
setlocal

set "WRAPPER_LOG=%ProgramData%\Bimba3D\Logs\runtime-bootstrap-wrapper.log"
set "PHASE_LOG=%ProgramData%\Bimba3D\Logs\runtime-bootstrap-gsplat.log"
set "BOOTSTRAP_ROOT=%ProgramData%\Bimba3D\runtime\bootstrap"
if not exist "%ProgramData%\Bimba3D\Logs" mkdir "%ProgramData%\Bimba3D\Logs" >nul 2>nul

set "INSTALL_DIR=%~1"
if "%INSTALL_DIR%"=="" set "INSTALL_DIR=%ProgramFiles%\Bimba3D"
if not exist "%INSTALL_DIR%\bimba3d_backend\requirements.windows.txt" (
	if exist "%ProgramFiles(x86)%\Bimba3D\bimba3d_backend\requirements.windows.txt" (
		set "INSTALL_DIR=%ProgramFiles(x86)%\Bimba3D"
	)
)

>>"%WRAPPER_LOG%" echo [%date% %time%] START install="%INSTALL_DIR%" phase="gsplat"
>>"%WRAPPER_LOG%" echo [%date% %time%] INFO phase_log="%PHASE_LOG%"
>>"%PHASE_LOG%" echo.
>>"%PHASE_LOG%" echo ===== [%date% %time%] RUN START phase=gsplat install="%INSTALL_DIR%" =====
if not exist "%BOOTSTRAP_ROOT%\runtime-bootstrap.ps1" (
	>>"%WRAPPER_LOG%" echo [%date% %time%] ERROR missing shared runtime bootstrap script at "%BOOTSTRAP_ROOT%\runtime-bootstrap.ps1"
	exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%BOOTSTRAP_ROOT%\runtime-bootstrap.ps1" -InstallDir "%INSTALL_DIR%" -Phase "gsplat" >>"%PHASE_LOG%" 2>&1
set "RC=%ERRORLEVEL%"
>>"%WRAPPER_LOG%" echo [%date% %time%] EXIT code=%RC% phase="gsplat"
>>"%PHASE_LOG%" echo ===== [%date% %time%] RUN END phase=gsplat exit=%RC% =====
exit /b %RC%
