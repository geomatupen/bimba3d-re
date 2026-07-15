@echo off
setlocal ENABLEEXTENSIONS

set "EARLY_TRACE=%ProgramData%\Bimba3D\Logs\colmap-early.log"
if "%EARLY_TRACE%"=="" set "EARLY_TRACE=C:\ProgramData\Bimba3D\Logs\colmap-early.log"
if not exist "%ProgramData%\Bimba3D\Logs" mkdir "%ProgramData%\Bimba3D\Logs" >nul 2>nul
if not exist "%ProgramData%\Bimba3D\Logs" (
  set "EARLY_TRACE=%WINDIR%\Temp\Bimba3D-colmap-early.log"
  if "%EARLY_TRACE%"=="" set "EARLY_TRACE=C:\Windows\Temp\Bimba3D-colmap-early.log"
)
echo ==== %DATE% %TIME% PRE-BOOT install-colmap.cmd ====>>"%EARLY_TRACE%" 2>nul
set "EARLY_TRACE_CACHE=%~dp0install-colmap-early.log"
echo ==== %DATE% %TIME% PRE-BOOT install-colmap.cmd ====>>"%EARLY_TRACE_CACHE%" 2>nul
echo SCRIPT_DIR=%~dp0>>"%EARLY_TRACE%" 2>nul
echo ARGS_COUNT_CHECK arg1=%~1 arg2=%~2 arg3=%~3 arg4=%~4>>"%EARLY_TRACE%" 2>nul

set "SAFE_TEMP=%TEMP%"
if "%SAFE_TEMP%"=="" set "SAFE_TEMP=%TMP%"
if "%SAFE_TEMP%"=="" set "SAFE_TEMP=%WINDIR%\Temp"
if "%SAFE_TEMP%"=="" set "SAFE_TEMP=C:\Windows\Temp"
echo SAFE_TEMP=%SAFE_TEMP%>>"%EARLY_TRACE%" 2>nul
set "EARLY_TRACE_TEMP=%SAFE_TEMP%\Bimba3D-colmap-early.log"
echo ==== %DATE% %TIME% PRE-BOOT install-colmap.cmd ====>>"%EARLY_TRACE_TEMP%" 2>nul

set "LOG_ROOT=C:\ProgramData\Bimba3D\Logs"
if not exist "%LOG_ROOT%" mkdir "%LOG_ROOT%" >nul 2>nul
if not exist "%LOG_ROOT%" (
  set "LOG_ROOT=%SAFE_TEMP%\Bimba3D\Logs"
  if not exist "%LOG_ROOT%" mkdir "%LOG_ROOT%" >nul 2>nul
)
set "LOG_TEST=%LOG_ROOT%\.__write_test__.tmp"
echo log-test >"%LOG_TEST%" 2>nul
if errorlevel 1 (
  set "LOG_ROOT=%SAFE_TEMP%\Bimba3D\Logs"
  if not exist "%LOG_ROOT%" mkdir "%LOG_ROOT%" >nul 2>nul
) else (
  del /q "%LOG_TEST%" >nul 2>nul
)
set "LOG_FILE=%LOG_ROOT%\colmap-wrapper.log"
echo ==== %DATE% %TIME% START install-colmap.cmd ====>>"%LOG_FILE%"
echo LOG_ROOT=%LOG_ROOT%>>"%EARLY_TRACE%" 2>nul
echo LOG_FILE=%LOG_FILE%>>"%EARLY_TRACE%" 2>nul

if "%~1"=="" (
  echo Usage: install-colmap.cmd ^<install-dir^> ^<colmap-zip^>
  echo ERROR missing arg1>>"%LOG_FILE%"
  exit /b 2
)

if "%~2"=="" (
  echo Usage: install-colmap.cmd ^<install-dir^> ^<colmap-zip^>
  echo ERROR missing arg2>>"%LOG_FILE%"
  exit /b 2
)

set "INSTALL_DIR=%~1"
set "CUDA_ZIP_FILE=%~2"
set "NOCUDA_ZIP_FILE=%~3"
set "BURN_LOG_FILE=%~4"
set "PS_SCRIPT=%~dp0install-colmap.ps1"
set "DETAIL_LOG=%LOG_ROOT%\colmap-install-detail.log"
if not "%BURN_LOG_FILE%"=="" (
  echo [COLMAP] WRAPPER_START %DATE% %TIME%>>"%BURN_LOG_FILE%"
)
where powershell >>"%EARLY_TRACE%" 2>&1
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" set "POWERSHELL_EXE=powershell"
for %%I in ("%DETAIL_LOG%") do set "DETAIL_DIR=%%~dpI"
if not "%DETAIL_DIR%"=="" (
  if not exist "%DETAIL_DIR%" mkdir "%DETAIL_DIR%" >nul 2>nul
)
if not exist "%DETAIL_DIR%" (
  set "DETAIL_LOG=%~dp0colmap-install-detail.log"
)
echo INSTALL_DIR=%INSTALL_DIR%>>"%LOG_FILE%"
echo CUDA_ZIP_FILE=%CUDA_ZIP_FILE%>>"%LOG_FILE%"
echo NOCUDA_ZIP_FILE=%NOCUDA_ZIP_FILE%>>"%LOG_FILE%"
echo BURN_LOG_FILE=%BURN_LOG_FILE%>>"%LOG_FILE%"
echo PS_SCRIPT=%PS_SCRIPT%>>"%LOG_FILE%"
echo DETAIL_LOG=%DETAIL_LOG%>>"%LOG_FILE%"
if not "%BURN_LOG_FILE%"=="" (
  echo [COLMAP] ==== %DATE% %TIME% START install-colmap.cmd ====>>"%BURN_LOG_FILE%"
  echo [COLMAP] INSTALL_DIR=%INSTALL_DIR%>>"%BURN_LOG_FILE%"
  echo [COLMAP] CUDA_ZIP_FILE=%CUDA_ZIP_FILE%>>"%BURN_LOG_FILE%"
  echo [COLMAP] NOCUDA_ZIP_FILE=%NOCUDA_ZIP_FILE%>>"%BURN_LOG_FILE%"
  echo [COLMAP] DETAIL_LOG=%DETAIL_LOG%>>"%BURN_LOG_FILE%"
)

if not exist "%PS_SCRIPT%" (
  echo ERROR: Missing helper script: "%PS_SCRIPT%"
  echo ERROR helper script missing>>"%LOG_FILE%"
  if not "%BURN_LOG_FILE%"=="" echo [COLMAP] ERROR helper script missing: "%PS_SCRIPT%">>"%BURN_LOG_FILE%"
  exit /b 10
)

if "%NOCUDA_ZIP_FILE%"=="" (
  echo POWERSHELL_LAUNCH mode=cuda-only script="%PS_SCRIPT%">>"%EARLY_TRACE%" 2>nul
  "%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -CudaZipPath "%CUDA_ZIP_FILE%" -InstallDir "%INSTALL_DIR%" -LogPath "%DETAIL_LOG%" -BurnLogPath "%BURN_LOG_FILE%"
) else (
  echo POWERSHELL_LAUNCH mode=cuda+nocuda script="%PS_SCRIPT%">>"%EARLY_TRACE%" 2>nul
  "%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -CudaZipPath "%CUDA_ZIP_FILE%" -NoCudaZipPath "%NOCUDA_ZIP_FILE%" -InstallDir "%INSTALL_DIR%" -LogPath "%DETAIL_LOG%" -BurnLogPath "%BURN_LOG_FILE%"
)
set "PS_RC=%ERRORLEVEL%"
echo POWERSHELL_EXITCODE=%PS_RC%>>"%EARLY_TRACE%" 2>nul

if not exist "%DETAIL_LOG%" (
  echo %DATE% %TIME% DETAIL_LOG_MISSING RC=%ERRORLEVEL%>"%SAFE_TEMP%\bimba3d-colmap-detail-missing.log" 2>nul
)

set "DETAIL_MIRROR=%LOG_ROOT%\colmap-install-detail.log"
if exist "%DETAIL_LOG%" (
  if /I not "%DETAIL_LOG%"=="%DETAIL_MIRROR%" copy /Y "%DETAIL_LOG%" "%DETAIL_MIRROR%" >nul 2>nul
  if not "%BURN_LOG_FILE%"=="" echo [COLMAP] DETAIL_LOG_MIRROR=%DETAIL_MIRROR%>>"%BURN_LOG_FILE%"
)

set "RC=%PS_RC%"
if "%RC%"=="" set "RC=1"
echo RC=%RC%>>"%LOG_FILE%"
echo ==== %DATE% %TIME% END install-colmap.cmd ====>>"%LOG_FILE%"
if not "%BURN_LOG_FILE%"=="" (
  echo [COLMAP] RC=%RC%>>"%BURN_LOG_FILE%"
  echo [COLMAP] ==== %DATE% %TIME% END install-colmap.cmd ====>>"%BURN_LOG_FILE%"
)
echo FINAL_RC=%RC%>>"%EARLY_TRACE%" 2>nul
exit /b %RC%
