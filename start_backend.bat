@echo off
echo ============================================================
echo Starting Bimba3D Backend Server
echo ============================================================
echo.
echo Backend will be available at: http://localhost:8005
echo Frontend (served by backend) at: http://localhost:8005
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

cd /d "%~dp0"
set PYTHONPATH=%CD%
set WORKER_MODE=local

python -m uvicorn bimba3d_backend.app.main:app --host 0.0.0.0 --port 8005 --reload

pause
