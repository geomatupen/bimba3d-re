@echo off
setlocal
cd /d "%~dp0"
powershell -NoExit -ExecutionPolicy Bypass -File ".\tools\convert_splats_gui.ps1"
