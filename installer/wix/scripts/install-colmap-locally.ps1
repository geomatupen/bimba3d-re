param(
    [string]$ZipPath = "$PSScriptRoot\..\payloads\colmap-x64-windows-cuda.zip",
    [string]$InstallDir = "C:\Program Files\Bimba3D\third_party\colmap"
)

$ErrorActionPreference = 'Stop'

$ZipPath = [System.IO.Path]::GetFullPath($ZipPath)
$logPath = Join-Path $env:TEMP ("bimba3d-colmap-install-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + ".log")

function Write-Log([string]$message) {
    $line = "$(Get-Date -Format o) $message"
    Write-Host $line
    Add-Content -Path $logPath -Value $line
}

Write-Log "Starting COLMAP local install"
Write-Log "ZipPath=$ZipPath"
Write-Log "InstallDir=$InstallDir"

if (-not (Test-Path -LiteralPath $ZipPath)) {
    throw "COLMAP zip not found: $ZipPath"
}

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    throw "Please run PowerShell as Administrator for COLMAP install. Log: $logPath"
}

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
Expand-Archive -LiteralPath $ZipPath -DestinationPath $InstallDir -Force

$colmapBat = Join-Path $InstallDir "COLMAP.bat"
if (-not (Test-Path -LiteralPath $colmapBat)) {
    $nested = Get-ChildItem -LiteralPath $InstallDir -Filter "COLMAP.bat" -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($nested) {
        $nestedRoot = Split-Path -Parent $nested.FullName
        Copy-Item -Path (Join-Path $nestedRoot "*") -Destination $InstallDir -Recurse -Force
    }
}

if (-not (Test-Path -LiteralPath $colmapBat)) {
    throw "COLMAP.bat not found after extraction. Log: $logPath"
}

[Environment]::SetEnvironmentVariable("COLMAP_EXE", $colmapBat, "Machine")
Write-Log "Installed COLMAP at $InstallDir"
Write-Log "Set machine environment variable COLMAP_EXE=$colmapBat"
Write-Log "Completed successfully"
Write-Host "COLMAP install complete. Log: $logPath" -ForegroundColor Green
