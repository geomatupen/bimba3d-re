param(
    [string]$InstallerPath = "$PSScriptRoot\..\Bimba3D-Setup.exe",
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [string]$CertificateThumbprint,
    [string]$PfxPath,
    [string]$PfxPassword
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $InstallerPath)) {
    throw "Installer not found: $InstallerPath"
}

$signtool = Get-Command signtool.exe -ErrorAction SilentlyContinue
if (-not $signtool) {
    throw "signtool.exe not found in PATH. Install Windows SDK signing tools."
}

$arguments = @('sign', '/fd', 'SHA256', '/td', 'SHA256', '/tr', $TimestampUrl)

if ($CertificateThumbprint) {
    $arguments += @('/sha1', $CertificateThumbprint)
} elseif ($PfxPath) {
    if (-not (Test-Path $PfxPath)) {
        throw "PFX file not found: $PfxPath"
    }
    $arguments += @('/f', $PfxPath)
    if ($PfxPassword) {
        $arguments += @('/p', $PfxPassword)
    }
} else {
    throw "Provide either -CertificateThumbprint or -PfxPath (with optional -PfxPassword)."
}

$arguments += $InstallerPath

Write-Host "Signing installer: $InstallerPath"
& signtool.exe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Signing failed with exit code $LASTEXITCODE"
}

Write-Host "Verifying signature"
& signtool.exe verify /pa /v $InstallerPath
if ($LASTEXITCODE -ne 0) {
    throw "Signature verification failed with exit code $LASTEXITCODE"
}

Write-Host "Installer signed and verified." -ForegroundColor Green
