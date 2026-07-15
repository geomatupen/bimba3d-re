param(
    [switch]$DownloadPayloads,
    [switch]$UpdateShaFromLocalFiles,
    [string]$ManifestPath = "$PSScriptRoot\..\payloads\payload-manifest.json",
    [string]$OutputExe = "$PSScriptRoot\..\Bimba3D-Setup.exe",
    [string]$CertificateThumbprint,
    [string]$PfxPath,
    [string]$PfxPassword,
    [switch]$SkipSigning
)

$ErrorActionPreference = 'Stop'

$downloadScript = Join-Path $PSScriptRoot "download-payloads.ps1"
$updateShaScript = Join-Path $PSScriptRoot "update-manifest-sha256.ps1"
$validateScript = Join-Path $PSScriptRoot "validate-payload-manifest.ps1"
$buildScript = Join-Path $PSScriptRoot "build-bundle.ps1"
$signScript = Join-Path $PSScriptRoot "sign-installer.ps1"

if ($DownloadPayloads) {
    Write-Host "Step 1/5: Downloading payloads"
    & powershell -ExecutionPolicy Bypass -File $downloadScript -ManifestPath $ManifestPath
    if ($LASTEXITCODE -ne 0) { throw "Payload download failed." }
}

if ($UpdateShaFromLocalFiles) {
    Write-Host "Step 2/5: Updating SHA256 values from local payload files"
    & powershell -ExecutionPolicy Bypass -File $updateShaScript -ManifestPath $ManifestPath -UpdateManifest
    if ($LASTEXITCODE -ne 0) { throw "SHA256 update failed." }
}

Write-Host "Step 3/5: Validating payload manifest"
& powershell -ExecutionPolicy Bypass -File $validateScript -ManifestPath $ManifestPath -RequireLocalFiles
if ($LASTEXITCODE -ne 0) { throw "Manifest validation failed." }

Write-Host "Step 4/5: Building installer bundle"
& powershell -ExecutionPolicy Bypass -File $buildScript -WxsPath "$PSScriptRoot\..\Bimba3D.Bundle.wxs" -OutputExe $OutputExe -CiStrict
if ($LASTEXITCODE -ne 0) { throw "Bundle build failed." }

if (-not $SkipSigning) {
    Write-Host "Step 5/5: Signing installer"
    if ($CertificateThumbprint) {
        & powershell -ExecutionPolicy Bypass -File $signScript -InstallerPath $OutputExe -CertificateThumbprint $CertificateThumbprint
    } elseif ($PfxPath) {
        & powershell -ExecutionPolicy Bypass -File $signScript -InstallerPath $OutputExe -PfxPath $PfxPath -PfxPassword $PfxPassword
    } else {
        throw "Signing is enabled, but no certificate input was provided. Use -CertificateThumbprint or -PfxPath, or pass -SkipSigning."
    }

    if ($LASTEXITCODE -ne 0) { throw "Signing failed." }
} else {
    Write-Warning "Signing skipped (-SkipSigning). Do not release unsigned installer."
}

Write-Host "Release pipeline completed: $OutputExe" -ForegroundColor Green
