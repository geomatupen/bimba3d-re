param(
    [string]$ManifestPath = "$PSScriptRoot\..\payloads\payload-manifest.json",
    [string]$OutputDir = "$PSScriptRoot\..\payloads",
    [switch]$SkipHashValidation
)

$ErrorActionPreference = 'Stop'

function Get-FileHashSha256([string]$Path) {
    (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

if (-not (Test-Path $ManifestPath)) {
    throw "Manifest file not found: $ManifestPath"
}

if (-not (Test-Path $OutputDir)) {
    New-Item -Path $OutputDir -ItemType Directory | Out-Null
}

$manifest = Get-Content -Path $ManifestPath -Raw | ConvertFrom-Json
if (-not $manifest.payloads) {
    throw "Manifest has no payload entries."
}

foreach ($payload in $manifest.payloads) {
    $target = Join-Path $OutputDir $payload.file
    Write-Host "==> $($payload.id)"

    if (-not (Test-Path $target)) {
        Write-Host "Downloading $($payload.url)"
        Invoke-WebRequest -Uri $payload.url -OutFile $target
    } else {
        Write-Host "File already exists, skipping download: $target"
    }

    if (-not $SkipHashValidation) {
        if ($payload.sha256 -and $payload.sha256 -ne 'REPLACE_WITH_SHA256') {
            $actual = Get-FileHashSha256 -Path $target
            $expected = $payload.sha256.ToLowerInvariant()
            if ($actual -ne $expected) {
                throw "SHA256 mismatch for $($payload.file). Expected: $expected, Actual: $actual"
            }
            Write-Host "SHA256 OK"
        } else {
            Write-Warning "No SHA256 pinned for $($payload.file). Set payload-manifest.json before release."
        }
    }
}

Write-Host "All payload steps completed."
