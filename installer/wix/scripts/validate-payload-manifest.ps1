param(
    [string]$ManifestPath = "$PSScriptRoot\..\payloads\payload-manifest.json",
    [switch]$RequireLocalFiles,
    [string]$PayloadDir = "$PSScriptRoot\..\payloads"
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $ManifestPath)) {
    throw "Manifest file not found: $ManifestPath"
}

$manifest = Get-Content -Path $ManifestPath -Raw | ConvertFrom-Json
if (-not $manifest.payloads -or $manifest.payloads.Count -eq 0) {
    throw "Manifest has no payload entries."
}

$issues = @()

foreach ($payload in $manifest.payloads) {
    if (-not $payload.id) { $issues += "Payload entry is missing id." }
    if (-not $payload.file) { $issues += "Payload '$($payload.id)' missing file." }
    $urlMissingOrPlaceholder = (-not $payload.url) -or ($payload.url -match '^https://REPLACE_WITH')
    $urlLocalOnly = ($payload.url -eq 'LOCAL_FILE')

    if ($urlMissingOrPlaceholder) {
        $issues += "Payload '$($payload.id)' has placeholder or missing url."
    } elseif (-not $RequireLocalFiles -and $urlLocalOnly) {
        $issues += "Payload '$($payload.id)' uses LOCAL_FILE url marker but RequireLocalFiles was not set."
    }

    if (-not $payload.sha256 -or $payload.sha256 -eq 'REPLACE_WITH_SHA256') {
        $issues += "Payload '$($payload.id)' has placeholder/missing sha256."
    } elseif ($payload.sha256 -notmatch '^[A-Fa-f0-9]{64}$') {
        $issues += "Payload '$($payload.id)' sha256 is not 64 hex chars."
    }

    if ($RequireLocalFiles) {
        $localPath = Join-Path $PayloadDir $payload.file
        if (-not (Test-Path $localPath)) {
            $issues += "Missing local payload file: $localPath"
        }
    }
}

if ($issues.Count -gt 0) {
    Write-Host "Payload manifest validation failed:" -ForegroundColor Red
    $issues | ForEach-Object { Write-Host " - $_" -ForegroundColor Red }
    exit 1
}

Write-Host "Payload manifest validation passed." -ForegroundColor Green
exit 0
