param(
    [string]$ManifestPath = "$PSScriptRoot\..\payloads\payload-manifest.json",
    [string]$PayloadDir = "$PSScriptRoot\..\payloads",
    [switch]$UpdateManifest
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $ManifestPath)) {
    throw "Manifest file not found: $ManifestPath"
}

$manifest = Get-Content -Path $ManifestPath -Raw | ConvertFrom-Json
if (-not $manifest.payloads) {
    throw "Manifest has no payload entries."
}

$changed = $false

foreach ($payload in $manifest.payloads) {
    $path = Join-Path $PayloadDir $payload.file
    if (Test-Path $path) {
        $hash = (Get-FileHash -Path $path -Algorithm SHA256).Hash.ToLowerInvariant()
        Write-Host "$($payload.file): $hash"
        if ($UpdateManifest) {
            if ($payload.sha256 -ne $hash) {
                $payload.sha256 = $hash
                $changed = $true
            }
        }
    } else {
        Write-Warning "Missing file, cannot hash: $path"
    }
}

if ($UpdateManifest -and $changed) {
    $manifest | ConvertTo-Json -Depth 10 | Set-Content -Path $ManifestPath -Encoding UTF8
    Write-Host "Manifest updated: $ManifestPath" -ForegroundColor Green
} elseif ($UpdateManifest) {
    Write-Host "Manifest unchanged." -ForegroundColor Yellow
}
