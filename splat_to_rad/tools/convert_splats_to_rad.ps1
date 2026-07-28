param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [string]$SparkRepoPath = "",
    [string]$OutputRoot = "",

    [switch]$Quality,
    [switch]$Chunked,
    [switch]$Force,
    [Nullable[int]]$MaxSh = $null
)

$ErrorActionPreference = "Stop"

function Resolve-ToolPath {
    param([string]$PathValue, [string]$DefaultRelativePath)

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot $DefaultRelativePath)).Path
    }

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return (Resolve-Path -LiteralPath $PathValue).Path
    }

    return (Resolve-Path -LiteralPath (Join-Path (Get-Location) $PathValue)).Path
}

function Assert-Command {
    param([string]$Name, [string]$InstallHint)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found. $InstallHint"
    }
}

function Get-SplatInputs {
    param([string]$ResolvedInput)

    $allowed = @(".ply", ".spz", ".splat", ".ksplat", ".sog", ".zip")
    $item = Get-Item -LiteralPath $ResolvedInput

    if (-not $item.PSIsContainer) {
        if ($allowed -notcontains $item.Extension.ToLowerInvariant()) {
            throw "Unsupported input extension '$($item.Extension)'. Supported: $($allowed -join ', ')"
        }
        return @($item)
    }

    return Get-ChildItem -LiteralPath $item.FullName -File |
        Where-Object { $allowed -contains $_.Extension.ToLowerInvariant() } |
        Sort-Object FullName
}

function Move-RadOutputs {
    param(
        [System.IO.FileInfo]$InputFile,
        [string]$ResolvedOutputRoot,
        [bool]$ForceOverwrite
    )

    $sceneName = [System.IO.Path]::GetFileNameWithoutExtension($InputFile.Name)
    $sceneDir = Join-Path $ResolvedOutputRoot $sceneName
    New-Item -ItemType Directory -Force -Path $sceneDir | Out-Null

    $sourceDir = $InputFile.DirectoryName
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($InputFile.Name)
    $outputs = @()
    $outputs += Get-ChildItem -LiteralPath $sourceDir -File -Filter "$stem-lod.rad" -ErrorAction SilentlyContinue
    $outputs += Get-ChildItem -LiteralPath $sourceDir -File -Filter "$stem-lod-*.radc" -ErrorAction SilentlyContinue

    if ($outputs.Count -eq 0) {
        throw "Spark finished but no RAD outputs were found beside '$($InputFile.FullName)'."
    }

    foreach ($output in $outputs) {
        $target = Join-Path $sceneDir $output.Name
        if (Test-Path -LiteralPath $target) {
            if (-not $ForceOverwrite) {
                throw "Output already exists: $target. Re-run with -Force to overwrite."
            }
            Remove-Item -LiteralPath $target -Force
        }
        Move-Item -LiteralPath $output.FullName -Destination $target
    }

    $manifest = [ordered]@{
        id = $sceneName
        source = $InputFile.Name
        format = "rad"
        entry = "$stem-lod.rad"
        chunked = [bool](Get-ChildItem -LiteralPath $sceneDir -File -Filter "$stem-lod-*.radc" -ErrorAction SilentlyContinue)
        generated_at = (Get-Date).ToUniversalTime().ToString("o")
        files = @(Get-ChildItem -LiteralPath $sceneDir -File | Sort-Object Name | ForEach-Object {
            [ordered]@{
                name = $_.Name
                bytes = $_.Length
            }
        })
    }

    $manifestPath = Join-Path $sceneDir "manifest.json"
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

    Write-Host "Scene written: $sceneDir"
    Write-Host "Temporary RAD files moved out of input folder."
}

Assert-Command -Name "node" -InstallHint "Install Node.js LTS from https://nodejs.org/."
Assert-Command -Name "npm" -InstallHint "Install Node.js LTS from https://nodejs.org/."
Assert-Command -Name "cargo" -InstallHint "Install Rust with rustup from https://rust-lang.org/tools/install/."

$resolvedInput = Resolve-Path -LiteralPath $InputPath
$resolvedSparkRepo = if ([string]::IsNullOrWhiteSpace($SparkRepoPath)) {
    Join-Path (Split-Path $PSScriptRoot -Parent) "vendor\spark"
} elseif ([System.IO.Path]::IsPathRooted($SparkRepoPath)) {
    $SparkRepoPath
} else {
    Join-Path (Get-Location) $SparkRepoPath
}

if (-not (Test-Path -LiteralPath $resolvedSparkRepo)) {
    throw "Spark repo not found at '$resolvedSparkRepo'. Clone it first: git clone https://github.com/sparkjsdev/spark `"$resolvedSparkRepo`""
}

$buildLodDir = Join-Path $resolvedSparkRepo "rust\build-lod"
if (-not (Test-Path -LiteralPath $buildLodDir)) {
    throw "Spark build-lod directory not found at '$buildLodDir'. Check that Spark was cloned correctly."
}

$resolvedOutputRoot = if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    Join-Path (Split-Path $PSScriptRoot -Parent) "processed"
} elseif ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot
} else {
    Join-Path (Get-Location) $OutputRoot
}
New-Item -ItemType Directory -Force -Path $resolvedOutputRoot | Out-Null

$inputs = Get-SplatInputs -ResolvedInput $resolvedInput.Path
if ($inputs.Count -eq 0) {
    throw "No supported splat files found in '$($resolvedInput.Path)'."
}

Write-Host "Converting $($inputs.Count) splat file(s) with Spark build-lod..."
Write-Host "Spark repo: $resolvedSparkRepo"
Write-Host "Output root: $resolvedOutputRoot"

foreach ($input in $inputs) {
    Write-Host ""
    Write-Host "Input: $($input.FullName)"

    $cargoArgs = @("run", "--release", "--", $input.FullName)
    if ($Quality) {
        $cargoArgs += "--quality"
    } else {
        $cargoArgs += "--quick"
    }
    if ($Chunked) {
        $cargoArgs += "--rad-chunked"
    }
    if ($null -ne $MaxSh) {
        $cargoArgs += "--max-sh=$MaxSh"
    }

    Push-Location $buildLodDir
    try {
        & cargo @cargoArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Spark build-lod failed for '$($input.FullName)' with exit code $LASTEXITCODE."
        }
    } finally {
        Pop-Location
    }

    Move-RadOutputs -InputFile $input -ResolvedOutputRoot $resolvedOutputRoot -ForceOverwrite:$Force.IsPresent
}

Write-Host ""
Write-Host "Done. Upload the processed scene folders to your static website only after local testing."
