param(
    [Parameter(Mandatory = $true)]
    [string]$CudaZipPath,
    [string]$NoCudaZipPath,
    [string]$InstallDir = "C:\ProgramData\Bimba3D\third_party\colmap",
    [string]$LogPath,
    [string]$BurnLogPath
)

$ErrorActionPreference = 'Stop'

function Get-SafeTempRoot {
    if ($env:TEMP -and $env:TEMP.Trim()) {
        return $env:TEMP
    }
    if ($env:TMP -and $env:TMP.Trim()) {
        return $env:TMP
    }
    if ($env:WINDIR -and $env:WINDIR.Trim()) {
        return (Join-Path $env:WINDIR 'Temp')
    }
    return 'C:\Windows\Temp'
}

$logRoot = "C:\ProgramData\Bimba3D\Logs"
try {
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
} catch {
    $logRoot = Join-Path (Get-SafeTempRoot) "Bimba3D\Logs"
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
}
if ($LogPath) {
    $logPath = $LogPath
} else {
    $logPath = Join-Path $logRoot ("bimba3d-colmap-install-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + ".log")
}

try {
    Add-Content -Path $logPath -Value ("{0} bootstrap" -f (Get-Date -Format o)) -ErrorAction Stop
} catch {
    $logRoot = Join-Path (Get-SafeTempRoot) "Bimba3D\Logs"
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    if ($LogPath) {
        $logPath = Join-Path $logRoot ([System.IO.Path]::GetFileName($LogPath))
    } else {
        $logPath = Join-Path $logRoot ("bimba3d-colmap-install-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + ".log")
    }
}

$compat = $null

function Resolve-CompatibilityContext {
    $resolverPath = Join-Path $PSScriptRoot 'compatibility-resolver.ps1'
    if (-not (Test-Path $resolverPath)) {
        $resolverCandidate = Get-ChildItem -Path $PSScriptRoot -Filter 'compatibility-resolver*.ps1' -File -ErrorAction SilentlyContinue |
            Sort-Object -Property Name |
            Select-Object -First 1
        if ($resolverCandidate) {
            $resolverPath = $resolverCandidate.FullName
        }
    }
    if (-not (Test-Path $resolverPath)) {
        throw "Compatibility resolver not found in $PSScriptRoot"
    }

    . $resolverPath
    $matrixPath = Join-Path $PSScriptRoot 'compatibility-matrix-colmap.json'
    return (Resolve-Bimba3DCompatibility -MatrixPath $matrixPath)
}

function Get-FallbackCompatibilityContext {
    $cudaDetected = $false
    try {
        $cudaDetected = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
    } catch {
        $cudaDetected = $false
    }

    return [pscustomobject]@{
        detectedCudaVersion = if ($cudaDetected) { 'detected' } else { 'none' }
        detectedVsMajor = 'unknown'
        colmapPreferredVariant = if ($cudaDetected) { 'cuda' } else { 'nocuda' }
        colmapAssetProfile = 'fallback'
        colmapCudaVersion = 'offline'
        colmapNoCudaVersion = 'offline'
        useDefaultStack = $false
        colmapCudaUrl = $null
        colmapNoCudaUrl = $null
    }
}

function Write-Log([string]$message) {
    $line = "$(Get-Date -Format o) $message"
    Write-Host $line
    try {
        Add-Content -Path $logPath -Value $line -ErrorAction Stop
    } catch {
        Write-Host "[COLMAP-LOG-FALLBACK] $line"
    }

    if ($BurnLogPath) {
        try {
            Add-Content -Path $BurnLogPath -Value ("[COLMAP] " + $line) -ErrorAction Stop
        } catch {
        }
    }
}

function Download-ColmapArchive {
    param(
        [string]$Url,
        [string]$AssetName
    )

    $downloadDir = Join-Path $env:ProgramData "Bimba3D\cache"
    New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null

    $destination = Join-Path $downloadDir $AssetName
    if (-not $Url) {
        throw "No download URL available for requested COLMAP variant."
    }

    Write-Log "STATUS: download-start variant-asset='$AssetName' url='$Url'"
    Invoke-WebRequest -Uri $Url -OutFile $destination -UseBasicParsing
    if (-not (Test-Path $destination)) {
        throw "Failed to download COLMAP archive: $Url"
    }

    $sizeBytes = 0
    try {
        $sizeBytes = (Get-Item -LiteralPath $destination -ErrorAction Stop).Length
    } catch {
        $sizeBytes = 0
    }
    Write-Log "STATUS: download-complete path='$destination' bytes=$sizeBytes"

    return $destination
}

function Get-ArchivePathForVariant {
    param(
        [string]$Variant,
        [ValidateSet('offline', 'online')]
        [string]$Source
    )

    if ($Variant -eq 'cuda') {
        if ($Source -eq 'offline') {
            if ($CudaZipPath -and (Test-Path -LiteralPath $CudaZipPath)) {
                return [System.IO.Path]::GetFullPath($CudaZipPath)
            }

            throw "Offline CUDA COLMAP payload is missing."
        }

        return Download-ColmapArchive -Url $compat.colmapCudaUrl -AssetName ([System.IO.Path]::GetFileName($compat.colmapCudaUrl))
    }

    if ($Source -eq 'offline') {
        if ($NoCudaZipPath -and (Test-Path -LiteralPath $NoCudaZipPath)) {
            return [System.IO.Path]::GetFullPath($NoCudaZipPath)
        }

        throw "Offline no-CUDA COLMAP payload is missing."
    }

    return Download-ColmapArchive -Url $compat.colmapNoCudaUrl -AssetName ([System.IO.Path]::GetFileName($compat.colmapNoCudaUrl))
}

function Expand-ColmapArchive {
    param([string]$ArchivePath)

    if (-not (Test-Path -LiteralPath $ArchivePath)) {
        throw "COLMAP zip not found: $ArchivePath"
    }

    Write-Log "STATUS: extract-start archive='$ArchivePath' installDir='$InstallDir'"

    if (Test-Path -LiteralPath $InstallDir) {
        Remove-Item -LiteralPath $InstallDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Expand-Archive -LiteralPath $ArchivePath -DestinationPath $InstallDir -Force

    $colmapBat = Join-Path $InstallDir "COLMAP.bat"
    if (-not (Test-Path -LiteralPath $colmapBat)) {
        $nested = Get-ChildItem -LiteralPath $InstallDir -Filter "COLMAP.bat" -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($nested) {
            Write-Log "Found nested COLMAP.bat at $($nested.FullName)"
            $nestedBat = $nested.FullName
            $launcher = @"
@echo off
call "$nestedBat" %*
"@
            Set-Content -Path $colmapBat -Value $launcher -Encoding ascii -Force
            Write-Log "Created launcher at $colmapBat -> $nestedBat"
        }
    }

    if (-not (Test-Path -LiteralPath $colmapBat)) {
        throw "COLMAP.bat not found after extraction."
    }

    Write-Log "STATUS: extract-complete colmapBat='$colmapBat'"

    return $colmapBat
}

function Test-ColmapExecutable {
    param([string]$ColmapBatPath)

    & $ColmapBatPath -h | Out-Null
    return ($LASTEXITCODE -eq 0)
}

try {
    Write-Host "[COLMAP] Log file: $logPath"
    try {
        $compat = Resolve-CompatibilityContext
    } catch {
        Write-Log "WARNING: compatibility resolver failed; using fallback compatibility context. Reason: $($_.Exception.Message)"
        $compat = Get-FallbackCompatibilityContext
    }
    Write-Log "Starting COLMAP install"
    Write-Log "CudaZipPath=$CudaZipPath"
    Write-Log "NoCudaZipPath=$NoCudaZipPath"
    Write-Log "InstallDir=$InstallDir"
    Write-Log "Resolved compatibility: CUDA=$($compat.detectedCudaVersion) VS=$($compat.detectedVsMajor) Variant=$($compat.colmapPreferredVariant) Profile=$($compat.colmapAssetProfile) CudaColmap=$($compat.colmapCudaVersion) NoCudaColmap=$($compat.colmapNoCudaVersion) DefaultStack=$($compat.useDefaultStack)"
    Write-Log "Resolved URLs: cuda='$($compat.colmapCudaUrl)' nocuda='$($compat.colmapNoCudaUrl)'"

    $installPlan = New-Object System.Collections.Generic.List[object]
    if ($compat.colmapPreferredVariant -eq 'cuda' -and $compat.useDefaultStack) {
        $installPlan.Add([pscustomobject]@{ Variant = 'cuda'; Source = 'offline'; Reason = 'default stack match' })
        $installPlan.Add([pscustomobject]@{ Variant = 'cuda'; Source = 'online'; Reason = 'refresh from online if needed' })
        $installPlan.Add([pscustomobject]@{ Variant = 'nocuda'; Source = 'online'; Reason = 'fallback when CUDA COLMAP is incompatible' })
        $installPlan.Add([pscustomobject]@{ Variant = 'nocuda'; Source = 'offline'; Reason = 'offline fallback when internet is unavailable' })
    } elseif ($compat.colmapPreferredVariant -eq 'cuda') {
        $installPlan.Add([pscustomobject]@{ Variant = 'cuda'; Source = 'offline'; Reason = 'CUDA detected: try bundled CUDA COLMAP first' })
        $installPlan.Add([pscustomobject]@{ Variant = 'cuda'; Source = 'online'; Reason = 'CUDA bundled candidate failed; try online CUDA build' })
        $installPlan.Add([pscustomobject]@{ Variant = 'nocuda'; Source = 'offline'; Reason = 'last fallback when CUDA candidates fail' })
        $installPlan.Add([pscustomobject]@{ Variant = 'nocuda'; Source = 'online'; Reason = 'last online fallback when offline no-CUDA unavailable' })
    } else {
        $installPlan.Add([pscustomobject]@{ Variant = 'nocuda'; Source = 'offline'; Reason = 'CUDA not preferred by compatibility profile' })
        $installPlan.Add([pscustomobject]@{ Variant = 'nocuda'; Source = 'online'; Reason = 'refresh from online if needed' })
    }

    $selectedVariant = $null
    $selectedSource = $null
    $colmapBat = $null
    foreach ($plan in $installPlan) {
        $variant = [string]$plan.Variant
        $source = [string]$plan.Source
        $reason = [string]$plan.Reason

        Write-Host "[COLMAP] Trying variant=$variant source=$source ($reason)..."
        Write-Log "STATUS: candidate-start variant='$variant' source='$source' reason='$reason'"

        try {
            $archivePath = Get-ArchivePathForVariant -Variant $variant -Source $source
            Write-Log "STATUS: candidate-archive variant='$variant' source='$source' archive='$archivePath'"
            $colmapBat = Expand-ColmapArchive -ArchivePath $archivePath
            $smokeOk = $false
            try {
                Write-Log "STATUS: smoke-test-start colmapBat='$colmapBat'"
                $smokeOk = Test-ColmapExecutable -ColmapBatPath $colmapBat
            } catch {
                $smokeOk = $false
            }

            if ($smokeOk) {
                Write-Host "[COLMAP] Installed variant=$variant source=$source successfully."
                Write-Log "STATUS: candidate-success variant='$variant' source='$source'"
                $selectedVariant = $variant
                $selectedSource = $source
                break
            }

            if (Test-Path -LiteralPath $colmapBat) {
                Write-Host "[COLMAP] Smoke test failed for variant=$variant source=$source, but executable exists. Accepting install with warning."
                Write-Log "STATUS: candidate-warning smoke-failed-but-exists variant='$variant' source='$source' colmapBat='$colmapBat'"
                $selectedVariant = $variant
                $selectedSource = $source
                break
            }

            Write-Host "[COLMAP] Smoke test failed for variant=$variant source=$source and executable was not found after extraction."
            Write-Log "STATUS: candidate-failed smoke-failed-and-missing variant='$variant' source='$source'"
        } catch {
            Write-Host "[COLMAP] Attempt failed for variant=$variant source=$source."
            Write-Log "STATUS: candidate-error variant='$variant' source='$source' error='$($_.Exception.Message)'"
        }
    }

    if (-not $selectedVariant) {
        throw 'Failed to install a compatible COLMAP variant.'
    }

    try {
        [Environment]::SetEnvironmentVariable("COLMAP_EXE", $colmapBat, "Machine")
        Write-Log "Set machine environment variable COLMAP_EXE=$colmapBat"
    } catch {
        Write-Log ("WARNING: failed to set machine COLMAP_EXE: " + $_.Exception.Message)
        try {
            [Environment]::SetEnvironmentVariable("COLMAP_EXE", $colmapBat, "User")
            Write-Log "Set user environment variable COLMAP_EXE=$colmapBat"
        } catch {
            Write-Log ("WARNING: failed to set user COLMAP_EXE: " + $_.Exception.Message)
            Write-Log "Continuing without persisted COLMAP_EXE environment variable."
        }
    }
    Write-Log "COLMAP installed at $InstallDir (variant=$selectedVariant source=$selectedSource)"
    Write-Log "Completed successfully"
    exit 0
} catch {
    Write-Log ("ERROR: " + $_.Exception.Message)
    if ($_.Exception.StackTrace) {
        Write-Log ("STACK: " + $_.Exception.StackTrace)
    }
    if ($_.Exception.InnerException) {
        Write-Log ("INNER: " + $_.Exception.InnerException.Message)
    }
    Write-Host "COLMAP install failed. See log: $logPath" -ForegroundColor Red
    exit 1
}
