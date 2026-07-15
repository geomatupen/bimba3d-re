param(
    [Parameter(Mandatory = $true)]
    [string]$InstallDir,
    [Parameter(Mandatory = $true)]
    [ValidateSet("prepare", "torch", "gsplat", "requirements")]
    [string]$Phase
)

$ErrorActionPreference = 'Stop'

$runtimeRoot = Join-Path $env:ProgramData "Bimba3D\runtime"
$venvDir = Join-Path $runtimeRoot ".venv"
$venvPy = Join-Path $venvDir "Scripts\python.exe"
$bootstrapState = Join-Path $runtimeRoot "bootstrap-state.txt"
$torchFlavorFile = Join-Path $runtimeRoot "torch-flavor.txt"
$backendRequirements = Join-Path $InstallDir "bimba3d_backend\requirements.windows.txt"

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
$matrixPath = Join-Path $PSScriptRoot 'compatibility-matrix-runtime.json'
if (-not (Test-Path $matrixPath)) {
    $matrixPath = Join-Path $PSScriptRoot 'compatibility-matrix.json'
}
$compat = Resolve-Bimba3DCompatibility -MatrixPath $matrixPath

$torchTrack = [string]$compat.torchTrack
$torchIndex = [string]$compat.torchIndexUrl
$torchVersion = [string]$compat.torchVersion
$torchvisionVersion = [string]$compat.torchvisionVersion
$torchaudioVersion = [string]$compat.torchaudioVersion
$torchCpuVersion = [string]$compat.torchCpuVersion
$torchvisionCpuVersion = [string]$compat.torchvisionCpuVersion
$torchaudioCpuVersion = [string]$compat.torchaudioCpuVersion
$gsplatVersion = [string]$compat.gsplatVersion
$gsplatSupportedTracks = @($compat.gsplatSupportedTorchTracks)
$selectedCudaVersion = [string]$compat.selectedCudaVersion
$selectedCudaPath = [string]$compat.selectedCudaPath
$recommendedCudaVersion = [string]$compat.recommendedCudaVersion
$recommendedCudaInstallerUrl = [string]$compat.recommendedCudaInstallerUrl
$cudaPreferredVersion = [string]$compat.cudaPreferred
$detectedVsMajor = [string]$compat.detectedVsMajor
$requiredVsMajor = [string]$compat.requiredVsMajor
$recommendedVsInstallerUrl = [string]$compat.recommendedVsInstallerUrl

function Write-PhaseEvent {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Step,
        [ValidateSet('RUN', 'STEP', 'SOURCE', 'INFO', 'WARN', 'ERROR')]
        [string]$Kind = 'INFO',
        [string]$Detail = ''
    )

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'
    $message = "[$timestamp] [$Phase] [$Kind] $Step"
    if ($Detail) {
        $message = "$message - $Detail"
    }
    Write-Host $message
}

function Ensure-RuntimeRoot {
    if (-not (Test-Path $runtimeRoot)) {
        New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
    }

    Ensure-RuntimeWritable -TargetPath $runtimeRoot
}

function Ensure-RuntimeWritable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    if (-not (Test-Path $TargetPath)) {
        return
    }

    try {
        & icacls $TargetPath /grant "*S-1-5-32-545:(OI)(CI)M" /T /C | Out-Null
    }
    catch {
        Write-Warning "Unable to update ACLs for runtime path '$TargetPath': $($_.Exception.Message)"
    }
}

function Resolve-PythonExecutable {
    $candidates = @(
        "C:\Program Files\Python312\python.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return @{ exe = $candidate; args = @() }
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return @{ exe = "python"; args = @() }
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        return @{ exe = "py"; args = @("-3") }
    }

    throw "Python is not installed. Please install Python 3.12+ and rerun setup."
}

function Ensure-Venv {
    Ensure-RuntimeRoot

    if (Test-Path $venvDir) {
        Ensure-RuntimeWritable -TargetPath $venvDir
    }

    if (-not (Test-Path $venvPy)) {
        Write-Host "Creating Python virtual environment..."
        $python = Resolve-PythonExecutable
        & $python.exe @($python.args + @("-m", "venv", $venvDir))
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create Python virtual environment."
        }
    }

    if (-not (Test-Path $venvPy)) {
        throw "Venv python not found at $venvPy"
    }

    Ensure-RuntimeWritable -TargetPath $venvDir
}

function Get-VsWherePath {
    $cmd = Get-Command vswhere.exe -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) {
        return $cmd.Source
    }

    $candidates = @()
    if ($env:ProgramFiles -and $env:ProgramFiles.Trim()) {
        $candidates += (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\Installer\vswhere.exe')
    }
    if (${env:ProgramFiles(x86)} -and ${env:ProgramFiles(x86)}.Trim()) {
        $candidates += (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe')
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Get-VsX64DevCmdPath {
    $vswherePath = Get-VsWherePath
    if (-not $vswherePath) {
        return $null
    }

    $installationPath = $null
    try {
        $installationPath = & $vswherePath -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath -latest 2>$null
    } catch {
    }

    if (-not $installationPath -or -not $installationPath.Trim()) {
        return $null
    }

    $installationPath = $installationPath.Trim()
    $vsDevCmd = Join-Path $installationPath 'Common7\Tools\VsDevCmd.bat'
    if (Test-Path $vsDevCmd) {
        return @{ path = $vsDevCmd; type = 'vsdevcmd' }
    }

    $vcvars64 = Join-Path $installationPath 'VC\Auxiliary\Build\vcvars64.bat'
    if (Test-Path $vcvars64) {
        return @{ path = $vcvars64; type = 'vcvars64' }
    }

    return $null
}

function Invoke-InVsX64Environment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandLine
    )

    $devCmd = Get-VsX64DevCmdPath
    if (-not $devCmd) {
        Write-Warning 'VS x64 developer command environment was not found.'
        return $false
    }

    $devCmdPath = [string]$devCmd.path
    $bootstrap = ''
    if ([string]$devCmd.type -eq 'vsdevcmd') {
        $bootstrap = "`"$devCmdPath`" -arch=x64 -host_arch=x64"
    } else {
        $bootstrap = "`"$devCmdPath`" x64"
    }

    $fullCommand = "$bootstrap && $CommandLine"
    & cmd.exe /d /c $fullCommand
    return ($LASTEXITCODE -eq 0)
}

function Install-PipTooling {
    Write-PhaseEvent -Step 'Install pip/setuptools/wheel' -Kind 'STEP'
    & $venvPy -m pip install --upgrade pip setuptools==69.5.1 wheel
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Python packaging tooling."
    }
}

function Install-CpuTorch {
    Write-PhaseEvent -Step 'Install CPU torch fallback' -Kind 'SOURCE' -Detail 'ONLINE_OR_CACHE'
    & $venvPy -m pip install --force-reinstall "torch==$torchCpuVersion" "torchvision==$torchvisionCpuVersion" "torchaudio==$torchaudioCpuVersion"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed to install CPU torch fallback runtime."
        return $false
    }

    return $true
}

function ConvertTo-SemVersion {
    param(
        [string]$Value
    )

    if (-not $Value) {
        return $null
    }

    $trimmed = $Value.Trim()
    if (-not $trimmed) {
        return $null
    }

    $match = [regex]::Match($trimmed, '(\d+)\.(\d+)(?:\.(\d+))?')
    if (-not $match.Success) {
        return $null
    }

    $major = [int]$match.Groups[1].Value
    $minor = [int]$match.Groups[2].Value
    $patch = 0
    if ($match.Groups[3].Success) {
        $patch = [int]$match.Groups[3].Value
    }

    return [version]::new($major, $minor, $patch)
}

function Get-InstalledMsvcToolsetVersion {
    $vswherePath = Get-VsWherePath
    if (-not $vswherePath) {
        return $null
    }

    $installationPath = $null
    try {
        $installationPath = & $vswherePath -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath -latest 2>$null
    }
    catch {
        return $null
    }

    if (-not $installationPath -or -not $installationPath.Trim()) {
        return $null
    }

    $toolsRoot = Join-Path $installationPath.Trim() 'VC\Tools\MSVC'
    if (-not (Test-Path $toolsRoot)) {
        return $null
    }

    $versions = @()
    Get-ChildItem -Path $toolsRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $parsed = ConvertTo-SemVersion -Value $_.Name
        if ($parsed) {
            $versions += [pscustomobject]@{ raw = $_.Name; parsed = $parsed }
        }
    }

    if (-not $versions -or $versions.Count -eq 0) {
        return $null
    }

    return ($versions | Sort-Object -Property parsed -Descending | Select-Object -First 1)
}

function Wait-ForCudaUpgradeIfInteractive {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DownloadUrl,
        [string]$RecommendedVersion
    )

    if (-not [Environment]::UserInteractive) {
        return
    }

    Write-Host "Open CUDA installer link: $DownloadUrl"
    try {
        Start-Process $DownloadUrl | Out-Null
    }
    catch {
        Write-Warning "Unable to open browser automatically. Open this URL manually: $DownloadUrl"
    }

    if ($RecommendedVersion) {
        Write-Host "Install CUDA $RecommendedVersion, then return to this window."
    }
    else {
        Write-Host "Install CUDA, then return to this window."
    }
    [void](Read-Host "Press Enter after CUDA installation is finished")
}

function Resolve-RecommendedCudaVersion {
    if ($cudaPreferredVersion -and $cudaPreferredVersion.Trim()) {
        return $cudaPreferredVersion.Trim()
    }
    if ($recommendedCudaVersion -and $recommendedCudaVersion.Trim()) {
        return $recommendedCudaVersion.Trim()
    }
    return '12.5'
}

function Resolve-RecommendedCudaInstallerUrl {
    if ($recommendedCudaInstallerUrl -and $recommendedCudaInstallerUrl.Trim()) {
        return $recommendedCudaInstallerUrl.Trim()
    }
    return 'https://developer.download.nvidia.com/compute/cuda/12.5.0/network_installers/cuda_12.5.0_windows_network.exe'
}

function Resolve-RequiredVsMajor {
    if ($requiredVsMajor -and $requiredVsMajor.Trim()) {
        return $requiredVsMajor.Trim()
    }
    return '17'
}

function Resolve-RecommendedVsInstallerUrl {
    if ($recommendedVsInstallerUrl -and $recommendedVsInstallerUrl.Trim()) {
        return $recommendedVsInstallerUrl.Trim()
    }
    return 'https://aka.ms/vs/17/release/vs_BuildTools.exe'
}

function Wait-ForToolchainUpgradeIfInteractive {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CudaDownloadUrl,
        [Parameter(Mandatory = $true)]
        [string]$CudaVersion,
        [Parameter(Mandatory = $true)]
        [string]$VsDownloadUrl,
        [Parameter(Mandatory = $true)]
        [string]$VsMajor
    )

    if (-not [Environment]::UserInteractive) {
        return
    }

    Write-Host "Open CUDA installer link: $CudaDownloadUrl"
    try {
        Start-Process $CudaDownloadUrl | Out-Null
    }
    catch {
        Write-Warning "Unable to open CUDA installer link automatically. Open this URL manually: $CudaDownloadUrl"
    }

    Write-Host "Open Visual Studio Build Tools installer link: $VsDownloadUrl"
    try {
        Start-Process $VsDownloadUrl | Out-Null
    }
    catch {
        Write-Warning "Unable to open VS installer link automatically. Open this URL manually: $VsDownloadUrl"
    }

    Write-Host "Install CUDA $CudaVersion and Visual Studio $VsMajor Build Tools (C++ x64), then return to this window."
    [void](Read-Host "Press Enter after toolchain installation is finished")
}

function Throw-GsplatToolchainUpgradeRequired {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Reason
    )

    $targetCudaVersion = Resolve-RecommendedCudaVersion
    $cudaDownloadUrl = Resolve-RecommendedCudaInstallerUrl
    $targetVsMajor = Resolve-RequiredVsMajor
    $vsDownloadUrl = Resolve-RecommendedVsInstallerUrl

    $message = @"
gsplat installation cannot continue due to an incompatible CUDA/Visual Studio toolchain.

Detected issue: $Reason

Required to continue:
- CUDA Toolkit $targetCudaVersion (x64)
- Visual Studio 2022 Build Tools (v$targetVsMajor) with C++ x64 tools

Install links (clickable):
- CUDA $targetCudaVersion installer: $cudaDownloadUrl
- Visual Studio 2022 Build Tools: $vsDownloadUrl

After installation finishes, rerun the Bimba3D installer.
"@
    Write-Error $message
    Wait-ForToolchainUpgradeIfInteractive -CudaDownloadUrl $cudaDownloadUrl -CudaVersion $targetCudaVersion -VsDownloadUrl $vsDownloadUrl -VsMajor $targetVsMajor
    throw $message
}

function Throw-GsplatCudaUpgradeRequired {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Reason,
        [string]$SupportedRangeText
    )

    $targetCudaVersion = Resolve-RecommendedCudaVersion
    $downloadUrl = Resolve-RecommendedCudaInstallerUrl

    $rangeSegment = ''
    if ($SupportedRangeText) {
        $rangeSegment = " gsplat supported CUDA range: $SupportedRangeText."
    }

    $message = @"
gsplat installation cannot continue because the installed CUDA version is outside the supported range.

Detected issue: $Reason$rangeSegment

Required to continue:
- Install CUDA Toolkit $targetCudaVersion (x64)

Install link (clickable):
- CUDA $targetCudaVersion installer: $downloadUrl

After CUDA installation finishes, rerun the Bimba3D installer.
"@
    Write-Error $message
    Wait-ForCudaUpgradeIfInteractive -DownloadUrl $downloadUrl -RecommendedVersion $targetCudaVersion
    throw $message
}

function Assert-GsplatCudaMsvcCompatibility {
    $downloadUrl = Resolve-RecommendedCudaInstallerUrl

    $cudaVersionText = $selectedCudaVersion
    if (-not $cudaVersionText -and $selectedCudaPath) {
        $pathMatch = [regex]::Match($selectedCudaPath, 'v(\d+\.\d+)')
        if ($pathMatch.Success) {
            $cudaVersionText = $pathMatch.Groups[1].Value
        }
    }

    $cudaVersion = ConvertTo-SemVersion -Value $cudaVersionText
    if (-not $cudaVersion) {
        return
    }

    $msvcInfo = Get-InstalledMsvcToolsetVersion
    if (-not $msvcInfo) {
        return
    }

    $minimumCudaForNewMsvc = [version]::new(12, 4, 0)
    $newMsvcThreshold = [version]::new(14, 44, 0)

    if ($cudaVersion -ge $minimumCudaForNewMsvc) {
        return
    }

    if ($msvcInfo.parsed -lt $newMsvcThreshold) {
        return
    }

    $targetCudaVersion = Resolve-RecommendedCudaVersion
    $message = @"
gsplat installation cannot continue due to a CUDA/Visual Studio compiler mismatch.

Detected toolchain:
- CUDA: $cudaVersionText
- MSVC toolset: $($msvcInfo.raw)

Failure signature:
- STL1002 / unexpected compiler version during gsplat CUDA extension build

Required to continue:
- Install CUDA Toolkit $targetCudaVersion (x64)

Install link (clickable):
- CUDA $targetCudaVersion installer: $downloadUrl

After CUDA installation finishes, rerun the Bimba3D installer.
"@
    Write-Error $message
    Wait-ForCudaUpgradeIfInteractive -DownloadUrl $downloadUrl -RecommendedVersion $targetCudaVersion
    throw $message
}

function Test-TorchImport {
    & $venvPy -c "import torch" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Invoke-PythonProbe {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Code,
        [switch]$SuppressStdErr
    )

    $originalErrorActionPreference = $ErrorActionPreference
    $hadNativePreference = Test-Path Variable:\PSNativeCommandUseErrorActionPreference
    if ($hadNativePreference) {
        $originalNativePreference = $PSNativeCommandUseErrorActionPreference
    }

    try {
        $ErrorActionPreference = 'Continue'
        if ($hadNativePreference) {
            $script:PSNativeCommandUseErrorActionPreference = $false
        }

        if ($SuppressStdErr) {
            $output = (& $venvPy -c $Code 2>$null | Out-String).Trim()
        }
        else {
            $output = (& $venvPy -c $Code 2>&1 | Out-String).Trim()
        }

        return @{
            output = $output
            exitCode = $LASTEXITCODE
        }
    }
    catch {
        return @{
            output = ""
            exitCode = 1
        }
    }
    finally {
        $ErrorActionPreference = $originalErrorActionPreference
        if ($hadNativePreference) {
            $script:PSNativeCommandUseErrorActionPreference = $originalNativePreference
        }
    }
}

function Get-ExistingTorchFlavor {
    $probeResult = Invoke-PythonProbe -Code "import torch; v=getattr(torch,'__version__',''); c=getattr(getattr(torch,'version',None),'cuda',None); ok=torch.cuda.is_available(); print(f'version={v};cuda={c};cuda_available={ok}'); raise SystemExit(0 if (c and ok) else 1)" -SuppressStdErr
    $probe = $probeResult.output
    if ($probe) {
        Write-Host "Existing torch probe: $probe"
    }

    if ($probeResult.exitCode -ne 0) {
        return $null
    }

    if ($probe -match 'version=([^;]+)') {
        $version = $Matches[1]
        if ($version -match '\+cu118') {
            return 'cu118'
        }
        if ($version -match '\+cu121') {
            return 'cu121'
        }
    }

    if ($probe -match 'cuda=(\d+\.\d+)') {
        $cudaVersion = [version]::new([int]$Matches[1].Split('.')[0], [int]$Matches[1].Split('.')[1])
        if ($cudaVersion -ge [version]::new(12, 1)) {
            return 'cu121'
        }
        if ($cudaVersion -ge [version]::new(11, 8)) {
            return 'cu118'
        }
    }

    return $null
}

function Get-ExistingGsplatVersion {
    $gsplatProbeResult = Invoke-PythonProbe -Code "import importlib.metadata as m; import gsplat; print(m.version('gsplat'))" -SuppressStdErr
    $gsplatVersionText = $gsplatProbeResult.output
    if ($gsplatProbeResult.exitCode -ne 0) {
        return $null
    }

    if ($gsplatVersionText) {
        return $gsplatVersionText
    }

    return $null
}

function Get-TorchInstallCandidates {
    $candidates = @()
    $existingTracks = @{}

    if ($torchTrack -and $torchIndex -and $torchVersion -and $torchvisionVersion -and $torchaudioVersion) {
        $primaryCandidate = [pscustomobject]@{
            track = $torchTrack
            indexUrl = $torchIndex
            torchVersion = $torchVersion
            torchvisionVersion = $torchvisionVersion
            torchaudioVersion = $torchaudioVersion
            reason = 'matrix-selected'
        }
        $candidates += $primaryCandidate
        $existingTracks[[string]$primaryCandidate.track] = $true
    }

    $fallbackMap = @(
        [pscustomobject]@{
            track = 'cu121'
            indexUrl = 'https://download.pytorch.org/whl/cu121'
            torchVersion = '2.5.1+cu121'
            torchvisionVersion = '0.20.1+cu121'
            torchaudioVersion = '2.5.1+cu121'
            reason = 'fallback-cu121'
        },
        [pscustomobject]@{
            track = 'cu118'
            indexUrl = 'https://download.pytorch.org/whl/cu118'
            torchVersion = '2.5.1+cu118'
            torchvisionVersion = '0.20.1+cu118'
            torchaudioVersion = '2.5.1+cu118'
            reason = 'fallback-cu118'
        }
    )

    foreach ($fallback in $fallbackMap) {
        $fallbackTrack = [string]$fallback.track
        if ($existingTracks.ContainsKey($fallbackTrack)) {
            continue
        }
        $candidates += $fallback
        $existingTracks[$fallbackTrack] = $true
    }

    return $candidates
}

function Install-Torch {
    Ensure-Venv

    Write-PhaseEvent -Step 'Torch runtime setup started' -Kind 'RUN' -Detail 'local-first mode'
    $torchFlavor = 'cpu'
    $torchProbeFile = Join-Path $runtimeRoot 'torch-probe.txt'
    $torchImportable = $false
    Write-PhaseEvent -Step 'Compatibility profile' -Kind 'INFO' -Detail "CUDA(selected)=$selectedCudaVersion CUDA(detected)=$($compat.detectedCudaVersion) VS=$($compat.detectedVsMajor) Track=$torchTrack DefaultStack=$($compat.useDefaultStack)"

    $torchCandidates = Get-TorchInstallCandidates
    $candidateTracks = @($torchCandidates | ForEach-Object { [string]$_.track })

    $existingFlavor = Get-ExistingTorchFlavor
    if ($existingFlavor -and ($candidateTracks -contains $existingFlavor)) {
        Write-PhaseEvent -Step 'Torch source selected' -Kind 'SOURCE' -Detail "LOCAL_EXISTING track=$existingFlavor"
        $torchFlavor = $existingFlavor
        Set-Content -Path $torchFlavorFile -Value $torchFlavor -Encoding ASCII
        return
    }

    Write-PhaseEvent -Step 'Torch source selected' -Kind 'SOURCE' -Detail 'ONLINE_OR_CACHE candidates (no local reusable CUDA torch)'

    $cudaTorchReady = $false
    foreach ($candidate in $torchCandidates) {
        Write-PhaseEvent -Step 'Torch candidate install' -Kind 'STEP' -Detail "track=$($candidate.track) reason=$($candidate.reason) index=$($candidate.indexUrl)"
        & $venvPy -m pip install --index-url $candidate.indexUrl --force-reinstall "torch==$($candidate.torchVersion)" "torchvision==$($candidate.torchvisionVersion)" "torchaudio==$($candidate.torchaudioVersion)"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Torch install failed for track '$($candidate.track)'."
            continue
        }

        $torchProbeResult = Invoke-PythonProbe -Code "import torch; v=getattr(torch,'__version__','unknown'); c=getattr(getattr(torch,'version',None),'cuda',None); ok=torch.cuda.is_available(); print(f'torch={v};cuda={c};cuda_available={ok}'); raise SystemExit(0 if (c and ok) else 1)" -SuppressStdErr
        $torchProbeOutput = $torchProbeResult.output
        if ($torchProbeOutput) {
            Set-Content -Path $torchProbeFile -Value $torchProbeOutput -Encoding ASCII
            Write-PhaseEvent -Step 'Torch probe result' -Kind 'INFO' -Detail $torchProbeOutput
        }

        if ($torchProbeResult.exitCode -eq 0) {
            $torchFlavor = [string]$candidate.track
            $cudaTorchReady = $true
            $torchImportable = $true
            break
        }

        if (Test-TorchImport) {
            $torchImportable = $true
        }

        Write-Warning "Installed torch for track '$($candidate.track)' but CUDA is not usable in this runtime context."
    }

    if (-not $cudaTorchReady) {
        Write-Warning "[Torch] No CUDA torch candidate succeeded; falling back to CPU torch."
        $cpuInstalled = Install-CpuTorch
        if ($cpuInstalled) {
            $torchImportable = $true
            $cpuProbeResult = Invoke-PythonProbe -Code "import torch; v=getattr(torch,'__version__','unknown'); c=getattr(getattr(torch,'version',None),'cuda',None); ok=torch.cuda.is_available(); print(f'torch={v};cuda={c};cuda_available={ok}')" -SuppressStdErr
            $cpuProbeOutput = $cpuProbeResult.output
            if ($cpuProbeOutput) {
                Set-Content -Path $torchProbeFile -Value $cpuProbeOutput -Encoding ASCII
                Write-PhaseEvent -Step 'Torch probe result' -Kind 'INFO' -Detail "cpu-fallback $cpuProbeOutput"
            }
        }
        elseif (-not $torchImportable -and -not (Test-TorchImport)) {
            throw "Failed to install torch runtime (CUDA candidates unavailable and CPU fallback install failed)."
        }
        else {
            Write-Warning "CPU fallback install failed, but torch is importable from a prior install; continuing."
        }
    }

    Set-Content -Path $torchFlavorFile -Value $torchFlavor -Encoding ASCII
}

function Install-Gsplat {
    Ensure-Venv
    Write-PhaseEvent -Step 'Gsplat runtime setup started' -Kind 'RUN' -Detail 'local-first mode'

    $torchFlavor = "cpu"
    if (Test-Path $torchFlavorFile) {
        $torchFlavor = (Get-Content $torchFlavorFile -Raw).Trim()
    }

    if ($torchFlavor -eq "cpu") {
        Write-Warning "Skipping gsplat build because CUDA torch is not active."
        return
    }

    if ($gsplatSupportedTracks.Count -gt 0 -and ($gsplatSupportedTracks -notcontains $torchFlavor)) {
        throw "gsplat is required for CUDA runtime, but torch track '$torchFlavor' is outside supported tracks: $($gsplatSupportedTracks -join ', ')."
    }

    if ($compat.PSObject.Properties.Name -contains 'gsplatCudaSupported' -and -not [bool]$compat.gsplatCudaSupported) {
        $reason = [string]$compat.gsplatUnsupportedReason
        if (-not $reason) {
            $reason = 'current CUDA version is outside matrix-supported gsplat range.'
        }
        $supportedRangeText = $null
        $gsplatMinCuda = [string]$compat.gsplatMinCuda
        $gsplatMaxCuda = [string]$compat.gsplatMaxCuda
        if ($gsplatMinCuda -and $gsplatMaxCuda) {
            $supportedRangeText = "$gsplatMinCuda to $gsplatMaxCuda"
        }
        elseif ($gsplatMinCuda) {
            $supportedRangeText = "$gsplatMinCuda+"
        }
        Throw-GsplatCudaUpgradeRequired -Reason $reason -SupportedRangeText $supportedRangeText
    }

    if ($compat.PSObject.Properties.Name -contains 'gsplatVsSupported' -and -not [bool]$compat.gsplatVsSupported) {
        $reason = [string]$compat.gsplatUnsupportedReason
        if (-not $reason) {
            $reason = "Visual Studio major $detectedVsMajor is incompatible for selected CUDA profile (requires VS $requiredVsMajor)."
        }
        Throw-GsplatToolchainUpgradeRequired -Reason $reason
    }

    if ($compat.PSObject.Properties.Name -contains 'gsplatBuildSupported' -and -not [bool]$compat.gsplatBuildSupported) {
        $reason = [string]$compat.gsplatUnsupportedReason
        if (-not $reason) {
            $reason = 'current toolchain is outside matrix-supported gsplat range.'
        }
        Throw-GsplatToolchainUpgradeRequired -Reason $reason
    }

    $existingGsplatVersion = Get-ExistingGsplatVersion
    if ($existingGsplatVersion -and ($existingGsplatVersion -eq $gsplatVersion)) {
        Write-PhaseEvent -Step 'Gsplat source selected' -Kind 'SOURCE' -Detail "LOCAL_EXISTING version=$existingGsplatVersion"
        return
    }

    if ($existingGsplatVersion) {
        Write-PhaseEvent -Step 'Gsplat source selected' -Kind 'SOURCE' -Detail "ONLINE_OR_CACHE rebuild (local version=$existingGsplatVersion required=$gsplatVersion)"
    }
    else {
        Write-PhaseEvent -Step 'Gsplat source selected' -Kind 'SOURCE' -Detail 'ONLINE_OR_CACHE build (no local reusable gsplat)'
    }

    if ($selectedCudaPath -and (Test-Path $selectedCudaPath)) {
        Write-PhaseEvent -Step 'CUDA toolkit for gsplat build' -Kind 'INFO' -Detail $selectedCudaPath
        $env:CUDA_HOME = $selectedCudaPath
        $env:CUDA_PATH = $selectedCudaPath
        $env:DISTUTILS_USE_SDK = '1'
        $cudaBinPath = Join-Path $selectedCudaPath 'bin'
        if ((Test-Path $cudaBinPath) -and -not (($env:Path -split ';') -contains $cudaBinPath)) {
            $env:Path = "$cudaBinPath;$env:Path"
        }
    }

    Assert-GsplatCudaMsvcCompatibility

    Write-PhaseEvent -Step 'Install ninja build dependency' -Kind 'STEP'
    & $venvPy -m pip install --force-reinstall ninja
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install ninja required for gsplat build."
    }

    Write-PhaseEvent -Step 'Gsplat build/install' -Kind 'SOURCE' -Detail "ONLINE_FORCED_SOURCE_BUILD version=$gsplatVersion"
    Write-PhaseEvent -Step 'Build environment' -Kind 'INFO' -Detail 'Visual Studio x64 developer command environment'
    $gsplatInstallCommand = "`"$venvPy`" -m pip install --force-reinstall --no-deps --no-cache-dir --no-binary=gsplat `"gsplat==$gsplatVersion`" --no-build-isolation -v"
    $gsplatOk = Invoke-InVsX64Environment -CommandLine $gsplatInstallCommand
    if (-not $gsplatOk) {
        throw "Failed to build/install gsplat $gsplatVersion using VS x64 toolchain."
    }

    & $venvPy -c "import gsplat; print('gsplat_import_ok=True')"
    if ($LASTEXITCODE -ne 0) {
        throw "gsplat module validation failed after installation."
    }
}

function Install-Requirements {
    Ensure-Venv

    if (-not (Test-Path $backendRequirements)) {
        throw "Backend requirements file not found: $backendRequirements"
    }

    Write-PhaseEvent -Step 'Backend requirements install' -Kind 'SOURCE' -Detail 'ONLINE_OR_CACHE (pip determines cache/network per package)'
    & $venvPy -m pip install --upgrade-strategy only-if-needed -r $backendRequirements
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install backend requirements."
    }

    $torchInstalled = & $venvPy -c "import torch; print(getattr(torch,'__version__','unknown'))"
    $torchCudaUsable = (& $venvPy -c "import torch; print(bool(getattr(getattr(torch,'version',None),'cuda',None) and torch.cuda.is_available()))" 2>$null)
    $torchFlavor = "unknown"
    if (Test-Path $torchFlavorFile) {
        $torchFlavor = (Get-Content $torchFlavorFile -Raw).Trim()
    }

    @(
        "TORCH_VERSION=$torchVersion"
        "TORCHVISION_VERSION=$torchvisionVersion"
        "TORCHAUDIO_VERSION=$torchaudioVersion"
        "TORCH_FLAVOR=$torchFlavor"
        "TORCH_INSTALLED=$torchInstalled"
        "TORCH_CUDA_USABLE=$torchCudaUsable"
        "GSPLAT_VERSION=$gsplatVersion"
        "CUDA_SELECTED=$selectedCudaVersion"
        "CUDA_SELECTED_PATH=$selectedCudaPath"
        "CUDA_DETECTED=$($compat.detectedCudaVersion)"
        "VS_DETECTED=$($compat.detectedVsMajor)"
        "DEFAULT_STACK=$($compat.useDefaultStack)"
    ) | Set-Content -Path $bootstrapState -Encoding ASCII

    $dataDir = Join-Path $env:ProgramData "Bimba3D\data\projects"
    if (-not (Test-Path $dataDir)) {
        New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
    }
}

Write-PhaseEvent -Step 'Phase started' -Kind 'RUN' -Detail "InstallDir=$InstallDir RuntimeRoot=$runtimeRoot"

try {
    switch ($Phase) {
        "prepare" {
            Ensure-Venv
            Install-PipTooling
        }
        "torch" {
            Install-Torch
        }
        "gsplat" {
            Install-Gsplat
        }
        "requirements" {
            Install-Requirements
        }
    }

    Write-PhaseEvent -Step 'Phase completed successfully' -Kind 'RUN'
}
catch {
    Write-PhaseEvent -Step 'Phase failed' -Kind 'ERROR' -Detail $_.Exception.Message
    throw
}
