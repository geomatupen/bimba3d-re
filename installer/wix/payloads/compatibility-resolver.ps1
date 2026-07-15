$ErrorActionPreference = 'Stop'

function ConvertTo-Bimba3DVersion {
    param([string]$VersionText)
    if (-not $VersionText) {
        return $null
    }

    try {
        return [Version]$VersionText
    } catch {
        return $null
    }
}

function Test-Bimba3DVersionAtLeast {
    param(
        [string]$Actual,
        [string]$Minimum
    )

    $actualVersion = ConvertTo-Bimba3DVersion -VersionText $Actual
    $minimumVersion = ConvertTo-Bimba3DVersion -VersionText $Minimum
    if (-not $actualVersion -or -not $minimumVersion) {
        return $false
    }

    return $actualVersion -ge $minimumVersion
}

function Test-Bimba3DVersionInRange {
    param(
        [string]$Actual,
        [string]$Minimum,
        [string]$Maximum
    )

    if (-not (Test-Bimba3DVersionAtLeast -Actual $Actual -Minimum $Minimum)) {
        return $false
    }

    if (-not $Maximum) {
        return $true
    }

    $actualVersion = ConvertTo-Bimba3DVersion -VersionText $Actual
    $maximumVersion = ConvertTo-Bimba3DVersion -VersionText $Maximum
    if (-not $actualVersion -or -not $maximumVersion) {
        return $false
    }

    return $actualVersion -le $maximumVersion
}

function Get-Bimba3DRegistryValue {
    param(
        [Microsoft.Win32.RegistryHive]$Hive,
        [string]$SubKey,
        [string]$ValueName
    )

    foreach ($view in @([Microsoft.Win32.RegistryView]::Registry64, [Microsoft.Win32.RegistryView]::Registry32)) {
        try {
            $base = [Microsoft.Win32.RegistryKey]::OpenBaseKey($Hive, $view)
            $key = $base.OpenSubKey($SubKey)
            if ($key) {
                $value = $key.GetValue($ValueName, $null)
                if ($null -ne $value -and [string]$value -ne '') {
                    return [string]$value
                }
            }
        } catch {
        }
    }

    return $null
}

function Get-Bimba3DVsWherePath {
    $cmd = Get-Command vswhere.exe -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) {
        return $cmd.Source
    }

    $candidates = @()
    if ($env:ProgramFiles) {
        $candidates += (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\Installer\vswhere.exe')
    }
    if (${env:ProgramFiles(x86)}) {
        $candidates += (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe')
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Get-Bimba3DInstalledVsMajor {
    function Convert-VsCatalogValueToMajor {
        param([int]$Value)

        $yearToMajor = @{
            2017 = 15
            2019 = 16
            2022 = 17
        }

        if ($yearToMajor.ContainsKey($Value)) {
            return [int]$yearToMajor[$Value]
        }

        if ($Value -gt 0 -and $Value -lt 100) {
            return $Value
        }

        return $null
    }

    $vswherePath = Get-Bimba3DVsWherePath
    if ($vswherePath) {
        try {
            $versionText = & $vswherePath -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property catalog_productLineVersion -latest 2>$null
            if ($LASTEXITCODE -eq 0 -and $versionText -and $versionText.Trim()) {
                $major = Convert-VsCatalogValueToMajor -Value ([int]$versionText.Trim())
                if ($major) {
                    return $major
                }
            }
        } catch {
        }

        try {
            $installationVersion = & $vswherePath -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion -latest 2>$null
            if ($LASTEXITCODE -eq 0 -and $installationVersion -and $installationVersion.Trim()) {
                $parsed = ConvertTo-Bimba3DVersion -VersionText $installationVersion.Trim()
                if ($parsed) {
                    return $parsed.Major
                }
            }
        } catch {
        }
    }

    return $null
}

function Test-Bimba3DHasVsX64Tools {
    $vswherePath = Get-Bimba3DVsWherePath
    if (-not $vswherePath) {
        return $false
    }

    try {
        $installationPath = & $vswherePath -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath -latest 2>$null
        if ($LASTEXITCODE -eq 0 -and $installationPath -and $installationPath.Trim()) {
            return $true
        }
    } catch {
    }

    return $false
}

function Get-Bimba3DInstalledCudaVersion {
    $versions = Get-Bimba3DInstalledCudaVersions
    if ($versions.Count -gt 0) {
        return $versions[0]
    }

    return $null
}

function Get-Bimba3DInstalledCudaVersions {
    $candidates = New-Object System.Collections.Generic.HashSet[string]

    if ($env:CUDA_PATH) {
        $match = [regex]::Match($env:CUDA_PATH, 'CUDA\\v(?<v>\d+\.\d+)')
        if ($match.Success) {
            [void]$candidates.Add($match.Groups['v'].Value)
        }
    }

    foreach ($cudaEnv in (Get-ChildItem Env:CUDA_PATH_V* -ErrorAction SilentlyContinue)) {
        if ($cudaEnv.Value) {
            $match = [regex]::Match($cudaEnv.Value, 'CUDA\\v(?<v>\d+\.\d+)')
            if ($match.Success) {
                [void]$candidates.Add($match.Groups['v'].Value)
            }
        }
    }

    $nvcc = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($nvcc -and $nvcc.Source) {
        try {
            $nvccOut = & $nvcc.Source --version 2>$null | Out-String
            $match = [regex]::Match($nvccOut, 'release\s+(?<v>\d+\.\d+)')
            if ($match.Success) {
                [void]$candidates.Add($match.Groups['v'].Value)
            }
        } catch {
        }

        $pathMatch = [regex]::Match($nvcc.Source, 'CUDA\\v(?<v>\d+\.\d+)')
        if ($pathMatch.Success) {
            [void]$candidates.Add($pathMatch.Groups['v'].Value)
        }
    }

    $cudaRoot = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path $cudaRoot) {
        foreach ($dir in (Get-ChildItem -Path $cudaRoot -Directory -ErrorAction SilentlyContinue)) {
            $match = [regex]::Match($dir.Name, '^v(?<v>\d+\.\d+)$')
            if ($match.Success) {
                [void]$candidates.Add($match.Groups['v'].Value)
            }
        }
    }

    $registryCudaRoot = 'SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA'
    foreach ($view in @([Microsoft.Win32.RegistryView]::Registry64, [Microsoft.Win32.RegistryView]::Registry32)) {
        try {
            $base = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine, $view)
            $root = $base.OpenSubKey($registryCudaRoot)
            if ($root) {
                foreach ($subName in $root.GetSubKeyNames()) {
                    $match = [regex]::Match($subName, '^v(?<v>\d+\.\d+)$')
                    if ($match.Success) {
                        [void]$candidates.Add($match.Groups['v'].Value)
                    }
                }
            }
        } catch {
        }
    }

    $parsedVersions = @()
    foreach ($candidate in $candidates) {
        $parsed = ConvertTo-Bimba3DVersion -VersionText $candidate
        if ($parsed) {
            $parsedVersions += [pscustomobject]@{
                text = "$($parsed.Major).$($parsed.Minor)"
                version = $parsed
            }
        }
    }

    if ($parsedVersions.Count -eq 0) {
        return @()
    }

    $ordered = $parsedVersions |
        Sort-Object -Property @{ Expression = { $_.version }; Descending = $true } |
        ForEach-Object { $_.text } |
        Select-Object -Unique

    return @($ordered)
}

function Get-Bimba3DCudaInstallPathByVersion {
    param([string]$VersionText)

    if (-not $VersionText) {
        return $null
    }

    $versionKey = $VersionText -replace '\\.', '_'
    $envVarName = "CUDA_PATH_V$versionKey"
    $envVarPath = [Environment]::GetEnvironmentVariable($envVarName, 'Process')
    if (-not $envVarPath) {
        $envVarPath = [Environment]::GetEnvironmentVariable($envVarName, 'Machine')
    }
    if (-not $envVarPath) {
        $envVarPath = [Environment]::GetEnvironmentVariable($envVarName, 'User')
    }
    if ($envVarPath -and (Test-Path (Join-Path $envVarPath 'bin\\nvcc.exe'))) {
        return $envVarPath
    }

    if ($env:CUDA_PATH) {
        $match = [regex]::Match($env:CUDA_PATH, 'CUDA\\v(?<v>\d+\.\d+)')
        if ($match.Success -and ($match.Groups['v'].Value -eq $VersionText) -and (Test-Path (Join-Path $env:CUDA_PATH 'bin\\nvcc.exe'))) {
            return $env:CUDA_PATH
        }
    }

    $registrySubKey = "SOFTWARE\\NVIDIA Corporation\\GPU Computing Toolkit\\CUDA\\v$VersionText"
    $registryPath = Get-Bimba3DRegistryValue -Hive LocalMachine -SubKey $registrySubKey -ValueName 'InstallDir'
    if ($registryPath -and (Test-Path (Join-Path $registryPath 'bin\\nvcc.exe'))) {
        return $registryPath
    }

    $defaultPath = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v$VersionText"
    if (Test-Path (Join-Path $defaultPath 'bin\\nvcc.exe')) {
        return $defaultPath
    }

    return $null
}

function Get-Bimba3DCudaInstallerUrl {
    param([string]$VersionText)

    if (-not $VersionText) {
        return $null
    }

    $parsed = ConvertTo-Bimba3DVersion -VersionText $VersionText
    if (-not $parsed) {
        return $null
    }

    $normalized = "$($parsed.Major).$($parsed.Minor).0"
    return "https://developer.download.nvidia.com/compute/cuda/$normalized/network_installers/cuda_$normalized`_windows_network.exe"
}

function Get-Bimba3DCompatibilityMatrix {
    param([string]$MatrixPath)

    if ([string]::IsNullOrWhiteSpace([string]$MatrixPath)) {
        $MatrixPath = $null
    }

    if (-not $MatrixPath) {
        $variantCandidates = @()
        try {
            $variantCandidates = Get-ChildItem -Path $PSScriptRoot -Filter 'compatibility-matrix*.json' -File -ErrorAction SilentlyContinue |
                Sort-Object -Property Name |
                ForEach-Object { $_.FullName }
        } catch {
        }

        $candidates = @(
            (Join-Path $PSScriptRoot 'compatibility-matrix.json')
        ) + @($variantCandidates) + @(
            (Join-Path $PSScriptRoot '..\..\..\compatibility-matrix.json')
        )

        $candidates = $candidates | Select-Object -Unique

        foreach ($candidate in $candidates) {
            if (Test-Path $candidate) {
                $MatrixPath = $candidate
                break
            }
        }
    }

    if ([string]::IsNullOrWhiteSpace([string]$MatrixPath) -or -not (Test-Path -LiteralPath $MatrixPath)) {
        throw "Compatibility matrix not found: $MatrixPath"
    }

    return Get-Content -Path $MatrixPath -Raw | ConvertFrom-Json
}

function Resolve-Bimba3DCompatibility {
    param(
        [string]$MatrixPath,
        [string]$CudaVersion,
        [Nullable[int]]$VsMajor
    )

    $matrix = Get-Bimba3DCompatibilityMatrix -MatrixPath $MatrixPath

    $installedCudaVersions = @()
    if (-not $CudaVersion) {
        $installedCudaVersions = @(Get-Bimba3DInstalledCudaVersions)
    }

    $detectedCudaVersion = $CudaVersion
    if (-not $detectedCudaVersion -and $installedCudaVersions.Count -gt 0) {
        $detectedCudaVersion = $installedCudaVersions[0]
    }

    $selectedCudaVersion = $CudaVersion
    if (-not $selectedCudaVersion -and $installedCudaVersions.Count -gt 0) {
        $cudaCandidatesWithTorchAndGsplat = @()
        $cudaCandidatesWithTorch = @()

        foreach ($candidateCuda in $installedCudaVersions) {
            $hasTorchProfile = $false
            foreach ($profile in $matrix.torchProfiles) {
                if (Test-Bimba3DVersionInRange -Actual $candidateCuda -Minimum $profile.minCuda -Maximum $profile.maxCuda) {
                    $hasTorchProfile = $true
                    break
                }
            }

            if (-not $hasTorchProfile) {
                continue
            }

            $cudaCandidatesWithTorch += $candidateCuda

            $gsplatMin = $null
            if ($matrix.gsplat -and $matrix.gsplat.minCuda) {
                $gsplatMin = [string]$matrix.gsplat.minCuda
            }
            $gsplatMax = $null
            if ($matrix.gsplat -and $matrix.gsplat.maxCuda) {
                $gsplatMax = [string]$matrix.gsplat.maxCuda
            }

            $supportsGsplat = $true
            if ($gsplatMin -and -not (Test-Bimba3DVersionAtLeast -Actual $candidateCuda -Minimum $gsplatMin)) {
                $supportsGsplat = $false
            }
            if ($supportsGsplat -and $gsplatMax -and -not (Test-Bimba3DVersionInRange -Actual $candidateCuda -Minimum $gsplatMin -Maximum $gsplatMax)) {
                $supportsGsplat = $false
            }

            if ($supportsGsplat) {
                $cudaCandidatesWithTorchAndGsplat += $candidateCuda
            }
        }

        if ($cudaCandidatesWithTorchAndGsplat.Count -gt 0) {
            $selectedCudaVersion = $cudaCandidatesWithTorchAndGsplat[0]
        } elseif ($cudaCandidatesWithTorch.Count -gt 0) {
            $selectedCudaVersion = $cudaCandidatesWithTorch[0]
        } else {
            $selectedCudaVersion = $installedCudaVersions[0]
        }
    }

    $selectedCudaPath = Get-Bimba3DCudaInstallPathByVersion -VersionText $selectedCudaVersion

    if ($null -eq $VsMajor) {
        $VsMajor = Get-Bimba3DInstalledVsMajor
    }
    $vsHasX64VcTools = Test-Bimba3DHasVsX64Tools

    $defaults = $matrix.defaults
    $cudaMeetsMinimum = $false
    if ($selectedCudaVersion) {
        $cudaMeetsMinimum = Test-Bimba3DVersionAtLeast -Actual $selectedCudaVersion -Minimum $defaults.cudaMin
    }

    $cudaNeedsUpgrade = -not $cudaMeetsMinimum
    $vsMeetsDefault = $false

    $gsplatMinCuda = $null
    if ($matrix.gsplat -and $matrix.gsplat.minCuda) {
        $gsplatMinCuda = [string]$matrix.gsplat.minCuda
    }

    $gsplatMaxCuda = $null
    if ($matrix.gsplat -and $matrix.gsplat.maxCuda) {
        $gsplatMaxCuda = [string]$matrix.gsplat.maxCuda
    }

    $gsplatSupportedVsMajors = @()
    if ($matrix.gsplat -and $matrix.gsplat.supportedVsMajors) {
        foreach ($vsMajorCandidate in @($matrix.gsplat.supportedVsMajors)) {
            if ($null -ne $vsMajorCandidate -and [string]$vsMajorCandidate -ne '') {
                $gsplatSupportedVsMajors += [int]$vsMajorCandidate
            }
        }
    }

    $recommendedCudaVersion = [string]$defaults.cudaPreferred
    if ($selectedCudaVersion -and $cudaMeetsMinimum) {
        $recommendedCudaVersion = [string]$selectedCudaVersion
    }

    if ($selectedCudaVersion -and $gsplatMinCuda -and $gsplatMaxCuda -and -not (Test-Bimba3DVersionInRange -Actual $selectedCudaVersion -Minimum $gsplatMinCuda -Maximum $gsplatMaxCuda)) {
        $recommendedCudaVersion = [string]$defaults.cudaPreferred
    }

    $recommendedCudaInstallerUrl = Get-Bimba3DCudaInstallerUrl -VersionText $recommendedCudaVersion
    if (-not $recommendedCudaInstallerUrl) {
        $recommendedCudaInstallerUrl = [string]$defaults.cudaRecommendedInstallerUrl
    }

    $resolved = [ordered]@{
        matrix = $matrix
        detectedCudaVersion = $detectedCudaVersion
        selectedCudaVersion = $selectedCudaVersion
        selectedCudaPath = $selectedCudaPath
        installedCudaVersions = ($installedCudaVersions -join ',')
        detectedVsMajor = $VsMajor
        vsHasX64VcTools = [bool]$vsHasX64VcTools
        cudaMin = [string]$defaults.cudaMin
        cudaPreferred = [string]$defaults.cudaPreferred
        recommendedCudaVersion = $recommendedCudaVersion
        recommendedCudaInstallerUrl = $recommendedCudaInstallerUrl
        defaultVsMajor = [int]$defaults.vsMajor
        requiredVsMajor = [int]$defaults.vsMajor
        recommendedVsInstallerUrl = [string]$defaults.vsRecommendedInstallerUrl
        selectedVsProfile = 'default'
        cudaIsInstalled = [bool]$selectedCudaVersion
        cudaMeetsMinimum = [bool]$cudaMeetsMinimum
        cudaNeedsUpgrade = [bool]$cudaNeedsUpgrade
        vsMeetsDefault = [bool]$vsMeetsDefault
        useDefaultStack = $false
        torchTrack = [string]$defaults.torchTrack
        torchIndexUrl = [string]$defaults.torchIndexUrl
        torchVersion = [string]$defaults.torchVersion
        torchvisionVersion = [string]$defaults.torchvisionVersion
        torchaudioVersion = [string]$defaults.torchaudioVersion
        torchCpuVersion = [string]$defaults.torchCpuVersion
        torchvisionCpuVersion = [string]$defaults.torchvisionCpuVersion
        torchaudioCpuVersion = [string]$defaults.torchaudioCpuVersion
        gsplatVersion = [string]$matrix.gsplat.version
        gsplatSupportedTorchTracks = @($matrix.gsplat.supportedTorchTracks)
        gsplatSupportedVsMajors = @($gsplatSupportedVsMajors)
        gsplatMinCuda = $gsplatMinCuda
        gsplatMaxCuda = $gsplatMaxCuda
        gsplatCudaSupported = $true
        gsplatVsSupported = $true
        gsplatBuildSupported = $true
        gsplatUnsupportedReason = ''
        colmapCudaVersion = [string]$matrix.colmap.cuda.version
        colmapCudaUrl = [string]$matrix.colmap.cuda.url
        colmapNoCudaVersion = [string]$matrix.colmap.nocuda.version
        colmapNoCudaUrl = [string]$matrix.colmap.nocuda.url
        colmapAssetProfile = 'default'
        colmapPreferredVariant = 'nocuda'
    }

    if ($resolved.cudaMeetsMinimum) {
        $resolved.colmapPreferredVariant = 'cuda'
    }

    if ($resolved.cudaMeetsMinimum -and $matrix.colmapProfiles) {
        foreach ($colmapProfile in $matrix.colmapProfiles) {
            if (Test-Bimba3DVersionInRange -Actual $selectedCudaVersion -Minimum $colmapProfile.minCuda -Maximum $colmapProfile.maxCuda) {
                if ($colmapProfile.name) {
                    $resolved.colmapAssetProfile = [string]$colmapProfile.name
                }

                if ($colmapProfile.cuda) {
                    if ($colmapProfile.cuda.version) {
                        $resolved.colmapCudaVersion = [string]$colmapProfile.cuda.version
                    }
                    if ($colmapProfile.cuda.url) {
                        $resolved.colmapCudaUrl = [string]$colmapProfile.cuda.url
                    }
                }

                if ($colmapProfile.nocuda) {
                    if ($colmapProfile.nocuda.version) {
                        $resolved.colmapNoCudaVersion = [string]$colmapProfile.nocuda.version
                    }
                    if ($colmapProfile.nocuda.url) {
                        $resolved.colmapNoCudaUrl = [string]$colmapProfile.nocuda.url
                    }
                }

                break
            }
        }
    }

    if ($resolved.cudaMeetsMinimum -and $matrix.vsProfiles) {
        foreach ($vsProfile in $matrix.vsProfiles) {
            if (Test-Bimba3DVersionInRange -Actual $selectedCudaVersion -Minimum $vsProfile.minCuda -Maximum $vsProfile.maxCuda) {
                if ($vsProfile.vsMajor) {
                    $resolved.requiredVsMajor = [int]$vsProfile.vsMajor
                }
                if ($vsProfile.vsRecommendedInstallerUrl) {
                    $resolved.recommendedVsInstallerUrl = [string]$vsProfile.vsRecommendedInstallerUrl
                }
                if ($vsProfile.name) {
                    $resolved.selectedVsProfile = [string]$vsProfile.name
                }
                break
            }
        }
    }

    if ($null -ne $VsMajor) {
        $resolved.vsMeetsDefault = ([int]$VsMajor -eq [int]$resolved.requiredVsMajor)
    } else {
        $resolved.vsMeetsDefault = $false
    }

    if ($resolved.cudaMeetsMinimum -and $VsMajor -eq [int]$defaults.vsMajor -and $selectedCudaVersion -eq [string]$defaults.cudaPreferred) {
        $resolved.useDefaultStack = $true
    }

    if ($resolved.cudaMeetsMinimum) {
        foreach ($profile in $matrix.torchProfiles) {
            if (Test-Bimba3DVersionInRange -Actual $selectedCudaVersion -Minimum $profile.minCuda -Maximum $profile.maxCuda) {
                $resolved.torchTrack = [string]$profile.track
                $resolved.torchIndexUrl = [string]$profile.indexUrl
                $resolved.torchVersion = [string]$profile.torchVersion
                $resolved.torchvisionVersion = [string]$profile.torchvisionVersion
                $resolved.torchaudioVersion = [string]$profile.torchaudioVersion
                break
            }
        }
    }

    if ($resolved.cudaIsInstalled -and $resolved.selectedCudaVersion) {
        if ($resolved.gsplatMinCuda -and -not (Test-Bimba3DVersionAtLeast -Actual $resolved.selectedCudaVersion -Minimum $resolved.gsplatMinCuda)) {
            $resolved.gsplatCudaSupported = $false
            $resolved.gsplatUnsupportedReason = "CUDA $($resolved.selectedCudaVersion) is below gsplat minimum $($resolved.gsplatMinCuda)"
        }

        if ($resolved.gsplatCudaSupported -and $resolved.gsplatMaxCuda -and -not (Test-Bimba3DVersionInRange -Actual $resolved.selectedCudaVersion -Minimum $resolved.gsplatMinCuda -Maximum $resolved.gsplatMaxCuda)) {
            $resolved.gsplatCudaSupported = $false
            $resolved.gsplatUnsupportedReason = "CUDA $($resolved.selectedCudaVersion) is above gsplat maximum $($resolved.gsplatMaxCuda)"
        }
    }

    if (-not $resolved.vsHasX64VcTools) {
        $resolved.gsplatVsSupported = $false
        $resolved.gsplatUnsupportedReason = 'Visual Studio C++ x64 tools component is missing.'
    } elseif ($resolved.gsplatSupportedVsMajors.Count -gt 0) {
        if (-not $resolved.detectedVsMajor) {
            $resolved.gsplatVsSupported = $false
            $resolved.gsplatUnsupportedReason = 'Visual Studio major version could not be detected for gsplat build.'
        } elseif ($resolved.gsplatSupportedVsMajors -notcontains [int]$resolved.detectedVsMajor) {
            $resolved.gsplatVsSupported = $false
            $resolved.gsplatUnsupportedReason = "Visual Studio major $($resolved.detectedVsMajor) is outside gsplat-supported majors: $($resolved.gsplatSupportedVsMajors -join ', ')."
        }
    }

    $resolved.gsplatBuildSupported = ([bool]$resolved.gsplatCudaSupported -and [bool]$resolved.gsplatVsSupported)

    [pscustomobject]$resolved
}
