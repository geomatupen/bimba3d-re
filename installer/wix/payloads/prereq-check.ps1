param(
    [switch]$ProbeOnly
)

$ErrorActionPreference = 'Stop'

$logRoot = 'C:\ProgramData\Bimba3D\Logs'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$logPath = Join-Path $logRoot 'prereq-check.log'

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
$matrixPath = Join-Path $PSScriptRoot 'compatibility-matrix-prereq.json'
$compat = Resolve-Bimba3DCompatibility -MatrixPath $matrixPath

function Write-Log([string]$message) {
    Add-Content -Path $logPath -Value "$(Get-Date -Format o) $message"
}

$vsRecommended = [string]$compat.recommendedVsInstallerUrl
$vsAllVersions = 'https://visualstudio.microsoft.com/vs/older-downloads/'
$cudaRecommended = [string]$compat.recommendedCudaInstallerUrl
$cudaAllVersions = 'https://developer.nvidia.com/cuda-toolkit-archive'
$cudaMinimum = [string]$compat.cudaMin
$cudaPreferred = [string]$compat.cudaPreferred
if (-not $cudaPreferred) {
    $cudaPreferred = '12.5'
}

$cudaPreferredInstaller = $null
if ($compat.matrix -and $compat.matrix.defaults -and $compat.matrix.defaults.cudaRecommendedInstallerUrl) {
    $cudaPreferredInstaller = [string]$compat.matrix.defaults.cudaRecommendedInstallerUrl
}
if (-not $cudaPreferredInstaller) {
    $cudaPreferredInstaller = 'https://developer.download.nvidia.com/compute/cuda/12.5.0/network_installers/cuda_12.5.0_windows_network.exe'
}

function Test-SupportedCudaVersion {
    param([string]$VersionText)

    if (-not $VersionText) {
        return $false
    }

    return Test-Bimba3DVersionAtLeast -Actual $VersionText -Minimum $cudaMinimum
}

function Get-CudaVersionFromPath {
    param([string]$PathText)

    if (-not $PathText) {
        return $null
    }

    $match = [regex]::Match($PathText, 'CUDA\\v(?<v>\d+\.\d+)')
    if ($match.Success) {
        return $match.Groups['v'].Value
    }

    return $null
}

function Get-RegistryValue {
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

function ConvertTo-SemVersion {
    param([string]$VersionText)

    if (-not $VersionText) {
        return $null
    }

    $match = [regex]::Match($VersionText, '(\d+)\.(\d+)(?:\.(\d+))?')
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
        $parsed = ConvertTo-SemVersion -VersionText $_.Name
        if ($parsed) {
            $versions += [pscustomobject]@{ raw = $_.Name; parsed = $parsed }
        }
    }

    if (-not $versions -or $versions.Count -eq 0) {
        return $null
    }

    return ($versions | Sort-Object -Property parsed -Descending | Select-Object -First 1)
}

function Get-ToolchainCompatibilityStatus {
    param(
        [Parameter(Mandatory = $true)]
        $Compatibility
    )

    $status = [ordered]@{
        compatible = $true
        reason = ''
        requireCudaUpgrade = $false
        requireVsUpgrade = $false
    }

    if ($Compatibility.PSObject.Properties.Name -contains 'gsplatBuildSupported' -and -not [bool]$Compatibility.gsplatBuildSupported) {
        $status.compatible = $false
        $status.reason = [string]$Compatibility.gsplatUnsupportedReason
        if (-not $status.reason) {
            $status.reason = 'gsplat toolchain support checks failed.'
        }

        if ($status.reason -match 'Visual Studio') {
            $status.requireVsUpgrade = $true
        }
        if ($status.reason -match 'CUDA') {
            $status.requireCudaUpgrade = $true
        }

        if (-not $status.requireCudaUpgrade -and -not $status.requireVsUpgrade) {
            $status.requireCudaUpgrade = $true
            $status.requireVsUpgrade = $true
        }

        return [pscustomobject]$status
    }

    $cudaVersion = ConvertTo-SemVersion -VersionText ([string]$Compatibility.selectedCudaVersion)
    if (-not $cudaVersion) {
        return [pscustomobject]$status
    }

    $msvcInfo = Get-InstalledMsvcToolsetVersion
    if (-not $msvcInfo) {
        return [pscustomobject]$status
    }

    $minimumCudaForNewMsvc = [version]::new(12, 4, 0)
    $newMsvcThreshold = [version]::new(14, 44, 0)

    if ($cudaVersion -lt $minimumCudaForNewMsvc -and $msvcInfo.parsed -ge $newMsvcThreshold) {
        $status.compatible = $false
        $status.reason = "Detected CUDA $($Compatibility.selectedCudaVersion) with MSVC toolset $($msvcInfo.raw). This combo fails gsplat build (STL1002) and requires CUDA 12.4+; install CUDA $cudaPreferred."
        $status.requireCudaUpgrade = $true
        $status.requireVsUpgrade = $false
        return [pscustomobject]$status
    }

    return [pscustomobject]$status
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

    $candidates = $candidates | Select-Object -Unique
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Test-BuildToolsInstalled {
    $buildToolsPath = Get-RegistryValue -Hive LocalMachine -SubKey 'SOFTWARE\Microsoft\VisualStudio\Setup\Instances\BuildTools_17' -ValueName 'InstallationPath'
    if ($buildToolsPath) {
        Write-Log "VS detected via BuildTools registry path: $buildToolsPath"
        return $true
    }

    $vs2022Sxs = Get-RegistryValue -Hive LocalMachine -SubKey 'SOFTWARE\Microsoft\VisualStudio\SxS\VS7' -ValueName '17.0'
    if ($vs2022Sxs) {
        Write-Log "VS detected via VS7 SxS registry path: $vs2022Sxs"
        return $true
    }

    $vswherePath = Get-VsWherePath
    if ($vswherePath) {
        try {
            $vcToolsPath = & $vswherePath -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath -latest 2>$null
            if ($LASTEXITCODE -eq 0 -and $vcToolsPath -and $vcToolsPath.Trim()) {
                Write-Log "VS detected via vswhere VC tools component: $($vcToolsPath.Trim())"
                return $true
            }
        } catch {
            Write-Log ("vswhere check failed: " + $_.Exception.Message)
        }
    }

    Write-Log 'VS not detected by registry or vswhere VC tools component check.'

    return $false
}

function Test-CudaInstalled {
    param(
        [Parameter(Mandatory = $true)]
        $Compatibility
    )

    $cudaVersion = [string]$Compatibility.selectedCudaVersion
    if (-not $cudaVersion) {
        Write-Log 'CUDA not detected by resolver checks (selectedCudaVersion empty).'
        return $false
    }

    if ([bool]$Compatibility.cudaMeetsMinimum) {
        Write-Log "CUDA detected: version $cudaVersion (minimum required: $cudaMinimum)"
        return $true
    }

    Write-Log "CUDA detected but too old: version $cudaVersion (minimum required: $cudaMinimum)"
    return $false
}

function Open-DependencyLinks {
    param(
        [bool]$openVs,
        [bool]$openCuda
    )

    if ($openVs) {
        Start-Process $vsRecommended
        Start-Process $vsAllVersions
    }

    if ($openCuda) {
        Start-Process $cudaPreferredInstaller
        Start-Process $cudaAllVersions
    }
}

function Show-PrereqDialog {
    param(
        [bool]$vsInstalled,
        [bool]$cudaInstalled,
        [Nullable[int]]$detectedVsMajor,
        [int]$requiredVsMajor,
        [string]$recommendedCudaVersion,
        [string]$compatibleCudaVersion,
        [string]$toolchainIssue,
        [bool]$needVsAction,
        [bool]$needCudaAction
    )

    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing

    $form = New-Object System.Windows.Forms.Form
    $form.Text = 'Bimba3D Setup - Prerequisites'
    $form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
    $form.Size = New-Object System.Drawing.Size(760, 380)
    $form.MinimumSize = $form.Size
    $form.MaximizeBox = $false
    $form.MinimizeBox = $false
    $form.TopMost = $true
    $form.AutoScroll = $true

    $title = New-Object System.Windows.Forms.Label
    $title.Text = "Bimba3D requires Visual Studio Build Tools and NVIDIA CUDA Toolkit before continuing. Recommended: VS Build Tools $requiredVsMajor, CUDA $recommendedCudaVersion."
    $title.AutoSize = $true
    $title.Location = New-Object System.Drawing.Point(16, 16)
    $form.Controls.Add($title)

    $status = New-Object System.Windows.Forms.Label
    $status.AutoSize = $true
    $status.Location = New-Object System.Drawing.Point(16, 46)
    $vsDetectedText = if ($detectedVsMajor) { "$detectedVsMajor" } else { 'Not detected' }
    $vsState = if (-not $vsInstalled) { 'Missing, unsupported major, or x64 tools missing' } elseif ($needVsAction) { 'Installed but incompatible with required profile' } else { 'Installed' }
    $cudaState = if (-not $cudaInstalled) { 'Missing or too old' } elseif ($needCudaAction) { 'Installed but incompatible with required profile' } else { 'Installed' }
    $status.Text = "Visual Studio Build Tools (required major = $requiredVsMajor with C++ x64 tools): $vsState`n" +
                   "Detected VS major: $vsDetectedText`n" +
                   "NVIDIA CUDA Toolkit (minimum $cudaMinimum): $cudaState"
    $form.Controls.Add($status)

    if ($toolchainIssue) {
        $issueLabel = New-Object System.Windows.Forms.Label
        $issueLabel.AutoSize = $true
        $issueLabel.Location = New-Object System.Drawing.Point(16, 82)
        $issueLabel.ForeColor = [System.Drawing.Color]::FromArgb(180, 30, 30)
        $issueLabel.Text = "Compatibility issue: $toolchainIssue"
        $form.Controls.Add($issueLabel)
    }

    if ($vsInstalled -and $cudaInstalled) {
        $okLabel = New-Object System.Windows.Forms.Label
        $okLabel.AutoSize = $true
        $okLabel.Location = New-Object System.Drawing.Point(16, 62)
        $okLabel.ForeColor = [System.Drawing.Color]::FromArgb(0, 120, 0)
        $okLabel.Text = "Using existing compatible toolchain (VS $detectedVsMajor, CUDA $compatibleCudaVersion). No download needed."
        $form.Controls.Add($okLabel)
    }

    $vsHint = New-Object System.Windows.Forms.Label
    $vsHint.AutoSize = $true
    $vsHint.Location = New-Object System.Drawing.Point(16, 98)
    $vsHint.Text = 'When installing Build Tools, select: Desktop development with C++ and MSVC v143 x64/x86 build tools.'
    $form.Controls.Add($vsHint)

    $y = 138
    if ($needVsAction) {
        $vsLabel = New-Object System.Windows.Forms.Label
        $vsLabel.AutoSize = $true
        $vsLabel.Location = New-Object System.Drawing.Point(16, $y)
        $vsLabel.Text = 'Visual Studio Build Tools links:'
        $form.Controls.Add($vsLabel)

        $vsRec = New-Object System.Windows.Forms.LinkLabel
        $vsRec.AutoSize = $true
        $vsRec.Location = New-Object System.Drawing.Point(36, ($y + 22))
        $vsRec.Text = "Recommended for this CUDA profile: VS Build Tools $requiredVsMajor (direct download)"
        $vsRec.Tag = $vsRecommended
        $vsRec.add_LinkClicked({ Start-Process $this.Tag })
        $form.Controls.Add($vsRec)

        $vsRecUrl = New-Object System.Windows.Forms.LinkLabel
        $vsRecUrl.AutoSize = $true
        $vsRecUrl.Location = New-Object System.Drawing.Point(36, ($y + 44))
        $vsRecUrl.Text = $vsRecommended
        $vsRecUrl.Tag = $vsRecommended
        $vsRecUrl.add_LinkClicked({ Start-Process $this.Tag })
        $form.Controls.Add($vsRecUrl)

        $vsAny = New-Object System.Windows.Forms.LinkLabel
        $vsAny.AutoSize = $true
        $vsAny.Location = New-Object System.Drawing.Point(36, ($y + 66))
        $vsAny.Text = 'Any versions page'
        $vsAny.Tag = $vsAllVersions
        $vsAny.add_LinkClicked({ Start-Process $this.Tag })
        $form.Controls.Add($vsAny)

        $y += 100
    }

    if ($needCudaAction) {
        $cudaLabel = New-Object System.Windows.Forms.Label
        $cudaLabel.AutoSize = $true
        $cudaLabel.Location = New-Object System.Drawing.Point(16, $y)
        $cudaLabel.Text = 'NVIDIA CUDA Toolkit links:'
        $form.Controls.Add($cudaLabel)

        $cudaRec = New-Object System.Windows.Forms.LinkLabel
        $cudaRec.AutoSize = $true
        $cudaRec.Location = New-Object System.Drawing.Point(36, ($y + 22))
        $cudaRec.Text = "Recommended (validated): CUDA $recommendedCudaVersion network installer (.exe)"
        $cudaRec.Tag = $cudaPreferredInstaller
        $cudaRec.add_LinkClicked({ Start-Process $this.Tag })
        $form.Controls.Add($cudaRec)

        $cudaRecUrl = New-Object System.Windows.Forms.LinkLabel
        $cudaRecUrl.AutoSize = $true
        $cudaRecUrl.Location = New-Object System.Drawing.Point(36, ($y + 44))
        $cudaRecUrl.Text = $cudaPreferredInstaller
        $cudaRecUrl.Tag = $cudaPreferredInstaller
        $cudaRecUrl.add_LinkClicked({ Start-Process $this.Tag })
        $form.Controls.Add($cudaRecUrl)

        $cudaAny = New-Object System.Windows.Forms.LinkLabel
        $cudaAny.AutoSize = $true
        $cudaAny.Location = New-Object System.Drawing.Point(36, ($y + 66))
        $cudaAny.Text = 'Any versions page'
        $cudaAny.Tag = $cudaAllVersions
        $cudaAny.add_LinkClicked({ Start-Process $this.Tag })
        $form.Controls.Add($cudaAny)

        $y += 100
    }

    $hint = New-Object System.Windows.Forms.Label
    $hint.AutoSize = $true
    $hint.Location = New-Object System.Drawing.Point(16, ($y + 8))
    $hint.Text = 'Install prerequisites, then click Retry Detection. Click Cancel to exit setup.'
    $form.Controls.Add($hint)

    $buttonsY = [Math]::Max(($y + 36), 325)

    $openBtn = New-Object System.Windows.Forms.Button
    $openBtn.Text = 'Open Missing Download Pages'
    $openBtn.Size = New-Object System.Drawing.Size(220, 30)
    $openBtn.Location = New-Object System.Drawing.Point(16, $buttonsY)
    $openBtn.Enabled = $needVsAction -or $needCudaAction
    $openBtn.Add_Click({ Open-DependencyLinks -openVs $needVsAction -openCuda $needCudaAction })
    $form.Controls.Add($openBtn)

    $retryBtn = New-Object System.Windows.Forms.Button
    $retryBtn.Text = 'Retry Detection'
    $retryBtn.Size = New-Object System.Drawing.Size(130, 30)
    $retryBtn.Location = New-Object System.Drawing.Point(470, $buttonsY)
    $retryBtn.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $form.Controls.Add($retryBtn)

    $cancelBtn = New-Object System.Windows.Forms.Button
    $cancelBtn.Text = 'Cancel Setup'
    $cancelBtn.Size = New-Object System.Drawing.Size(110, 30)
    $cancelBtn.Location = New-Object System.Drawing.Point(614, $buttonsY)
    $cancelBtn.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
    $form.Controls.Add($cancelBtn)

    $form.AcceptButton = $retryBtn
    $form.CancelButton = $cancelBtn

    $result = $form.ShowDialog()
    $form.Dispose()
    return $result
}

while ($true) {
    $compat = Resolve-Bimba3DCompatibility -MatrixPath $matrixPath
    $requiredVsMajor = [int]$compat.requiredVsMajor
    $vsRecommended = [string]$compat.recommendedVsInstallerUrl
    $recommendedCudaVersion = $cudaPreferred
    $cudaRecommended = $cudaPreferredInstaller

    $detectedVsMajor = $compat.detectedVsMajor
    $vsHasX64Tools = [bool]$compat.vsHasX64VcTools
    $vsInstalled = ([bool]$compat.vsMeetsDefault -and $vsHasX64Tools)

    $cudaInstalled = Test-CudaInstalled -Compatibility $compat
    $toolchain = Get-ToolchainCompatibilityStatus -Compatibility $compat
    $toolchainReason = [string]$toolchain.reason
    $cudaVsCompatible = [bool]$toolchain.compatible

    $needVsAction = ((-not $vsInstalled) -or [bool]$toolchain.requireVsUpgrade)
    $needCudaAction = ((-not $cudaInstalled) -or [bool]$toolchain.requireCudaUpgrade)

    $prereqSatisfied = ($vsInstalled -and $cudaInstalled -and $cudaVsCompatible -and -not $needVsAction -and -not $needCudaAction)

    Write-Log "Detected VS=$vsInstalled (detectedMajor=$detectedVsMajor requiredMajor=$requiredVsMajor hasX64VcTools=$vsHasX64Tools profile=$($compat.selectedVsProfile)) CUDA=$cudaInstalled GSPLAT_TOOLCHAIN_OK=$cudaVsCompatible"
    if (-not $cudaVsCompatible) {
        Write-Log "Toolchain mismatch reason: $toolchainReason"
    }

    if ($prereqSatisfied) {
        Write-Log "Using existing compatible installations: VS major $detectedVsMajor (x64 VC tools present), CUDA $($compat.selectedCudaVersion). No prerequisite download required."
    }

    if ($ProbeOnly) {
        Write-Host "VSInstalled=$vsInstalled"
        Write-Host "VSDetectedMajor=$detectedVsMajor"
        Write-Host "VSRequiredMajor=$requiredVsMajor"
        Write-Host "VSHasX64VcTools=$vsHasX64Tools"
        Write-Host "CUDAInstalled=$cudaInstalled"
        Write-Host "GsplatToolchainCompatible=$cudaVsCompatible"
        Write-Host "NeedVsAction=$needVsAction"
        Write-Host "NeedCudaAction=$needCudaAction"
        if (-not $cudaVsCompatible -and $toolchainReason) {
            Write-Host "GsplatToolchainIssue=$toolchainReason"
        }
        if ($prereqSatisfied) {
            Write-Host "UsingExistingCompatibleToolchain=True"
            Write-Host "CompatibleCudaVersion=$($compat.selectedCudaVersion)"
        }
        if ($prereqSatisfied) {
            exit 0
        }
        exit 1
    }

    if ($prereqSatisfied) {
        Write-Log 'All prerequisites detected. Continuing setup.'
        exit 0
    }

    $result = Show-PrereqDialog -vsInstalled $vsInstalled -cudaInstalled $cudaInstalled -detectedVsMajor $detectedVsMajor -requiredVsMajor $requiredVsMajor -recommendedCudaVersion $recommendedCudaVersion -compatibleCudaVersion ([string]$compat.selectedCudaVersion) -toolchainIssue $toolchainReason -needVsAction $needVsAction -needCudaAction $needCudaAction
    if ($result -eq [System.Windows.Forms.DialogResult]::Cancel) {
        throw 'User cancelled setup during prerequisite check.'
    }
}
