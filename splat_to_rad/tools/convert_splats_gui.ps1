Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$converterPath = Join-Path $scriptDir "convert_splats_to_rad.ps1"
$workspaceRoot = Split-Path -Parent $scriptDir

$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$owner.Size = New-Object System.Drawing.Size(1, 1)
$owner.ShowInTaskbar = $false
$owner.Opacity = 0
$owner.Show()

function Show-Question {
    param(
        [string]$Message,
        [string]$Title,
        [System.Windows.Forms.MessageBoxButtons]$Buttons = [System.Windows.Forms.MessageBoxButtons]::YesNo
    )

    return [System.Windows.Forms.MessageBox]::Show(
        $owner,
        $Message,
        $Title,
        $Buttons,
        [System.Windows.Forms.MessageBoxIcon]::Question
    )
}

function Select-InputPath {
    $inputMessage = @(
        "Select Yes to choose a folder containing multiple splat files for batch conversion.",
        "Select No to choose one splat file.",
        "",
        "The next dialog selects the input splats. After that, you will choose the output folder for converted RAD files."
    ) -join [Environment]::NewLine

    $choice = Show-Question `
        -Buttons ([System.Windows.Forms.MessageBoxButtons]::YesNoCancel) `
        -Title "Select input splats" `
        -Message $inputMessage

    if ($choice -eq [System.Windows.Forms.DialogResult]::Cancel) {
        return $null
    }

    if ($choice -eq [System.Windows.Forms.DialogResult]::Yes) {
        $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
        $dialog.Description = "Select input folder containing splat files for batch conversion"
        if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
            return $dialog.SelectedPath
        }
        return $null
    }

    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = "Select one input splat file"
    $dialog.Filter = "Gaussian splat files (*.ply;*.spz;*.splat;*.ksplat;*.sog;*.zip)|*.ply;*.spz;*.splat;*.ksplat;*.sog;*.zip|All files (*.*)|*.*"
    $dialog.Multiselect = $false
    if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
        return $dialog.FileName
    }
    return $null
}

function Select-Folder {
    param(
        [string]$Description,
        [string]$InitialPath = ""
    )

    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = $Description
    if ($InitialPath -and (Test-Path -LiteralPath $InitialPath)) {
        $dialog.SelectedPath = $InitialPath
    }
    if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
        return $dialog.SelectedPath
    }
    return $null
}

$inputPath = Select-InputPath
if (-not $inputPath) {
    Write-Host "Cancelled: no input selected."
    $owner.Close()
    exit 0
}

$defaultOutput = Join-Path $workspaceRoot "processed"
$outputRoot = Select-Folder -Description "Select output folder where converted RAD files will be placed" -InitialPath $defaultOutput
if (-not $outputRoot) {
    Write-Host "Cancelled: no output folder selected."
    $owner.Close()
    exit 0
}

$defaultSparkRepo = Join-Path $workspaceRoot "vendor\spark"
$sparkRepoDescription = "Select the local Spark repository folder. Usually this is inside the converter workspace: $defaultSparkRepo"
$sparkRepo = Select-Folder -Description $sparkRepoDescription -InitialPath $defaultSparkRepo
if (-not $sparkRepo) {
    Write-Host "Cancelled: no Spark repository selected."
    $owner.Close()
    exit 0
}

$qualityChoice = Show-Question `
    -Title "Conversion quality" `
    -Message "Use higher-quality conversion? It is slower but better for final public splats."

$chunkChoice = Show-Question `
    -Title "Chunked streaming output" `
    -Message "Create chunked RAD output for static streaming? Recommended for large splats."

$forceChoice = Show-Question `
    -Title "Overwrite output" `
    -Message "Overwrite existing converted files if the output scene already exists?"

$argsList = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $converterPath,
    "-InputPath", $inputPath,
    "-SparkRepoPath", $sparkRepo,
    "-OutputRoot", $outputRoot
)

if ($qualityChoice -eq [System.Windows.Forms.DialogResult]::Yes) {
    $argsList += "-Quality"
}
if ($chunkChoice -eq [System.Windows.Forms.DialogResult]::Yes) {
    $argsList += "-Chunked"
}
if ($forceChoice -eq [System.Windows.Forms.DialogResult]::Yes) {
    $argsList += "-Force"
}

Write-Host "Input splats: $inputPath"
Write-Host "Spark repo: $sparkRepo"
Write-Host "Output RAD files: $outputRoot"
Write-Host ""

$inputItem = Get-Item -LiteralPath $inputPath
if ($inputItem.PSIsContainer) {
    $supported = @(".ply", ".spz", ".splat", ".ksplat", ".sog", ".zip")
    $inputCount = @(Get-ChildItem -LiteralPath $inputItem.FullName -File | Where-Object { $supported -contains $_.Extension.ToLowerInvariant() }).Count
    Write-Host "Batch input folder contains $inputCount supported splat file(s)."
} else {
    Write-Host "Single input file selected."
}
Write-Host "Conversion progress will appear in this PowerShell window."
Write-Host ""

$startChoice = Show-Question `
    -Title "Start conversion" `
    -Message "Start converting now? Progress will be shown in the PowerShell window."
if ($startChoice -ne [System.Windows.Forms.DialogResult]::Yes) {
    Write-Host "Cancelled: conversion was not started."
    $owner.Close()
    exit 0
}

& powershell @argsList

if ($LASTEXITCODE -eq 0) {
    [System.Windows.Forms.MessageBox]::Show(
        $owner,
        "Conversion completed. Output folder:`n$outputRoot",
        "Splat conversion complete",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Information
    ) | Out-Null
} else {
    [System.Windows.Forms.MessageBox]::Show(
        $owner,
        "Conversion failed. Check the PowerShell window for details.",
        "Splat conversion failed",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error
    ) | Out-Null
}

$owner.Close()
