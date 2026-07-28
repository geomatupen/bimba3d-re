# Splat To RAD Converter

This folder contains helper scripts for converting Gaussian splat files into Spark `.rad` / `.radc` files for static web viewing.

The purpose is to prepare large splats locally, then upload only web-ready processed files to the thesis website or another asset host. This avoids making the browser download a very large raw `.splat`, `.ply`, `.spz`, or `.ksplat` file before visualization can begin.

## What This Uses

The conversion is handled by Spark's open-source `build-lod` tool.

- Spark website: https://sparkjs.dev/
- Spark GitHub repository: https://github.com/sparkjsdev/spark
- Spark LoD documentation: https://sparkjs.dev/docs/lod-getting-started/
- License: Spark is published under the MIT License. Keep Spark's license and attribution when using or distributing Spark-derived tooling or viewer code.

Attribution text to use in the website or documentation:

```text
Interactive Gaussian Splatting visualization uses Spark, an open-source 3D Gaussian Splatting renderer for Three.js by World Labs, distributed under the MIT License.
```

## Folder Layout

```text
splat_to_rad/
  raw/                  optional local input folder, ignored by Git
  processed/            optional local output folder, ignored by Git
  vendor/
    spark/              local Spark clone, ignored by Git
  tools/
    convert_splats_gui.ps1
    convert_splats_to_rad.ps1
  start_converter_gui.cmd
```

You do not need to put large splats in this folder. Input and output paths can be on any drive, for example:

```text
Input folder:  E:\Thesis\splats
Output folder: E:\Thesis\splats_rad
```

## Supported Input Files

The wrapper accepts:

- `.ply`
- `.spz`
- `.splat`
- `.ksplat`
- `.sog`
- `.zip` for SOG zip files

Selecting one file converts one scene. Selecting a folder converts all supported files directly inside that folder.

## Install Requirements

Install these once on the machine where conversion will run:

- Git
- Node.js LTS, including `npm`
- Rust, installed with `rustup`, including `cargo`
- Spark repository cloned locally

On Windows, if Rust fails with a linker or MSVC error, install Visual Studio Build Tools with the Desktop development with C++ workload.

## Windows Setup

From PowerShell:

```powershell
cd D:\bimba3d-re\splat_to_rad
git clone https://github.com/sparkjsdev/spark .\vendor\spark
```

Check tools:

```powershell
git --version
node -v
npm -v
cargo --version
```

If `cargo` is missing, install Rust from:

```text
https://www.rust-lang.org/tools/install
```

## Windows GUI Conversion

Double-click:

```text
D:\bimba3d-re\splat_to_rad\start_converter_gui.cmd
```

The GUI asks for:

- input splat file, or an input splats folder for batch conversion
- output folder where converted RAD files will be placed
- local Spark repository folder, usually `D:\bimba3d-re\splat_to_rad\vendor\spark`
- quick or quality conversion
- chunked RAD output
- overwrite choice
- final confirmation before processing starts

Recommended choices for thesis visualization:

```text
Input splats: E:\Thesis\splats
Output RAD:   E:\Thesis\splats_rad
Mode:         Quality
Chunked:      Yes
Overwrite:    No, unless re-running intentionally
```

## Windows Command Line

Convert one file:

```powershell
cd D:\bimba3d-re\splat_to_rad
powershell -ExecutionPolicy Bypass -File .\tools\convert_splats_to_rad.ps1 `
  -InputPath "E:\Thesis\splats\pix4d-forensic_model-ridge-7k_checkpoint-007000.splat" `
  -SparkRepoPath ".\vendor\spark" `
  -OutputRoot "E:\Thesis\splats_rad" `
  -Quality `
  -Chunked
```

Batch convert a folder:

```powershell
cd D:\bimba3d-re\splat_to_rad
powershell -ExecutionPolicy Bypass -File .\tools\convert_splats_to_rad.ps1 `
  -InputPath "E:\Thesis\splats" `
  -SparkRepoPath ".\vendor\spark" `
  -OutputRoot "E:\Thesis\splats_rad" `
  -Quality `
  -Chunked
```

## Linux Setup

Install the required tools using your system package manager. Example for Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y git curl nodejs npm
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

Clone Spark:

```bash
cd /path/to/bimba3d-re/splat_to_rad
git clone https://github.com/sparkjsdev/spark ./vendor/spark
```

Check tools:

```bash
git --version
node -v
npm -v
cargo --version
```

## Linux Command Line

The converter script is PowerShell, so install PowerShell Core (`pwsh`) if it is not already available.

Batch convert a folder:

```bash
cd /path/to/bimba3d-re/splat_to_rad
pwsh -ExecutionPolicy Bypass -File ./tools/convert_splats_to_rad.ps1 \
  -InputPath "/mnt/e/Thesis/splats" \
  -SparkRepoPath "./vendor/spark" \
  -OutputRoot "/mnt/e/Thesis/splats_rad" \
  -Quality \
  -Chunked
```

For a normal Linux drive path, replace `/mnt/e/Thesis/...` with your actual folder path.

## Output Structure

For each input file, the script creates one output folder:

```text
E:\Thesis\splats_rad\
  pix4d-forensic_model-ridge-7k_checkpoint-007000\
    pix4d-forensic_model-ridge-7k_checkpoint-007000-lod.rad
    pix4d-forensic_model-ridge-7k_checkpoint-007000-lod-000.radc
    pix4d-forensic_model-ridge-7k_checkpoint-007000-lod-001.radc
    manifest.json
```

The `manifest.json` is generated by this wrapper to help the website or a later packaging step know which files belong to each converted scene.

Spark's `build-lod` tool creates the `.rad` and `.radc` files beside the input splat first. This wrapper moves those generated files into the selected output folder, so successful conversions should leave the input folder with only the original splat files.

## Quick Vs Quality

Quick mode is useful for testing that conversion works.

Quality mode is recommended for public thesis visualization because it builds a better LoD structure, but it takes longer.

## Chunked RAD

Use chunked output for large scenes.

Chunked output creates:

- one `.rad` entry file
- several `.radc` chunk files

This is better for static hosting because the viewer can stream scene data progressively instead of waiting for one large raw splat file.

## Static Hosting Notes

For cPanel, object storage, or any static host, check that the server supports:

- large files
- HTTP Range requests
- `.rad` and `.radc` file delivery
- CORS headers if the viewer and splat files are on different domains

If cPanel struggles with large files or range requests, keep the thesis website on cPanel and host only the converted RAD assets on object storage or a CDN.

## Do Not Commit Large Files

This converter workspace ignores raw and processed splat data:

```text
raw/
processed/
vendor/spark/
```

Keep large staged folders such as `E:\Thesis\splats` and `E:\Thesis\splats_rad` outside the Git repository.
