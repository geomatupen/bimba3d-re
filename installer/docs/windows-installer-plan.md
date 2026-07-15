# Bimba3D Windows One-Click Installer Plan (Chainer)

## Goal
Provide a **single installer experience** where the user runs one `Bimba3D-Setup.exe` and the installer handles all dependencies automatically.

## UX Requirements
- One branded installer executable.
- One progress window for all steps.
- No manual dependency steps for users.
- Automatic skip for dependencies already installed.
- Auto-resume after reboot if required.
- Install logs in a predictable folder for support.

## Recommended Installer Stack
- **Bootstrapper/Chainer:** WiX Burn (WiX v4)
- **App payload:** Existing frontend/backend packaged as your app MSI/EXE payload
- **Dependency installers:** vendor installers invoked silently

## High-Level Install Sequence
1. Preflight checks
   - Admin privileges
   - OS compatibility (Windows 10/11 x64)
   - Optional: NVIDIA GPU presence check
2. Install/verify Microsoft VC++ runtime
3. Install/verify CUDA prerequisites (or required redistributable runtime set)
4. Install/verify COLMAP binaries
5. Install Bimba3D app payload
6. Write config (`COLMAP_EXE`, app paths)
7. Final health check and finish

## Packaging Strategy
- **Online bootstrapper (recommended first):**
  - Small installer downloads missing vendor packages.
  - Better for legal compliance and version updates.
- **Offline full bundle (optional second):**
  - Large package contains all installers.
  - Useful for air-gapped or low-connectivity environments.

## Legal-Safe Distribution Approach
- Keep your installer as a **single launcher** but source Microsoft/NVIDIA installers from official channels where possible.
- Bundle COLMAP with BSD notices and preserve license texts.
- Include a `THIRD_PARTY_NOTICES` file in your app install directory.

## Detection + Idempotency Rules
- Every dependency package must have:
  - Detect condition (registry/file version)
  - Install command
  - Exit code handling (including reboot-required)
- If detected, package is skipped and marked complete.

## Reboot and Resume
- Burn handles `3010` (reboot required).
- Configure bundle to:
  - save state,
  - trigger reboot prompt,
  - resume automatically and continue remaining chain items.

## Logging and Support
- Bundle logs to `%ProgramData%\Bimba3D\Logs`.
- Keep one master chain log + per-package logs.
- Add quick support script to collect logs for troubleshooting.

## Security + Integrity
- Use HTTPS download URLs.
- Pin SHA-256 for every downloaded payload.
- Validate checksum before execution.
- Sign your bootstrapper and app payload with code signing certificate.

## Acceptance Criteria
- Fresh machine install succeeds with one executable.
- Re-running installer is safe (no broken duplicates).
- Existing dependencies are skipped.
- Reboot path resumes automatically.
- Installed app launches and can run local mode.

## Implementation Order
1. Freeze dependency versions.
2. Create Burn bundle with detect/install for each package.
3. Add app payload package.
4. Add config step for `COLMAP_EXE`.
5. Build online installer.
6. QA on clean Windows 10/11 (GPU and non-GPU paths).
7. Add offline bundle variant.
