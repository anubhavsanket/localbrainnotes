# Bootstrap the Rust toolchain + build prerequisites for the Tauri desktop shell.
#
# Tauri v2 on Windows needs:
#   1. Rust toolchain (rustup -> MSVC toolchain)
#   2. Visual Studio Build Tools with the "Desktop development with C++" workload
#      (provides link.exe / the MSVC CRT)
#   3. WebView2 runtime (present on stock Windows 10/11)
#
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File desktop\bootstrap-toolchain.ps1
# Afterwards:  cd desktop; npm run tauri dev
#
# NOTE: both installers are multi-GB and can take 20-60 minutes. A new terminal
# is required after rustup installs (PATH changes).

$ErrorActionPreference = "Stop"

Write-Host "=== LocalBrain desktop - Rust toolchain bootstrap ===" -ForegroundColor Cyan

# --- 1. rustup + Rust MSVC toolchain ---------------------------------------
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "[1/3] Installing Rust via rustup (MSVC toolchain)..." -ForegroundColor Yellow
    $rustupInit = "$env:TEMP\rustup-init.exe"
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $rustupInit
    & $rustupInit -y --default-toolchain stable --profile default
    # Make cargo available in this session (and persist for future ones).
    $env:CARGO_HOME = "$env:USERPROFILE\.cargo"
    $env:RUSTUP_HOME = "$env:USERPROFILE\.rustup"
    $env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
} else {
    Write-Host "[1/3] cargo already installed: $(cargo --version)" -ForegroundColor Green
}

# --- 2. Visual Studio Build Tools (MSVC linker) -----------------------------
if (-not (Get-Command cl -ErrorAction SilentlyContinue) -and
    -not (Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe")) {
    Write-Host "[2/3] Installing Visual Studio Build Tools (C++ workload)..." -ForegroundColor Yellow
    Write-Host "      Launching winget - the workload is several GB." -ForegroundColor DarkYellow
    winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
} else {
    Write-Host "[2/3] MSVC build tools already present (or vswhere found)." -ForegroundColor Green
}

# --- 3. WebView2 -------------------------------------------------------------
if (Test-Path "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}") {
    Write-Host "[3/3] WebView2 runtime present." -ForegroundColor Green
} else {
    Write-Host "[3/3] WebView2 runtime NOT detected - Tauri on Windows needs it." -ForegroundColor Yellow
    Write-Host "      Get it at: https://developer.microsoft.com/microsoft-edge/webview2/"
}

Write-Host ""
Write-Host "Bootstrap complete (or already satisfied)." -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Open a NEW terminal (PATH changed by rustup)." -ForegroundColor White
Write-Host "  2. cd desktop" -ForegroundColor White
Write-Host "  3. npm run tauri dev   # first build compiles ~400 crates (15-30 min)" -ForegroundColor White
