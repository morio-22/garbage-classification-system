$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

Write-Host "Preparing Windows installer..."

$exePath = Join-Path $PSScriptRoot "dist\GarbageClassificationSystem\GarbageClassificationSystem.exe"
if (-not (Test-Path -LiteralPath $exePath)) {
    Write-Host "Packaged exe was not found. Running build_exe.ps1 first..."
    & (Join-Path $PSScriptRoot "build_exe.ps1")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$candidatePaths = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Inno\ISCC.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
)

$isccPath = $candidatePaths | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $isccPath) {
    Write-Host ""
    Write-Host "[ERROR] Inno Setup 6 was not found."
    Write-Host "Download: https://jrsoftware.org/isinfo.php"
    exit 1
}

Write-Host "Using Inno Setup compiler: $isccPath"
& $isccPath "installer\setup.iss"
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup compile failed."
}

Write-Host ""
Write-Host "Installer build complete:"
Write-Host "installer\Output\GarbageClassificationSystem_Setup.exe"
