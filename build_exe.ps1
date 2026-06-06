$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

Write-Host "Preparing desktop inference application folder..."

$modelPath = Join-Path $PSScriptRoot "models\garbage_resnet18.pth"
if (-not (Test-Path -LiteralPath $modelPath)) {
    Write-Host ""
    Write-Host "[ERROR] models\garbage_resnet18.pth was not found."
    Write-Host "Download it from GitHub Release, or run python train.py first."
    Write-Host "Release: https://github.com/morio-22/garbage-classification-system/releases/tag/garbage-resnet18-realwaste-v1"
    exit 1
}

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1
$pythonPrefix = @()
if ($pythonCommand) {
    $pythonExe = $pythonCommand.Source
} else {
    $pyCommand = Get-Command py -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $pyCommand) {
        Write-Host ""
        Write-Host "[ERROR] python or py was not found. Install Python 3.10+ and enable Add Python to PATH."
        exit 1
    }

    $pythonExe = $pyCommand.Source
    $pythonPrefix = @("-3")
}

function Invoke-Python {
    param([string[]] $Arguments)

    & $script:pythonExe @script:pythonPrefix @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($Arguments -join ' ')"
    }
}

Write-Host "Using Python: $pythonExe $($pythonPrefix -join ' ')"

Invoke-Python @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Python @("-m", "pip", "install", "-r", "requirements-desktop.txt")
Invoke-Python @("-m", "pip", "install", "pyinstaller")

$pyinstallerArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onedir",
    "--name", "GarbageClassificationSystem",
    "--add-data", "app.py;.",
    "--add-data", "predict.py;.",
    "--add-data", "feedback.py;.",
    "--add-data", "review_feedback.py;.",
    "--add-data", "models\garbage_resnet18.pth;models",
    "--collect-all", "streamlit",
    "--collect-binaries", "torch",
    "--collect-binaries", "torchvision",
    "--collect-data", "torchvision",
    "--collect-data", "PIL",
    "--hidden-import", "streamlit.web.bootstrap",
    "--hidden-import", "torch",
    "--hidden-import", "torchvision",
    "--hidden-import", "torchvision.models.resnet",
    "--hidden-import", "torchvision.transforms._presets",
    "--exclude-module", "prepare_dataset",
    "--exclude-module", "prepare_trashnet",
    "--exclude-module", "prepare_realwaste",
    "--exclude-module", "train",
    "--exclude-module", "matplotlib",
    "--exclude-module", "huggingface_hub",
    "--exclude-module", "scipy",
    "--exclude-module", "tensorboard",
    "--exclude-module", "torch.utils.tensorboard",
    "run_app.py"
)
Invoke-Python $pyinstallerArgs

Write-Host ""
Write-Host "Build complete:"
Write-Host "dist\GarbageClassificationSystem\GarbageClassificationSystem.exe"
Write-Host ""
Write-Host "Test this exe first. If it opens the web app correctly, run build_installer.ps1."
