$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PreferredPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
$VenvPath = Join-Path $ProjectRoot ".venv_local"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PreferredPython)) {
    winget install `
        --id Python.Python.3.12 `
        --exact `
        --scope user `
        --silent `
        --accept-package-agreements `
        --accept-source-agreements `
        --disable-interactivity
}

if (-not (Test-Path -LiteralPath $PreferredPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.13 --version *> $null
        if ($LASTEXITCODE -eq 0) {
            $PreferredPython = "py -3.13"
        } else {
            $PreferredPython = "py -3.10"
        }
    } else {
        throw "Python 3.12 was not found and the Python launcher is unavailable."
    }
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    if ($PreferredPython.StartsWith("py ")) {
        $parts = $PreferredPython.Split(" ")
        & $parts[0] $parts[1] -m venv $VenvPath
    } else {
        & $PreferredPython -m venv $VenvPath
    }
}

& $VenvPython -m pip install --upgrade pip "setuptools<82" wheel
& $VenvPython -m pip install -e "${ProjectRoot}[solver,langgraph]" "uvicorn[standard]>=0.30,<1"
& $VenvPython -m pip check
& $VenvPython (Join-Path $PSScriptRoot "check_environment.py")

Write-Host ""
Write-Host "LogicGuard environment is ready."
Write-Host "Activate with: .\.venv_local\Scripts\Activate.ps1"
