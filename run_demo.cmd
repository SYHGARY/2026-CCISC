@echo off
setlocal
if exist ".venv_local\Scripts\python.exe" (
  ".venv_local\Scripts\python.exe" scripts\run_api.py
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\run_api.py
) else (
  echo Project virtual environment not found.
  echo Run: powershell -ExecutionPolicy Bypass -File scripts\setup_environment.ps1
  exit /b 1
)
