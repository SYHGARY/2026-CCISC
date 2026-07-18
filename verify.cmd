@echo off
setlocal
if exist ".venv_local\Scripts\python.exe" (
  set "PYTHON=.venv_local\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  echo Project virtual environment not found.
  echo Run: powershell -ExecutionPolicy Bypass -File scripts\setup_environment.ps1
  exit /b 1
)

"%PYTHON%" scripts\check_environment.py
if errorlevel 1 exit /b %errorlevel%
"%PYTHON%" scripts\verify_project.py
if errorlevel 1 exit /b %errorlevel%
