@echo off
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo Please install uv first: https://docs.astral.sh/uv/
  pause
  exit /b 1
)
uv sync --frozen
if errorlevel 1 exit /b 1
uv run python scripts/demo_inference.py --phase both
pause
