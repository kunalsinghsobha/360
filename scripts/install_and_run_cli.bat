@echo off
if "%~1"=="" (
  echo Usage: scripts\install_and_run_cli.bat path\to\report.xlsx
  exit /b 1
)
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
customer360-run --input "%~1" --outdir output
echo Outputs saved to .\output
