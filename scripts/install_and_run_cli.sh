#!/usr/bin/env bash
set -e
if [ -z "$1" ]; then
  echo "Usage: scripts/install_and_run_cli.sh path/to/report.xlsx"
  exit 1
fi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
customer360-run --input "$1" --outdir output
echo "Outputs saved to ./output"
