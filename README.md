
# Customer 360 — Full Package

End-to-end Customer 360 pipeline and UI (Streamlit), built from your modules:
**dedupe → resolution → unit freezing → finance consolidation → rollups & scoring → dashboard.**

## Contents
```
customer360/                 # package (utils.py, units.py, scoring.py, resolution.py, __init__.py)
streamlit_app.py             # Streamlit entry
gradio_app.py                # Optional Gradio entry (lightweight local tester)
run_pipeline.py              # CLI runner
requirements.txt             # dependencies
pyproject.toml, setup.py     # installable project
scripts/                     # one-click launchers (Windows + macOS/Linux)
README.md, .gitignore
```

## Quickstart (Windows)
```bat
scripts\install_and_run_streamlit.bat
```
- This creates `.venv`, installs deps, installs package in editable mode, and launches the app.
- App opens at: http://localhost:8501 (or nearby port)

CLI:
```bat
scripts\install_and_run_cli.bat "C:\path\to\Salesforce_report.xlsx"
```

Gradio (optional fast tester):
```bat
scripts\install_and_run_gradio.bat
```

## Quickstart (macOS / Linux)
```bash
chmod +x scripts/*.sh
./scripts/install_and_run_streamlit.sh
# or
./scripts/install_and_run_cli.sh path/to/Salesforce_report.xlsx
# or
./scripts/install_and_run_gradio.sh
```

## Manual install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
streamlit run streamlit_app.py
```

## Notes
- If you previously saw `ModuleNotFoundError: customer360.resolution`, this package ships a robust `__init__.py` that
  works whether installed or run in-place.
- For very large CSVs, consider saving as `.csv` and using a fast reader (PyArrow is included).

## Support
- Streamlit app: `streamlit_app.py`
- CLI: `customer360-run --input file.xlsx --outdir output`
- Optional Gradio app: `gradio_app.py`
