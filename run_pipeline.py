# run_pipeline.py
import os, sys, importlib
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def _import_or_load(module_path: str, fallback_file: str, attr: str = None):
    try:
        mod = importlib.import_module(module_path)
    except Exception:
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_path, fallback_file)
        if spec is None or spec.loader is None:
            raise
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[module_path] = mod
    return getattr(mod, attr) if attr else mod

RESOLUTION_PY = os.path.join(ROOT, "customer360", "resolution.py")
UNITS_PY      = os.path.join(ROOT, "customer360", "units.py")
SCORING_PY    = os.path.join(ROOT, "customer360", "scoring.py")

resolve_customers = _import_or_load("customer360.resolution", RESOLUTION_PY, "resolve_customers")
build_units       = _import_or_load("customer360.units", UNITS_PY, "build_units")
_scoring_mod      = _import_or_load("customer360.scoring", SCORING_PY, None)
compute_customer_rollups = getattr(_scoring_mod, "compute_customer_rollups")
ai_like_summary         = getattr(_scoring_mod, "ai_like_summary")

import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser(description="Customer 360 CLI")
    p.add_argument("--input", required=True, help="Path to CSV/XLSX")
    p.add_argument("--outdir", default="output", help="Output directory")
    args = p.parse_args()

    if args.input.lower().endswith(".csv"):
        df = pd.read_csv(args.input, encoding="utf-8-sig")
    else:
        df = pd.read_excel(args.input)

    df_units, units_master, unit_bookings_map = build_units(df)
    df["customer_master_id"] = resolve_customers(df)
    rollups = compute_customer_rollups(df_units, units_master)

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    df_units.to_csv(out / "df_units.csv", index=False)
    units_master.to_csv(out / "units_master.csv", index=False)
    rollups.to_csv(out / "customer_rollups.csv", index=False)

    try:
        summary_lines = [ai_like_summary(r) for _, r in rollups.head(100).iterrows()]
        (out / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    except Exception as e:
        (out / "summary.txt").write_text(f"Summary failed: {e}", encoding="utf-8")

if __name__ == "__main__":
    main()
