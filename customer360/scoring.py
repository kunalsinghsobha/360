# customer360/scoring.py
from __future__ import annotations
import numpy as np
import pandas as pd

from .utils import pct_rank


def _first_nonnull_row_value(row: pd.Series, cols: list[str]):
    for c in cols:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return np.nan


def compute_customer_rollups(df_units: pd.DataFrame, units_master: pd.DataFrame) -> pd.DataFrame:
    """
    Per-customer aggregates + 0–100 score. Robust to missing columns.
    Requires df_units to have customer_master_id and unit_master_id.
    """
    if "unit_master_id" not in df_units.columns:
        raise KeyError("`unit_master_id` missing in df_units (ensure build_units() was applied).")
    if "customer_master_id" not in df_units.columns:
        # Failsafe so we never crash
        df_units = df_units.copy()
        df_units["customer_master_id"] = "SYN::" + df_units.index.astype(str)

    # Normalize types for fields we aggregate
    for c in ["Total Receipts Amount", "Total Amount Balance", "Overdue %"]:
        if c in df_units.columns:
            df_units[c] = pd.to_numeric(df_units[c], errors="coerce")
    for c in ["Booking: Last Modified Date", "Booking Date As Per The SBTR", "Booking Date"]:
        if c in df_units.columns:
            df_units[c] = pd.to_datetime(df_units[c], errors="coerce")

    # Row-level paid proxy: pick first non-null among typical "paid" cols to avoid double counting
    paid_priority = [
        "Total Receipts Amount",
        "Total Receipt Amount",
        "Total Amount Paid ( Unit Price )",
        "Paid Amount For Process",
    ]
    if any(c in df_units.columns for c in paid_priority):
        df_units = df_units.copy()
        df_units["__paid_row"] = df_units.apply(
            lambda r: _first_nonnull_row_value(r, paid_priority), axis=1
        )
        df_units["__paid_row"] = pd.to_numeric(df_units["__paid_row"], errors="coerce")
    else:
        df_units["__paid_row"] = np.nan

    # Group per customer x unit
    def _last_nonnull(s: pd.Series):
        s = s.dropna()
        return s.iloc[-1] if len(s) else np.nan

    per_cu = (
        df_units.groupby(["customer_master_id", "unit_master_id"])
        .agg(
            total_paid=("__paid_row", "sum"),
            balance_latest=("Total Amount Balance", _last_nonnull)
            if "Total Amount Balance" in df_units.columns
            else ("unit_master_id", "size"),
            overdue_latest=("Overdue %", _last_nonnull)
            if "Overdue %" in df_units.columns
            else ("unit_master_id", "size"),
            last_mod=("Booking: Last Modified Date", "max")
            if "Booking: Last Modified Date" in df_units.columns
            else ("unit_master_id", "first"),
            first_book=("Booking Date As Per The SBTR", "min")
            if "Booking Date As Per The SBTR" in df_units.columns
            else ("Booking Date", "min")
            if "Booking Date" in df_units.columns
            else ("unit_master_id", "first"),
        )
        .reset_index()
    )

    # Join per-unit investment from units_master
    if len(units_master) and "unit_master_id" in units_master.columns:
        inv = units_master[["unit_master_id", "purchase_price_max"]].rename(
            columns={"purchase_price_max": "unit_invest"}
        )
        per_cu = per_cu.merge(inv, on="unit_master_id", how="left")
    else:
        per_cu["unit_invest"] = np.nan

    # Customer totals
    cust = (
        per_cu.groupby("customer_master_id")
        .agg(
            n_units=("unit_master_id", "nunique"),
            total_investment=("unit_invest", "sum"),
            total_paid=("total_paid", "sum"),
            balance_latest=("balance_latest", "sum"),
            overdue_pct=("overdue_latest", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            last_activity=("last_mod", "max"),
            first_booking=("first_book", "min"),
        )
        .reset_index()
    )

    cust["last_activity"] = pd.to_datetime(cust["last_activity"], errors="coerce")
    cust["first_booking"] = pd.to_datetime(cust["first_booking"], errors="coerce")
    cust["days_inactive"] = (pd.Timestamp.now() - cust["last_activity"]).dt.days
    cust["months_tenure"] = (pd.Timestamp.now() - cust["first_booking"]).dt.days / 30.0

    inv_rank = pct_rank(pd.to_numeric(cust["total_investment"], errors="coerce")).fillna(0.0)
    units_rank = pct_rank(pd.to_numeric(cust["n_units"], errors="coerce")).fillna(0.0)
    overdue_norm = pd.to_numeric(cust["overdue_pct"], errors="coerce").fillna(0.0) / 100.0
    pay_score = (1.0 - overdue_norm).clip(0.0, 1.0)
    recency_score = (1.0 - (pd.to_numeric(cust["days_inactive"], errors="coerce").fillna(9999) / 180.0)).clip(0.0, 1.0)
    tenure_rank = pct_rank(pd.to_numeric(cust["months_tenure"], errors="coerce")).fillna(0.0)

    # Stage proxy from Title Transfer %
    if len(units_master) and "Title Transfer" in units_master.columns:
        tt = units_master[["unit_master_id", "Title Transfer"]].copy()
        tt["tt_int"] = tt["Title Transfer"].astype(str).str.lower().isin(["yes", "y", "true", "1"]).astype(int)
        per_cu2 = per_cu.merge(tt, on="unit_master_id", how="left")
        stage_pct = (
            per_cu2.groupby("customer_master_id")["tt_int"].mean()
        ).reindex(cust["customer_master_id"]).fillna(0.0)
    else:
        stage_pct = pd.Series(0.0, index=cust["customer_master_id"])

    # No DNC source here; caller can merge a real flag later. Use 0 to avoid penalizing.
    w = {"investment": 0.35, "units": 0.15, "payment": 0.20, "recency": 0.15, "tenure": 0.05, "engagement": 0.05, "stage": 0.05}
    score01 = (
        w["investment"] * inv_rank
        + w["units"] * units_rank
        + w["payment"] * pay_score
        + w["recency"] * recency_score
        + w["tenure"] * tenure_rank
        + w["engagement"] * 1.0
        + w["stage"] * stage_pct
    ).clip(0.0, 1.0)
    cust["customer_value_score"] = (score01 * 100.0).round(1)
    return cust


def ai_like_summary(row) -> str:
    """
    Natural language AI-like summary for a customer.
    Only includes values that exist, written as full sentences.
    """
    import pandas as pd

    sentences = []

    # --- Name & Nationality ---
    name = row.get("Primary Applicant Name") or row.get("display_name") or "Customer"
    if pd.notna(row.get("Nationality")) and str(row["Nationality"]).strip():
        sentences.append(f"**AI Summary:** {name} is from {row['Nationality']}.")
    else:
        sentences.append(f"**AI Summary:** {name}.")

    # --- Contact ---
    if pd.notna(row.get("Primary Applicant Email")) and str(row["Primary Applicant Email"]).strip():
        sentences.append(f"Their email is {row['Primary Applicant Email']}.")
    if pd.notna(row.get("Primary Mobile Number")) and str(row["Primary Mobile Number"]).strip():
        sentences.append(f"Their phone number is {row['Primary Mobile Number']}.")

    # --- Ownership & Finance ---
    own_parts = []
    if row.get("n_units", 0) > 0:
        own_parts.append(f"owns {int(row['n_units'])} unit(s)")
    if pd.notna(row.get("total_investment")) and row["total_investment"] > 0:
        own_parts.append(f"with a total investment of AED {row['total_investment']:,.0f}")
    if pd.notna(row.get("total_paid")) and row["total_paid"] > 0:
        own_parts.append(f"has paid ~AED {row['total_paid']:,.0f}")
    if pd.notna(row.get("balance_latest")) and row["balance_latest"] > 0:
        own_parts.append(f"with an outstanding balance of ~AED {row['balance_latest']:,.0f}")
    if own_parts:
        sentences.append(f"The customer {' '.join(own_parts)}.")

    # --- Payment / Bank Info ---
    payinfo = []
    if pd.notna(row.get("Payment Plan")) and str(row["Payment Plan"]).strip():
        payinfo.append(f"Payment plan is {row['Payment Plan']}")
    if pd.notna(row.get("Mode of Funding")) and str(row["Mode of Funding"]).strip():
        payinfo.append(f"funded via {row['Mode of Funding']}")
    if pd.notna(row.get("Bank Name (Flat Cost)")) and str(row["Bank Name (Flat Cost)"]).strip():
        payinfo.append(f"banking with {row['Bank Name (Flat Cost)']}")
    if pd.notna(row.get("IBAN (Flat Cost)")) and str(row["IBAN (Flat Cost)"]).strip():
        payinfo.append(f"IBAN {row['IBAN (Flat Cost)']}")
    if pd.notna(row.get("Green Channel")) and str(row["Green Channel"]).strip():
        payinfo.append(f"Green Channel: {row['Green Channel']}")
    if payinfo:
        sentences.append(" ".join(payinfo) + ".")

    # --- Activity ---
    if pd.notna(row.get("days_inactive")):
        d = int(row["days_inactive"])
        if d <= 30:
            sentences.append(f"The customer was recently active (~{d} days ago).")
        else:
            sentences.append(f"The customer has been inactive for ~{d} days.")

    # Join as paragraph
    return " ".join([s for s in sentences if s and str(s).strip()])
