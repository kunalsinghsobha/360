# customer360/units.py
from __future__ import annotations
import hashlib
import re
from typing import List

import numpy as np
import pandas as pd

from .utils import nz, safe_to_datetime


def _stage_norm(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = x.strip().lower()
    if "title" in x:
        return "Title Transfer"
    if "handover" in x:
        return "Handover"
    if "execut" in x:
        return "SPA Executed"
    if "sent" in x:
        return "SPA Sent"
    if "qualif" in x:
        return "Qualified"
    if "book" in x:
        return "Booked"
    if "prospect" in x:
        return "Prospect"
    return x.title() if x else ""


STAGE_ORDER = ["Title Transfer", "Handover", "SPA Executed", "SPA Sent", "Qualified", "Booked", "Prospect"]
_STAGE_RANK = {s.lower(): i for i, s in enumerate(STAGE_ORDER, start=1)}


def _stage_score(x: str) -> int:
    x = _stage_norm(x)
    return 100 - _STAGE_RANK.get(x.lower(), 99)


def _yes_no_to_int(x) -> int:
    if isinstance(x, str) and x.strip().lower() in ("yes", "y", "true", "1"):
        return 1
    if x in (1, True):
        return 1
    return 0


def _norm_project(x: str) -> str:
    return re.sub(r"\s+", " ", nz(x).lower().replace("\u00a0", " ")).strip()


def _norm_tower(x: str) -> str:
    x = re.sub(r"\s+", " ", nz(x).lower().replace("\u00a0", " ")).strip()
    x = re.sub(r"\b(tower|twr|blk|block|bldg|building)\b\.?", "", x)
    x = re.sub(r"[^a-z0-9]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    parts: List[str] = []
    for p in x.split():
        parts.append(str(int(p)) if p.isdigit() else p)
    return " ".join(parts)


def _norm_unit(x: str) -> str:
    x = re.sub(r"\s+", " ", nz(x).lower().replace("\u00a0", " ")).strip()
    letters = re.findall(r"[a-zA-Z]+", x)
    digits = re.findall(r"\d+", x)
    flat4 = digits[-1][-4:] if digits else ""
    prefix = (letters[0].lower() if letters else "")
    if flat4:
        try:
            return f"{prefix}{int(flat4)}"
        except Exception:
            return f"{prefix}{flat4}"
    return re.sub(r"[^a-z0-9]", "", x)


def _unit_key_row(r: pd.Series) -> str:
    """
    Build a robust unit key:
    Project | Sub Project | Tower | Floor | Unit
    – uses multiple fields so different units don't collapse.
    – if all core parts are blank, fall back to a UNIQUE key so rows don't all merge.
    """
    proj = _norm_project(r.get("Project Name", ""))
    subp = _norm_project(r.get("Sub Project", ""))  # << new, present in your file
    tower = _norm_tower(r.get("Tower", "") or r.get("Tower Name", ""))
    floor = nz(r.get("Floor", "")).strip().lower()
    unit  = _norm_unit(r.get("Unit Name", "") or r.get("Unit", ""))

    # if we have any of the core pieces, return the composite key
    if proj or subp or tower or floor or unit:
        return "|".join([proj, subp, tower, floor, unit])

    # otherwise, create a UNIQUE fallback to avoid collapsing everything
    # include a few weak identifiers + row index to remain stable and unique
    weak = (
        nz(r.get("Booking: Booking Name", "")) or
        nz(r.get("Opportunity: Opportunity Name", "")) or
        nz(r.get("Opportunity: Account ID", ""))
    ).lower()
    return f"__FALLBACK__|{weak}|row{r.name}"


def _make_unit_id(key: str) -> str:
    return "UNIT::" + hashlib.sha1((key or "").encode("utf-8")).hexdigest()[:16]


def build_units(df: pd.DataFrame):
    df = df.copy()

    # --- robust numeric coercion for all likely numeric fields ---
    numeric_candidates = [
        "Purchase Price",
        "Saleable Area",
        "PSF Rate",
        "Total Receipts Amount",
        "Total Receipt Amount",
        "Total Amount Paid ( Unit Price )",
        "Advance Payment",
        "Advance Payments",
        "Amount On Account",
        "Token Amount",
        "Paid Amount For Process",
        "Total Amount Balance",
        "Total Milestone Due Till Today",
        "Total Payment Due",
        "Due Payments This Month",
        "Shortfall Amount for DLD",
        "DLD Amount",
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # >>> ADD THIS BLOCK DIRECTLY BELOW <<<
    # --- derive missing Area / PSF from Purchase Price where possible ---
    if "Purchase Price" in df.columns:
        if "Saleable Area" in df.columns and "PSF Rate" in df.columns:
            # If area is missing but price & psf exist → area = price / psf
            need_area = df["Saleable Area"].isna() & df["PSF Rate"].notna() & (df["PSF Rate"] != 0)
            df.loc[need_area, "Saleable Area"] = df.loc[need_area, "Purchase Price"] / df.loc[need_area, "PSF Rate"]

            # If psf is missing but price & area exist → psf = price / area
            need_psf = df["PSF Rate"].isna() & df["Saleable Area"].notna() & (df["Saleable Area"] != 0)
            df.loc[need_psf, "PSF Rate"] = df.loc[need_psf, "Purchase Price"] / df.loc[need_psf, "Saleable Area"]

     # --- date parsing ---
    date_cols = [
        "Booking: Last Modified Date",
        "Milestone Due Till Date",
        "Milestone Due Date",
        "Booking Date As Per The SBTR",
        "Booking Date",
        "Last Action Attempt Date",
        "Handover Date",
        "10 % Collected Date as per GL Date",
        "10 % Collected Date as Per Receipt Date",
        "20 % Collected Date as per GL Date",
        "20 % Collected Date as per Receipt Date",
        "Date of Title Transfer Completion",
    ]
    for c in date_cols:
        if c in df.columns:
            df[c] = safe_to_datetime(df[c])

    have = [c for c in date_cols if c in df.columns]
    df["__ordering_date"] = df[have].max(axis=1) if have else pd.NaT

    # --- stage/title/handover features ---
    df["__stage_score"] = df.get("Stage of Booking", pd.Series("", index=df.index)).map(_stage_score)
    df["__title_transfer_int"] = df.get("Title Transfer", pd.Series("", index=df.index)).map(_yes_no_to_int)
    df["__handover_present"] = df.get("Handover Date", pd.Series(pd.NaT, index=df.index)).notna().astype(int)

    # --- unit id ---
    df["__unit_key"] = df.apply(_unit_key_row, axis=1)
    df["unit_master_id"] = df["__unit_key"].map(_make_unit_id)

    # --- score tuples for canonical booking ---
    def _score_tuple(r: pd.Series):
        return (
            r.get("__title_transfer_int", 0),
            r.get("__handover_present", 0),
            r.get("__stage_score", 0),
            r.get("__ordering_date", pd.Timestamp.min),
        )

    df["__score_tuple"] = df.apply(_score_tuple, axis=1)
    idx = df.groupby("unit_master_id")["__score_tuple"].idxmax() if len(df) else pd.Series(dtype="int64")
    df["is_canonical_booking"] = False
    if len(idx):
        df.loc[idx, "is_canonical_booking"] = True

    # --- ensure needed columns exist to avoid KeyErrors ---
    finance_cols_sum = [
        "Total Receipts Amount",
        "Total Receipt Amount",
        "Total Amount Paid ( Unit Price )",
        "Advance Payment",
        "Advance Payments",
        "Token Amount",
        "Amount On Account",
        "Paid Amount For Process",
    ]
    finance_cols_latest = [
        "Balance",   # << added
        "Total Milestone Due Till Today",  # << added
        "Total Amount Balance",
        "Total Payment Due",
        "Overdue %",
        "Overdue Percentage Till Past 30 Days %",
        "Due Payments This Month",
        "Shortfall Amount for DLD",
        "DLD Amount",
    ]

    finance_cols_mode = ["Payment Plan", "Mode of Funding", "Bank Name (Flat Cost)", "IBAN (Flat Cost)"]

    needed = (
        finance_cols_sum
        + finance_cols_latest
        + finance_cols_mode
        + [
            "Project Name",
            "Tower",
            "Tower Name",
            "Unit Name",
            "Unit",
            "Booking: Booking Name",
            "Stage of Booking",
            "Current Status",
            "Title Transfer",
            "Handover Date",
            "Booking Date",
            "Booking Date As Per The SBTR",
            "Booking: Last Modified Date",
            "Purchase Price",
            "Flat Typology",
            "Saleable Area",
            "PSF Rate",
            "Green Channel",
            "Nationality",
        ]
    )
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    # --- aggregations ---
    sum_df = df.groupby("unit_master_id")[finance_cols_sum].sum(min_count=1).reset_index()

    # latest (by ordering date; fallback to last index if all NaT)
    latest_idx = []
    for uid, g in df.groupby("unit_master_id"):
        g2 = g.dropna(subset=["__ordering_date"])
        latest_idx.append(g2["__ordering_date"].idxmax() if len(g2) else g.index.max())
    latest_df = (
        df.loc[latest_idx, ["unit_master_id"] + finance_cols_latest].reset_index(drop=True)
        if len(df)
        else pd.DataFrame(columns=["unit_master_id"] + finance_cols_latest)
    )

    # mode for categoricals
    def _mode_nonnull(s):
        s = s.dropna()
        return s.mode().iloc[0] if len(s) else np.nan

    mode_df = df.groupby("unit_master_id")[finance_cols_mode].agg(_mode_nonnull).reset_index()

    # canonical columns
    canon_cols = [
        "unit_master_id",
        "Project Name",
        "Tower",
        "Tower Name",
        "Unit Name",
        "Unit",
        "Booking: Booking Name",
        "Stage of Booking",
        "Current Status",
        "Title Transfer",
        "Handover Date",
        "Booking Date",
        "Booking Date As Per The SBTR",
        "Booking: Last Modified Date",
        "Purchase Price",
        "Flat Typology",
        "Saleable Area",
        "PSF Rate",
        "Green Channel",
        "Nationality",
    ]
    canon_df = df.loc[idx, canon_cols].copy() if len(idx) else pd.DataFrame(columns=canon_cols)
    canon_df.rename(
        columns={
            "Booking: Booking Name": "canonical_booking_name",
            "Stage of Booking": "canonical_stage",
            "Current Status": "canonical_status",
            "Booking Date": "canonical_booking_date",
            "Booking Date As Per The SBTR": "canonical_booking_date_sbtr",
            "Booking: Last Modified Date": "canonical_last_modified",
        },
        inplace=True,
    )

    # purchase price stats
    pp = (
        df.groupby("unit_master_id")["Purchase Price"]
        .agg(["max", "median"])
        .reset_index()
        .rename(columns={"max": "purchase_price_max", "median": "purchase_price_median"})
    )

    # merge to units_master
    units_master = (
        canon_df.merge(sum_df, on="unit_master_id", how="left")
        .merge(latest_df, on="unit_master_id", how="left")
        .merge(mode_df, on="unit_master_id", how="left")
        .merge(pp, on="unit_master_id", how="left")
        if len(df)
        else pd.DataFrame()
    )

    if len(units_master):
        units_master["tower_name"] = units_master["Tower"].fillna(units_master["Tower Name"])
        units_master["unit_name"] = units_master["Unit Name"].fillna(units_master["Unit"])

    # --- robust aggregated Area & PSF fill (ensures KPIs never show zero if data exists) ---
    if "Saleable Area" in df.columns:
        area_sum = (
            df.groupby("unit_master_id")["Saleable Area"].sum(min_count=1).rename("agg_area_sum").reset_index()
        )
    else:
        area_sum = pd.DataFrame(columns=["unit_master_id", "agg_area_sum"])

    if {"Saleable Area", "PSF Rate"}.issubset(df.columns):
        valid = df[["unit_master_id", "Saleable Area", "PSF Rate"]].copy()
        valid["__area_psf"] = valid["Saleable Area"] * valid["PSF Rate"]
        w = (
            valid.groupby("unit_master_id")["__area_psf"].sum(min_count=1)
            / valid.groupby("unit_master_id")["Saleable Area"].sum(min_count=1)
        )
        w = w.replace([np.inf, -np.inf], np.nan).rename("agg_psf_weighted").reset_index()
    else:
        w = pd.DataFrame(columns=["unit_master_id", "agg_psf_weighted"])

    if isinstance(units_master, pd.DataFrame) and len(units_master):
        units_master = units_master.merge(area_sum, on="unit_master_id", how="left").merge(
            w, on="unit_master_id", how="left"
        )
        if "Saleable Area" in units_master.columns:
            units_master["Saleable Area"] = units_master["Saleable Area"].fillna(
                units_master["agg_area_sum"]
            )
        if "PSF Rate" in units_master.columns:
            units_master["PSF Rate"] = units_master["PSF Rate"].fillna(
                units_master["agg_psf_weighted"]
            )
        units_master.drop(columns=["agg_area_sum", "agg_psf_weighted"], inplace=True, errors="ignore")

    # --- booking map ---
    if "Booking: Booking Name" in df.columns:
        unit_bookings_map = (
            df[["unit_master_id", "Booking: Booking Name", "is_canonical_booking"]]
            .dropna(subset=["unit_master_id"])
            .drop_duplicates()
            .rename(columns={"Booking: Booking Name": "booking_name"})
        )
    else:
        unit_bookings_map = pd.DataFrame(columns=["unit_master_id", "booking_name", "is_canonical_booking"])

    return df, units_master, unit_bookings_map
