# customer360/utils.py
from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import List

import numpy as np
import pandas as pd


# ---------- Basic normalizers ----------
def nz(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def clean_spaces(x: str) -> str:
    return re.sub(r"\s+", " ", nz(x).replace("\u00a0", " ")).strip()


def norm_email(x: str) -> str:
    x = nz(x).lower()
    if "@" not in x:
        return x
    local, domain = x.split("@", 1)
    local = re.sub(r"\+.*$", "", local)
    local = local.replace(".", "")
    return f"{local}@{domain}"


def norm_phone(x) -> str:
    return re.sub(r"\D", "", nz(x))


def last7(x) -> str:
    s = re.sub(r"\D", "", nz(x))
    return s[-7:] if s else ""


def norm_name(x: str) -> str:
    x = nz(x).lower()
    x = re.sub(r"[^\w\s]", "", x)
    x = re.sub(r"\b(mr|mrs|ms|dr)\b\.?", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def pct_rank(series: pd.Series):
    if series is None or len(series) == 0:
        return pd.Series([], dtype=float)
    return series.rank(pct=True, method="max")


def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce")


# ---------- Finance enrichment & helpers ----------
def to_num(sv, drop_zero_for_avg: bool = False) -> pd.Series:
    v = pd.to_numeric(sv, errors="coerce")
    if drop_zero_for_avg:
        v = v.replace(0, np.nan)
    return v


def finance_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Adds numeric copies + area*psf check."""
    df = df.copy()
    df["paid_unit"] = to_num(df.get("Total Amount Paid ( Unit Price )"))
    df["balance"] = to_num(df.get("Total Amount Balance") if "Total Amount Balance" in df.columns else df.get("Balance"))
    df["milestone_tt"] = to_num(df.get("Total Milestone Due Till Today"))
    df["purchase"] = to_num(df.get("Purchase Price"))
    df["area"] = to_num(df.get("Saleable Area"))
    df["psf"] = to_num(df.get("PSF Rate"))
    df["area_psf"] = df["area"] * df["psf"]
    tol = 0.05
    df["price_vs_area_psf_diff"] = (df["purchase"] - df["area_psf"]).abs()
    df["flag_price_area_psf_mismatch"] = (df["purchase"] > 0) & (
        (df["price_vs_area_psf_diff"] / df["purchase"]) > tol
    )
    # numeric mirrors for common paid columns
    for c in [
        "Total Receipts Amount",
        "Total Receipt Amount",
        "Advance Payment",
        "Advance Payments",
        "Token Amount",
        "Amount On Account",
        "Paid Amount For Process",
    ]:
        if c in df.columns:
            df[c + "_num"] = to_num(df[c])
    return df


def pretty_money(x) -> str:
    try:
        x = float(x)
    except Exception:
        return "AED 0"
    if x >= 1e12:
        return f"AED {x/1e12:.1f} T"
    if x >= 1e9:
        return f"AED {x/1e9:.1f} B"
    if x >= 1e6:
        return f"AED {x/1e6:.1f} M"
    if x >= 1e3:
        return f"AED {x/1e3:.1f} K"
    return f"AED {x:,.0f}"


# ---------- Pre-dedupe: Name → Booking → Unit ----------
def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _name_key(x: str) -> str:
    x = clean_spaces(x).lower()
    x = re.sub(r"[^\w\s]", "", x)
    x = re.sub(r"\b(mr|mrs|ms|dr)\b\.?", "", x)
    return re.sub(r"\s+", " ", x).strip()


def _tower_norm(x: str) -> str:
    x = clean_spaces(x).lower()
    x = re.sub(r"\b(tower|twr|blk|block|bldg|building)\b\.?", "", x)
    x = re.sub(r"[^a-z0-9]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    parts = []
    for p in x.split():
        parts.append(str(int(p)) if p.isdigit() else p)
    return " ".join(parts)


def _unit_norm(x: str) -> str:
    x = clean_spaces(x).lower()
    letters = re.findall(r"[a-z]+", x)
    digits = re.findall(r"\d+", x)
    flat4 = digits[-1][-4:] if digits else ""
    prefix = letters[0] if letters else ""
    if flat4:
        try:
            return f"{prefix}{int(flat4)}"
        except Exception:
            return f"{prefix}{flat4}"
    return re.sub(r"[^a-z0-9]", "", x)


def _project_norm(x: str) -> str:
    return clean_spaces(x).lower()


def _unit_key_norm_row(r: pd.Series) -> str:
    proj = _project_norm(r.get("Project Name", ""))
    subp = _project_norm(r.get("Sub Project", ""))
    tw = _tower_norm(r.get("Tower", "") or r.get("Tower Name", ""))
    floor = nz(r.get("Floor", "")).strip().lower()
    un = _unit_norm(r.get("Unit Name", "") or r.get("Unit", ""))
    if proj or subp or tw or floor or un:
        return f"{proj}|{subp}|{tw}|{floor}|{un}"
    # unique fallback so rows never all merge
    bk = nz(r.get("Booking: Booking Name", ""))
    opp = nz(r.get("Opportunity: Opportunity Name", ""))
    acc = nz(r.get("Opportunity: Account ID", ""))
    fallback = bk or opp or acc or str(r.name)
    return f"__FALLBACK__|{fallback}|row{r.name}"


def prededupe_name_booking_unit(
    df: pd.DataFrame, name_thr: float = 0.92, unit_thr: float = 0.995
) -> pd.DataFrame:
    """
    Pre-dedupe in 3 passes: Name cluster → Booking → Unit cluster.
    Vectorized string handling (no scalar broadcasting), strict unit clustering.
    """
    if not len(df):
        return df
    df = df.copy()

    # --- Vectorized name keys & blocks ---
    if "Primary Applicant Name" in df.columns:
        df["__name_key"] = df["Primary Applicant Name"].astype(str).map(_name_key)
    else:
        df["__name_key"] = pd.Series([""] * len(df), index=df.index, dtype="object")
    df["__name_block"] = df["__name_key"].str[:2].fillna("")

    # --- Vectorized booking normalization ---
    if "Booking: Booking Name" in df.columns:
        df["__book_norm"] = df["Booking: Booking Name"].astype(str).map(clean_spaces).str.lower()
    else:
        df["__book_norm"] = pd.Series([""] * len(df), index=df.index, dtype="object")

    # --- Unit normalized text (row-wise because it combines multiple columns) ---
    df["__unit_text"] = df.apply(_unit_key_norm_row, axis=1).replace("", np.nan)

    # --- Modified date for "keep latest" ---
    df["__mod"] = safe_to_datetime(df.get("Booking: Last Modified Date"))

    keep_idx: List[int] = []

    # Group by small name blocks for speed
    for _, g in df.groupby("__name_block", dropna=False):
        idxs = list(g.index)
        centers: List[str] = []
        assigned: dict[int, int] = {}

        # Name clustering per block (fuzzy)
        for i in idxs:
            nk = df.at[i, "__name_key"]
            if not nk:
                assigned[i] = len(centers)
                centers.append(nk)
                continue
            placed = False
            for c_id, c in enumerate(centers):
                if _sim(nk, c) >= name_thr:
                    assigned[i] = c_id
                    placed = True
                    break
            if not placed:
                assigned[i] = len(centers)
                centers.append(nk)

        clusters = defaultdict(list)
        for i, c_id in assigned.items():
            clusters[c_id].append(i)

        for _, rows in clusters.items():
            sub = df.loc[rows].copy()

            # De-dupe booking within name cluster → keep latest modified per booking
            if "__book_norm" in sub:
                sub = (
                    sub.sort_values(["__book_norm", "__mod"], ascending=[True, False])
                    .drop_duplicates(subset=["__book_norm"], keep="first")
                )

            if len(sub) <= 1:
                keep_idx.extend(sub.index.tolist())
                continue

            # ---- STRICT unit clustering (prefer exact; only merge if extremely similar) ----
            rows2 = list(sub.index)
            u_centers: List[str] = []
            u_assign: dict[int, int] = {}

            for i in rows2:
                ut = nz(sub.at[i, "__unit_text"])
                if not ut:
                    # no unit key → treat each as its own cluster (avoid merge)
                    u_assign[i] = len(u_centers)
                    u_centers.append(f"__EMPTY__|{i}")
                    continue

                placed = False
                # exact match first
                for u_id, u_c in enumerate(u_centers):
                    if ut == u_c:
                        u_assign[i] = u_id
                        placed = True
                        break

                if not placed:
                    # only if *extremely* close consider same cluster
                    for u_id, u_c in enumerate(u_centers):
                        if _sim(ut, u_c) >= unit_thr:
                            u_assign[i] = u_id
                            placed = True
                            break

                if not placed:
                    u_assign[i] = len(u_centers)
                    u_centers.append(ut)

            # Within each unit-cluster keep the most recently modified row
            unit_groups = defaultdict(list)
            for i, u_id in u_assign.items():
                unit_groups[u_id].append(i)

            for _, rws in unit_groups.items():
                sub2 = sub.loc[rws]
                if sub2["__mod"].notna().any():
                    i_keep = sub2["__mod"].idxmax()
                else:
                    i_keep = sub2.index[0]
                keep_idx.append(i_keep)

    out = (
        df.loc[sorted(set(keep_idx))]
        .drop(columns=["__name_key", "__name_block", "__book_norm", "__unit_text", "__mod"], errors="ignore")
        .reset_index(drop=True)
    )
    return out


# ---------- Dates / Comms panel ----------
DATE_PANEL_COLS = [
    "Booking Date",
    "Booking: Last Modified Date",
    "10 % Collected Date as per GL Date",
    "10 % Collected Date as Per Receipt Date",
    "20 % Collected Date as per GL Date",
    "20 % Collected Date as Per Receipt Date",
    "Milestone Due Date",
]


def date_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for c in DATE_PANEL_COLS:
        if c in df.columns:
            out[c] = pd.to_datetime(df[c], errors="coerce")
    out["Do Not Contact"] = df.get("Do Not Contact")
    out["Payment Plan"] = df.get("Payment Plan")
    return out


def iqr_trim(series: pd.Series, k: float = 1.5) -> pd.Series:
    v = to_num(series, drop_zero_for_avg=True).dropna()
    if v.empty:
        return v
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    return v[(v >= low) & (v <= high)]


# ---------- Owner tagging ----------
def tag_owner_rank(df_units: pd.DataFrame) -> pd.DataFrame:
    """
    Assign owner rank per unit using earliest booking date.
    Robust to NaT dates and missing values — never casts NaN to int.
    """
    df = df_units.copy()

    # Build first_seen (SBTR date preferred, else Booking Date)
    if "Booking Date As Per The SBTR" in df.columns:
        df["__first_seen"] = pd.to_datetime(df["Booking Date As Per The SBTR"], errors="coerce")
    else:
        df["__first_seen"] = pd.to_datetime(df.get("Booking Date"), errors="coerce")

    # Keep unique (unit, customer) pairs
    base = (
        df[["unit_master_id", "customer_master_id", "__first_seen"]]
        .dropna(subset=["unit_master_id", "customer_master_id"])
        .drop_duplicates()
    )

    if base.empty:
        return df

    # Deterministic ordering: by unit, then date (NaT at end), then customer id
    base_sorted = base.sort_values(
        ["unit_master_id", "__first_seen", "customer_master_id"], kind="mergesort"
    )

    # Owner rank is just the 1-based position within each unit after sorting
    base_sorted["owner_rank"] = base_sorted.groupby("unit_master_id").cumcount() + 1

    # Ordinal labels
    def _ordinal(n):
        try:
            n = int(n)
        except Exception:
            return None
        return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")

    base_sorted["owner_label"] = base_sorted["owner_rank"].map(_ordinal)

    # Merge back onto the unit rows
    out = df.merge(
        base_sorted[["unit_master_id", "customer_master_id", "owner_rank", "owner_label"]],
        on=["unit_master_id", "customer_master_id"],
        how="left",
    )
    return out

# ---------- Validation ----------
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    issues = []
    eml = df.get("Primary Applicant Email")
    if eml is not None:
        bad_emails = (~eml.astype(str).str.contains("@", na=False)).sum()
        if bad_emails:
            issues.append(("Invalid emails", int(bad_emails)))

    ph = df.get("Primary Mobile Number")
    if isinstance(ph, pd.Series):
        digits = ph.astype(str).str.replace(r"\D", "", regex=True)
        bad_phone = (digits.str.len() < 7).sum()
        issues.append(("Suspicious phones (<7 digits)", int(bad_phone)))

    for c in [
        "Purchase Price",
        "Total Amount Paid ( Unit Price )",
        "Total Amount Balance",
        "Balance",
        "Total Receipts Amount",
        "Total Receipt Amount",
    ]:
        if c in df.columns:
            neg = (to_num(df[c]) < 0).sum()
            if neg:
                issues.append((f"Negative values in {c}", int(neg)))

    b = pd.to_datetime(df.get("Booking Date"), errors="coerce")
    m10 = pd.to_datetime(
        df.get("10 % Collected Date as per GL Date"), errors="coerce"
    )
    wrong = ((m10.notna()) & (b.notna()) & (m10 < b)).sum()
    if wrong:
        issues.append(("10% GL Date before Booking Date", int(wrong)))

    df2 = finance_flags(df)
    mismatch = int(df2["flag_price_area_psf_mismatch"].sum())
    if mismatch:
        issues.append(("Purchase ≠ Area×PSF (>|5%| diff)", mismatch))

    if "Booking: Booking Name" in df.columns:
        d = (
            df.assign(__nk=_name_key(df.get("Primary Applicant Name", "")))
            .groupby(["__nk", "Booking: Booking Name"])
            .size()
            .reset_index(name="n")
        )
        dup_rows = int(d[d["n"] > 1]["n"].sum())
        if dup_rows:
            issues.append(("Duplicate Booking Names per applicant", dup_rows))

    return pd.DataFrame(issues, columns=["Issue", "Count"])
