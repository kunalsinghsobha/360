# streamlit_app.py — Customer 360 (Upgraded, Clean)
from __future__ import annotations
import os, sys, re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Reduce font size of KPI metrics
st.markdown(
    """
    <style>
    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 20px !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Ensure local package is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Package imports
from customer360.utils import (
    finance_flags,
    pretty_money,
    prededupe_name_booking_unit,
    date_panel,
    iqr_trim,
    validate_data,
    tag_owner_rank,
)
from customer360.units import build_units
from customer360.resolution import resolve_customers, fallback_customer_id
from customer360.scoring import compute_customer_rollups, ai_like_summary


# ---------------- Streamlit page ----------------
st.set_page_config(page_title="Customer 360 — Upgraded", layout="wide")
st.title("Customer 360 — Upgraded Build")
st.caption("Upload your Salesforce CSV/XLSX. Toggle **Enhanced** for fuzzy de-duplication; **Fast** uses deterministic keys.")


with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Salesforce report (.csv / .xlsx)", type=["csv", "xlsx", "xls"])

    st.header("Deduplication")
    use_enhanced = st.toggle(
        "Enhanced de-duplication (fuzzy)",
        value=True,
        help="Turn off for Fast mode (faster on very large files)."
    )
    st.caption("Enhanced uses name/email/phone similarity; Fast uses Account ID → Email → Phone.")

    st.header("Performance")
    st.write("Enhanced auto-disables for files > 50k rows to keep the app snappy.")


# ---------------- Robust loader ----------------
def _read_excel(file):
    try:
        return pd.read_excel(file, engine="openpyxl")
    except Exception:
        return pd.read_excel(file)


def load_file(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    name = getattr(file, "name", "") or ""
    ext = os.path.splitext(name)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = _read_excel(file)
        df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
        return df.dropna(how="all").reset_index(drop=True)

    try:
        df = pd.read_csv(file, encoding="utf-8-sig")
    except Exception:
        try: file.seek(0)
        except Exception: pass
        try:
            df = pd.read_csv(file, encoding="latin-1")
        except Exception:
            try: file.seek(0)
            except Exception: pass
            df = pd.read_csv(file, encoding="utf-8-sig", sep=None, engine="python")

    if df.shape[1] <= 1:
        try: file.seek(0)
        except Exception: pass
        df = pd.read_csv(file, encoding="latin-1", sep=None, engine="python")

    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df.dropna(how="all").reset_index(drop=True)


# ---------------- KPIs ----------------
def topline_metrics(df_units: pd.DataFrame, units_master: pd.DataFrame, customers: pd.DataFrame):
    """
    Topline KPIs (strict, no duplication).
    Formulas:
      1. Total Purchase Price         = Σ "Purchase Price"
      2. Total Anticipated Investment = Σ "Balance"
      3. Total Revenue                = Σ "Purchase Price" – Σ "Balance"
      4. Collection %                 = (Total Revenue ÷ Total Purchase Price) × 100
      5. Avg PSF Rate                 = (Σ "Purchase Price") ÷ (Σ "Saleable Area")
      6. Total Saleable Area          = Σ "Saleable Area"
      7. Number of Buyers             = count(customers)
      8. Total Units                  = count(unique unit_master_id)
    """

    def num(series):
        return pd.to_numeric(series, errors="coerce").fillna(0) if series is not None else pd.Series(dtype=float)

    if units_master is None or not len(units_master):
        return {
            "Total Purchase Price": "AED 0",
            "Total Anticipated Investment": "AED 0",
            "Total Revenue": "AED 0",
            "Collection %": "0%",
            "Avg PSF Rate": "AED 0",
            "Total Saleable Area": "0 sq.ft",
            "Number of Buyers": "0",
            "Total Units": "0",
        }

    # ---- Finance KPIs ----
    purchase = num(units_master.get("Purchase Price"))
    balance  = num(units_master.get("Balance"))

    total_purchase = float(purchase.sum())
    total_balance  = float(balance.sum())
    total_revenue  = total_purchase - total_balance
    pct_collected  = (total_revenue / total_purchase * 100) if total_purchase > 0 else 0

    # ---- Area / PSF ----
    area = num(units_master.get("Saleable Area"))
    total_area = float(area.sum())
    avg_psf = (total_purchase / total_area) if total_area > 0 else 0

    # ---- Counts ----
    n_units  = units_master["unit_master_id"].nunique() if "unit_master_id" in units_master.columns else len(units_master)
    n_buyers = len(customers) if customers is not None else 0

    return {
        "Total Purchase Price": pretty_money(total_purchase),
        "Total Anticipated Investment": pretty_money(total_balance),
        "Total Revenue": pretty_money(total_revenue),
        "Collection %": f"{pct_collected:.1f}%",
        "Avg PSF Rate": f"AED {avg_psf:,.0f}",
        "Total Saleable Area": f"{total_area:,.0f} sq.ft",
        "Number of Buyers": f"{n_buyers:,}",
        "Total Units": f"{n_units:,}",
    }


# ---------------- Orchestration ----------------
def prepare_data(df: pd.DataFrame, enhanced: bool):
    df = df.copy()
    rowcount = len(df)
    if enhanced and rowcount > 50_000: enhanced = False

    with st.status("Processing…", expanded=True) as s:
        s.write("Step 0/4: Pre-dedupe (Name → Booking → Unit)")
        df = prededupe_name_booking_unit(df)

        s.write(f"Step 1/4: Resolving customers {'(enhanced)' if enhanced else '(fast)'}")
        df["customer_master_id"] = resolve_customers(df) if enhanced else fallback_customer_id(df)

        s.write("Step 2/4: Freezing units & consolidating finance")
        df_units, units_master, unit_bookings_map = build_units(df)

        s.write("Step 3/4: Computing customer rollups & aliases")
        dnc_series = df_units.get("Do Not Contact", pd.Series(False, index=df_units.index))
        dnc = (df_units.assign(dnc=dnc_series.astype(str).str.lower().isin(["1","true","yes","y"]))
                      .groupby("customer_master_id")["dnc"].max().rename("do_not_contact_flag")).reset_index()

        customers = compute_customer_rollups(df_units, units_master)
        customers = customers.merge(dnc, on="customer_master_id", how="left").fillna({"do_not_contact_flag": 0})

        candidate_name_cols = ["Primary Applicant Name","Applicant Name","Name","Customer Name","Primary Applicant"]
        name_col = next((c for c in candidate_name_cols if c in df_units.columns), None)
        if name_col:
            display = (df_units.groupby("customer_master_id")[name_col]
                       .agg(lambda s: s.dropna().value_counts().index[0] if len(s.dropna()) else None)
                       .rename("display_name").reset_index())
        else:
            eml = df_units.get("Primary Applicant Email", pd.Series("", index=df_units.index)).astype(str)
            alias_series = (eml.str.split("@").str[0].str.replace(r"[._\-]+"," ", regex=True).str.title())
            phn = df_units.get("Primary Mobile Number", pd.Series("", index=df_units.index)).astype(str)
            masked = pd.Series(phn).str.replace(r".*(\d{4})$", r"****\\1", regex=True)
            alias_series = alias_series.where(alias_series.str.strip() != "", masked)
            display = (pd.DataFrame({"customer_master_id": df_units["customer_master_id"], "display_name": alias_series})
                       .dropna().drop_duplicates())

        customers = customers.merge(display, on="customer_master_id", how="left")
        customers["display_name"] = customers["display_name"].fillna("Customer")

        # --- Merge core contact / identity fields into customers ---
        contact_fields = [
            "Primary Applicant Name",
            "Primary Applicant Email",
            "Primary Mobile Number",
            "Nationality",
            "Unit Status",
            "Unit Name",
            "Unit",
            "Payment Plan",
            "Green Channel",
            "Bank Name (Flat Cost)",
            "IBAN (Flat Cost)",
            "Mode of Funding",
        ]

        contact_df = (
            df_units.groupby("customer_master_id")[contact_fields]
            .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else None)
            .reset_index()
        )

        customers = customers.merge(contact_df, on="customer_master_id", how="left")

        s.update(label="Done", state="complete")
        return df, df_units, units_master, unit_bookings_map, customers, enhanced, rowcount


# ---------------- Filters (form + apply) ----------------
def apply_filters(df_units: pd.DataFrame, units_master: pd.DataFrame, customers: pd.DataFrame):
    def _opts(col):
        if col in df_units.columns:
            vals = df_units[col].dropna().astype(str)
            vals = [x for x in vals.unique().tolist() if x.strip()]
            return sorted(vals)
        return []

    def _within_range(val, lo, hi):
        try:
            if pd.isna(val): return True
            return lo <= val <= hi
        except Exception: return True

    def _expand_range(lo, hi):
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi): hi = lo
        if hi <= lo: lo, hi = lo-1, hi+1
        return float(lo), float(hi)

    um_pp_series = pd.to_numeric(units_master.get("purchase_price_max"), errors="coerce") if len(units_master) else pd.Series(dtype=float)
    inv_lo = float(um_pp_series.min(skipna=True)) if um_pp_series.notna().any() else 0.0
    inv_hi = float(um_pp_series.max(skipna=True)) if um_pp_series.notna().any() else 0.0
    inv_lo, inv_hi = _expand_range(inv_lo, inv_hi)

    pp_series_all = pd.to_numeric(df_units.get("Purchase Price"), errors="coerce")
    pp_lo = float(pp_series_all.min(skipna=True)) if pp_series_all.notna().any() else 0.0
    pp_hi = float(pp_series_all.max(skipna=True)) if pp_series_all.notna().any() else 0.0
    pp_lo, pp_hi = _expand_range(pp_lo, pp_hi)

    bmin = pd.to_datetime(df_units.get("Booking Date"), errors="coerce").min()
    bmax = pd.to_datetime(df_units.get("Booking Date"), errors="coerce").max()

    SEARCHABLE_FIELDS = [
        "Primary Applicant Name","Primary Applicant Email","Primary Mobile Number",
        "Opportunity: Account ID","Opportunity: Account Name","Booking: Booking Name",
        "Project Name","Flat Typology","Nationality","Tower","Tower Name","Unit Name","Unit"
    ]
    existing_search_fields = [c for c in SEARCHABLE_FIELDS if c in df_units.columns]

    with st.sidebar:
        st.header("Filters")
        with st.form("filters_form", clear_on_submit=False):
            f_unit_status = st.multiselect("Unit Status", _opts("Current Status"))
            f_stage       = st.multiselect("Stage of Booking", _opts("Stage of Booking"))
            f_project     = st.multiselect("Project Name", _opts("Project Name"))
            f_typology    = st.multiselect("Flat Typology", _opts("Flat Typology"))
            f_nat         = st.multiselect("Nationality", _opts("Nationality"))
            f_green       = st.multiselect("Green Channel", _opts("Green Channel"))
            f_tt          = st.multiselect("Title Transfer", _opts("Title Transfer"))

            br = st.date_input("Booking Date range",
                               value=(bmin.date() if pd.notna(bmin) else None,
                                      bmax.date() if pd.notna(bmax) else None))

            f_unit_inv = st.slider("Investment per unit (Purchase Price Max, AED)", inv_lo, inv_hi, (inv_lo, inv_hi))
            f_purchase = st.slider("Purchase Price (AED)", pp_lo, pp_hi, (pp_lo, pp_hi))

            st.subheader("Find people")
            q_text = st.text_input("Search text (name, email, phone, etc.)", "")
            q_fields = st.multiselect("Search fields", existing_search_fields, default=existing_search_fields[:5])

            applied = st.form_submit_button("Apply filters")

        if st.button("Clear filters"):
            st.session_state.pop("active_filters", None)
            st.experimental_rerun()

    if applied:
        st.session_state["active_filters"] = {
            "f_unit_status": f_unit_status, "f_stage": f_stage, "f_project": f_project,
            "f_typology": f_typology, "f_nat": f_nat, "f_green": f_green, "f_tt": f_tt,
            "br": br, "f_unit_inv": f_unit_inv, "f_purchase": f_purchase,
            "q_text": q_text, "q_fields": q_fields
        }

    state = st.session_state.get("active_filters", {
        "f_unit_status":[],"f_stage":[],"f_project":[],"f_typology":[],"f_nat":[],"f_green":[],"f_tt":[],
        "br": (bmin.date() if pd.notna(bmin) else None, bmax.date() if pd.notna(bmax) else None),
        "f_unit_inv": (inv_lo, inv_hi),"f_purchase": (pp_lo, pp_hi),
        "q_text": "","q_fields": existing_search_fields[:5]
    })

    def _isin(col, choices):
        if not choices: return pd.Series(True, index=df_units.index)
        return df_units.get(col, pd.Series(index=df_units.index)).astype(str).isin(choices)

    mask = pd.Series(True, index=df_units.index)
    mask &= _isin("Current Status", state["f_unit_status"])
    mask &= _isin("Stage of Booking", state["f_stage"])
    mask &= _isin("Project Name", state["f_project"])
    mask &= _isin("Flat Typology", state["f_typology"])
    mask &= _isin("Nationality", state["f_nat"])
    mask &= _isin("Green Channel", state["f_green"])
    mask &= _isin("Title Transfer", state["f_tt"])

    br_val = state["br"]
    if isinstance(br_val,(list,tuple)) and len(br_val)==2 and all(br_val):
        start,end = [pd.Timestamp(d) for d in br_val]
        bd = pd.to_datetime(df_units.get("Booking Date"), errors="coerce")
        mask &= (bd>=start) & (bd<=end)

    if len(units_master) and "unit_master_id" in df_units.columns:
        um_map = units_master.set_index("unit_master_id")["purchase_price_max"]
        lo,hi = state["f_unit_inv"]
        mask &= df_units["unit_master_id"].map(lambda x: _within_range(um_map.get(x,np.nan), lo,hi))

    lo2,hi2 = state["f_purchase"]
    pp_series = pd.to_numeric(df_units.get("Purchase Price"), errors="coerce")
    mask &= pp_series.map(lambda v: _within_range(v, lo2,hi2))

    q_text = (state["q_text"] or "").strip().lower()
    if q_text and state["q_fields"]:
        combo = pd.Series("", index=df_units.index, dtype="object")
        for c in state["q_fields"]:
            if c in df_units.columns:
                combo = (combo+" "+df_units[c].astype(str).str.lower().fillna(""))
        for tok in re.split(r"\s+", q_text):
            if tok: mask &= combo.str.contains(re.escape(tok), na=False)

    sel_uids = df_units.loc[mask,"unit_master_id"].dropna().astype(str).unique().tolist() if "unit_master_id" in df_units.columns else []
    units_master_f = units_master[units_master["unit_master_id"].astype(str).isin(sel_uids)].copy() if len(units_master) else units_master

    sel_cids = df_units.loc[mask,"customer_master_id"].dropna().astype(str).unique().tolist() if "customer_master_id" in df_units.columns else []
    customers_f = customers[customers["customer_master_id"].astype(str).isin(sel_cids)].copy() if customers is not None else customers

    return mask, units_master_f, customers_f, state


# ---------------- Main ----------------
if uploaded is None:
    st.info("Please upload your Salesforce CSV/XLSX to proceed.")
    st.stop()

df_raw = load_file(uploaded)
df_raw = finance_flags(df_raw)

with st.expander("🔎 Debug: Input snapshot", expanded=False):
    st.write("Shape:", df_raw.shape)
    st.write("Columns:", list(df_raw.columns)[:50], "…")
    st.dataframe(df_raw.head(10), use_container_width=True)

df_raw, df_units, units_master, unit_bookings_map, customers, enhanced_used, nrows = prepare_data(df_raw, use_enhanced)
st.toast(f"Loaded {nrows:,} rows • Mode: {'Enhanced' if enhanced_used else 'Fast'}", icon="✅")

if "customer_master_id" not in df_units.columns or df_units["customer_master_id"].isna().all():
    fb = fallback_customer_id(df_raw).astype(str)
    df_units = df_units.copy()
    df_units["customer_master_id"] = fb if len(fb)==len(df_units) else "SYN::"+df_units.index.astype(str)

# --- Apply filters
mask, units_master_f, customers_f, _fstate = apply_filters(df_units, units_master, customers)

# --- Topline KPIs
st.markdown("### Topline KPIs (filtered)")
KPI = topline_metrics(df_units.loc[mask], units_master_f, customers_f)
k1,k2,k3,k4 = st.columns(4); k5,k6,k7,k8 = st.columns(4)
k1.metric("Total Purchase Price", KPI["Total Purchase Price"])
k2.metric("Avg PSF Rate", KPI["Avg PSF Rate"])
k3.metric("Total Saleable Area", KPI["Total Saleable Area"])
k4.metric("Number of Buyers", KPI["Number of Buyers"])
k5.metric("Total Units", KPI["Total Units"])
k6.metric("Total Anticipated Investment", KPI["Total Anticipated Investment"])
k7.metric("Total Revenue", KPI["Total Revenue"])
k8.metric("Collection %", KPI["Collection %"])
st.markdown("---")

# --- Secondary KPIs
# ========= Distribution Charts =========
st.markdown("### 📊 Customer Distributions (filtered)")

col1, col2, col3 = st.columns(3)

# Work only on filtered subset
df_units_f = df_units.loc[mask].copy()

# 1. Flat Typology Distribution (Pie chart, top 10 + Others)
with col1:
    if "Flat Typology" in df_units_f.columns:
        typ_counts = df_units_f["Flat Typology"].fillna("Unknown").value_counts()
        top10 = typ_counts.nlargest(10)
        others = typ_counts.iloc[10:].sum()
        if others > 0:
            top10["Others"] = others
        fig1 = px.pie(values=top10.values, names=top10.index,
                      title="Distribution by Flat Typology (Top 10)")
        st.plotly_chart(fig1, use_container_width=True)

# 2. Stage of Booking (Histogram)
with col2:
    if "Stage of Booking" in df_units_f.columns:
        stage_counts = df_units_f["Stage of Booking"].fillna("Unknown").value_counts()
        fig2 = px.bar(x=stage_counts.index, y=stage_counts.values,
                      title="Stage of Booking Distribution")
        fig2.update_layout(xaxis_title="Stage of Booking", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

# 3. Top 10 Nationalities by Typology
with col3:
    if {"Nationality", "Flat Typology"}.issubset(df_units_f.columns):
        nat_typ_counts = (
            df_units_f.groupby(["Nationality", "Flat Typology"]).size()
            .reset_index(name="count")
        )
        nat_totals = nat_typ_counts.groupby("Nationality")["count"].sum().nlargest(10).index
        nat_typ_counts = nat_typ_counts[nat_typ_counts["Nationality"].isin(nat_totals)]
        fig3 = px.bar(
            nat_typ_counts, x="Nationality", y="count", color="Flat Typology",
            title="Top 10 Nationalities by Typology (Filtered)", barmode="stack"
        )
        st.plotly_chart(fig3, use_container_width=True)


# --- Customers + details
left, right = st.columns([0.45, 0.55])
with left:
    st.subheader("Customers")
    if not len(customers_f):
        st.info("No customers in current filter.")
        sel_name = None
    else:
        show_cols = [
            "display_name", "customer_value_score", "n_units",
            "total_investment", "total_paid", "balance_latest",
            "overdue_pct", "days_inactive"
        ]
        for c in show_cols:
            if c not in customers_f.columns:
                customers_f[c] = np.nan
        table = customers_f[show_cols].sort_values(
            ["customer_value_score", "total_investment"],
            ascending=[False, False]
        ).copy()
        table = table.rename(columns={
            "display_name": "Name",
            "customer_value_score": "Score",
            "n_units": "Units",
            "total_investment": "Investment (AED)",
            "total_paid": "Paid (AED)",
            "balance_latest": "Balance (AED)",
            "overdue_pct": "Overdue %",
            "days_inactive": "Days Inactive"
        })
        st.dataframe(table, use_container_width=True, height=440)
        sel_name = st.selectbox("Select a customer to open details",
                                customers_f["display_name"].tolist())

with right:
    if len(customers_f) and sel_name:
        row = customers_f.loc[customers_f["display_name"] == sel_name].iloc[0]
        st.subheader(f"Customer Details — {row['display_name']}")
        st.caption(f"ID: {row['customer_master_id']}")
        st.write(ai_like_summary(row))

        have_cols = [
            "customer_master_id", "unit_master_id", "Project Name", "Tower",
            "Unit Name", "Unit", "Flat Typology", "Saleable Area", "PSF Rate",
            "Title Transfer", "Handover Date", "Current Status", "Stage of Booking",
            "Total Receipts Amount", "Total Amount Balance", "Balance",
            "Booking Date As Per The SBTR", "Booking Date",
            "10 % Collected Date as per GL Date", "10 % Collected Date as Per Receipt Date"
        ]
        have_cols = [c for c in have_cols if c in df_units.columns]

        cu = df_units.loc[
            mask & (df_units["customer_master_id"] == row["customer_master_id"]),
            have_cols
        ].copy()

        if len(cu):
            cu = tag_owner_rank(cu)
            for c in ["Booking Date As Per The SBTR", "Booking Date"]:
                if c in cu.columns:
                    cu[c] = pd.to_datetime(cu[c], errors="coerce")
            cu["first_seen"] = cu[["Booking Date As Per The SBTR", "Booking Date"]].min(axis=1)

            # --- Unit details aggregation (robust, no KeyErrors) ---
            agg = (
                cu.groupby("unit_master_id")
                  .agg(
                      project=("Project Name", "first"),
                      tower=("Tower", "first"),
                      unit=("Unit Name", lambda s: s.dropna().iloc[0] if len(s.dropna()) else None),
                      typology=("Flat Typology", "first"),
                      stage=("Stage of Booking", "last"),
                      status=("Current Status", "last"),
                      balance=("Balance", "last") if "Balance" in cu.columns
                              else ("Total Amount Balance", "last"),
                  )
                  .reset_index()
            )

            # Merge canonical finance info from units_master
            if "unit_master_id" in units_master.columns:
                agg = agg.merge(
                    units_master[["unit_master_id", "purchase_price_max", "Total Amount Balance"]],
                    on="unit_master_id",
                    how="left"
                )
                agg["purchase_price"] = agg.get("purchase_price", agg.get("purchase_price_max"))
                agg["balance"] = agg["balance"].fillna(agg["Total Amount Balance"])
                agg.drop(columns=["purchase_price_max", "Total Amount Balance"],
                         inplace=True, errors="ignore")

            # Render nicely
            for _, u in agg.iterrows():
                st.info(
                    f"**{u.get('project','')} • {u.get('tower','')} • {u.get('unit','')}**  \n"
                    f"Type: {u.get('typology','')} | Stage: {u.get('stage','')} | Status: {u.get('status','')}  \n"
                    f"Purchase Price: AED {float(u.get('purchase_price',0) or 0):,.0f} | "
                    f"Balance: AED {float(u.get('balance',0) or 0):,.0f}"
                )
        else:
            st.info("No owned units in the current filter.")

# --- Validation
with st.expander("🧪 Validation Suite"):
    issues=validate_data(df_raw)
    if len(issues):
        st.warning("Potential data issues detected:"); st.dataframe(issues,use_container_width=True,height=240)
    else: st.success("No issues detected by the automated checks.")
