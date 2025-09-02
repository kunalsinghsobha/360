# customer360/resolution.py
from __future__ import annotations
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from .utils import norm_email, norm_phone, last7, norm_name

ACC = "Opportunity: Account ID"
EMAIL = "Primary Applicant Email"
PHONE = "Primary Mobile Number"


def fallback_customer_id(df: pd.DataFrame) -> pd.Series:
    """Deterministic fallback: ACC → EML → PHN → SYN."""
    acc = df.get(ACC, pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    eml = df.get(EMAIL, pd.Series("", index=df.index)).astype(str)
    phn = df.get(PHONE, pd.Series("", index=df.index)).astype(str)

    # normalize
    eml = eml.map(norm_email)
    phn = phn.map(norm_phone)

    cid = acc.where(acc != "", pd.NA)
    cid = cid.fillna("EML::" + eml).where(eml != "", pd.NA)
    cid = cid.fillna("PHN::" + phn).where(phn != "", pd.NA)
    cid = cid.fillna(pd.Series([f"SYN::{i}" for i in range(len(df))], index=df.index))
    return cid.astype(str)


# ---------------- Enhanced (fuzzy) helpers ----------------

def _coalesce_str(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """
    Return the first non-empty string across the provided columns, row-wise.
    Always returns a pandas Series (never a numpy array).
    Empty means '' after strip.
    """
    out = pd.Series([""] * len(df), index=df.index, dtype="object")
    for c in cols:
        if c in df.columns:
            s = df[c].astype(str)
            out = out.where(out.str.strip() != "", s)
    return out


def _prepare_for_resolution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build normalized columns used for fuzzy matching.
    Uses Series.where (not np.where) to avoid numpy arrays (which lack .str).
    """
    out = df.copy()

    acc = _coalesce_str(df, [ACC]).str.strip().str.lower()
    eml = _coalesce_str(df, [EMAIL, "Future Correspondence Email"])
    phn = _coalesce_str(df, [PHONE, "Future Correspondence Phone", "Future Correspondence Contact"])
    nam = _coalesce_str(df, ["Primary Applicant Name", "Future Correspondence Name", "Opportunity: Account Name"])
    nat = _coalesce_str(df, ["Nationality"]).str.upper()

    out["_acc_norm"] = acc
    out["_email_norm"] = eml.map(norm_email)
    out["_phone_norm"] = phn.map(norm_phone)
    out["_name_norm"] = nam.map(norm_name)
    out["_nat_norm"] = nat
    out["_phone_last7"] = out["_phone_norm"].map(last7)

    def _domain(x: str) -> str:
        x = str(x)
        return x.split("@", 1)[1] if "@" in x else ""

    out["_email_domain"] = out["_email_norm"].map(_domain)
    return out


def _exact(a: str, b: str) -> float:
    return 1.0 if a and b and a == b else 0.0


def _name_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _phone_sim(p1: str, p2: str) -> float:
    if not p1 or not p2:
        return 0.0
    if p1 == p2:
        return 1.0
    return 0.7 if last7(p1) and last7(p1) == last7(p2) else 0.0


def _score_pair(a: pd.Series, b: pd.Series) -> float:
    # Account ID exact match overrides
    if _exact(a["_acc_norm"], b["_acc_norm"]) == 1.0 and a["_acc_norm"]:
        return 1.0

    s_email = _exact(a["_email_norm"], b["_email_norm"])
    if s_email == 0 and a["_email_domain"] and a["_email_domain"] == b["_email_domain"]:
        la = a["_email_norm"].split("@")[0] if "@" in a["_email_norm"] else a["_email_norm"]
        lb = b["_email_norm"].split("@")[0] if "@" in b["_email_norm"] else b["_email_norm"]
        if SequenceMatcher(None, la, lb).ratio() >= 0.9:
            s_email = 0.8

    s_phone = _phone_sim(a["_phone_norm"], b["_phone_norm"])
    s_name = _name_sim(a["_name_norm"], b["_name_norm"])
    s_nat = _exact(a["_nat_norm"], b["_nat_norm"])

    score = (
        0.55 * _exact(a["_acc_norm"], b["_acc_norm"])
        + 0.20 * s_email
        + 0.15 * s_phone
        + 0.08 * s_name
        + 0.02 * s_nat
    )
    strong = (s_email >= 0.8) + (s_phone >= 0.7) + (s_name >= 0.92)
    if score < 0.85 and strong >= 2:
        return 0.86
    return score


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def _build_blocks_pos(X: pd.DataFrame):
    blocks = defaultdict(list)
    for i, r in X.reset_index(drop=True).iterrows():
        added = False
        if r["_acc_norm"]:
            blocks[("ACC", r["_acc_norm"])].append(i)
            added = True
        if r["_email_norm"]:
            blocks[("EML", r["_email_norm"])].append(i)
            added = True
        if r["_phone_last7"]:
            blocks[("P7", r["_phone_last7"])].append(i)
            added = True
        if r["_email_domain"] and r["_nat_norm"]:
            blocks[("DOMNAT", (r["_email_domain"], r["_nat_norm"]))].append(i)
            added = True
        if not added:
            blocks[("ROW", i)].append(i)
    return blocks


def resolve_customers(df: pd.DataFrame, auto_thr: float = 0.85) -> pd.Series:
    """
    Enhanced (fuzzy) resolver with union-find; switches to deterministic if >50k rows.
    Returns a Series of stable customer_master_id strings.
    """
    X = _prepare_for_resolution(df).reset_index(drop=True)
    n = len(X)
    if n == 0:
        return pd.Series([], dtype=str)

    # For very large datasets, fall back to deterministic to keep performance acceptable
    if n > 50_000:
        fb = X["_acc_norm"].where(X["_acc_norm"] != "", pd.NA)
        fb = fb.fillna("EML::" + X["_email_norm"]).where(X["_email_norm"] != "", fb)
        fb = fb.fillna("PHN::" + X["_phone_norm"]).where(X["_phone_norm"] != "", fb)
        return fb.fillna(pd.Series([f"SYN::{i}" for i in range(n)])).astype(str)

    uf = _UF(n)
    blocks = _build_blocks_pos(X)
    MAX_BLOCK = 500  # avoid O(n^2) explosions

    for _, idxs in blocks.items():
        if len(idxs) <= 1 or len(idxs) > MAX_BLOCK:
            continue
        L = list(idxs)
        for ii in range(len(L)):
            a = X.iloc[L[ii]]
            for jj in range(ii + 1, len(L)):
                b = X.iloc[L[jj]]
                if _score_pair(a, b) >= auto_thr:
                    uf.union(L[ii], L[jj])

    roots = [uf.find(i) for i in range(n)]
    groups = defaultdict(list)
    for pos, r in enumerate(roots):
        groups[r].append(pos)

    master = {}
    for root, pos_list in groups.items():
        grp = X.iloc[pos_list]
        acc = grp["_acc_norm"].replace("", pd.NA).dropna()
        eml = grp["_email_norm"].replace("", pd.NA).dropna()
        phn = grp["_phone_norm"].replace("", pd.NA).dropna()
        if len(acc):
            mid = f"ACC::{acc.mode().iloc[0]}"
        elif len(eml):
            mid = f"EML::{eml.mode().iloc[0]}"
        elif len(phn):
            mid = f"PHN::{phn.mode().iloc[0]}"
        else:
            mid = f"SYN::{root}"
        master[root] = mid

    return pd.Series([master[roots[i]] for i in range(n)], index=X.index)
