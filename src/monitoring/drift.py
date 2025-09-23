from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2
from typing import List, Tuple, Dict

DRIFT_ALPHA_DEFAULT = 0.05

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _align_cats(ref: pd.Series, prod: pd.Series) -> Tuple[pd.Index, np.ndarray, np.ndarray]:
    cats = pd.Index(sorted(set(ref.dropna().unique()) | set(prod.dropna().unique())))
    ref_counts = ref.value_counts().reindex(cats, fill_value=0).values
    prod_counts = prod.value_counts().reindex(cats, fill_value=0).values
    return cats, ref_counts, prod_counts

def chi_square_test(ref: pd.Series, prod: pd.Series) -> Tuple[float, float]:
    cats, ref_c, prod_c = _align_cats(ref, prod)
    ref_n, prod_n = ref_c.sum(), prod_c.sum()
    if ref_n == 0 or prod_n == 0 or len(cats) == 0:
        return np.nan, np.nan
    ref_props = ref_c / ref_n
    expected = ref_props * prod_n
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = ((prod_c - expected) ** 2 / np.where(expected == 0, 1, expected)).sum()
    dof = max(len(cats) - 1, 1)
    pval = 1 - chi2.cdf(stat, dof)
    return float(stat), float(pval)

def ks_test(ref: pd.Series, prod: pd.Series) -> Tuple[float, float]:
    a = ref.dropna().astype(float)
    b = prod.dropna().astype(float)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    stat, pval = ks_2samp(a, b, alternative="two-sided", mode="auto")
    return float(stat), float(pval)

def detect_drift(df_ref: pd.DataFrame, df_prod: pd.DataFrame, alpha: float = DRIFT_ALPHA_DEFAULT,
                 columns: List[str] | None = None) -> pd.DataFrame:
    if columns is None:
        columns = sorted(set(df_ref.columns) & set(df_prod.columns))
    rows = []
    for col in columns:
        r, p = df_ref[col], df_prod[col]
        if _is_numeric(r) and _is_numeric(p):
            stat, pval, test = *ks_test(r, p), "ks"
        else:
            stat, pval, test = *chi_square_test(r.astype("string"), p.astype("string")), "chi2"
        rows.append({
            "feature": col, "test": test, "statistic": stat, "p_value": pval,
            "alpha": alpha, "drift": bool(pval <= alpha) if pval == pval else False,
            "ref_n": int(r.notna().sum()), "prod_n": int(p.notna().sum())
        })
    out = pd.DataFrame(rows).sort_values(["drift","p_value"], ascending=[False,True])
    out.attrs["drift_rate"] = float(out["drift"].mean()) if len(out) else 0.0
    return out
