"""
EFA Utility Functions — Exploratory Factor Analysis
Handles suitability checks, factor extraction, loading diagnostics, and iterative purification.
"""

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# 1. EFA Suitability Tests
# ─────────────────────────────────────────

def check_efa_suitability(df: pd.DataFrame) -> dict:
    """
    Run KMO and Bartlett's test of sphericity.
    Returns a dict with scores, p-values, and pass/fail flags.
    """
    kmo_all, kmo_model = calculate_kmo(df)
    bartlett_chi2, bartlett_p = calculate_bartlett_sphericity(df)

    kmo_label = _kmo_label(kmo_model)

    return {
        "kmo_model": round(float(kmo_model), 4),
        "kmo_all": kmo_all,
        "kmo_label": kmo_label,
        "kmo_pass": kmo_model >= 0.6,
        "bartlett_chi2": round(float(bartlett_chi2), 4),
        "bartlett_p": round(float(bartlett_p), 6),
        "bartlett_pass": bartlett_p < 0.05,
        "overall_pass": kmo_model >= 0.6 and bartlett_p < 0.05,
    }


def _kmo_label(kmo: float) -> str:
    if kmo >= 0.90:
        return "Marvellous"
    elif kmo >= 0.80:
        return "Meritorious"
    elif kmo >= 0.70:
        return "Middling"
    elif kmo >= 0.60:
        return "Mediocre"
    elif kmo >= 0.50:
        return "Miserable"
    else:
        return "Unacceptable"


# ─────────────────────────────────────────
# 2. Determine Number of Factors
# ─────────────────────────────────────────

def determine_n_factors(df: pd.DataFrame) -> dict:
    """
    Compute eigenvalues and suggest n_factors via Kaiser criterion (eigenvalue > 1).
    Also returns all eigenvalues for scree plot.
    """
    fa = FactorAnalyzer(n_factors=min(len(df.columns), len(df) - 1), rotation=None)
    fa.fit(df)
    ev, v = fa.get_eigenvalues()

    kaiser_n = int(np.sum(ev > 1))
    kaiser_n = max(1, kaiser_n)  # at least 1

    return {
        "eigenvalues": ev.tolist(),
        "suggested_n": kaiser_n,
    }


# ─────────────────────────────────────────
# 3. Run EFA
# ─────────────────────────────────────────

def run_efa(df: pd.DataFrame, n_factors: int, rotation: str = "varimax") -> dict:
    """
    Fit EFA with given n_factors and rotation. Returns loadings, communalities,
    and variance explained.
    """
    n_factors = min(n_factors, len(df.columns) - 1)
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
    fa.fit(df)

    factor_labels = [f"F{i+1}" for i in range(n_factors)]
    loadings = pd.DataFrame(fa.loadings_, index=df.columns, columns=factor_labels)
    communalities = pd.Series(fa.get_communalities(), index=df.columns, name="Communality")

    variance = fa.get_factor_variance()
    variance_df = pd.DataFrame(
        variance,
        index=["SS Loadings", "Proportion Var", "Cumulative Var"],
        columns=factor_labels,
    ).T

    return {
        "loadings": loadings,
        "communalities": communalities,
        "variance": variance_df,
        "fa_object": fa,
        "n_factors": n_factors,
    }


# ─────────────────────────────────────────
# 4. Diagnose Loadings — Ranked Problem List
# ─────────────────────────────────────────

def diagnose_loadings(loadings: pd.DataFrame, communalities: pd.Series,
                      loading_threshold: float = 0.4,
                      communality_threshold: float = 0.3) -> pd.DataFrame:
    """
    For each variable, assess:
      - max loading
      - number of factors with |loading| >= threshold  (cross-loading if >1)
      - communality
      - issue type: 'Cross-Loader', 'Weak Loader', 'Low Communality', or 'OK'
      - severity score (for ranking)
    Returns a DataFrame sorted by severity descending.
    """
    records = []

    for var in loadings.index:
        row = loadings.loc[var]
        abs_row = np.abs(row)
        max_load = abs_row.max()
        n_high = int((abs_row >= loading_threshold).sum())
        comm = float(communalities[var])

        issues = []
        severity = 0.0

        if comm < communality_threshold:
            issues.append("Low Communality")
            severity += (communality_threshold - comm) * 3  # weighted

        if n_high == 0:
            issues.append("Weak Loader")
            severity += (loading_threshold - max_load) * 2
        elif n_high > 1:
            issues.append("Cross-Loader")
            # severity = gap between top two loadings (smaller gap = worse)
            sorted_loads = sorted(abs_row.values, reverse=True)
            gap = sorted_loads[0] - sorted_loads[1]
            severity += (1 - gap) * 2

        issue_str = ", ".join(issues) if issues else "OK"

        records.append({
            "Variable": var,
            "Max Loading": round(max_load, 4),
            "# Factors ≥ Threshold": n_high,
            "Communality": round(comm, 4),
            "Issue": issue_str,
            "Severity": round(severity, 4),
            "Recommend Drop": len(issues) > 0,
        })

    df_diag = pd.DataFrame(records).sort_values("Severity", ascending=False).reset_index(drop=True)
    return df_diag


# ─────────────────────────────────────────
# 5. Re-run EFA after drops
# ─────────────────────────────────────────

def rerun_efa_after_drops(df: pd.DataFrame, drop_vars: list,
                          n_factors: int, rotation: str = "varimax") -> tuple:
    """
    Drop selected variables, re-check suitability, re-run EFA.
    Returns (cleaned_df, suitability_result, efa_result, diagnostics).
    """
    cleaned = df.drop(columns=drop_vars, errors="ignore")

    suit = check_efa_suitability(cleaned)
    efa = run_efa(cleaned, n_factors=n_factors, rotation=rotation)
    diag = diagnose_loadings(efa["loadings"], efa["communalities"])

    return cleaned, suit, efa, diag
