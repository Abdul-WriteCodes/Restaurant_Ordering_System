"""
EFA Utility Module (Production Safe)
- KMO & Bartlett checks
- Robust preprocessing
- Eigenvalue estimation
- EFA execution
- Loading diagnostics
"""

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# 1. CORE CLEANER (SINGLE SOURCE OF TRUTH)
# ─────────────────────────────────────────

def prepare_efa_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strict preprocessing to guarantee EFA compatibility.
    """

    df = df.copy()

    # Force numeric conversion (fixes hidden strings, commas, etc.)
    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop columns with too many missing values
    df = df.loc[:, df.isna().mean() < 0.4]

    # Drop rows with missing values
    df = df.dropna()

    # Remove constant columns
    df = df.loc[:, df.nunique() > 1]

    # Remove near-zero variance columns
    df = df.loc[:, df.std() > 1e-8]

    return df


# ─────────────────────────────────────────
# 2. SUITABILITY TESTS (KMO + BARTLETT)
# ─────────────────────────────────────────

def check_efa_suitability(df: pd.DataFrame) -> dict:
    """
    KMO and Bartlett test after cleaning.
    """

    df_clean = prepare_efa_input(df)

    if df_clean.shape[1] < 3:
        return {
            "overall_pass": False,
            "reason": "Need at least 3 numeric variables",
            "shape": df_clean.shape
        }

    if df_clean.shape[0] < 10:
        return {
            "overall_pass": False,
            "reason": "Need at least 10 observations",
            "shape": df_clean.shape
        }

    kmo_all, kmo_model = calculate_kmo(df_clean)
    bartlett_chi2, bartlett_p = calculate_bartlett_sphericity(df_clean)

    return {
        "kmo_model": float(kmo_model),
        "bartlett_p": float(bartlett_p),
        "kmo_pass": kmo_model >= 0.6,
        "bartlett_pass": bartlett_p < 0.05,
        "overall_pass": (kmo_model >= 0.6 and bartlett_p < 0.05),
        "shape": df_clean.shape
    }


# ─────────────────────────────────────────
# 3. EIGENVALUE ESTIMATION (FACTOR COUNT)
# ─────────────────────────────────────────

def determine_n_factors(df: pd.DataFrame) -> dict:
    """
    Kaiser criterion + eigenvalues.
    """

    df_clean = prepare_efa_input(df)

    if df_clean.shape[1] < 3:
        raise ValueError("Need ≥3 variables for EFA")

    if df_clean.shape[0] < 10:
        raise ValueError("Need ≥10 observations for EFA")

    fa = FactorAnalyzer(
        n_factors=min(df_clean.shape[1], df_clean.shape[0] - 1),
        rotation=None
    )

    fa.fit(df_clean)

    eigenvalues, _ = fa.get_eigenvalues()

    return {
        "eigenvalues": eigenvalues.tolist(),
        "suggested_n": max(1, int(np.sum(eigenvalues > 1))),
        "shape": df_clean.shape
    }


# ─────────────────────────────────────────
# 4. RUN EFA MODEL
# ─────────────────────────────────────────

def run_efa(df: pd.DataFrame, n_factors: int, rotation: str = "varimax") -> dict:
    """
    Fit EFA model safely.
    """

    df_clean = prepare_efa_input(df)

    if df_clean.shape[1] < 3:
        raise ValueError("Not enough variables after cleaning")

    n_factors = min(n_factors, df_clean.shape[1] - 1)

    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
    fa.fit(df_clean)

    factor_labels = [f"F{i+1}" for i in range(n_factors)]

    loadings = pd.DataFrame(
        fa.loadings_,
        index=df_clean.columns,
        columns=factor_labels
    )

    communalities = pd.Series(
        fa.get_communalities(),
        index=df_clean.columns,
        name="Communality"
    )

    variance = fa.get_factor_variance()

    variance_df = pd.DataFrame(
        variance,
        index=["SS Loadings", "Proportion Var", "Cumulative Var"],
        columns=factor_labels
    ).T

    return {
        "loadings": loadings,
        "communalities": communalities,
        "variance": variance_df,
        "fa_object": fa,
        "n_factors": n_factors,
        "shape": df_clean.shape
    }


# ─────────────────────────────────────────
# 5. LOADINGS DIAGNOSTICS
# ─────────────────────────────────────────

def diagnose_loadings(loadings: pd.DataFrame,
                      communalities: pd.Series,
                      threshold: float = 0.4) -> pd.DataFrame:

    records = []

    for var in loadings.index:
        vals = loadings.loc[var].abs()

        max_loading = vals.max()
        n_high = (vals >= threshold).sum()
        comm = communalities[var]

        issue = "OK"

        if comm < 0.3:
            issue = "Low Communality"
        elif n_high == 0:
            issue = "Weak Loader"
        elif n_high > 1:
            issue = "Cross-Loader"

        records.append({
            "Variable": var,
            "Max Loading": round(max_loading, 3),
            "Communality": round(comm, 3),
            "Issue": issue
        })

    return pd.DataFrame(records).sort_values(
        by="Max Loading", ascending=False
    ).reset_index(drop=True)


# ─────────────────────────────────────────
# 6. OPTIONAL: SAFE PIPELINE WRAPPER
# ─────────────────────────────────────────

def efa_pipeline(df: pd.DataFrame, n_factors: int = None):
    """
    End-to-end safe EFA pipeline.
    """

    df_clean = prepare_efa_input(df)

    suitability = check_efa_suitability(df_clean)

    if not suitability["overall_pass"]:
        return {
            "status": "failed",
            "reason": suitability
        }

    if n_factors is None:
        n_factors = determine_n_factors(df_clean)["suggested_n"]

    efa_result = run_efa(df_clean, n_factors)

    diagnostics = diagnose_loadings(
        efa_result["loadings"],
        efa_result["communalities"]
    )

    return {
        "status": "success",
        "suitability": suitability,
        "efa": efa_result,
        "diagnostics": diagnostics
    }
