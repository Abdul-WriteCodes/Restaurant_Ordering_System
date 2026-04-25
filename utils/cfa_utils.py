"""
CFA Utility Functions — Confirmatory Factor Analysis
Handles model string construction, fitting, fit index extraction, and modification indices.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# 1. Build CFA Model Description from EFA Loadings
# ─────────────────────────────────────────

def build_cfa_model(loadings: pd.DataFrame, threshold: float = 0.4) -> tuple[str, dict]:
    """
    Convert EFA loading matrix to semopy model string.
    Each variable is assigned to the factor it loads highest on (if >= threshold).
    Returns (model_string, factor_to_vars_dict).
    """
    factor_vars = {}

    for var in loadings.index:
        abs_row = np.abs(loadings.loc[var])
        max_val = abs_row.max()
        if max_val >= threshold:
            best_factor = abs_row.idxmax()
            factor_vars.setdefault(best_factor, []).append(var)

    # Remove factors with < 2 indicators (CFA requires ≥ 2, ideally ≥ 3)
    factor_vars = {f: v for f, v in factor_vars.items() if len(v) >= 2}

    model_lines = []
    for factor, vars_ in factor_vars.items():
        model_lines.append(f"{factor} =~ " + " + ".join(vars_))

    model_str = "\n".join(model_lines)
    return model_str, factor_vars


# ─────────────────────────────────────────
# 2. Run CFA and Extract Fit Indices
# ─────────────────────────────────────────

def run_cfa(df: pd.DataFrame, model_str: str) -> dict:
    """
    Fit CFA using semopy. Extracts fit indices: CFI, TLI, RMSEA, SRMR, chi2, df, p.
    Returns a result dict with fit_indices, parameter estimates, and any error message.
    """
    try:
        from semopy import Model
        from semopy.stats import calc_stats

        model = Model(model_str)
        model.fit(df)

        try:
            stats = calc_stats(model)
            fit_indices = _extract_fit_indices(stats)
        except Exception as stats_err:
            fit_indices = {"error": str(stats_err)}

        estimates = model.inspect()

        return {
            "success": True,
            "fit_indices": fit_indices,
            "estimates": estimates,
            "model_str": model_str,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "fit_indices": {},
            "estimates": None,
            "model_str": model_str,
            "error": str(e),
        }


def _extract_fit_indices(stats) -> dict:
    """
    Parse semopy calc_stats output (DataFrame or Series) into a clean dict.
    """
    fit = {}

    if isinstance(stats, pd.DataFrame):
        flat = stats.iloc[0] if len(stats) > 0 else pd.Series()
    elif isinstance(stats, pd.Series):
        flat = stats
    else:
        return {"raw": str(stats)}

    # Normalise column names
    flat.index = [str(c).strip().upper() for c in flat.index]

    mapping = {
        "CFI": ["CFI"],
        "TLI": ["TLI", "NNFI"],
        "RMSEA": ["RMSEA"],
        "SRMR": ["SRMR"],
        "Chi2": ["CHI2", "CHISQ", "CHI-SQUARE", "X2"],
        "df": ["DF", "DOF"],
        "p_value": ["P-VALUE", "PVALUE", "P_VALUE", "P(CHI2)"],
        "AIC": ["AIC"],
        "BIC": ["BIC"],
    }

    for label, candidates in mapping.items():
        for c in candidates:
            if c in flat.index:
                try:
                    fit[label] = round(float(flat[c]), 4)
                except Exception:
                    fit[label] = flat[c]
                break

    return fit


# ─────────────────────────────────────────
# 3. Assess CFA Fit Against User Thresholds
# ─────────────────────────────────────────

def assess_cfa_fit(fit_indices: dict, thresholds: dict) -> dict:
    """
    Compare fit indices against user-defined thresholds.
    thresholds = {"CFI": 0.95, "TLI": 0.95, "RMSEA": 0.06, "SRMR": 0.08}
    Returns assessment dict per index.
    """
    assessment = {}

    higher_better = ["CFI", "TLI"]
    lower_better = ["RMSEA", "SRMR"]

    for idx in higher_better:
        if idx in fit_indices and idx in thresholds:
            val = fit_indices[idx]
            thresh = thresholds[idx]
            assessment[idx] = {
                "value": val,
                "threshold": thresh,
                "pass": val >= thresh,
                "direction": "≥",
            }

    for idx in lower_better:
        if idx in fit_indices and idx in thresholds:
            val = fit_indices[idx]
            thresh = thresholds[idx]
            assessment[idx] = {
                "value": val,
                "threshold": thresh,
                "pass": val <= thresh,
                "direction": "≤",
            }

    n_pass = sum(1 for v in assessment.values() if v["pass"])
    n_total = len(assessment)
    overall_pass = n_pass == n_total and n_total > 0

    return {
        "indices": assessment,
        "n_pass": n_pass,
        "n_total": n_total,
        "overall_pass": overall_pass,
    }


# ─────────────────────────────────────────
# 4. Generate Modification Index Suggestions
# ─────────────────────────────────────────

def get_modification_suggestions(cfa_result: dict, fit_assessment: dict) -> list[str]:
    """
    Return human-readable suggestions when CFA fit is poor.
    """
    suggestions = []

    indices = fit_assessment.get("indices", {})

    if "RMSEA" in indices and not indices["RMSEA"]["pass"]:
        suggestions.append(
            f"RMSEA = {indices['RMSEA']['value']:.3f} exceeds threshold. "
            "Consider freeing residual covariances between items that may share method variance, "
            "or removing items with high modification indices."
        )

    if "CFI" in indices and not indices["CFI"]["pass"]:
        suggestions.append(
            f"CFI = {indices['CFI']['value']:.3f} is below threshold. "
            "Check whether some indicators load significantly on multiple factors. "
            "Consider adding cross-loadings or removing weak indicators."
        )

    if "SRMR" in indices and not indices["SRMR"]["pass"]:
        suggestions.append(
            f"SRMR = {indices['SRMR']['value']:.3f} exceeds threshold. "
            "Large residual correlations exist. Review residual correlation matrix for systematic patterns."
        )

    if not suggestions and not fit_assessment.get("overall_pass"):
        suggestions.append(
            "Model fit is inadequate. Consider: (1) revising factor structure, "
            "(2) removing low-communality items, or (3) allowing correlated residuals "
            "for items sharing common wording."
        )

    return suggestions
