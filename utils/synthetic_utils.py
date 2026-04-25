"""
Synthetic Data Generation
Two modes:
  1. Factor-Structure Preserving — simulate from latent factors (statistically rigorous)
  2. Correlation Preserving — multivariate normal from empirical covariance (simpler)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# 1. Factor-Structure Preserving Synthesis
# ─────────────────────────────────────────

def generate_factor_based(df: pd.DataFrame, efa_result: dict,
                           n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data that preserves the latent factor structure.
    Algorithm:
      1. Simulate latent factor scores from N(0, I)
      2. Multiply by loading matrix to get common variance
      3. Add unique variance (sqrt(1 - communality)) as residuals
      4. Re-scale to original variable means and stds
    """
    np.random.seed(seed)

    loadings = efa_result["loadings"].values          # shape: (p, k)
    communalities = efa_result["communalities"].values # shape: (p,)
    n_factors = efa_result["n_factors"]
    columns = efa_result["loadings"].index.tolist()
    p = len(columns)

    # Simulate factor scores: (n_samples, n_factors)
    factor_scores = np.random.multivariate_normal(
        mean=np.zeros(n_factors),
        cov=np.eye(n_factors),
        size=n_samples
    )

    # Common part: (n_samples, p)
    common = factor_scores @ loadings.T

    # Unique part: residual variance per variable
    unique_var = np.maximum(1 - communalities, 1e-6)
    unique = np.random.normal(0, np.sqrt(unique_var), size=(n_samples, p))

    # Combined standardised synthetic data
    synthetic_std = common + unique

    # Re-scale to original distribution
    orig_mean = df[columns].mean().values
    orig_std = df[columns].std().values
    orig_std = np.where(orig_std == 0, 1.0, orig_std)

    # Standardise synthetic then rescale
    syn_mean = synthetic_std.mean(axis=0)
    syn_std = synthetic_std.std(axis=0)
    syn_std = np.where(syn_std == 0, 1.0, syn_std)

    synthetic_rescaled = ((synthetic_std - syn_mean) / syn_std) * orig_std + orig_mean

    return pd.DataFrame(synthetic_rescaled, columns=columns)


# ─────────────────────────────────────────
# 2. Correlation-Preserving Synthesis
# ─────────────────────────────────────────

def generate_correlation_based(df: pd.DataFrame, n_samples: int = 500,
                                seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data via multivariate normal using empirical
    covariance matrix (not correlation matrix — corrects the original code's bug).
    """
    np.random.seed(seed)

    columns = df.columns.tolist()
    mean = df.mean().values
    cov = df.cov().values  # ← use covariance, NOT correlation

    # Ensure positive semi-definiteness (numerical safety)
    cov = _make_psd(cov)

    synthetic = np.random.multivariate_normal(mean=mean, cov=cov, size=n_samples)
    return pd.DataFrame(synthetic, columns=columns)


def _make_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Force a matrix to be positive semi-definite by clipping negative eigenvalues.
    """
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ─────────────────────────────────────────
# 3. Validation — Check synthetic vs original
# ─────────────────────────────────────────

def validate_synthetic(original: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """
    Compare means, stds, and correlations between original and synthetic datasets.
    Returns a summary DataFrame.
    """
    cols = original.columns.tolist()
    records = []

    for col in cols:
        records.append({
            "Variable": col,
            "Orig Mean": round(original[col].mean(), 4),
            "Syn Mean": round(synthetic[col].mean(), 4),
            "Orig Std": round(original[col].std(), 4),
            "Syn Std": round(synthetic[col].std(), 4),
            "Mean Δ": round(abs(original[col].mean() - synthetic[col].mean()), 4),
            "Std Δ": round(abs(original[col].std() - synthetic[col].std()), 4),
        })

    return pd.DataFrame(records)
