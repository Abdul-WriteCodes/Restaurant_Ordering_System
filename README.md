# 🔬 EFA / CFA Analyser

A rigorous, researcher-grade Streamlit app for Exploratory and Confirmatory Factor Analysis.

## Features

- **EFA Suitability**: KMO + Bartlett's test with labelled interpretation
- **Scree Plot**: Eigenvalue visualisation with Kaiser criterion overlay
- **EFA**: Varimax/oblimin/promax rotation, auto-suggested n_factors (overridable)
- **Item Diagnostics**: Ranked table of cross-loaders, weak loaders, low communality items
- **User-Guided Remediation**: You choose what to drop — app re-runs EFA after each purge
- **CFA**: Auto-derived from EFA structure, fit indices (CFI, TLI, RMSEA, SRMR) with user-adjustable thresholds
- **Modification Suggestions**: Plain-English guidance when CFA fit is poor
- **Synthetic Data**: Factor-structure-preserving OR correlation-preserving generation
- **Export Bundle**: Cleaned CSV, synthetic CSV, HTML report, EFA loadings, CFA model syntax — all as a ZIP

## Project Structure

```
efa_cfa_app/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── config.toml         # Dark theme config
└── utils/
    ├── __init__.py
    ├── efa_utils.py         # EFA: suitability, extraction, diagnostics
    ├── cfa_utils.py         # CFA: model building, fitting, fit assessment
    ├── synthetic_utils.py   # Synthetic data generation (2 modes)
    ├── plot_utils.py        # All Plotly visualisations
    └── report_utils.py      # HTML report generation
```

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this entire folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your repo, branch, and set **Main file path** to `app.py`
5. Click **Deploy** — done

> **Note**: semopy requires scipy ≥ 1.11. If deployment fails, pin `scipy==1.11.4` in requirements.txt.

## Dataset Requirements

- CSV or Excel (.xlsx)
- All analysis columns must be **numeric**
- Minimum **5 variables**, recommended **≥ 100 observations**
- Missing values are automatically dropped

## Statistical Notes

### EFA Suitability
| KMO Value | Label |
|-----------|-------|
| ≥ 0.90 | Marvellous |
| ≥ 0.80 | Meritorious |
| ≥ 0.70 | Middling |
| ≥ 0.60 | Mediocre |
| < 0.60 | Unacceptable |

### CFA Fit Index Defaults (adjustable in-app)
| Index | Default Threshold |
|-------|------------------|
| CFI | ≥ 0.95 |
| TLI | ≥ 0.95 |
| RMSEA | ≤ 0.06 |
| SRMR | ≤ 0.08 |

### Synthetic Data Modes
- **Factor-Based**: Simulates latent factor scores → multiplies by loading matrix → adds unique variance → rescales. Preserves psychometric structure. Best for validity studies.
- **Correlation-Based**: Draws from multivariate normal using empirical covariance matrix. Faster but doesn't model latent structure explicitly.
