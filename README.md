# 🔬 EFA / CFA Analyser

A rigorous, researcher-grade Streamlit app for Exploratory and Confirmatory Factor Analysis.
Single-file architecture — deploy to Streamlit Cloud in under 2 minutes.

---

## 🚀 Deploy to Streamlit Cloud

1. Push this folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch, set **Main file path** → `app.py`
4. Click **Deploy**

That's it. No server setup, no Docker, no config beyond what's here.

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

> **Python 3.10+** recommended. If `semopy` install fails, try:
> `pip install semopy --no-build-isolation`

---

## 📁 File Structure

```
efa_cfa_single/
├── app.py                  ← Everything in one file (~700 lines)
├── requirements.txt        ← All dependencies
├── README.md               ← This file
└── .streamlit/
    └── config.toml         ← Dark theme + upload size config
```

---

## 🔬 Pipeline Overview

| Step | What happens |
|------|-------------|
| **1. Upload** | CSV or Excel; numeric columns auto-selected; missing rows dropped |
| **2. Suitability** | KMO (labelled Marvellous→Unacceptable) + Bartlett's test + Scree plot |
| **3. EFA** | Auto n_factors via Kaiser λ>1 (user-overridable); varimax/oblimin/promax/etc |
| **4. Diagnostics** | Ranked table: Cross-Loaders, Weak Loaders, Low Communality — you choose what to drop |
| **5. CFA** | EFA-informed confirmatory model; CFI/TLI/RMSEA/SRMR with adjustable thresholds |
| **6. Synthetic** | Factor-structure-preserving OR correlation-preserving generation |
| **7. Export** | Cleaned CSV + Synthetic CSV + HTML report + EFA loadings + CFA syntax → ZIP |

---

## 📐 Statistical Notes

### KMO Interpretation (Kaiser, 1974)
| Score | Label |
|-------|-------|
| ≥ 0.90 | Marvellous |
| ≥ 0.80 | Meritorious |
| ≥ 0.70 | Middling |
| ≥ 0.60 | Mediocre |
| < 0.60 | Unacceptable |

### Default CFA Fit Thresholds (adjustable via sidebar sliders)
| Index | Default | Direction |
|-------|---------|-----------|
| CFI | 0.95 | ≥ |
| TLI | 0.95 | ≥ |
| RMSEA | 0.06 | ≤ |
| SRMR | 0.08 | ≤ |

### Synthetic Data Modes
- **Factor-Based** — Simulates latent scores → multiplies by loading matrix → adds unique variance → rescales to original distribution. Preserves psychometric structure. Best for structural validity and simulation studies.
- **Correlation-Based** — Draws from multivariate normal using empirical **covariance matrix** (not correlation — this was a bug in common implementations). Faster, simpler, preserves pairwise relationships.

### Item Diagnostic Severity Scoring
Problems are ranked by a composite severity score:
- **Low Communality** (<threshold): weighted 3× — most damaging to factor structure
- **Weak Loader** (max |loading| < threshold): weighted 2× the shortfall
- **Cross-Loader** (≥2 factors above threshold): weighted by inverse of loading gap (closer gaps = worse)

---

## ⚠️ Known Limitations

- CFA fit index parsing depends on `semopy.stats.calc_stats()` output format — if indices show as missing, the semopy version may format them differently. Check `semopy` changelog if upgrading.
- For very small datasets (n < 100), EFA and CFA results may be unstable. Interpret with caution.
- The EFA→CFA transition assigns each item to its **primary factor only**. In real research, cross-loadings should be theoretically justified before being forced into a CFA model.
- `semopy` fitting can take 10–30 seconds on larger datasets — this is normal.

---

## 📚 References

- Kaiser, H.F. (1974). An index of factorial simplicity. *Psychometrika*, 39, 31–36.
- Hu, L., & Bentler, P.M. (1999). Cutoff criteria for fit indexes in covariance structure analysis. *Structural Equation Modeling*, 6(1), 1–55.
- Tabachnick, B.G., & Fidell, L.S. (2013). *Using Multivariate Statistics* (6th ed.).
