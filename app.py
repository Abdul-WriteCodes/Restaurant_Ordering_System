"""
EFA / CFA Analyser — Streamlit App
Full pipeline: upload → suitability → EFA → diagnose → user-guided fix → CFA → export
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
from datetime import datetime

# ── Local utils ──────────────────────────────────────────────────────────────
from utils.efa_utils import (
    check_efa_suitability,
    determine_n_factors,
    run_efa,
    diagnose_loadings,
    rerun_efa_after_drops,
)
from utils.cfa_utils import (
    build_cfa_model,
    run_cfa,
    assess_cfa_fit,
    get_modification_suggestions,
)
from utils.synthetic_utils import (
    generate_factor_based,
    generate_correlation_based,
    validate_synthetic,
)
from utils.report_utils import generate_html_report
from utils.plot_utils import (
    plot_scree,
    plot_loading_heatmap,
    plot_communalities,
    plot_fit_indices,
    plot_correlation_matrix,
    plot_synthetic_comparison,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EFA / CFA Analyser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark academic aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --accent: #6c8dfa;
  --accent2: #a78bfa;
  --green: #34d399;
  --red: #f87171;
  --yellow: #fbbf24;
  --border: #2d3148;
  --surface: #1a1d27;
  --muted: #94a3b8;
}

html, body, [class*="css"] {
  font-family: 'Crimson Pro', Georgia, serif;
}

.stApp { background: #0f1117; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: #13161f !important;
  border-right: 1px solid var(--border);
}

/* Headings */
h1 { color: var(--accent) !important; letter-spacing: -0.5px; }
h2 { color: var(--accent2) !important; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
h3 { color: #e2e8f0 !important; }

/* Step badges */
.step-badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: linear-gradient(135deg, #1a1d27, #2d3148);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  padding: 10px 18px; border-radius: 6px;
  color: #e2e8f0; font-size: 1.05rem;
  margin-bottom: 16px; width: 100%;
}
.step-num {
  background: var(--accent); color: #0f1117;
  border-radius: 50%; width: 26px; height: 26px;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 0.85rem; font-weight: bold; flex-shrink: 0;
}

/* Status pills */
.pill-pass {
  background: rgba(52,211,153,0.15); color: var(--green);
  border: 1px solid var(--green); border-radius: 20px;
  padding: 2px 12px; font-size: 0.85rem; font-weight: 600;
  display: inline-block;
}
.pill-fail {
  background: rgba(248,113,113,0.15); color: var(--red);
  border: 1px solid var(--red); border-radius: 20px;
  padding: 2px 12px; font-size: 0.85rem; font-weight: 600;
  display: inline-block;
}
.pill-warn {
  background: rgba(251,191,36,0.15); color: var(--yellow);
  border: 1px solid var(--yellow); border-radius: 20px;
  padding: 2px 12px; font-size: 0.85rem; font-weight: 600;
  display: inline-block;
}

/* Metric cards */
.metric-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px 20px; text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 600; color: var(--accent); }
.metric-label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }

/* Info box */
.info-box {
  background: rgba(108,141,250,0.08); border: 1px solid rgba(108,141,250,0.25);
  border-radius: 6px; padding: 12px 16px; color: #c7d2fe;
  font-size: 0.92rem; margin: 8px 0;
}

/* Warning box */
.warn-box {
  background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.3);
  border-radius: 6px; padding: 12px 16px; color: #fde68a;
  font-size: 0.92rem; margin: 8px 0;
}

/* Code / model string */
code, pre { font-family: 'JetBrains Mono', monospace !important; }

/* Divider */
.section-divider {
  border: none; border-top: 1px solid var(--border);
  margin: 28px 0;
}

/* Table styling */
[data-testid="stDataFrame"] { border-radius: 6px; overflow: hidden; }

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, #1d2035, #2d3560) !important;
  border: 1px solid var(--accent) !important;
  color: var(--accent) !important;
  font-family: 'Crimson Pro', serif !important;
  font-size: 1rem !important;
  border-radius: 6px !important;
  transition: all 0.2s;
}
.stButton > button:hover {
  background: linear-gradient(135deg, var(--accent), #4f6fe8) !important;
  color: #0f1117 !important;
  border-color: transparent !important;
}

/* Download button */
.stDownloadButton > button {
  background: linear-gradient(135deg, #1a2d1a, #1f3b1f) !important;
  border: 1px solid var(--green) !important;
  color: var(--green) !important;
  border-radius: 6px !important;
}

/* Sidebar header */
.sidebar-header {
  font-size: 1.3rem; color: var(--accent2);
  font-weight: 600; margin-bottom: 4px;
}
.sidebar-sub { font-size: 0.82rem; color: var(--muted); margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "df_original": None,
        "df_working": None,
        "suitability": None,
        "n_factors_auto": None,
        "eigenvalues": None,
        "efa_result": None,
        "diagnostics": None,
        "dropped_vars": [],
        "efa_done": False,
        "cfa_result": None,
        "fit_assessment": None,
        "synthetic_factor": None,
        "synthetic_corr": None,
        "syn_validation": None,
        "report_html": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
S = st.session_state


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">🔬 EFA / CFA Analyser</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Rigorous factor analysis pipeline for researchers</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")

    loading_threshold = st.slider(
        "Loading threshold",
        min_value=0.30, max_value=0.60, value=0.40, step=0.05,
        help="Minimum |loading| to consider a variable as loading on a factor.",
    )
    communality_threshold = st.slider(
        "Communality threshold",
        min_value=0.20, max_value=0.60, value=0.30, step=0.05,
        help="Variables below this communality are flagged for removal.",
    )
    rotation_method = st.selectbox(
        "EFA Rotation",
        ["varimax", "oblimin", "promax", "quartimax", "equamax"],
        help="Varimax (orthogonal) is most common. Oblimin/Promax for oblique structures.",
    )

    st.markdown("---")
    st.markdown("### 📐 CFA Fit Thresholds")

    cfi_thresh = st.slider("CFI ≥", 0.80, 0.99, 0.95, 0.01)
    tli_thresh = st.slider("TLI ≥", 0.80, 0.99, 0.95, 0.01)
    rmsea_thresh = st.slider("RMSEA ≤", 0.04, 0.15, 0.06, 0.01)
    srmr_thresh = st.slider("SRMR ≤", 0.04, 0.15, 0.08, 0.01)

    cfa_thresholds = {
        "CFI": cfi_thresh,
        "TLI": tli_thresh,
        "RMSEA": rmsea_thresh,
        "SRMR": srmr_thresh,
    }

    st.markdown("---")
    st.markdown("### 🧪 Synthetic Data")
    syn_n = st.number_input("Sample size", min_value=50, max_value=10000, value=500, step=50)
    syn_seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)

    st.markdown("---")
    if st.button("🔄 Reset All", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        _init_state()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🔬 EFA / CFA Analyser")
st.markdown(
    "_Upload a dataset and run a complete Exploratory and Confirmatory Factor Analysis pipeline. "
    "The app diagnoses issues, lets you guide remediation, and exports clean + synthetic datasets._"
)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ═════════════════════════════════════════
# STEP 1 — Upload Dataset
# ═════════════════════════════════════════
st.markdown('<div class="step-badge"><span class="step-num">1</span> Upload Dataset</div>', unsafe_allow_html=True)

col_up1, col_up2 = st.columns([2, 1])
with col_up1:
    uploaded = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="All columns will be treated as numeric variables. Rows with missing values are dropped.",
    )

with col_up2:
    st.markdown('<div class="info-box">💡 <b>Dataset requirements</b><br>• Numeric variables only<br>• Minimum 5 variables<br>• Recommended ≥ 100 observations<br>• No missing values (auto-dropped)</div>', unsafe_allow_html=True)

if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)

        # Keep only numeric columns
        df_numeric = df_raw.select_dtypes(include=[np.number]).dropna()

        if len(df_numeric.columns) < 3:
            st.error("❌ Need at least 3 numeric variables. Check your file.")
            st.stop()

        if len(df_numeric) < 30:
            st.warning("⚠️ Fewer than 30 observations — results may be unreliable.")

        S.df_original = df_numeric.copy()
        S.df_working = df_numeric.copy()
        S.dropped_vars = []
        # Reset downstream state when new file uploaded
        S.suitability = None; S.efa_result = None
        S.efa_done = False; S.cfa_result = None

    except Exception as e:
        st.error(f"❌ Could not parse file: {e}")
        st.stop()

if S.df_original is None:
    st.markdown('<div class="warn-box">👆 Upload a dataset to begin.</div>', unsafe_allow_html=True)
    st.stop()

# Preview
with st.expander("📋 Dataset Preview", expanded=True):
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{len(S.df_original)}</div><div class="metric-label">Observations</div></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{len(S.df_original.columns)}</div><div class="metric-label">Variables</div></div>', unsafe_allow_html=True)
    with col_c:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{len(S.df_working.columns)}</div><div class="metric-label">Working Variables</div></div>', unsafe_allow_html=True)
    with col_d:
        dropped_n = len(S.dropped_vars)
        st.markdown(f'<div class="metric-card"><div class="metric-val">{dropped_n}</div><div class="metric-label">Dropped</div></div>', unsafe_allow_html=True)

    st.dataframe(S.df_working.head(10), use_container_width=True)

    tab_corr1, tab_corr2 = st.tabs(["Descriptive Statistics", "Correlation Matrix"])
    with tab_corr1:
        st.dataframe(S.df_working.describe().T.round(4), use_container_width=True)
    with tab_corr2:
        st.plotly_chart(plot_correlation_matrix(S.df_working), use_container_width=True)


st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ═════════════════════════════════════════
# STEP 2 — EFA Suitability
# ═════════════════════════════════════════
st.markdown('<div class="step-badge"><span class="step-num">2</span> EFA Suitability Tests</div>', unsafe_allow_html=True)

if st.button("▶ Run Suitability Tests", use_container_width=False):
    with st.spinner("Running KMO and Bartlett's tests…"):
        S.suitability = check_efa_suitability(S.df_working)
        ev_info = determine_n_factors(S.df_working)
        S.n_factors_auto = ev_info["suggested_n"]
        S.eigenvalues = ev_info["eigenvalues"]

if S.suitability:
    suit = S.suitability
    overall_html = '<span class="pill-pass">✓ SUITABLE FOR EFA</span>' if suit["overall_pass"] else '<span class="pill-fail">✗ EFA SUITABILITY ISSUES</span>'
    st.markdown(f"**Overall:** {overall_html}", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        kmo_pill = "pill-pass" if suit["kmo_pass"] else "pill-fail"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">{suit['kmo_model']}</div>
          <div class="metric-label">KMO Score</div>
          <div style="margin-top:8px">
            <span class="{kmo_pill}">{suit['kmo_label']}</span>
          </div>
          <div style="color:var(--muted); font-size:0.8rem; margin-top:6px">Threshold ≥ 0.60</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        bart_pill = "pill-pass" if suit["bartlett_pass"] else "pill-fail"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">{suit['bartlett_p']}</div>
          <div class="metric-label">Bartlett p-value</div>
          <div style="margin-top:8px">
            <span class="{bart_pill}">χ² = {suit['bartlett_chi2']}</span>
          </div>
          <div style="color:var(--muted); font-size:0.8rem; margin-top:6px">Threshold p &lt; 0.05</div>
        </div>
        """, unsafe_allow_html=True)

    if not suit["overall_pass"]:
        st.markdown("""
        <div class="warn-box">
        ⚠️ <b>Suitability concerns detected.</b> You can still proceed with EFA, but results should be interpreted with caution.
        Consider removing variables that are uncorrelated with others.
        </div>
        """, unsafe_allow_html=True)

    # Scree plot
    st.markdown("#### Scree Plot")
    st.plotly_chart(plot_scree(S.eigenvalues, S.n_factors_auto), use_container_width=True)
    st.markdown(f'<div class="info-box">🔢 Kaiser criterion suggests <b>{S.n_factors_auto}</b> factor(s). You can override this below.</div>', unsafe_allow_html=True)


st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ═════════════════════════════════════════
# STEP 3 — EFA
# ═════════════════════════════════════════
st.markdown('<div class="step-badge"><span class="step-num">3</span> Exploratory Factor Analysis (EFA)</div>', unsafe_allow_html=True)

if S.suitability is None:
    st.markdown('<div class="warn-box">Complete Step 2 first.</div>', unsafe_allow_html=True)
else:
    n_factors_override = st.number_input(
        "Number of factors to extract",
        min_value=1,
        max_value=max(1, len(S.df_working.columns) - 1),
        value=S.n_factors_auto or 3,
        step=1,
        help="Default is Kaiser-suggested. Override if theory or scree plot suggests otherwise.",
    )

    if st.button("▶ Run EFA", use_container_width=False):
        with st.spinner("Fitting factor model…"):
            S.efa_result = run_efa(S.df_working, n_factors=n_factors_override, rotation=rotation_method)
            S.diagnostics = diagnose_loadings(
                S.efa_result["loadings"],
                S.efa_result["communalities"],
                loading_threshold=loading_threshold,
                communality_threshold=communality_threshold,
            )
            S.efa_done = True
            S.cfa_result = None  # invalidate downstream

    if S.efa_done and S.efa_result:
        efa = S.efa_result
        diag = S.diagnostics

        tab_load, tab_comm, tab_var = st.tabs(["Factor Loadings", "Communalities", "Variance Explained"])

        with tab_load:
            st.plotly_chart(plot_loading_heatmap(efa["loadings"], threshold=loading_threshold), use_container_width=True)
            st.dataframe(efa["loadings"].round(3), use_container_width=True)

        with tab_comm:
            st.plotly_chart(plot_communalities(efa["communalities"], comm_threshold=communality_threshold), use_container_width=True)

        with tab_var:
            st.dataframe(efa["variance"].round(4), use_container_width=True)
            total_var = efa["variance"]["Cumulative Var"].iloc[-1] * 100
            st.markdown(f'<div class="info-box">📊 Total variance explained: <b>{total_var:.1f}%</b></div>', unsafe_allow_html=True)


st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ═════════════════════════════════════════
# STEP 4 — Item Diagnostics & User-Guided Drop
# ═════════════════════════════════════════
st.markdown('<div class="step-badge"><span class="step-num">4</span> Item Diagnostics & Remediation</div>', unsafe_allow_html=True)

if not S.efa_done or S.diagnostics is None:
    st.markdown('<div class="warn-box">Complete Step 3 first.</div>', unsafe_allow_html=True)
else:
    diag = S.diagnostics
    problematic = diag[diag["Recommend Drop"]].reset_index(drop=True)
    clean_vars = diag[~diag["Recommend Drop"]]["Variable"].tolist()

    st.markdown(f"**{len(problematic)} variable(s) flagged** out of {len(diag)} — sorted by severity.")

    # Colour-coded diagnostics table
    def _colour_issue(val):
        if val == "OK":
            return "color: #34d399"
        elif "Cross" in str(val):
            return "color: #f87171"
        elif "Weak" in str(val):
            return "color: #fbbf24"
        elif "Communality" in str(val):
            return "color: #fb923c"
        return ""

    styled = diag.style.applymap(_colour_issue, subset=["Issue"]) \
                       .format({"Max Loading": "{:.3f}", "Communality": "{:.3f}", "Severity": "{:.3f}"}) \
                       .background_gradient(subset=["Severity"], cmap="Reds")
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("#### Select Variables to Drop")
    st.markdown('<div class="info-box">📌 Pre-selected = recommended by severity ranking. You can add or remove selections. Dropping and re-running EFA updates all downstream results.</div>', unsafe_allow_html=True)

    # Pre-tick recommended drops but let user override
    all_vars = S.df_working.columns.tolist()
    default_drops = problematic["Variable"].tolist()

    selected_drops = st.multiselect(
        "Variables to remove from analysis",
        options=all_vars,
        default=[v for v in default_drops if v in all_vars],
        help="Select variables to drop. Remaining variables will be used for CFA.",
    )

    col_drop1, col_drop2 = st.columns(2)
    with col_drop1:
        if st.button("▶ Apply Drops & Re-run EFA", disabled=(len(selected_drops) == 0)):
            with st.spinner("Dropping variables and re-running EFA…"):
                cleaned, new_suit, new_efa, new_diag = rerun_efa_after_drops(
                    S.df_working,
                    drop_vars=selected_drops,
                    n_factors=S.efa_result["n_factors"],
                    rotation=rotation_method,
                )
                S.df_working = cleaned
                S.dropped_vars = list(set(S.dropped_vars + selected_drops))
                S.suitability = new_suit
                S.efa_result = new_efa
                S.diagnostics = new_diag
                S.cfa_result = None
                st.success(f"✓ Dropped {len(selected_drops)} variable(s). EFA re-run complete.")
                st.rerun()

    with col_drop2:
        if st.button("▶ Proceed Without Drops"):
            st.markdown('<div class="info-box">✓ Proceeding with current variables to CFA.</div>', unsafe_allow_html=True)

    if S.dropped_vars:
        st.markdown(f"**Total dropped so far:** `{', '.join(S.dropped_vars)}`")


st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ═════════════════════════════════════════
# STEP 5 — CFA
# ═════════════════════════════════════════
st.markdown('<div class="step-badge"><span class="step-num">5</span> Confirmatory Factor Analysis (CFA)</div>', unsafe_allow_html=True)

if not S.efa_done or S.efa_result is None:
    st.markdown('<div class="warn-box">Complete EFA (Step 3) first.</div>', unsafe_allow_html=True)
else:
    loadings = S.efa_result["loadings"]
    model_str, factor_vars = build_cfa_model(loadings, threshold=loading_threshold)

    st.markdown("#### CFA Model Specification (auto-derived from EFA)")
    st.code(model_str, language="text")

    if not factor_vars:
        st.error("❌ No factors have ≥ 2 indicators. Cannot build CFA model. Try lowering the loading threshold or dropping fewer variables.")
    else:
        st.markdown("""
        <div class="info-box">
        ℹ️ The model is derived by assigning each item to its <b>primary factor</b> (highest absolute loading ≥ threshold).
        Items without a strong loading are excluded. This is a confirmatory re-specification of the exploratory structure —
        appropriate for academic validation purposes.
        </div>
        """, unsafe_allow_html=True)

        if st.button("▶ Run CFA", use_container_width=False):
            with st.spinner("Fitting CFA model (this may take a moment)…"):
                S.cfa_result = run_cfa(S.df_working, model_str)
                if S.cfa_result["success"]:
                    S.fit_assessment = assess_cfa_fit(
                        S.cfa_result["fit_indices"],
                        thresholds=cfa_thresholds,
                    )
                else:
                    S.fit_assessment = None

    if S.cfa_result:
        if not S.cfa_result["success"]:
            st.error(f"❌ CFA failed: {S.cfa_result['error']}")
            st.markdown('<div class="warn-box">Common causes: too few observations per variable, singular covariance matrix, or model identification issues. Try increasing observations or reducing the number of factors.</div>', unsafe_allow_html=True)
        else:
            fa = S.fit_assessment
            overall_badge = "pill-pass" if fa["overall_pass"] else "pill-fail"
            overall_text = "ADEQUATE FIT" if fa["overall_pass"] else "INADEQUATE FIT"
            st.markdown(f"""
            **CFA Fit:** <span class="{overall_badge}">{overall_text}</span>
            &nbsp; <span style="color:var(--muted); font-size:0.9rem">{fa['n_pass']}/{fa['n_total']} indices passed</span>
            """, unsafe_allow_html=True)

            # Fit index chart
            st.plotly_chart(plot_fit_indices(fa), use_container_width=True)

            # Fit index table
            fit_records = []
            for idx, data in fa["indices"].items():
                fit_records.append({
                    "Index": idx,
                    "Value": data["value"],
                    "Threshold": f"{data['direction']} {data['threshold']}",
                    "Status": "✓ PASS" if data["pass"] else "✗ FAIL",
                })
            st.dataframe(pd.DataFrame(fit_records), use_container_width=True, hide_index=True)

            # Raw fit indices
            raw_fi = S.cfa_result["fit_indices"]
            extra_cols = st.columns(3)
            for i, (k, v) in enumerate([kv for kv in raw_fi.items() if kv[0] not in ("CFI","TLI","RMSEA","SRMR")]):
                with extra_cols[i % 3]:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{v}</div><div class="metric-label">{k}</div></div>', unsafe_allow_html=True)

            # Parameter estimates
            with st.expander("📊 Parameter Estimates"):
                if S.cfa_result["estimates"] is not None:
                    st.dataframe(S.cfa_result["estimates"], use_container_width=True)

            # Suggestions if fit is poor
            if not fa["overall_pass"]:
                suggestions = get_modification_suggestions(S.cfa_result, fa)
                st.markdown("#### 🔧 Modification Suggestions")
                for s in suggestions:
                    st.markdown(f'<div class="warn-box">⚠️ {s}</div>', unsafe_allow_html=True)
                st.markdown('<div class="info-box">💡 After modifications, return to Step 4 to drop additional items or Step 3 to extract different factors, then re-run CFA.</div>', unsafe_allow_html=True)


st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ═════════════════════════════════════════
# STEP 6 — Synthetic Data Generation
# ═════════════════════════════════════════
st.markdown('<div class="step-badge"><span class="step-num">6</span> Synthetic Data Generation</div>', unsafe_allow_html=True)

if S.efa_result is None:
    st.markdown('<div class="warn-box">Complete EFA (Step 3) first.</div>', unsafe_allow_html=True)
else:
    syn_col1, syn_col2 = st.columns(2)

    with syn_col1:
        st.markdown("#### Factor-Structure Preserving")
        st.markdown('<div class="info-box">Simulates from latent factor scores × loadings + unique variance. Preserves the <b>psychometric structure</b>. Recommended for structural validity studies.</div>', unsafe_allow_html=True)
        if st.button("▶ Generate (Factor-Based)", use_container_width=True):
            with st.spinner("Simulating from factor structure…"):
                S.synthetic_factor = generate_factor_based(
                    S.df_working, S.efa_result, n_samples=syn_n, seed=syn_seed
                )
                S.syn_validation = validate_synthetic(S.df_working, S.synthetic_factor)
            st.success(f"✓ Generated {syn_n} synthetic observations.")

    with syn_col2:
        st.markdown("#### Correlation Preserving")
        st.markdown('<div class="info-box">Multivariate normal from empirical <b>covariance matrix</b>. Simpler and faster. Preserves pairwise relationships but not latent structure explicitly.</div>', unsafe_allow_html=True)
        if st.button("▶ Generate (Correlation-Based)", use_container_width=True):
            with st.spinner("Sampling from multivariate normal…"):
                S.synthetic_corr = generate_correlation_based(
                    S.df_working, n_samples=syn_n, seed=syn_seed
                )
                S.syn_validation = validate_synthetic(S.df_working, S.synthetic_corr)
            st.success(f"✓ Generated {syn_n} synthetic observations.")

    active_syn = S.synthetic_factor or S.synthetic_corr

    if active_syn is not None:
        syn_display = S.synthetic_factor if S.synthetic_factor is not None else S.synthetic_corr

        tab_s1, tab_s2, tab_s3 = st.tabs(["Preview", "Validation Summary", "Distribution Comparison"])
        with tab_s1:
            st.dataframe(syn_display.head(10).round(3), use_container_width=True)
        with tab_s2:
            if S.syn_validation is not None:
                st.dataframe(S.syn_validation, use_container_width=True, hide_index=True)
        with tab_s3:
            st.plotly_chart(
                plot_synthetic_comparison(S.df_working, syn_display),
                use_container_width=True
            )


st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ═════════════════════════════════════════
# STEP 7 — Export Bundle
# ═════════════════════════════════════════
st.markdown('<div class="step-badge"><span class="step-num">7</span> Export Bundle</div>', unsafe_allow_html=True)

has_efa = S.efa_result is not None
has_cfa = S.cfa_result is not None and S.cfa_result["success"]
has_syn = (S.synthetic_factor is not None) or (S.synthetic_corr is not None)

if not has_efa:
    st.markdown('<div class="warn-box">Complete EFA (Step 3) to enable exports.</div>', unsafe_allow_html=True)
else:
    col_e1, col_e2, col_e3 = st.columns(3)

    # ── Cleaned Dataset CSV ──────────────────
    with col_e1:
        st.markdown("##### 🗃️ Cleaned Dataset")
        cleaned_csv = S.df_working.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇ Download cleaned_data.csv\n({len(S.df_working)} rows × {len(S.df_working.columns)} cols)",
            data=cleaned_csv,
            file_name="cleaned_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Synthetic Dataset CSV ──────────────────
    with col_e2:
        st.markdown("##### 🧪 Synthetic Dataset")
        syn_export = S.synthetic_factor if S.synthetic_factor is not None else S.synthetic_corr
        if syn_export is not None:
            syn_csv = syn_export.to_csv(index=False).encode("utf-8")
            syn_label = "factor_based" if S.synthetic_factor is not None else "correlation_based"
            st.download_button(
                label=f"⬇ Download synthetic_data.csv\n({len(syn_export)} rows × {len(syn_export.columns)} cols)",
                data=syn_csv,
                file_name=f"synthetic_{syn_label}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.markdown('<div class="warn-box">Generate synthetic data in Step 6 first.</div>', unsafe_allow_html=True)

    # ── HTML Report ──────────────────────────
    with col_e3:
        st.markdown("##### 📄 Analysis Report")
        if has_cfa:
            if st.button("🔨 Build HTML Report", use_container_width=True):
                with st.spinner("Compiling report…"):
                    S.report_html = generate_html_report(
                        original_df=S.df_original,
                        cleaned_df=S.df_working,
                        suitability=S.suitability,
                        efa_result=S.efa_result,
                        diagnostics=S.diagnostics,
                        dropped_vars=S.dropped_vars,
                        cfa_result=S.cfa_result,
                        fit_assessment=S.fit_assessment,
                        cfa_thresholds=cfa_thresholds,
                        synthetic_validation=S.syn_validation,
                        model_str=S.cfa_result.get("model_str", ""),
                    )
                st.success("✓ Report ready.")

            if S.report_html:
                st.download_button(
                    label="⬇ Download analysis_report.html",
                    data=S.report_html.encode("utf-8"),
                    file_name="efa_cfa_report.html",
                    mime="text/html",
                    use_container_width=True,
                )
        else:
            st.markdown('<div class="warn-box">Run CFA (Step 5) to enable report generation.</div>', unsafe_allow_html=True)

    # ── Full ZIP Bundle ──────────────────────
    st.markdown("---")
    st.markdown("##### 📦 Download Everything as ZIP")

    can_zip = True  # cleaned CSV is always available if EFA done

    if st.button("⬇ Download Full Bundle (.zip)", use_container_width=False):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("cleaned_data.csv", S.df_working.to_csv(index=False))

            if syn_export is not None:
                zf.writestr("synthetic_data.csv", syn_export.to_csv(index=False))

            if S.report_html:
                zf.writestr("efa_cfa_report.html", S.report_html)

            if has_cfa and S.cfa_result.get("model_str"):
                zf.writestr("cfa_model.txt", S.cfa_result["model_str"])

            # EFA loadings
            if S.efa_result:
                zf.writestr("efa_loadings.csv", S.efa_result["loadings"].round(4).to_csv())
                zf.writestr("efa_communalities.csv", S.efa_result["communalities"].round(4).to_frame().to_csv())

        zip_buffer.seek(0)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            label="⬇ efa_cfa_bundle.zip",
            data=zip_buffer.getvalue(),
            file_name=f"efa_cfa_bundle_{ts_str}.zip",
            mime="application/zip",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; color:#475569; font-size:0.82rem">'
    'EFA/CFA Analyser &nbsp;|&nbsp; Built for researchers. '
    'Results should be interpreted within the theoretical framework of your study.'
    '</div>',
    unsafe_allow_html=True,
)
