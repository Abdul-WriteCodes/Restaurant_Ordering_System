"""
Report Generator — produces a self-contained HTML analysis report.
Covers: dataset overview, EFA suitability, EFA results, CFA results, synthetic data summary.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import base64
import io


def generate_html_report(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    suitability: dict,
    efa_result: dict,
    diagnostics: pd.DataFrame,
    dropped_vars: list,
    cfa_result: dict,
    fit_assessment: dict,
    cfa_thresholds: dict,
    synthetic_validation: pd.DataFrame | None = None,
    model_str: str = "",
) -> str:
    """
    Build and return a complete HTML report as a string.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections = []

    sections.append(_html_header(ts))
    sections.append(_section_overview(original_df, cleaned_df, dropped_vars))
    sections.append(_section_efa_suitability(suitability))
    sections.append(_section_efa_results(efa_result, diagnostics))
    sections.append(_section_cfa_results(cfa_result, fit_assessment, cfa_thresholds, model_str))
    if synthetic_validation is not None:
        sections.append(_section_synthetic(synthetic_validation))
    sections.append(_html_footer())

    return "\n".join(sections)


# ─────────────────────────────────────────
# HTML Sections
# ─────────────────────────────────────────

def _html_header(ts: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EFA/CFA Analysis Report</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2d3148;
    --accent: #6c8dfa;
    --accent2: #a78bfa;
    --green: #34d399;
    --red: #f87171;
    --yellow: #fbbf24;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --font-mono: 'Courier New', monospace;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Georgia', serif;
    line-height: 1.7;
    padding: 40px 20px;
  }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 2.2rem; color: var(--accent); margin-bottom: 4px; letter-spacing: -0.5px; }}
  h2 {{ font-size: 1.3rem; color: var(--accent2); margin: 36px 0 14px; border-bottom: 1px solid var(--border); padding-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }}
  h3 {{ font-size: 1.05rem; color: var(--text); margin: 20px 0 8px; }}
  .meta {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 32px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px 24px; margin-bottom: 16px; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; }}
  .badge-pass {{ background: rgba(52,211,153,0.15); color: var(--green); border: 1px solid var(--green); }}
  .badge-fail {{ background: rgba(248,113,113,0.15); color: var(--red); border: 1px solid var(--red); }}
  .badge-warn {{ background: rgba(251,191,36,0.15); color: var(--yellow); border: 1px solid var(--yellow); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 10px; }}
  th {{ background: var(--border); color: var(--accent); padding: 8px 12px; text-align: left; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid var(--border); color: var(--text); }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(108,141,250,0.05); }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; margin-top: 12px; }}
  .stat-box {{ background: var(--border); border-radius: 6px; padding: 14px 16px; }}
  .stat-label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }}
  .stat-value {{ font-size: 1.6rem; font-weight: bold; color: var(--accent); margin-top: 4px; }}
  .stat-sub {{ font-size: 0.75rem; color: var(--muted); }}
  pre {{ background: #111; border: 1px solid var(--border); padding: 14px; border-radius: 6px; font-family: var(--font-mono); font-size: 0.85rem; overflow-x: auto; color: #a5f3fc; }}
  .row-ok td:first-child {{ border-left: 3px solid var(--green); }}
  .row-warn td:first-child {{ border-left: 3px solid var(--yellow); }}
  .row-fail td:first-child {{ border-left: 3px solid var(--red); }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 32px 0; }}
  footer {{ color: var(--muted); font-size: 0.8rem; margin-top: 48px; text-align: center; }}
</style>
</head>
<body>
<div class="container">
<h1>EFA / CFA Analysis Report</h1>
<p class="meta">Generated: {ts} &nbsp;|&nbsp; EFA–CFA Analyser</p>
"""


def _section_overview(original_df, cleaned_df, dropped_vars) -> str:
    n_orig = len(original_df.columns)
    n_clean = len(cleaned_df.columns)
    n_rows = len(cleaned_df)
    n_dropped = len(dropped_vars)

    dropped_list = ", ".join(dropped_vars) if dropped_vars else "None"

    return f"""
<h2>1. Dataset Overview</h2>
<div class="stat-grid">
  <div class="stat-box"><div class="stat-label">Original Variables</div><div class="stat-value">{n_orig}</div></div>
  <div class="stat-box"><div class="stat-label">Variables After Cleaning</div><div class="stat-value">{n_clean}</div></div>
  <div class="stat-box"><div class="stat-label">Observations</div><div class="stat-value">{n_rows}</div></div>
  <div class="stat-box"><div class="stat-label">Variables Dropped</div><div class="stat-value">{n_dropped}</div></div>
</div>
<div class="card" style="margin-top:14px">
  <strong>Dropped Variables:</strong> <span style="color:var(--muted)">{dropped_list}</span>
</div>
"""


def _section_efa_suitability(suit: dict) -> str:
    kmo_badge = "badge-pass" if suit["kmo_pass"] else "badge-fail"
    bart_badge = "badge-pass" if suit["bartlett_pass"] else "badge-fail"
    overall_badge = "badge-pass" if suit["overall_pass"] else "badge-fail"
    overall_text = "PASS" if suit["overall_pass"] else "FAIL"

    return f"""
<h2>2. EFA Suitability</h2>
<div class="card">
  <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px">
    <span>Overall Suitability:</span>
    <span class="badge {overall_badge}">{overall_text}</span>
  </div>
  <table>
    <thead><tr><th>Test</th><th>Value</th><th>Threshold</th><th>Result</th></tr></thead>
    <tbody>
      <tr>
        <td>KMO (Kaiser-Meyer-Olkin)</td>
        <td>{suit['kmo_model']} — <em>{suit['kmo_label']}</em></td>
        <td>≥ 0.60</td>
        <td><span class="badge {kmo_badge}">{'PASS' if suit['kmo_pass'] else 'FAIL'}</span></td>
      </tr>
      <tr>
        <td>Bartlett's Test of Sphericity</td>
        <td>χ² = {suit['bartlett_chi2']}, p = {suit['bartlett_p']}</td>
        <td>p &lt; 0.05</td>
        <td><span class="badge {bart_badge}">{'PASS' if suit['bartlett_pass'] else 'FAIL'}</span></td>
      </tr>
    </tbody>
  </table>
</div>
"""


def _section_efa_results(efa_result: dict, diagnostics: pd.DataFrame) -> str:
    loadings = efa_result["loadings"]
    variance = efa_result["variance"]
    communalities = efa_result["communalities"]
    n_factors = efa_result["n_factors"]

    # Loadings table
    load_rows = ""
    for var in loadings.index:
        cells = "".join(f"<td>{loadings.loc[var, col]:.3f}</td>" for col in loadings.columns)
        comm_val = communalities[var]
        comm_color = "var(--green)" if comm_val >= 0.5 else ("var(--yellow)" if comm_val >= 0.3 else "var(--red)")
        load_rows += f"<tr><td>{var}</td>{cells}<td style='color:{comm_color}'>{comm_val:.3f}</td></tr>"

    load_headers = "".join(f"<th>{col}</th>" for col in loadings.columns)

    # Variance explained
    var_rows = ""
    for idx, row in variance.iterrows():
        var_rows += f"<tr><td>{idx}</td><td>{row['SS Loadings']:.4f}</td><td>{row['Proportion Var']*100:.2f}%</td><td>{row['Cumulative Var']*100:.2f}%</td></tr>"

    # Diagnostics table
    diag_rows = ""
    for _, r in diagnostics.iterrows():
        row_class = "row-ok" if r["Issue"] == "OK" else "row-fail"
        rec = "✓ Drop" if r["Recommend Drop"] else "—"
        diag_rows += f"""<tr class="{row_class}">
          <td>{r['Variable']}</td>
          <td>{r['Max Loading']}</td>
          <td>{r['# Factors ≥ Threshold']}</td>
          <td>{r['Communality']}</td>
          <td>{r['Issue']}</td>
          <td>{r['Severity']}</td>
          <td>{rec}</td>
        </tr>"""

    return f"""
<h2>3. Exploratory Factor Analysis (EFA)</h2>
<div class="card">
  <strong>Number of Factors Extracted:</strong> {n_factors}
</div>

<h3>Factor Loadings (Varimax Rotation)</h3>
<div class="card" style="overflow-x:auto">
  <table>
    <thead><tr><th>Variable</th>{load_headers}<th>Communality</th></tr></thead>
    <tbody>{load_rows}</tbody>
  </table>
</div>

<h3>Variance Explained</h3>
<div class="card" style="overflow-x:auto">
  <table>
    <thead><tr><th>Factor</th><th>SS Loadings</th><th>Proportion Var</th><th>Cumulative Var</th></tr></thead>
    <tbody>{var_rows}</tbody>
  </table>
</div>

<h3>Item Diagnostics</h3>
<div class="card" style="overflow-x:auto">
  <table>
    <thead><tr><th>Variable</th><th>Max Load</th><th># Factors ≥ Threshold</th><th>Communality</th><th>Issue</th><th>Severity</th><th>Recommended</th></tr></thead>
    <tbody>{diag_rows}</tbody>
  </table>
</div>
"""


def _section_cfa_results(cfa_result: dict, fit_assessment: dict,
                          thresholds: dict, model_str: str) -> str:
    if not cfa_result["success"]:
        return f"""
<h2>4. Confirmatory Factor Analysis (CFA)</h2>
<div class="card">
  <span class="badge badge-fail">CFA FAILED</span>
  <p style="margin-top:10px; color:var(--red)">{cfa_result['error']}</p>
</div>
"""

    overall = fit_assessment.get("overall_pass", False)
    overall_badge = "badge-pass" if overall else "badge-fail"
    overall_text = "ADEQUATE FIT" if overall else "INADEQUATE FIT"
    n_pass = fit_assessment.get("n_pass", 0)
    n_total = fit_assessment.get("n_total", 0)

    fit_rows = ""
    for idx, data in fit_assessment.get("indices", {}).items():
        row_class = "row-ok" if data["pass"] else "row-fail"
        thresh_str = f"{data['direction']} {data['threshold']}"
        fit_rows += f"""<tr class="{row_class}">
          <td>{idx}</td>
          <td>{data['value']}</td>
          <td>{thresh_str}</td>
          <td><span class="badge {'badge-pass' if data['pass'] else 'badge-fail'}">{'PASS' if data['pass'] else 'FAIL'}</span></td>
        </tr>"""

    return f"""
<h2>4. Confirmatory Factor Analysis (CFA)</h2>
<div class="card">
  <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px">
    <span>Model Fit:</span>
    <span class="badge {overall_badge}">{overall_text}</span>
    <span style="color:var(--muted); font-size:0.85rem">{n_pass}/{n_total} indices passed</span>
  </div>
  <table>
    <thead><tr><th>Index</th><th>Value</th><th>Threshold</th><th>Result</th></tr></thead>
    <tbody>{fit_rows}</tbody>
  </table>
</div>

<h3>Model Specification</h3>
<pre>{model_str}</pre>
"""


def _section_synthetic(validation: pd.DataFrame) -> str:
    rows = ""
    for _, r in validation.iterrows():
        mean_ok = r["Mean Δ"] < 0.1
        std_ok = r["Std Δ"] < 0.2
        row_class = "row-ok" if (mean_ok and std_ok) else "row-warn"
        rows += f"""<tr class="{row_class}">
          <td>{r['Variable']}</td>
          <td>{r['Orig Mean']}</td><td>{r['Syn Mean']}</td>
          <td>{r['Orig Std']}</td><td>{r['Syn Std']}</td>
          <td>{r['Mean Δ']}</td><td>{r['Std Δ']}</td>
        </tr>"""

    return f"""
<h2>5. Synthetic Data Validation</h2>
<div class="card" style="overflow-x:auto">
  <table>
    <thead><tr>
      <th>Variable</th>
      <th>Orig Mean</th><th>Syn Mean</th>
      <th>Orig Std</th><th>Syn Std</th>
      <th>Mean Δ</th><th>Std Δ</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
"""


def _html_footer() -> str:
    return """
<hr>
<footer>
  EFA/CFA Analyser &nbsp;|&nbsp; Report generated automatically. Results should be interpreted in context of research design and theoretical framework.
</footer>
</div>
</body>
</html>
"""
