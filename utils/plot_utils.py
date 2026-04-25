"""
Plotting Utilities — all Plotly/Seaborn charts used in the Streamlit app.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ─────────────────────────────────────────
# Colour palette (matches dark theme)
# ─────────────────────────────────────────
ACCENT = "#6c8dfa"
ACCENT2 = "#a78bfa"
GREEN = "#34d399"
RED = "#f87171"
YELLOW = "#fbbf24"
BG = "#0f1117"
SURFACE = "#1a1d27"
BORDER = "#2d3148"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"

LAYOUT_BASE = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(color=TEXT, family="Georgia, serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=[ACCENT, ACCENT2, GREEN, YELLOW, RED, "#38bdf8", "#fb923c"],
)


# ─────────────────────────────────────────
# 1. Scree Plot
# ─────────────────────────────────────────

def plot_scree(eigenvalues: list, suggested_n: int) -> go.Figure:
    n = len(eigenvalues)
    x = list(range(1, n + 1))

    colors = [GREEN if i < suggested_n else MUTED for i in range(n)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=eigenvalues,
        marker_color=colors,
        name="Eigenvalue",
        hovertemplate="Factor %{x}<br>Eigenvalue: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=eigenvalues,
        mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=7, color=ACCENT),
        name="Trend",
    ))
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color=YELLOW,
        annotation_text="Kaiser criterion (λ=1)",
        annotation_font_color=YELLOW,
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"Scree Plot — Suggested Factors: <b>{suggested_n}</b>", font=dict(color=ACCENT2)),
        xaxis=dict(title="Factor", gridcolor=BORDER, tickmode="linear"),
        yaxis=dict(title="Eigenvalue", gridcolor=BORDER),
        showlegend=False,
        height=360,
    )
    return fig


# ─────────────────────────────────────────
# 2. Loading Heatmap
# ─────────────────────────────────────────

def plot_loading_heatmap(loadings: pd.DataFrame, threshold: float = 0.4) -> go.Figure:
    z = loadings.values
    variables = loadings.index.tolist()
    factors = loadings.columns.tolist()

    # Highlight strong loadings
    annotations = []
    for i, var in enumerate(variables):
        for j, fac in enumerate(factors):
            val = z[i, j]
            ann_color = "white" if abs(val) >= threshold else MUTED
            annotations.append(dict(
                x=fac, y=var,
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(color=ann_color, size=11),
            ))

    fig = go.Figure(go.Heatmap(
        z=z,
        x=factors,
        y=variables,
        colorscale=[
            [0.0, "#1e1b4b"],
            [0.25, "#312e81"],
            [0.5, SURFACE],
            [0.75, "#1d4ed8"],
            [1.0, ACCENT],
        ],
        zmid=0,
        zmin=-1, zmax=1,
        colorbar=dict(
            title="Loading",
            tickfont=dict(color=TEXT),
            titlefont=dict(color=TEXT),
        ),
        hovertemplate="%{y} → %{x}<br>Loading: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Factor Loading Heatmap", font=dict(color=ACCENT2)),
        annotations=annotations,
        height=max(300, len(variables) * 32 + 100),
        xaxis=dict(side="top"),
    )
    return fig


# ─────────────────────────────────────────
# 3. Communality Bar Chart
# ─────────────────────────────────────────

def plot_communalities(communalities: pd.Series,
                        comm_threshold: float = 0.3) -> go.Figure:
    colors = [GREEN if v >= 0.5 else (YELLOW if v >= comm_threshold else RED)
              for v in communalities.values]

    fig = go.Figure(go.Bar(
        x=communalities.index.tolist(),
        y=communalities.values,
        marker_color=colors,
        hovertemplate="%{x}<br>Communality: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=comm_threshold, line_dash="dash", line_color=RED,
                  annotation_text=f"Threshold ({comm_threshold})", annotation_font_color=RED)
    fig.add_hline(y=0.5, line_dash="dot", line_color=YELLOW,
                  annotation_text="Good (0.50)", annotation_font_color=YELLOW)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Communalities per Variable", font=dict(color=ACCENT2)),
        xaxis=dict(title="Variable", gridcolor=BORDER, tickangle=-35),
        yaxis=dict(title="Communality", gridcolor=BORDER, range=[0, 1]),
        height=380,
    )
    return fig


# ─────────────────────────────────────────
# 4. CFA Fit Index Gauge / Bar
# ─────────────────────────────────────────

def plot_fit_indices(fit_assessment: dict) -> go.Figure:
    indices_data = fit_assessment.get("indices", {})
    if not indices_data:
        return go.Figure()

    labels, values, thresholds, colors, directions = [], [], [], [], []

    for idx, data in indices_data.items():
        labels.append(idx)
        values.append(data["value"])
        thresholds.append(data["threshold"])
        colors.append(GREEN if data["pass"] else RED)
        directions.append(data["direction"])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Fit Index Value",
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ))

    # Threshold markers
    for i, (thresh, direction) in enumerate(zip(thresholds, directions)):
        fig.add_annotation(
            x=labels[i], y=thresh,
            text=f"Threshold: {thresh}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=YELLOW,
            font=dict(color=YELLOW, size=10),
            ay=-30,
        )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="CFA Fit Indices", font=dict(color=ACCENT2)),
        xaxis=dict(title="Index", gridcolor=BORDER),
        yaxis=dict(title="Value", gridcolor=BORDER),
        showlegend=False,
        height=380,
    )
    return fig
    

    # 7. Annotations (safe formatting)
    annotations = []
    for i, r in enumerate(corr.index):
        for j, c in enumerate(corr.columns):
            val = corr.iloc[i, j]

            annotations.append(dict(
                x=c,
                y=r,
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(
                    size=9,
                    color="white" if abs(val) > 0.5 else "black"
                )
            ))

    fig.update_layout(
        title=title,
        annotations=annotations,
        height=max(350, len(cols) * 30 + 120),
        xaxis=dict(tickangle=-35),
    )

    return fig




# ─────────────────────────────────────────
# 5. Correlation Matrix Heatmap
# ─────────────────────────────────────────
"""
def plot_correlation_matrix(df: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    corr = df.corr().round(2)
    cols = corr.columns.tolist()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=cols, y=cols,
        colorscale="RdBu",
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="r", tickfont=dict(color=TEXT), titlefont=dict(color=TEXT)),
        hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
    ))

    annotations = []
    for i, r in enumerate(corr.index):
        for j, c in enumerate(corr.columns):
            val = corr.loc[r, c]
            annotations.append(dict(
                x=c, y=r, text=f"{val:.2f}", showarrow=False,
                font=dict(size=9, color="white" if abs(val) > 0.5 else TEXT)
            ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(color=ACCENT2)),
        annotations=annotations,
        height=max(350, len(cols) * 30 + 120),
        xaxis=dict(tickangle=-35),
    )
    return fig
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_correlation_matrix(df: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    # 1. Keep numeric only
    df_num = df.select_dtypes(include=[np.number])

    # 2. Remove constant columns (zero variance breaks correlation)
    df_num = df_num.loc[:, df_num.nunique() > 1]

    # 3. Safety check
    if df_num.shape[1] < 2:
        fig = go.Figure()
        fig.update_layout(title="Not enough numeric variables for correlation matrix")
        return fig

    # 4. Correlation
    corr = df_num.corr()

    # 5. Clean invalid values
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0).round(2)

    cols = corr.columns.tolist()

    # 6. Heatmap
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=cols,
        y=cols,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
    ))

    # 7. Annotations (safe formatting)
    annotations = []
    for i, r in enumerate(corr.index):
        for j, c in enumerate(corr.columns):
            val = corr.iloc[i, j]

            annotations.append(dict(
                x=c,
                y=r,
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(
                    size=9,
                    color="white" if abs(val) > 0.5 else "black"
                )
            ))

    fig.update_layout(
        title=title,
        annotations=annotations,
        height=max(350, len(cols) * 30 + 120),
        xaxis=dict(tickangle=-35),
    )

    return fig





# ─────────────────────────────────────────
# 6. Synthetic vs Original Distribution
# ─────────────────────────────────────────

def plot_synthetic_comparison(original: pd.DataFrame, synthetic: pd.DataFrame,
                               max_vars: int = 6) -> go.Figure:
    cols = original.columns[:max_vars].tolist()
    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=cols,
                        shared_xaxes=False)

    for i, col in enumerate(cols):
        row = i // ncols + 1
        col_idx = i % ncols + 1

        fig.add_trace(go.Histogram(
            x=original[col], name="Original", nbinsx=20,
            marker_color=ACCENT, opacity=0.6,
            showlegend=(i == 0),
        ), row=row, col=col_idx)

        fig.add_trace(go.Histogram(
            x=synthetic[col], name="Synthetic", nbinsx=20,
            marker_color=ACCENT2, opacity=0.6,
            showlegend=(i == 0),
        ), row=row, col=col_idx)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Original vs Synthetic Distributions", font=dict(color=ACCENT2)),
        barmode="overlay",
        height=300 * nrows,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center",
                    font=dict(color=TEXT)),
    )
    return fig
