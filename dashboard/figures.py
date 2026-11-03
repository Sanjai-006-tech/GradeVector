"""Interactive Plotly charts for the dashboard.

Each one mirrors a Matplotlib chart in analyzer/visualize.py (same colours, same
meaning) and adds hover details, zoom and live updates. Only figure building
lives here; every number comes from analyzer.run_analysis.
"""

import itertools

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyzer.config import (ATTENDANCE_COL, AXIS, BLUE_RAMP, CRITICAL, DIVERGING, GOOD, GRID,
                             INK, INK_MUTED, INK_SECONDARY, NAME_COL, OVERALL_COL, SUBJECT_COLORS,
                             SUBJECT_LABELS, SUBJECTS, SURFACE)
from analyzer.prediction import predict_score

FONT = "Segoe UI, system-ui, -apple-system, sans-serif"
LABELS = [SUBJECT_LABELS[s] for s in SUBJECTS]
COLORS = [SUBJECT_COLORS[s] for s in SUBJECTS]
DIVERGING_SCALE = [[i / (len(DIVERGING) - 1), c] for i, c in enumerate(DIVERGING)]
SEQUENTIAL_SCALE = [[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)]
LIGHT_BLUE, DARK_BLUE = BLUE_RAMP[0], BLUE_RAMP[-2]


def styled(fig, height=380, title=None, subtitle=None, right=8):
    """Shared look: warm off-white surface, light grid, left-aligned bold title."""
    top = 20 + (34 if title else 0) + (22 if subtitle else 0)
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=right, t=top, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK_SECONDARY),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, color=INK)),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
        bargap=0.35,
    )
    if title:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.005, xanchor="left", y=0.98, yanchor="top",
                                     font=dict(size=16, color=INK),
                                     subtitle=dict(text=subtitle or "", font=dict(size=12, color=INK_SECONDARY))))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, automargin=True)
    return fig


def _reference_line(fig, value, text):
    """Dashed horizontal line labelled just outside the right edge (needs a right margin)."""
    fig.add_hline(y=value, line_dash="dash", line_color=INK_MUTED, line_width=1)
    fig.add_annotation(x=1, xref="paper", y=value, yref="y", text=text, showarrow=False, xanchor="left",
                       xshift=6, font=dict(size=11, color=INK_SECONDARY))


# ---- Overview / statistics -------------------------------------------------

def class_averages(df, pass_mark, a_mark):
    means, stds = df[SUBJECTS].mean(), df[SUBJECTS].std()
    fig = go.Figure(go.Bar(
        x=LABELS, y=means, marker=dict(color=COLORS, line=dict(color=SURFACE, width=2)),
        error_y=dict(type="data", array=stds, color=INK_SECONDARY, thickness=1.3, width=8),
        text=[f"<b>{m:.1f}</b>" for m in means], textposition="inside", insidetextanchor="start",
        textfont=dict(color="white", size=14), customdata=stds,
        hovertemplate="<b>%{x}</b><br>Mean %{y:.1f}<br>SD %{customdata:.1f}<extra></extra>"))
    _reference_line(fig, pass_mark, f"Pass mark {pass_mark}")
    _reference_line(fig, a_mark, f"A grade {a_mark}")
    fig.update_yaxes(range=[0, 100], title="Average score")
    fig.update_xaxes(showgrid=False)
    return styled(fig, 360, "Class average per subject", "Whiskers show one standard deviation either side",
                  right=70)


def grade_mix(grade_distribution):
    probs = grade_distribution["Probability"]
    fig = go.Figure(go.Bar(
        x=grade_distribution.index, y=probs, marker=dict(color=BLUE_RAMP[::-1][:len(probs)]),
        text=[f"{p:.0%}" if p else "" for p in probs], textposition="outside",
        customdata=grade_distribution["Students"],
        hovertemplate="Grade %{x}<br>%{customdata} students (%{y:.0%})<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", range=[0, probs.max() * 1.25 + 0.01])
    fig.update_xaxes(showgrid=False, title="Grade from overall average")
    return styled(fig, 360, "Overall letter grade mix", "Share of the class in each band")


def score_histograms(df, pass_mark, a_mark):
    low = min(30, 5 * int(df[SUBJECTS].min().min() // 5))
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[f"{SUBJECT_LABELS[s]}  ·  SD {df[s].std():.1f}" for s in SUBJECTS])
    for col, subject in enumerate(SUBJECTS, start=1):
        scores = df[subject]
        fig.add_trace(go.Histogram(
            x=scores, xbins=dict(start=low, end=100, size=5), marker=dict(color=SUBJECT_COLORS[subject],
                                                                          line=dict(color=SURFACE, width=1.5)),
            name=SUBJECT_LABELS[subject], showlegend=False,
            hovertemplate="Scores %{x}<br>%{y} students<extra></extra>"), row=1, col=col)
        fig.add_vrect(x0=low, x1=pass_mark, fillcolor=CRITICAL, opacity=0.06, line_width=0, layer="below",
                      row=1, col=col)
        fig.add_vrect(x0=a_mark, x1=100, fillcolor=GOOD, opacity=0.06, line_width=0, layer="below",
                      row=1, col=col)
        fig.add_vline(x=scores.mean(), line_color=INK, line_width=1.6, row=1, col=col)
        fig.add_vline(x=scores.median(), line_color=INK, line_width=1.6, line_dash="dot", row=1, col=col)
    fig.update_xaxes(range=[low, 100], showgrid=False)
    fig.update_yaxes(title="Students", row=1, col=1)
    fig.update_annotations(font=dict(size=13, color=INK))
    return styled(fig, 330, "How scores are spread",
                  "Solid line = mean, dotted = median · red band below the pass mark, green band = A grade")


def attendance_scatter(df, corr, trend, highlight=None):
    x, y = df[ATTENDANCE_COL], df[OVERALL_COL]
    xs = np.linspace(x.min() - 2, min(100, x.max() + 2), 60)
    band = np.array([predict_score(trend, v) for v in xs])
    fig = go.Figure([
        go.Scatter(x=xs, y=band[:, 2], line=dict(width=0), hoverinfo="skip", showlegend=False),
        go.Scatter(x=xs, y=band[:, 1], line=dict(width=0), fill="tonexty",
                   fillcolor="rgba(42,120,214,0.10)", hoverinfo="skip", name="95% prediction band"),
        go.Scatter(x=xs, y=band[:, 0], line=dict(color=INK_SECONDARY, width=2), name="Least-squares line",
                   hoverinfo="skip"),
        go.Scatter(x=x, y=y, mode="markers", name="Student",
                   marker=dict(size=11, color=SUBJECT_COLORS["Math_Score"], line=dict(color=SURFACE, width=1.5)),
                   customdata=np.column_stack([df[NAME_COL], df["Overall_Grade"]]),
                   hovertemplate="<b>%{customdata[0]}</b><br>Attendance %{x}%<br>"
                                 "Overall %{y:.1f} (grade %{customdata[1]})<extra></extra>"),
    ])
    if highlight is not None:
        fig.add_trace(go.Scatter(x=[x.iloc[highlight]], y=[y.iloc[highlight]], mode="markers+text",
                                 text=[df[NAME_COL].iloc[highlight]], textposition="top center",
                                 marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color=INK, width=2.5)),
                                 hoverinfo="skip", showlegend=False))
    fig.add_annotation(x=0.01, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top", align="left",
                       showarrow=False, bgcolor="white", bordercolor=GRID, borderpad=6,
                       text=f"r = {corr['r']:.2f} ({corr['strength']})<br>r² = {corr['r_squared']:.2f}<br>"
                            f"overall ≈ {corr['slope']:.2f} × attendance + {corr['intercept']:.1f}",
                       font=dict(color=INK, size=12))
    fig.update_xaxes(title="Attendance (%)")
    fig.update_yaxes(title="Overall average")
    styled(fig, 420, "Attendance vs overall performance",
           "Shaded band: where 95% of students at that attendance are expected to score")
    # Legend in the empty lower-right corner instead of under the axis title
    return fig.update_layout(legend=dict(orientation="v", yanchor="bottom", y=0.02, xanchor="right", x=0.99,
                                         bgcolor="rgba(255,255,255,0.85)"))


def correlation_heatmap(matrix):
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=list(matrix.columns), y=list(matrix.index), zmin=-1, zmax=1,
        colorscale=DIVERGING_SCALE, texttemplate="%{z:.2f}", textfont=dict(size=12),
        hovertemplate="%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>", colorbar=dict(thickness=12, outlinewidth=0)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return styled(fig, 420, "Correlation matrix", "Pearson r between every pair of columns")


# ---- Inference -------------------------------------------------------------

def resampling_chart(inference):
    corr = inference["correlation"]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08, subplot_titles=[
        "Permutation test: r when attendance is shuffled",
        f"Bootstrap: r across resampled classes (95% CI {corr['ci_low']:.2f}–{corr['ci_high']:.2f})"])
    fig.add_trace(go.Histogram(x=corr["null"], xbins=dict(start=-1, end=1, size=0.04), marker_color=INK_MUTED,
                               name="Shuffled", hovertemplate="r ≈ %{x}<br>%{y} shuffles<extra></extra>"), 1, 1)
    fig.add_vline(x=corr["r"], line_color=SUBJECT_COLORS["Math_Score"], line_width=3, row=1, col=1,
                  annotation_text=f"real r = {corr['r']:.2f}<br>{corr['p_text']}", annotation_position="top left",
                  annotation_font=dict(color=INK, size=12))
    fig.add_vrect(x0=corr["ci_low"], x1=corr["ci_high"], fillcolor=SUBJECT_COLORS["Math_Score"], opacity=0.1,
                  line_width=0, layer="below", row=1, col=2)
    fig.add_trace(go.Histogram(x=corr["bootstrap"], nbinsx=40, marker_color=SUBJECT_COLORS["Math_Score"],
                               name="Bootstrap", hovertemplate="r ≈ %{x}<br>%{y} resamples<extra></extra>"), 1, 2)
    fig.update_traces(marker_line=dict(color=SURFACE, width=1))
    fig.update_xaxes(range=[-1, 1], row=1, col=1)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(showlegend=False)
    fig.update_annotations(font_size=13)
    return styled(fig, 340)


def mean_ci_chart(mean_ci):
    fig = go.Figure()
    colors = dict(zip(LABELS, COLORS))
    for label, row in mean_ci.iloc[::-1].iterrows():
        fig.add_trace(go.Scatter(
            x=[row["Mean"]], y=[label], mode="markers", showlegend=False,
            marker=dict(size=13, color=colors.get(label, INK_MUTED), line=dict(color=SURFACE, width=1.5)),
            error_x=dict(type="data", symmetric=False, array=[row["CI_High"] - row["Mean"]],
                         arrayminus=[row["Mean"] - row["CI_Low"]], color=colors.get(label, INK_MUTED),
                         thickness=4, width=0),
            hovertemplate=f"<b>{label}</b><br>mean {row['Mean']:.1f}<br>"
                          f"95% CI {row['CI_Low']:.1f} to {row['CI_High']:.1f}<extra></extra>"))
    fig.update_xaxes(title="Class mean")
    fig.update_yaxes(showgrid=False)
    return styled(fig, 300, "Class means with bootstrap 95% intervals")


# ---- Probability -----------------------------------------------------------

def probability_chart(probability_ci, probabilities):
    labels = list(probability_ci.index)
    colors = [dict(zip(LABELS, COLORS)).get(label, INK_MUTED) for label in labels]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=["P(pass)", "P(A grade): observed vs normal-curve model"])
    for col, (p, low, high) in enumerate([("P(Pass)", "Pass_Low", "Pass_High"),
                                          ("P(A grade)", "A_Low", "A_High")], start=1):
        values = probability_ci[p]
        fig.add_trace(go.Bar(
            x=labels, y=values, marker_color=colors, name="Observed", showlegend=col == 2,
            text=[f"{v:.0%}" for v in values], textposition="inside", insidetextanchor="start",
            textfont=dict(color="white"),
            error_y=dict(type="data", symmetric=False, array=probability_ci[high] - values,
                         arrayminus=values - probability_ci[low], color=INK_SECONDARY, thickness=1.2, width=6),
            customdata=np.column_stack([probability_ci[low], probability_ci[high]]),
            hovertemplate="<b>%{x}</b><br>%{y:.1%}<br>95% CI %{customdata[0]:.1%} to %{customdata[1]:.1%}"
                          "<extra></extra>"), 1, col)
    fig.add_trace(go.Scatter(
        x=labels, y=probabilities.loc[labels, "P(A) normal model"], mode="markers", name="Normal-curve estimate",
        marker=dict(symbol="diamond-open", size=13, color=INK, line=dict(width=2)),
        hovertemplate="<b>%{x}</b><br>normal model %{y:.1%}<extra></extra>"), 1, 2)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero", row=1, col=2)
    fig.update_xaxes(showgrid=False)
    fig.update_annotations(font_size=13)
    return styled(fig, 380, "Probabilities for a randomly selected student",
                  "Whiskers are 95% Wilson confidence intervals")


def conditional_chart(conditional):
    bands = list(conditional.index)
    fig = go.Figure([
        go.Bar(x=bands, y=conditional["P(pass all)"], name="P(pass all subjects)", marker_color=LIGHT_BLUE,
               text=[f"{v:.0%}" for v in conditional["P(pass all)"]], textposition="outside",
               hovertemplate="%{x}<br>P(pass all) %{y:.0%}<extra></extra>"),
        go.Bar(x=bands, y=conditional["P(at least one A)"], name="P(at least one A)", marker_color=DARK_BLUE,
               text=[f"{v:.0%}" for v in conditional["P(at least one A)"]], textposition="outside",
               hovertemplate="%{x}<br>P(≥1 A) %{y:.0%}<extra></extra>"),
    ])
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.15])
    fig.update_xaxes(showgrid=False)
    return styled(fig, 380, "Conditional probability: P(event | attendance band)")


# ---- Study groups ----------------------------------------------------------

def buddy_map(results, groups, highlight=None):
    lin, df = results["linear_algebra"], results["data"]
    coords, explained = lin["pca_coords"], lin["pca_explained"]
    group_of = {i: n for n, g in enumerate(groups, start=1) for i in g}
    edge_x, edge_y = [], []
    for g in groups:
        for a, b in itertools.combinations(g, 2):
            edge_x += [coords[a, 0], coords[b, 0], None]
            edge_y += [coords[a, 1], coords[b, 1], None]
    custom = np.column_stack([df[NAME_COL], df[SUBJECTS].to_numpy(), df[OVERALL_COL],
                              [group_of.get(i, "-") for i in range(len(df))]])
    fig = go.Figure([
        go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color=AXIS, width=2), hoverinfo="skip",
                   showlegend=False),
        go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="markers+text", text=df[NAME_COL],
                   textposition="top center", textfont=dict(size=10, color=INK_SECONDARY),
                   marker=dict(size=13, color=df[OVERALL_COL], colorscale=SEQUENTIAL_SCALE,
                               line=dict(color=SURFACE, width=1.5),
                               colorbar=dict(title=dict(text="Overall", side="right"), thickness=12,
                                             outlinewidth=0)),
                   customdata=custom, showlegend=False,
                   hovertemplate="<b>%{customdata[0]}</b> · group %{customdata[5]}<br>"
                                 "Math %{customdata[1]} · Science %{customdata[2]} · English %{customdata[3]}<br>"
                                 "Overall %{customdata[4]:.1f}<extra></extra>"),
    ])
    if highlight is not None:
        members = next((g for g in groups if highlight in g), [highlight])
        fig.add_trace(go.Scatter(x=coords[members, 0], y=coords[members, 1], mode="markers", hoverinfo="skip",
                                 marker=dict(size=24, color="rgba(0,0,0,0)", line=dict(color=INK, width=2.5)),
                                 showlegend=False))
    fig.update_xaxes(title=f"Component 1: overall level ({explained[0]:.0%} of variation)  →  stronger")
    fig.update_yaxes(title=f"Component 2 ({explained[1]:.0%})")
    return styled(fig, 560, "Study Buddy map",
                  "Standardised grade vectors projected to 2-D with SVD · lines join students who study together")


def similarity_heatmap(sim, names, groups):
    order = [i for g in groups for i in g]
    labels = [names[i] for i in order]
    fig = go.Figure(go.Heatmap(
        z=sim[np.ix_(order, order)], x=labels, y=labels, zmin=-1, zmax=1, colorscale=DIVERGING_SCALE,
        colorbar=dict(title=dict(text="cos θ", side="right"), thickness=12, outlinewidth=0),
        hovertemplate="%{y} & %{x}<br>cos θ = %{z:.3f}<extra></extra>"))
    start = 0
    for g in groups:
        fig.add_shape(type="rect", x0=start - 0.5, x1=start + len(g) - 0.5, y0=start - 0.5,
                      y1=start + len(g) - 0.5, line=dict(color=INK, width=1.5))
        start += len(g)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=10))
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10), tickangle=-90)
    return styled(fig, 620, "Cosine similarity matrix", "Ordered by group; boxes mark each group")


def cosine_comparison(raw_values, std_values):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.16, subplot_titles=[
        f"Raw scores: {raw_values.min():.2f} to {raw_values.max():.2f}",
        f"Standardised (z-scores): {std_values.min():.2f} to {std_values.max():.2f}"])
    for row, (values, color) in enumerate([(raw_values, INK_MUTED),
                                           (std_values, SUBJECT_COLORS["Math_Score"])], start=1):
        fig.add_trace(go.Histogram(x=values, xbins=dict(start=-1, end=1, size=0.05), marker_color=color,
                                   marker_line=dict(color=SURFACE, width=1), showlegend=False,
                                   hovertemplate="cos θ ≈ %{x}<br>%{y} pairs<extra></extra>"), row, 1)
    fig.update_xaxes(range=[-1, 1], showgrid=False)
    fig.update_xaxes(title="Cosine similarity between two students", row=2, col=1)
    fig.update_annotations(font_size=13)
    return styled(fig, 380)


# ---- Early warning ---------------------------------------------------------

def logistic_chart(df, early_warning, key, attendance=None):
    fit = early_warning[key]
    is_risk = key == "risk"
    color = CRITICAL if is_risk else GOOD
    outcome = early_warning["needs_support"] if is_risk else early_warning["any_a"]
    x = df[ATTENDANCE_COL].to_numpy(float)
    xs = np.linspace(min(40, x.min()), 100, 200)
    jitter = np.random.default_rng(3).uniform(0, 0.03, len(x)) * np.where(outcome == 1, -1, 1)
    yes, no = ("Needs support", "On track") if is_risk else ("Has an A", "No A")
    fig = go.Figure([
        go.Scatter(x=xs, y=fit["model"].predict_proba(xs), mode="lines", line=dict(color=color, width=3),
                   name="Model", hovertemplate="Attendance %{x:.0f}%<br>P = %{y:.1%}<extra></extra>"),
        go.Scatter(x=x, y=outcome + jitter, mode="markers", name="Students",
                   marker=dict(size=9, color=np.where(outcome == 1, color, INK_MUTED),
                               line=dict(color=SURFACE, width=1)),
                   customdata=np.column_stack([df[NAME_COL], np.where(outcome == 1, yes, no),
                                               fit["loo_probs"]]),
                   hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>Attendance %{x}%<br>"
                                 "Out-of-sample P = %{customdata[2]:.1%}<extra></extra>"),
    ])
    _reference_line(fig, 0.5, "50%")
    if attendance is not None:
        p = float(fit["model"].predict_proba(attendance))
        fig.add_vline(x=attendance, line_color=INK, line_width=1.5, line_dash="dot")
        fig.add_trace(go.Scatter(x=[attendance], y=[p], mode="markers+text", text=[f"<b>{p:.0%}</b>"],
                                 textposition="middle right" if is_risk else "middle left",
                                 textfont=dict(size=15, color=INK),
                                 marker=dict(size=15, color=color, line=dict(color="white", width=2)),
                                 hoverinfo="skip", showlegend=False))
    title = (f"P(needs support): overall below {early_warning['risk_mark']} or a failed subject"
             if is_risk else "P(at least one A grade)")
    m = fit["metrics"]
    fig.update_layout(showlegend=False)
    fig.update_yaxes(tickformat=".0%", range=[-0.07, 1.07], title="Probability")
    fig.update_xaxes(range=[xs[0], 100], title="Attendance (%)")
    return styled(fig, 400, title, f"Leave-one-out accuracy {m['accuracy']:.0%} "
                                   f"(majority guess {m['baseline_accuracy']:.0%}) · AUC {m['auc']:.2f}",
                  right=44)


# ---- Student explorer ------------------------------------------------------

def student_radar(df, i, buddy=None):
    closed = LABELS + LABELS[:1]

    def ring(values):
        values = list(values)
        return values + values[:1]

    fig = go.Figure([
        go.Scatterpolar(r=ring(df[SUBJECTS].mean()), theta=closed, name="Class average", fill="toself",
                        line=dict(color=INK_MUTED, width=1.5), fillcolor="rgba(137,135,129,0.15)",
                        hovertemplate="%{theta}: %{r:.1f}<extra>Class average</extra>"),
    ])
    if buddy is not None:
        fig.add_trace(go.Scatterpolar(r=ring(df.iloc[buddy][SUBJECTS]), theta=closed,
                                      name=f"Buddy: {df.iloc[buddy][NAME_COL]}",
                                      line=dict(color=SUBJECT_COLORS["Science_Score"], width=2, dash="dash"),
                                      hovertemplate="%{theta}: %{r}<extra>buddy</extra>"))
    fig.add_trace(go.Scatterpolar(r=ring(df.iloc[i][SUBJECTS]), theta=closed, name=df.iloc[i][NAME_COL],
                                  fill="toself", line=dict(color=SUBJECT_COLORS["Math_Score"], width=3),
                                  fillcolor="rgba(42,120,214,0.18)",
                                  hovertemplate="%{theta}: %{r}<extra></extra>"))
    fig.update_layout(polar=dict(bgcolor=SURFACE, radialaxis=dict(range=[0, 100], gridcolor=GRID, linecolor=AXIS,
                                                                  tickvals=[50, 100], angle=30, tickangle=30,
                                                                  tickfont=dict(size=10, color=INK_MUTED)),
                                 angularaxis=dict(gridcolor=GRID, linecolor=AXIS, rotation=90, direction="clockwise",
                                                  tickfont=dict(size=13, color=INK))))
    return styled(fig, 380, "Score profile")


def student_strengths(z_row):
    fig = go.Figure(go.Bar(
        x=z_row, y=LABELS, orientation="h", marker_color=COLORS,
        text=[f"{z:+.2f} SD" for z in z_row], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.2f} standard deviations from the class average<extra></extra>"))
    edge = max(2.5, float(np.abs(z_row).max()) + 1.4)
    fig.add_vline(x=0, line_color=INK_SECONDARY, line_width=1.2)
    fig.update_xaxes(range=[-edge, edge], title="z-score (0 = class average)")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return styled(fig, 260, "Relative to the class", "The grade vector used for buddy matching")
