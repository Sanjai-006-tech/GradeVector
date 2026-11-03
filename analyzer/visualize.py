"""All charts for the project, saved as PNG files.

Every subject keeps the same colour in every chart (Math blue, Science orange,
English aqua), so a reader learns the key once.
"""

import itertools
import os

import matplotlib

matplotlib.use("Agg")  # render to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter

from .config import (A_GRADE_MARK, ATTENDANCE_COL, AXIS, BLUE_RAMP, CRITICAL,
                     DIVERGING, GOOD, GRID, INK, INK_MUTED, INK_SECONDARY,
                     NAME_COL, OVERALL_COL, PASS_MARK, SUBJECT_COLORS,
                     SUBJECT_LABELS, SUBJECTS, SURFACE)

DIVERGING_CMAP = LinearSegmentedColormap.from_list("gv_diverging", DIVERGING)
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("gv_sequential", BLUE_RAMP)


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def _save(fig, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _threshold_line(ax, value, label):
    """Dashed horizontal reference line, labelled just outside the right edge."""
    ax.axhline(value, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(label, xy=(1, value), xycoords=("axes fraction", "data"),
                xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                fontsize=8.5, color=INK_SECONDARY, annotation_clip=False)


def _place_labels(ax, xs, ys, labels, fontsize=8):
    """Label scatter points, trying a few positions around each point so that
    no two labels overlap (a simple greedy layout)."""
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    pt = fig.dpi / 72
    char_w, height = fontsize * 0.55 * pt, fontsize * 1.25 * pt
    candidates = [(7, 3, "left"), (7, -11, "left"), (-7, 3, "right"), (-7, -11, "right"),
                  (0, 8, "center"), (0, -18, "center"), (12, 12, "left"), (-12, 12, "right"),
                  (12, -22, "left"), (-12, -22, "right")]
    # Treat every marker as an obstacle so labels never cover a dot
    r = 5 * pt
    placed = [(px - r, py - r, px + r, py + r) for px, py in (to_px(p) for p in zip(xs, ys))]
    for x, y, text in zip(xs, ys, labels):
        px, py = to_px((x, y))
        width = len(text) * char_w
        for dx, dy, ha in candidates:
            left = px + dx * pt - (width if ha == "right" else width / 2 if ha == "center" else 0)
            box = (left, py + dy * pt, left + width, py + dy * pt + height)
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in placed):
                break
        placed.append(box)
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=fontsize, color=INK_SECONDARY, zorder=4)


# ---- Phase 1 ---------------------------------------------------------------

def class_averages_chart(df, out_dir, pass_mark=PASS_MARK, a_mark=A_GRADE_MARK):
    """Bar chart of the class average per subject, with +/-1 SD whiskers."""
    means = df[SUBJECTS].mean()
    stds = df[SUBJECTS].std()
    labels = [SUBJECT_LABELS[s] for s in SUBJECTS]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, means, width=0.55, color=[SUBJECT_COLORS[s] for s in SUBJECTS],
                  edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.errorbar(labels, means, yerr=stds, fmt="none", ecolor=INK_SECONDARY,
                elinewidth=1.2, capsize=6, zorder=3)
    for bar, mean, std in zip(bars, means, stds):
        ax.annotate(f"{mean:.1f}", xy=(bar.get_x() + bar.get_width() / 2, mean + std),
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=11, fontweight="bold", color=INK)
    _threshold_line(ax, pass_mark, f"Pass mark ({pass_mark})")
    _threshold_line(ax, a_mark, f"A grade ({a_mark})")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Average score (out of 100)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Class average per subject", pad=24)
    ax.text(0, 1.025, "Whiskers show one standard deviation either side of the mean",
            transform=ax.transAxes, fontsize=8.5, color=INK_SECONDARY)
    return _save(fig, out_dir, "01_class_averages.png")


# ---- Phase 2 ---------------------------------------------------------------

def distribution_chart(df, out_dir, pass_mark=PASS_MARK, a_mark=A_GRADE_MARK):
    """Histogram of each subject with mean and median marked."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    bins = np.arange(30, 105, 5)
    for ax, subject in zip(axes, SUBJECTS):
        scores = df[subject]
        ax.hist(scores, bins=bins, color=SUBJECT_COLORS[subject], edgecolor=SURFACE,
                linewidth=2, zorder=2)
        ax.axvline(scores.mean(), color=INK, linewidth=1.6, zorder=3, label=f"Mean {scores.mean():.1f}")
        ax.axvline(scores.median(), color=INK, linewidth=1.6, linestyle=(0, (2, 2)), zorder=3,
                   label=f"Median {scores.median():.1f}")
        ax.axvspan(0, pass_mark, color=CRITICAL, alpha=0.06, zorder=0)
        ax.axvspan(a_mark, 100, color=GOOD, alpha=0.06, zorder=0)
        ax.set_xlim(30, 100)
        ax.set_title(f"{SUBJECT_LABELS[subject]}  (SD {scores.std():.1f})", fontsize=11)
        ax.set_xlabel("Score")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Number of students")
    fig.suptitle("How scores are spread in each subject", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.text(0.01, 0.955, f"Red band = below pass mark ({pass_mark}),  green band = A grade ({a_mark}+)",
             fontsize=8.5, color=INK_SECONDARY)
    fig.tight_layout()
    return _save(fig, out_dir, "02_score_distributions.png")


def attendance_scatter(df, corr, out_dir):
    """Attendance vs overall score with the least-squares trend line."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x, y = df[ATTENDANCE_COL], df[OVERALL_COL]
    ax.scatter(x, y, s=64, color=SUBJECT_COLORS["Math_Score"], edgecolor=SURFACE,
               linewidth=1.5, zorder=3)
    xs = np.linspace(x.min() - 2, x.max() + 2, 50)
    ax.plot(xs, corr["slope"] * xs + corr["intercept"], color=INK_SECONDARY, linewidth=2, zorder=2)
    # Name the students furthest from the trend line
    residuals = y - (corr["slope"] * x + corr["intercept"])
    for i in residuals.abs().nlargest(4).index:
        ax.annotate(df.loc[i, NAME_COL], (x[i], y[i]), xytext=(6, -3),
                    textcoords="offset points", fontsize=8, color=INK_SECONDARY)
    ax.text(0.02, 0.97,
            f"r = {corr['r']:.2f}   ({corr['strength']})\n"
            f"r² = {corr['r_squared']:.2f}\n"
            f"overall ≈ {corr['slope']:.2f} × attendance + {corr['intercept']:.1f}",
            transform=ax.transAxes, va="top", fontsize=9.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=GRID))
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Overall average score")
    ax.set_title("Attendance vs overall performance")
    return _save(fig, out_dir, "03_attendance_vs_performance.png")


def correlation_heatmap(corr_matrix, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(corr_matrix.values, cmap=DIVERGING_CMAP, vmin=-1, vmax=1)
    labels = list(corr_matrix.columns)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr_matrix.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > 0.6 else INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_MUTED)
    ax.set_title("Correlation matrix (Pearson r)")
    return _save(fig, out_dir, "04_correlation_heatmap.png")


# ---- Phase 3 ---------------------------------------------------------------

def probability_chart(prob_table, grade_dist, out_dir, pass_mark=PASS_MARK, a_mark=A_GRADE_MARK):
    """Pass probability, A-grade probability (observed vs normal model), grade mix."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1, 1.25, 1]})
    names = list(prob_table.index)
    colors = [SUBJECT_COLORS[s] for s in SUBJECTS] + [INK_MUTED]

    ax = axes[0]
    bars = ax.bar(names, prob_table["P(Pass)"], color=colors, width=0.6, edgecolor=SURFACE,
                  linewidth=2, zorder=2)
    ax.bar_label(bars, labels=[f"{p:.0%}" for p in prob_table["P(Pass)"]], padding=3,
                 fontsize=9, color=INK)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(f"P(pass)  -  score ≥ {pass_mark}", fontsize=11)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    x = np.arange(len(names))
    width = 0.38
    observed = ax.bar(x - width / 2, prob_table["P(A grade)"], width, color=colors,
                      edgecolor=SURFACE, linewidth=2, zorder=2)
    model = ax.bar(x + width / 2, prob_table["P(A) normal model"], width, color="white",
                   edgecolor=colors, hatch="////", linewidth=1.2, zorder=2)
    ax.bar_label(observed, labels=[f"{p:.0%}" for p in prob_table["P(A grade)"]], padding=3, fontsize=8.5)
    ax.bar_label(model, labels=[f"{p:.0%}" for p in prob_table["P(A) normal model"]], padding=3,
                 fontsize=8.5, color=INK_SECONDARY)
    ax.set_xticks(x, names)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(prob_table["P(A grade)"].max(), prob_table["P(A) normal model"].max()) * 1.3 + 0.02)
    ax.set_title(f"P(A grade)  -  score ≥ {a_mark}", fontsize=11)
    ax.legend(handles=[Patch(facecolor=INK_MUTED, label="Observed in class"),
                       Patch(facecolor="white", edgecolor=INK_MUTED, hatch="////",
                             label="Normal-curve estimate")], loc="upper left")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    grade_colors = BLUE_RAMP[::-1][:len(grade_dist)]
    bars = ax.bar(grade_dist.index, grade_dist["Probability"], color=grade_colors, width=0.62,
                  edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.bar_label(bars, labels=[f"{p:.0%}" if p else "" for p in grade_dist["Probability"]],
                 padding=3, fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, grade_dist["Probability"].max() * 1.25)
    ax.set_title("Overall letter grade mix", fontsize=11)
    ax.set_xlabel("Grade (from overall average)")
    ax.grid(axis="x", visible=False)

    fig.suptitle("Empirical probabilities for a randomly selected student", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "05_probabilities.png")


# ---- Phase 4 ---------------------------------------------------------------

def cosine_comparison_chart(raw_values, std_values, out_dir):
    """Why we standardise: raw cosines are crowded near 1, standardised ones spread out."""
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.4), sharex=True)
    bins = np.linspace(-1, 1, 41)
    panels = [
        (raw_values, INK_MUTED, "Raw scores"),
        (std_values, SUBJECT_COLORS["Math_Score"], "Standardised scores (z-scores) - used for matching"),
    ]
    for ax, (values, color, label) in zip(axes, panels):
        ax.hist(values, bins=bins, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        ax.set_title(f"{label}:  cosine from {values.min():.2f} to {values.max():.2f}",
                     fontsize=10.5, pad=6)
        ax.set_ylabel("Student pairs")
        ax.grid(axis="x", visible=False)
    axes[1].set_xlabel("Cosine similarity between two students")
    fig.suptitle("Raw grade vectors all look alike; standardised ones spread out", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "06_cosine_raw_vs_standardised.png")


def similarity_heatmap(sim, names, groups, out_dir):
    """Student-by-student cosine similarity, ordered so each group sits together."""
    order = [i for g in groups for i in g]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim[np.ix_(order, order)], cmap=DIVERGING_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)), [names[i] for i in order], rotation=90, fontsize=8)
    ax.set_yticks(range(len(order)), [names[i] for i in order], fontsize=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    start = 0
    for g in groups:
        ax.add_patch(Rectangle((start - 0.5, start - 0.5), len(g), len(g), fill=False,
                               edgecolor=INK, linewidth=1.6))
        start += len(g)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Cosine similarity", color=INK_SECONDARY)
    cbar.outline.set_visible(False)
    ax.set_title("Cosine similarity of standardised grade vectors (boxes = study groups)")
    return _save(fig, out_dir, "07_similarity_heatmap.png")


def buddy_map(coords, explained, loadings, df, pairs, out_dir):
    """2-D map of students (via SVD); lines join each study-buddy pair."""
    fig, ax = plt.subplots(figsize=(10, 7.2))
    for group in pairs:  # an odd class leaves one trio, so join every two members
        for a, b in itertools.combinations(group, 2):
            ax.plot(coords[[a, b], 0], coords[[a, b], 1], color=AXIS, linewidth=2, zorder=1)
    points = ax.scatter(coords[:, 0], coords[:, 1], c=df[OVERALL_COL], cmap=SEQUENTIAL_CMAP,
                        s=80, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    cbar = fig.colorbar(points, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Overall average", color=INK_SECONDARY)
    cbar.outline.set_visible(False)

    # Describe component 2 in words from its subject weights (loadings)
    up = [SUBJECT_LABELS[s] for s, w in zip(SUBJECTS, loadings[1]) if w > 0.2]
    down = [SUBJECT_LABELS[s] for s, w in zip(SUBJECTS, loadings[1]) if w < -0.2]
    ax.set_xlabel(f"Component 1: overall level, all subjects together ({explained[0]:.0%} of variation)"
                  "   → stronger")
    ax.set_ylabel(f"Component 2 ({explained[1]:.0%} of variation)\n"
                  f"↑ relatively stronger in {' & '.join(up)}   "
                  f"↓ relatively stronger in {' & '.join(down)}")
    ax.axhline(0, color=GRID, linewidth=1, zorder=0)
    ax.axvline(0, color=GRID, linewidth=1, zorder=0)
    ax.margins(0.08)
    ax.text(0, 1.01, "Each dot is a student's standardised grade vector projected to 2-D with SVD; "
            "grey lines join study-buddy pairs", transform=ax.transAxes, fontsize=8.5, color=INK_SECONDARY)
    ax.set_title("Study Buddy map: students with similar profiles sit close together", pad=22)
    _place_labels(ax, coords[:, 0], coords[:, 1], df[NAME_COL].tolist())
    return _save(fig, out_dir, "08_study_buddy_map.png")


# ---- Statistical inference -------------------------------------------------

def inference_chart(inference, out_dir, a_mark=A_GRADE_MARK):
    """Permutation test and bootstrap for r, plus P(A) with Wilson intervals."""
    corr = inference["correlation"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1, 1]})

    ax = axes[0]
    beaten = int(np.sum(np.abs(corr["null"]) >= abs(corr["r"])))
    ax.hist(corr["null"], bins=np.linspace(-1, 1, 51), color=INK_MUTED, edgecolor=SURFACE,
            linewidth=1, zorder=2)
    ax.axvline(corr["r"], color=SUBJECT_COLORS["Math_Score"], linewidth=2.2, zorder=3)
    ax.annotate(f"real data\nr = {corr['r']:.2f}", xy=(corr["r"], 0.5), xycoords=("data", "axes fraction"),
                xytext=(-8, 0), textcoords="offset points", ha="right", fontsize=9, color=INK)
    ax.set_xlim(-1, 1)
    ax.set_title("Could chance produce this link?", fontsize=11)
    ax.set_xlabel("r after shuffling attendance")
    ax.set_ylabel("Shuffles")
    ax.text(0.02, 0.97, f"{beaten:,} of {len(corr['null']):,} shuffles reach |r| ≥ {abs(corr['r']):.2f}\n"
            f"{corr['p_text']}", transform=ax.transAxes, va="top", fontsize=8.5, color=INK_SECONDARY)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    boot = corr["bootstrap"]
    ax.axvspan(corr["ci_low"], corr["ci_high"], color=SUBJECT_COLORS["Math_Score"], alpha=0.1, zorder=0)
    ax.hist(boot, bins=40, color=SUBJECT_COLORS["Math_Score"], edgecolor=SURFACE, linewidth=1, zorder=2)
    for edge in (corr["ci_low"], corr["ci_high"]):
        ax.axvline(edge, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=3)
    ax.set_title(f"How precise is r?  95% CI {corr['ci_low']:.2f} to {corr['ci_high']:.2f}", fontsize=11)
    ax.set_xlabel("r in bootstrap resamples of the class")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    ci = inference["probability_ci"]
    labels = list(ci.index)[::-1]  # first subject at the top
    colors = {SUBJECT_LABELS[s]: SUBJECT_COLORS[s] for s in SUBJECTS}
    for y, label in enumerate(labels):
        row = ci.loc[label]
        color = colors.get(label, INK_MUTED)
        ax.plot([row["A_Low"], row["A_High"]], [y, y], color=color, linewidth=3, solid_capstyle="round", zorder=2)
        ax.scatter(row["P(A grade)"], y, s=70, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"{row['P(A grade)']:.0%}  ({row['A_Low']:.0%}-{row['A_High']:.0%})",
                    (row["A_High"], y), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK_SECONDARY)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(f"P(A grade, ≥ {a_mark}) with 95% Wilson intervals", fontsize=11)
    ax.grid(axis="y", visible=False)

    fig.suptitle("How sure can we be? Resampling the class 10,000 times", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "09_statistical_confidence.png")


# ---- Phase 5 ---------------------------------------------------------------

def early_warning_chart(df, early_warning, out_dir):
    """Fitted logistic curves for 'needs support' and 'at least one A' against attendance."""
    panels = [
        ("risk", early_warning["needs_support"], CRITICAL,
         f"P(needs support)  -  overall below {early_warning['risk_mark']} or a failed subject"),
        ("a_grade", early_warning["any_a"], GOOD, "P(at least one A grade)"),
    ]
    panels = [p for p in panels if early_warning[p[0]]]
    if not panels:
        return None
    attendance = df[ATTENDANCE_COL].to_numpy(float)
    xs = np.linspace(min(50, attendance.min()), 100, 200)
    jitter = np.random.default_rng(3).uniform(-0.025, 0.025, len(attendance))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 4.4), squeeze=False)
    for ax, (key, outcome, color, title) in zip(axes[0], panels):
        fit = early_warning[key]
        model, metrics = fit["model"], fit["metrics"]
        ax.plot(xs, model.predict_proba(xs), color=color, linewidth=2.4, zorder=3)
        ax.scatter(attendance, outcome + jitter * np.where(outcome == 1, -1, 1), s=36,
                   color=np.where(outcome == 1, color, INK_MUTED), edgecolor=SURFACE,
                   linewidth=1, zorder=4, alpha=0.9)
        _threshold_line(ax, 0.5, "50%")
        x50 = model.attendance_for(0.5)
        if x50 is not None and xs[0] <= x50 <= 100:
            ax.axvline(x50, color=INK_MUTED, linewidth=1, linestyle=(0, (2, 2)), zorder=1)
            # Put the label on the side where the curve is below 50%, so it never crosses it
            side = -1 if model.slope > 0 else 1
            ax.annotate(f"50/50 at {x50:.0f}% attendance", (x50, 0.5), xytext=(6 * side, 8),
                        textcoords="offset points", ha="left" if side > 0 else "right",
                        fontsize=8.5, color=INK)
        if key == "risk" and "attendance_low_risk" in early_warning:
            low = early_warning["attendance_low_risk"]
            ax.axvspan(xs[0], early_warning["attendance_50"], color=CRITICAL, alpha=0.07, zorder=0)
            ax.axvspan(early_warning["attendance_50"], low, color=CRITICAL, alpha=0.03, zorder=0)
            ax.annotate(f"below 20% risk\nfrom {low:.0f}% attendance", (low, 0.2), xytext=(8, 26),
                        textcoords="offset points", fontsize=8.5, color=INK_SECONDARY, va="center")
        ax.set_xlim(xs[0], 100)
        ax.set_ylim(-0.06, 1.06)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Attendance (%)")
        ax.set_title(title, fontsize=11)
        ax.text(0, -0.24, f"Leave-one-out check: accuracy {metrics['accuracy']:.0%} "
                f"(guessing the majority: {metrics['baseline_accuracy']:.0%}),  AUC {metrics['auc']:.2f}",
                transform=ax.transAxes, fontsize=8.5, color=INK_SECONDARY)
    axes[0][0].set_ylabel("Predicted probability")
    fig.suptitle("Early warning: logistic regression on attendance (dots = actual students)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "10_early_warning.png")


def make_all_charts(results, out_dir):
    """Draw every chart from the results of pipeline.run_analysis."""
    apply_style()
    df = results["data"]
    cfg = results["settings"]
    lin = results["linear_algebra"]
    names = df[NAME_COL].tolist()
    charts = [
        class_averages_chart(df, out_dir, cfg["pass_mark"], cfg["a_mark"]),
        distribution_chart(df, out_dir, cfg["pass_mark"], cfg["a_mark"]),
        attendance_scatter(df, results["attendance_correlation"], out_dir),
        correlation_heatmap(results["correlation_matrix"], out_dir),
        probability_chart(results["probabilities"], results["grade_distribution"], out_dir,
                          cfg["pass_mark"], cfg["a_mark"]),
        cosine_comparison_chart(lin["raw_pair_values"], lin["std_pair_values"], out_dir),
        similarity_heatmap(lin["similarity"], names, lin["groups"], out_dir),
        buddy_map(lin["pca_coords"], lin["pca_explained"], lin["pca_loadings"], df,
                  lin["pairs"], out_dir),
        inference_chart(results["inference"], out_dir, cfg["a_mark"]),
        early_warning_chart(df, results["early_warning"], out_dir),
    ]
    return [path for path in charts if path]  # a model too small to fit draws nothing
