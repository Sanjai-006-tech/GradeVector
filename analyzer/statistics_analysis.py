"""Phase 2 - Statistical Analysis: central tendency, spread and correlation."""

import numpy as np
import pandas as pd

from .config import ATTENDANCE_COL, NAME_COL, OVERALL_COL, SUBJECTS, SUBJECT_LABELS


def describe_subjects(df):
    """Mean, median, standard deviation and range for every subject.

    Standard deviation is the sample version (n - 1 in the denominator),
    since one class is a sample of all the students who could take the course.
    """
    cols = SUBJECTS + [OVERALL_COL]
    summary = pd.DataFrame({
        "Mean": df[cols].mean(),
        "Median": df[cols].median(),
        "Std_Dev": df[cols].std(ddof=1),
        "Min": df[cols].min(),
        "Max": df[cols].max(),
    })
    summary["Range"] = summary["Max"] - summary["Min"]
    # Coefficient of variation: spread relative to the mean (comparable across subjects)
    summary["CV_%"] = summary["Std_Dev"] / summary["Mean"] * 100
    return summary.rename(index={**SUBJECT_LABELS, OVERALL_COL: "Overall"}).round(2)


def pearson_r(x, y):
    """Pearson correlation written out from its definition.

    r = sum((x - x_mean)(y - y_mean)) / sqrt(sum((x - x_mean)^2) * sum((y - y_mean)^2))
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx, dy = x - x.mean(), y - y.mean()
    return float((dx @ dy) / np.sqrt((dx @ dx) * (dy @ dy)))


def describe_strength(r):
    size = abs(r)
    if size >= 0.7:
        strength = "strong"
    elif size >= 0.4:
        strength = "moderate"
    elif size >= 0.2:
        strength = "weak"
    else:
        return "no meaningful"
    return f"{strength} {'positive' if r > 0 else 'negative'}"


def attendance_correlation(df):
    """How strongly does attendance track overall performance?

    Returns r, r^2, the least-squares line (overall = slope * attendance + intercept)
    and a plain-English interpretation.
    """
    x, y = df[ATTENDANCE_COL], df[OVERALL_COL]
    r = pearson_r(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "r": r,
        "r_squared": r ** 2,
        "slope": float(slope),
        "intercept": float(intercept),
        "strength": describe_strength(r),
        "per_subject": {SUBJECT_LABELS[s]: pearson_r(df[ATTENDANCE_COL], df[s]) for s in SUBJECTS},
        "interpretation": (
            f"There is a {describe_strength(r)} correlation (r = {r:.2f}) between attendance and "
            f"overall score. Attendance explains about {r ** 2:.0%} of the variation in "
            f"overall scores; each extra 10% of attendance goes with about "
            f"{slope * 10:.1f} more marks on average."
        ),
    }


def correlation_matrix(df):
    cols = SUBJECTS + [ATTENDANCE_COL, OVERALL_COL]
    labels = {**SUBJECT_LABELS, ATTENDANCE_COL: "Attendance", OVERALL_COL: "Overall"}
    return df[cols].corr().rename(index=labels, columns=labels)


def top_performers(df, n=5, ascending=False):
    cols = [NAME_COL, *SUBJECTS, ATTENDANCE_COL, OVERALL_COL, "Overall_Grade"]
    return df.sort_values(OVERALL_COL, ascending=ascending)[cols].head(n).reset_index(drop=True)


def strongest_subject_counts(df):
    """How many students have each subject as their personal best."""
    best = df[SUBJECTS].idxmax(axis=1).map(SUBJECT_LABELS)
    return best.value_counts().reindex(SUBJECT_LABELS.values(), fill_value=0)
