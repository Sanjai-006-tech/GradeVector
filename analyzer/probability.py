"""Phase 3 - Pass/Fail Odds: empirical probabilities from the class data.

Empirical probability = (number of students meeting the condition) / (total students).
For comparison we also fit a normal distribution to each subject and read the
same probability off the bell curve, to see whether that model fits this class.
"""

import math

import pandas as pd

from .config import (ATTENDANCE_BANDS, GRADE_BANDS, OVERALL_COL, SUBJECTS,
                     SUBJECT_LABELS)


def empirical_probability(scores, threshold):
    """P(score >= threshold), counted directly from the data."""
    scores = pd.Series(scores)
    return float((scores >= threshold).sum() / len(scores))


def normal_probability(mean, std, threshold):
    """P(score >= threshold) under a normal model N(mean, std^2).

    Scores are whole numbers, so we apply a continuity correction of 0.5:
    P(X >= 80) on integers ~ P(Z >= (79.5 - mean) / std) on the bell curve.
    """
    z = (threshold - 0.5 - mean) / std
    return 0.5 * math.erfc(z / math.sqrt(2))


def probability_table(df, pass_mark, a_mark):
    """Per-subject probabilities of passing and of an A grade."""
    rows = {}
    for col in SUBJECTS + [OVERALL_COL]:
        scores = df[col]
        rows[SUBJECT_LABELS.get(col, "Overall")] = {
            "P(Pass)": empirical_probability(scores, pass_mark),
            "P(Fail)": 1 - empirical_probability(scores, pass_mark),
            "P(A grade)": empirical_probability(scores, a_mark),
            "P(A) normal model": normal_probability(scores.mean(), scores.std(), a_mark),
            "Students with A": int((scores >= a_mark).sum()),
        }
    return pd.DataFrame(rows).T


def combined_events(df, pass_mark, a_mark):
    """Probabilities of events that span all three subjects."""
    scores = df[SUBJECTS]
    passed_all = (scores >= pass_mark).all(axis=1)
    any_a = (scores >= a_mark).any(axis=1)

    # If passing each subject were independent, P(pass all) would be the
    # product of the three individual probabilities. Comparing the two tests
    # whether failing one subject makes failing another more likely.
    p_each = [(df[s] >= pass_mark).mean() for s in SUBJECTS]
    return {
        "P(pass all three subjects)": float(passed_all.mean()),
        "P(pass all) if independent": float(math.prod(p_each)),
        "P(at least one A)": float(any_a.mean()),
        "P(A in all three subjects)": float((scores >= a_mark).all(axis=1).mean()),
        "P(fail at least one subject)": float(1 - passed_all.mean()),
    }


def conditional_by_attendance(df, pass_mark, a_mark):
    """P(event | attendance band): does showing up change the odds?"""
    rows = []
    for _, band in ATTENDANCE_BANDS:
        group = df[df["Attendance_Band"] == band]
        if group.empty:
            continue
        scores = group[SUBJECTS]
        rows.append({
            "Attendance band": band,
            "Students": len(group),
            "P(pass all)": float((scores >= pass_mark).all(axis=1).mean()),
            "P(at least one A)": float((scores >= a_mark).any(axis=1).mean()),
            "Avg overall": float(group[OVERALL_COL].mean()),
        })
    return pd.DataFrame(rows).set_index("Attendance band")


def grade_distribution(df):
    """Share of students in each overall letter grade."""
    order = [grade for _, grade in GRADE_BANDS]
    counts = df["Overall_Grade"].value_counts().reindex(order, fill_value=0)
    return pd.DataFrame({"Students": counts, "Probability": counts / len(df)})
