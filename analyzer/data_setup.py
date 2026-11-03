"""Phase 1 - Data Setup: load the CSV, report and fill missing values."""

import numpy as np
import pandas as pd

from .config import (ATTENDANCE_BANDS, ATTENDANCE_COL, GRADE_BANDS, NAME_COL,
                     OVERALL_COL, SUBJECTS, SUBJECT_LABELS)


def load_data(path):
    """Read the student CSV (a path or an open file) and check it has the columns we need.

    Text in a number column (for example "absent") becomes a missing value, so
    it is repaired like any other gap.
    """
    df = pd.read_csv(path)
    required = [NAME_COL, *SUBJECTS, ATTENDANCE_COL]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")
    df = df[required].copy()
    numeric = [*SUBJECTS, ATTENDANCE_COL]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df


def missing_value_report(df):
    """Count missing values per column (only columns that have any)."""
    counts = df.isna().sum()
    return counts[counts > 0]


def impute_missing(df, strategy="regression"):
    """Fill missing subject scores.

    strategy="median"     -> use the subject's class median (simple, robust).
    strategy="regression" -> predict the score from the student's other subjects
                             and attendance with least squares (X @ w = y),
                             fitted on the students who have no gaps.

    Returns (clean_df, log) where log lists every value that was filled.
    """
    clean = df.copy()
    log = []
    # Attendance is a predictor for the scores, so fill it first (median)
    gaps = clean[ATTENDANCE_COL].isna()
    if gaps.any():
        fill = round(clean[ATTENDANCE_COL].median())
        clean.loc[gaps, ATTENDANCE_COL] = fill
        log += [{"Student": name, "Subject": "Attendance", "Filled_Value": int(fill), "Method": "median"}
                for name in clean.loc[gaps, NAME_COL]]
    for subject in SUBJECTS:
        gaps = clean[subject].isna()
        if not gaps.any():
            continue

        if strategy == "median":
            fill = np.full(gaps.sum(), clean[subject].median())
        elif strategy == "regression":
            predictors = [s for s in SUBJECTS if s != subject] + [ATTENDANCE_COL]
            train = clean.loc[~gaps].dropna(subset=predictors)
            # Design matrix with a column of ones for the intercept
            X = np.column_stack([np.ones(len(train)), train[predictors].to_numpy(float)])
            weights, *_ = np.linalg.lstsq(X, train[subject].to_numpy(float), rcond=None)
            X_gap = np.column_stack([np.ones(gaps.sum()),
                                     clean.loc[gaps, predictors].fillna(clean[predictors].median())])
            fill = X_gap @ weights
        else:
            raise ValueError("strategy must be 'median' or 'regression'")

        fill = np.clip(np.round(fill), 0, 100)
        clean.loc[gaps, subject] = fill
        for name, value in zip(clean.loc[gaps, NAME_COL], fill):
            log.append({"Student": name, "Subject": SUBJECT_LABELS[subject],
                        "Filled_Value": int(value), "Method": strategy})

    clean[SUBJECTS] = clean[SUBJECTS].round().astype(int)
    return clean, pd.DataFrame(log, columns=["Student", "Subject", "Filled_Value", "Method"])


def letter_grade(score):
    for cutoff, grade in GRADE_BANDS:
        if score >= cutoff:
            return grade
    return GRADE_BANDS[-1][1]


def attendance_band(pct):
    for cutoff, band in ATTENDANCE_BANDS:
        if pct >= cutoff:
            return band
    return ATTENDANCE_BANDS[-1][1]


def add_derived_columns(df):
    """Add overall average, overall letter grade and attendance band."""
    out = df.copy()
    out[OVERALL_COL] = out[SUBJECTS].mean(axis=1).round(2)
    out["Overall_Grade"] = out[OVERALL_COL].apply(letter_grade)
    out["Attendance_Band"] = out[ATTENDANCE_COL].apply(attendance_band)
    return out


def class_averages(df):
    """Mean score per subject - the data behind the Phase 1 bar chart."""
    return df[SUBJECTS].mean().rename(index=SUBJECT_LABELS)
