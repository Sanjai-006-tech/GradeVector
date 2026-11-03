"""Statistical inference: how sure can we be about numbers from one class of 30?

One class is a small sample, so every figure in Phases 2 and 3 carries sampling
error. This module puts error bars on those figures with no statistics library,
using two resampling ideas and one formula:

  Bootstrap         Resample the students with replacement thousands of times and
                    recompute the statistic each time. The middle 95% of those
                    values is a 95% confidence interval.
  Permutation test  Shuffle one column to break any real link, recompute the
                    statistic many times, and count how often chance alone does
                    as well as the real data. That share is the p-value.
  Wilson interval   A confidence interval for a proportion that stays sensible for
                    small classes and for proportions near 0% or 100%.
"""

import math

import numpy as np
import pandas as pd

from .config import ATTENDANCE_BANDS, ATTENDANCE_COL, OVERALL_COL, SUBJECTS, SUBJECT_LABELS
from .statistics_analysis import pearson_r

N_RESAMPLES = 10_000
SEED = 7
Z_95 = 1.959964  # standard normal quantile for a two-sided 95% interval


def wilson_interval(successes, n, z=Z_95):
    """95% confidence interval for a proportion successes / n (Wilson score).

    Unlike p +/- 1.96 * sqrt(p(1 - p) / n), it never leaves [0, 1] and does not
    collapse to zero width when every student passes.
    """
    if n == 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _rowwise_r(X, Y):
    """Pearson r for every row of two equally shaped matrices at once."""
    dx = X - X.mean(axis=1, keepdims=True)
    dy = Y - Y.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (dx * dy).sum(axis=1) / np.sqrt((dx * dx).sum(axis=1) * (dy * dy).sum(axis=1))


def _resample_index(rng, n, trials):
    """trials x n matrix of row numbers drawn with replacement."""
    return rng.integers(0, n, size=(trials, n))


def bootstrap_mean_ci(values, trials=N_RESAMPLES, seed=SEED):
    """(mean, low, high): percentile bootstrap 95% interval for the mean."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[_resample_index(rng, len(values), trials)].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def bootstrap_correlation(x, y, trials=N_RESAMPLES, seed=SEED):
    """Bootstrap distribution of Pearson r, resampling (x, y) pairs together."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    idx = _resample_index(np.random.default_rng(seed), len(x), trials)
    r = _rowwise_r(x[idx], y[idx])
    return r[~np.isnan(r)]  # a resample where every x is equal has no defined r


def permutation_test_correlation(x, y, trials=N_RESAMPLES, seed=SEED):
    """Two-sided p-value for H0: no link between x and y.

    Shuffling y destroys any real relationship, so the shuffled r values show
    what correlation chance alone produces. The +1 counts the observed data as
    one of the arrangements, so the p-value is never exactly zero.
    Returns (p_value, null_distribution).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    shuffled = rng.permuted(np.tile(y, (trials, 1)), axis=1)
    null = _rowwise_r(np.tile(x, (trials, 1)), shuffled)
    observed = pearson_r(x, y)
    p_value = (np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (trials + 1)
    return float(p_value), null


def permutation_test_means(a, b, trials=N_RESAMPLES, seed=SEED):
    """Two-sided p-value for H0: groups a and b have the same mean."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    pooled = rng.permuted(np.tile(np.concatenate([a, b]), (trials, 1)), axis=1)
    null = pooled[:, :len(a)].mean(axis=1) - pooled[:, len(a):].mean(axis=1)
    observed = a.mean() - b.mean()
    return float((np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (trials + 1))


def cohens_d(a, b):
    """Effect size: difference in means measured in pooled standard deviations."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                       / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled else float("inf")


def format_p(p, trials=N_RESAMPLES):
    """p-value as text; resampling cannot resolve anything below 1 / trials."""
    return f"p < {1 / trials:g}" if p <= 1 / trials else f"p = {p:.4f}"


# ---- Summaries used by the pipeline ---------------------------------------

def mean_intervals(df):
    """Bootstrap 95% interval for the class mean of each subject."""
    rows = {}
    for col in SUBJECTS + [OVERALL_COL]:
        mean, low, high = bootstrap_mean_ci(df[col])
        rows[SUBJECT_LABELS.get(col, "Overall")] = {"Mean": mean, "CI_Low": low, "CI_High": high,
                                                    "Margin": (high - low) / 2}
    return pd.DataFrame(rows).T


def probability_intervals(df, pass_mark, a_mark):
    """Wilson 95% intervals for P(pass) and P(A) in each subject."""
    n = len(df)
    rows = {}
    for col in SUBJECTS + [OVERALL_COL]:
        passed, a_grades = int((df[col] >= pass_mark).sum()), int((df[col] >= a_mark).sum())
        pass_low, pass_high = wilson_interval(passed, n)
        a_low, a_high = wilson_interval(a_grades, n)
        rows[SUBJECT_LABELS.get(col, "Overall")] = {
            "P(Pass)": passed / n, "Pass_Low": pass_low, "Pass_High": pass_high,
            "P(A grade)": a_grades / n, "A_Low": a_low, "A_High": a_high,
        }
    return pd.DataFrame(rows).T


def band_comparison(df):
    """Do high-attendance students really score more than low-attendance ones?"""
    high_label, low_label = ATTENDANCE_BANDS[0][1], ATTENDANCE_BANDS[-1][1]
    high = df.loc[df["Attendance_Band"] == high_label, OVERALL_COL].to_numpy(float)
    low = df.loc[df["Attendance_Band"] == low_label, OVERALL_COL].to_numpy(float)
    if len(high) < 2 or len(low) < 2:
        return None
    # Bootstrap each band separately to get an interval for the difference
    rng = np.random.default_rng(SEED)
    diffs = (high[_resample_index(rng, len(high), N_RESAMPLES)].mean(axis=1)
             - low[_resample_index(rng, len(low), N_RESAMPLES)].mean(axis=1))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return {
        "high_band": high_label, "low_band": low_label,
        "n_high": len(high), "n_low": len(low),
        "mean_high": float(high.mean()), "mean_low": float(low.mean()),
        "difference": float(high.mean() - low.mean()),
        "ci_low": float(ci_low), "ci_high": float(ci_high),
        "p_value": permutation_test_means(high, low),
        "cohens_d": cohens_d(high, low),
    }


def inference_summary(df, pass_mark, a_mark):
    x, y = df[ATTENDANCE_COL], df[OVERALL_COL]
    boot = bootstrap_correlation(x, y)
    p_value, null = permutation_test_correlation(x, y)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_ci": mean_intervals(df),
        "probability_ci": probability_intervals(df, pass_mark, a_mark),
        "correlation": {"r": pearson_r(x, y), "ci_low": float(low), "ci_high": float(high),
                        "p_value": p_value, "p_text": format_p(p_value),
                        "bootstrap": boot, "null": null},
        "band_test": band_comparison(df),
        "n_resamples": N_RESAMPLES,
    }
