"""Phase 5 - Early Warning: logistic regression written from scratch.

Attendance is known weeks before exam results, so it can act as an early-warning
signal. This module models the chance of an outcome given attendance x:

    P(outcome | x) = 1 / (1 + e^-(b0 + b1 x))

and fits b0 and b1 by maximum likelihood with Newton's method. Every Newton
step is a linear-algebra problem: solve H d = g for the step d, where

    g = X^T (p - y) + L w        gradient of the penalised log-loss
    H = X^T W X + L              Hessian, with W = diag(p (1 - p))

L is a small L2 penalty on the slope only. It keeps the slope finite when
attendance separates the two outcomes perfectly, which is easy in a class of 30.

Two outcomes are modelled:
    needs support     overall average below the risk mark, or any failed subject
    at least one A    a score at or above the A-grade mark in any subject

Each model is checked by leave-one-out cross-validation: refit without one
student, predict that student, and repeat for everybody. That measures how well
the model flags a student it has never seen.
"""

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import ATTENDANCE_COL, NAME_COL, OVERALL_COL, RISK_LEVELS, RISK_MARK, SUBJECTS

L2_PENALTY = 0.1
MIN_CASES = 2                  # students needed in each outcome group to fit a model
LOW_RISK = RISK_LEVELS[-2][0]  # below this P(needs support) a student is low risk
WHAT_IF_ATTENDANCE = [60, 70, 80, 90, 100]


def sigmoid(z):
    """1 / (1 + e^-z), written with tanh so it never overflows."""
    return 0.5 * (1 + np.tanh(np.asarray(z, dtype=float) / 2))


@dataclass
class LogisticModel:
    intercept: float  # b0, on the raw attendance scale
    slope: float      # b1, change in log-odds per attendance point
    iterations: int

    def predict_proba(self, attendance):
        return sigmoid(self.intercept + self.slope * np.asarray(attendance, dtype=float))

    def attendance_for(self, p):
        """Attendance at which the predicted probability equals p (the curve inverted)."""
        if self.slope == 0:
            return None
        return (math.log(p / (1 - p)) - self.intercept) / self.slope

    def odds_ratio(self, step=10):
        """Factor the odds are multiplied by for every `step` extra attendance points."""
        return math.exp(self.slope * step)


def fit_logistic(x, y, l2=L2_PENALTY, max_iter=100, tol=1e-10):
    """Maximum-likelihood logistic regression of y (0/1) on one feature x."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if y.min() == y.max():
        raise ValueError("both outcomes must appear in the data")
    mean, std = x.mean(), x.std()
    if std == 0:
        raise ValueError("attendance has no spread")
    # Work on standardised attendance so the Hessian is well conditioned
    X = np.column_stack([np.ones(len(x)), (x - mean) / std])
    L = np.diag([0.0, l2])  # the intercept is never penalised

    def loss(w):
        z = X @ w
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * w @ L @ w

    w = np.zeros(2)
    for iteration in range(1, max_iter + 1):
        p = sigmoid(X @ w)
        gradient = X.T @ (p - y) + L @ w
        hessian = X.T @ (X * (p * (1 - p))[:, None]) + L
        step = np.linalg.solve(hessian, gradient)
        # Halve the step if a full Newton step would make the fit worse
        t, current = 1.0, loss(w)
        while loss(w - t * step) > current and t > 1e-8:
            t /= 2
        w = w - t * step
        if np.abs(t * step).max() < tol:
            break
    b0, b1 = w
    # Back to raw attendance: b0 + b1 (x - mean) / std
    return LogisticModel(intercept=float(b0 - b1 * mean / std), slope=float(b1 / std),
                         iterations=iteration)


def leave_one_out(x, y, l2=L2_PENALTY):
    """Out-of-sample probability for each student from a model that never saw them."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    probs = np.empty(len(x))
    keep = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        keep[i] = False
        probs[i] = fit_logistic(x[keep], y[keep], l2).predict_proba(x[i])
        keep[i] = True
    return probs


def roc_auc(scores, y):
    """Chance that a random positive case scores above a random negative one (ties count half)."""
    scores, y = np.asarray(scores, dtype=float), np.asarray(y, dtype=bool)
    diff = scores[y][:, None] - scores[~y][None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def evaluate(probs, y, threshold=0.5):
    """Accuracy, sensitivity, specificity and AUC of probabilities against outcomes."""
    predicted, actual = np.asarray(probs) >= threshold, np.asarray(y).astype(bool)
    tp, tn = int((predicted & actual).sum()), int((~predicted & ~actual).sum())
    fp, fn = int((predicted & ~actual).sum()), int((~predicted & actual).sum())
    return {
        "accuracy": (tp + tn) / len(actual),
        "baseline_accuracy": float(max(actual.mean(), 1 - actual.mean())),  # always guess the majority
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "auc": roc_auc(probs, actual),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


# ---- Expected score with a prediction interval ----------------------------

def t_critical(dof, level=0.95):
    """Two-sided critical value of Student's t distribution.

    Uses the Cornish-Fisher expansion around the normal quantile, accurate to
    about 0.001 for dof >= 5, so no statistics library is needed.
    """
    z = NormalDist().inv_cdf(0.5 + level / 2)
    terms = [
        (z ** 3 + z) / 4,
        (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96,
        (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384,
        (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160,
    ]
    return z + sum(term / dof ** (k + 1) for k, term in enumerate(terms))


def fit_trend(x, y):
    """Least-squares line y = slope * x + intercept, plus what a prediction interval needs."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    return {"slope": float(slope), "intercept": float(intercept), "n": len(x),
            "residual_sd": float(np.sqrt(residuals @ residuals / (len(x) - 2))),
            "x_mean": float(x.mean()), "sxx": float(((x - x.mean()) ** 2).sum())}


def predict_score(trend, attendance, level=0.95):
    """(expected, low, high): overall score for one new student at this attendance.

    The interval covers a single student, not just the average, so it adds the
    scatter around the line to the uncertainty in the line itself:
        se = s * sqrt(1 + 1/n + (x0 - x_mean)^2 / Sxx)
    """
    expected = trend["intercept"] + trend["slope"] * attendance
    se = trend["residual_sd"] * math.sqrt(1 + 1 / trend["n"]
                                          + (attendance - trend["x_mean"]) ** 2 / trend["sxx"])
    margin = t_critical(trend["n"] - 2, level) * se
    return expected, expected - margin, expected + margin


# ---- Putting it together ---------------------------------------------------

def risk_level(p):
    for cutoff, label in RISK_LEVELS:
        if p >= cutoff:
            return label
    return RISK_LEVELS[-1][1]


def needs_support(df, pass_mark, risk_mark=RISK_MARK):
    return ((df[OVERALL_COL] < risk_mark) | (df[SUBJECTS] < pass_mark).any(axis=1)).to_numpy(int)


def at_least_one_a(df, a_mark):
    return (df[SUBJECTS] >= a_mark).any(axis=1).to_numpy(int)


def _fit_and_validate(x, y):
    """Fit on everyone and validate by leave-one-out; None if a group is too small."""
    if min(y.sum(), len(y) - y.sum()) < MIN_CASES:
        return None
    loo = leave_one_out(x, y)
    return {"model": fit_logistic(x, y), "loo_probs": loo, "metrics": evaluate(loo, y),
            "cases": int(y.sum())}


def what_if(early_warning, attendance):
    """Everything the models say about a student with this attendance."""
    expected, low, high = predict_score(early_warning["trend"], attendance)
    risk, a_grade = early_warning["risk"], early_warning["a_grade"]
    p_risk = float(risk["model"].predict_proba(attendance)) if risk else math.nan
    return {
        "Attendance": attendance,
        "Expected_Overall": expected, "Overall_Low": low, "Overall_High": high,
        "P_Needs_Support": p_risk,
        "Risk_Level": risk_level(p_risk) if risk else "n/a",
        "P_At_Least_One_A": float(a_grade["model"].predict_proba(attendance)) if a_grade else math.nan,
    }


def risk_register(df, risk, a_grade, outcome):
    """One row per student, most at risk first."""
    attendance = df[ATTENDANCE_COL].to_numpy(float)
    p_risk = risk["model"].predict_proba(attendance)
    target = risk["model"].attendance_for(LOW_RISK)
    reachable = target is not None and risk["model"].slope < 0 and target <= 100
    levels = [risk_level(p) for p in p_risk]
    gap = [max(0, math.ceil(target - a)) if reachable and level != "Low" else 0
           for a, level in zip(attendance, levels)]
    register = pd.DataFrame({
        "Student": df[NAME_COL].to_numpy(),
        "Attendance_Pct": attendance.astype(int),
        "Overall_Avg": df[OVERALL_COL].to_numpy(),
        "Needs_Support": np.where(outcome == 1, "Yes", "No"),
        "P_Needs_Support": p_risk.round(3),
        "Risk_Level": levels,
        "Attendance_Gain_Needed": gap,
        "P_At_Least_One_A": (a_grade["model"].predict_proba(attendance).round(3)
                             if a_grade else np.nan),
    })
    return register.sort_values(["P_Needs_Support", "Overall_Avg"],
                                ascending=[False, True]).reset_index(drop=True)


def model_card(early_warning):
    """One row per fitted model: coefficients and leave-one-out performance."""
    rows = {}
    for key, label in [("risk", "Needs support"), ("a_grade", "At least one A")]:
        fit = early_warning[key]
        if not fit:
            continue
        model, m = fit["model"], fit["metrics"]
        rows[label] = {"Students": fit["cases"], "Intercept": model.intercept, "Slope": model.slope,
                       "Odds_x_per_10pts": model.odds_ratio(), "LOO_Accuracy": m["accuracy"],
                       "Majority_Baseline": m["baseline_accuracy"], "Sensitivity": m["sensitivity"],
                       "Specificity": m["specificity"], "AUC": m["auc"]}
    card = pd.DataFrame(rows).T
    if not card.empty:
        card["Students"] = card["Students"].astype(int)
    return card


def build_early_warning(df, pass_mark, a_mark, risk_mark=RISK_MARK):
    x = df[ATTENDANCE_COL].to_numpy(float)
    support = needs_support(df, pass_mark, risk_mark)
    any_a = at_least_one_a(df, a_mark)
    risk = _fit_and_validate(x, support)
    a_grade = _fit_and_validate(x, any_a)
    result = {
        "risk_mark": risk_mark,
        "needs_support": support,
        "any_a": any_a,
        "risk": risk,
        "a_grade": a_grade,
        "trend": fit_trend(x, df[OVERALL_COL]),
        "observed_range": (float(x.min()), float(x.max())),
        "register": risk_register(df, risk, a_grade, support) if risk else None,
    }
    if risk and risk["model"].slope < 0:
        result["attendance_50"] = risk["model"].attendance_for(0.5)
        result["attendance_low_risk"] = risk["model"].attendance_for(LOW_RISK)
    result["what_if"] = pd.DataFrame([what_if(result, a) for a in WHAT_IF_ATTENDANCE]).set_index("Attendance")
    result["model_card"] = model_card(result)
    return result
