"""Unit tests for the maths behind each phase.  Run with:  python -m pytest"""

import math
import os

import numpy as np
import pandas as pd
import pytest

from analyzer import run_analysis
from analyzer.config import SUBJECTS
from analyzer.data_setup import impute_missing, letter_grade, load_data
from analyzer.inference import (bootstrap_mean_ci, cohens_d, permutation_test_correlation,
                                wilson_interval)
from analyzer.prediction import (fit_logistic, fit_trend, predict_score, roc_auc, sigmoid,
                                 t_critical)
from analyzer.probability import empirical_probability, normal_probability
from analyzer.statistics_analysis import pearson_r
from analyzer.study_buddy import (cosine_similarity, cosine_similarity_matrix,
                                  form_groups, mentor_pairs, random_baseline, standardize)

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "student_data.csv")


# ---- Phase 1 ---------------------------------------------------------------

@pytest.fixture
def small_class():
    return pd.DataFrame({
        "Student_Name": ["A", "B", "C", "D", "E"],
        "Math_Score": [80, np.nan, 60, 70, 90],
        "Science_Score": [78, 55, 62, 71, 88],
        "English_Score": [82, 60, 65, 72, 91],
        "Attendance_Pct": [95, 70, 75, 85, 99],
    })


@pytest.mark.parametrize("strategy", ["median", "regression"])
def test_imputation_fills_every_gap(small_class, strategy):
    clean, log = impute_missing(small_class, strategy=strategy)
    assert clean[SUBJECTS].isna().sum().sum() == 0
    assert len(log) == 1 and log.iloc[0]["Student"] == "B"
    assert clean[SUBJECTS].min().min() >= 0 and clean[SUBJECTS].max().max() <= 100


def test_median_imputation_uses_class_median(small_class):
    clean, _ = impute_missing(small_class, strategy="median")
    assert clean.loc[1, "Math_Score"] == 75  # median of 80, 60, 70, 90


def test_text_in_number_columns_becomes_a_repaired_gap(tmp_path):
    path = tmp_path / "messy.csv"
    path.write_text("Student_Name,Math_Score,Science_Score,English_Score,Attendance_Pct\n"
                    "A,80,78,82,95\nB,absent,55,60,\nC,60,62,65,75\nD,70,71,72,85\n")
    raw = load_data(path)
    assert raw.isna().sum().sum() == 2  # "absent" and the empty attendance
    clean, log = impute_missing(raw)
    assert clean.isna().sum().sum() == 0
    assert set(log["Subject"]) == {"Math", "Attendance"}


def test_regression_imputation_follows_profile(small_class):
    # Student B is weakest in every other subject, so the estimate should be low
    clean, _ = impute_missing(small_class, strategy="regression")
    assert clean.loc[1, "Math_Score"] < 65


@pytest.mark.parametrize("score, grade", [(95, "A"), (80, "A"), (79.9, "B"), (60, "C"), (40, "E"), (12, "F")])
def test_letter_grade_boundaries(score, grade):
    assert letter_grade(score) == grade


# ---- Phase 2 ---------------------------------------------------------------

def test_pearson_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=50)
    y = 2 * x + rng.normal(size=50)
    assert pearson_r(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])


def test_pearson_perfect_relationships():
    x = [1, 2, 3, 4]
    assert pearson_r(x, [2, 4, 6, 8]) == pytest.approx(1.0)
    assert pearson_r(x, [8, 6, 4, 2]) == pytest.approx(-1.0)


# ---- Phase 3 ---------------------------------------------------------------

def test_empirical_probability_counts():
    assert empirical_probability([30, 50, 80, 90], 80) == 0.5
    assert empirical_probability([30, 50, 80, 90], 101) == 0.0


def test_normal_probability_is_half_at_the_mean():
    # With the continuity correction, threshold 80.5 sits exactly on a mean of 80
    assert normal_probability(80, 10, 80.5) == pytest.approx(0.5)
    assert normal_probability(50, 10, 200) == pytest.approx(0.0, abs=1e-9)


# ---- Inference -------------------------------------------------------------

@pytest.mark.parametrize("k, n, low, high", [(0, 10, 0.0, 0.2775), (10, 10, 0.7225, 1.0),
                                             (5, 10, 0.2366, 0.7634), (15, 30, 0.3315, 0.6685)])
def test_wilson_interval_textbook_values(k, n, low, high):
    assert wilson_interval(k, n) == pytest.approx((low, high), abs=1e-4)


def test_bootstrap_interval_contains_mean_and_is_reproducible():
    values = np.random.default_rng(0).normal(70, 10, size=40)
    mean, low, high = bootstrap_mean_ci(values)
    assert low < mean < high
    assert bootstrap_mean_ci(values) == (mean, low, high)
    # roughly mean +/- 1.96 * standard error
    assert (high - low) / 2 == pytest.approx(1.96 * values.std() / np.sqrt(40), rel=0.15)


def test_permutation_test_separates_signal_from_noise():
    rng = np.random.default_rng(1)
    x = rng.normal(size=30)
    p_signal, _ = permutation_test_correlation(x, 2 * x + rng.normal(scale=0.3, size=30), trials=2000)
    p_noise, null = permutation_test_correlation(x, rng.normal(size=30), trials=2000)
    assert p_signal == pytest.approx(1 / 2001)  # no shuffle comes close
    assert p_noise > 0.05
    assert abs(null.mean()) < 0.02  # shuffled correlations centre on zero


def test_cohens_d_known_value():
    assert cohens_d([4, 5, 6], [1, 2, 3]) == pytest.approx(3.0)


# ---- Phase 4 ---------------------------------------------------------------

def test_cosine_similarity_known_angles():
    assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
    assert cosine_similarity([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert cosine_similarity([1, 2, 3], [-1, -2, -3]) == pytest.approx(-1.0)
    assert cosine_similarity([1, 1, 0], [1, 0, 0]) == pytest.approx(math.cos(math.pi / 4))


def test_similarity_matrix_matches_pairwise_definition():
    X = np.array([[70, 80, 60], [50, 55, 90], [88, 72, 64], [61, 61, 61]], dtype=float)
    S = cosine_similarity_matrix(X)
    assert np.allclose(np.diag(S), 1)
    assert np.allclose(S, S.T)
    for i in range(len(X)):
        for j in range(len(X)):
            assert S[i, j] == pytest.approx(cosine_similarity(X[i], X[j]))


def test_standardize_gives_zero_mean_unit_std():
    X = np.array([[70, 80, 60], [50, 55, 90], [88, 72, 64]], dtype=float)
    Z = standardize(X)
    assert np.allclose(Z.mean(axis=0), 0)
    assert np.allclose(Z.std(axis=0), 1)


@pytest.mark.parametrize("size", [2, 3, 4, 7])
def test_groups_cover_every_student_once(size):
    rng = np.random.default_rng(size)
    S = cosine_similarity_matrix(rng.normal(size=(30, 3)))
    groups = form_groups(S, size=size)
    members = sorted(i for g in groups for i in g)
    assert members == list(range(30))
    assert all(size <= len(g) < 2 * size for g in groups)


def test_mentor_pairs_cover_an_odd_sized_class():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"Student_Name": list("ABCDEFG"),
                       **{s: rng.integers(40, 95, size=7) for s in SUBJECTS}})
    df["Overall_Avg"] = df[SUBJECTS].mean(axis=1)
    Z = standardize(df[SUBJECTS].to_numpy(float))
    table = mentor_pairs(cosine_similarity_matrix(Z), df, Z)
    assert len(table) == 4  # 7 students -> two pairs and a trio: 3 mentors, 4 mentees
    assert sorted(set(table["Mentor"]) | set(table["Mentee"])) == list("ABCDEFG")


# ---- Phase 5 ---------------------------------------------------------------

@pytest.mark.parametrize("dof, expected", [(5, 2.5706), (10, 2.2281), (28, 2.0484), (120, 1.9799)])
def test_t_critical_matches_tables(dof, expected):
    assert t_critical(dof) == pytest.approx(expected, abs=1e-3)


def test_logistic_regression_recovers_known_parameters():
    rng = np.random.default_rng(2)
    x = rng.uniform(50, 100, size=5000)
    y = (rng.random(5000) < sigmoid(-20 + 0.25 * x)).astype(int)
    model = fit_logistic(x, y, l2=0)
    assert model.intercept == pytest.approx(-20, rel=0.1)
    assert model.slope == pytest.approx(0.25, rel=0.1)


def test_logistic_fit_satisfies_score_equations():
    # At the maximum-likelihood solution the gradient X^T (p - y) is zero
    x = np.array([60, 62, 65, 70, 72, 75, 80, 85, 90, 95], dtype=float)
    y = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 0])
    model = fit_logistic(x, y, l2=0)
    residual = model.predict_proba(x) - y
    assert residual.sum() == pytest.approx(0, abs=1e-8)
    assert (residual * x).sum() == pytest.approx(0, abs=1e-6)


def test_logistic_penalty_keeps_separable_data_finite():
    x = np.array([60, 62, 64, 66, 80, 85, 90, 95], dtype=float)
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    model = fit_logistic(x, y)
    assert np.isfinite(model.slope) and model.slope < 0
    assert model.attendance_for(0.5) == pytest.approx(73, abs=4)  # between the two groups
    assert float(model.predict_proba(model.attendance_for(0.2))) == pytest.approx(0.2)


def test_roc_auc_extremes():
    y = np.array([0, 0, 1, 1])
    assert roc_auc([0.1, 0.2, 0.8, 0.9], y) == 1.0
    assert roc_auc([0.9, 0.8, 0.2, 0.1], y) == 0.0
    assert roc_auc([0.5, 0.5, 0.5, 0.5], y) == 0.5


def test_prediction_interval_widens_away_from_the_data():
    x = np.arange(60, 101, 2, dtype=float)
    trend = fit_trend(x, 0.8 * x + np.random.default_rng(3).normal(scale=4, size=len(x)))
    def width(a):
        _, low, high = predict_score(trend, a)
        return high - low
    assert width(trend["x_mean"]) < width(100) < width(130)


def test_full_pipeline_on_project_data():
    results = run_analysis(DATA)
    df = results["data"]
    lin = results["linear_algebra"]
    assert len(df) == 30 and df[SUBJECTS].isna().sum().sum() == 0
    assert 0 <= results["probabilities"]["P(Pass)"].min() <= 1
    assert len(lin["pairs"]) == 15
    # Matching should clearly beat random pairing
    assert lin["pair_score"] > random_baseline(lin["similarity"], 2) + 0.5
    # The attendance link is real, and the early-warning model beats guessing on unseen students
    assert results["inference"]["correlation"]["p_value"] < 0.001
    risk = results["early_warning"]["risk"]
    assert risk["metrics"]["accuracy"] > risk["metrics"]["baseline_accuracy"]
    assert risk["metrics"]["auc"] > 0.9
    register = results["early_warning"]["register"]
    assert register["P_Needs_Support"].is_monotonic_decreasing
