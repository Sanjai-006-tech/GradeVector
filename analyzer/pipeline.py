"""Runs all five phases in order and collects every result in one dictionary."""

from . import data_setup, inference, prediction, probability, statistics_analysis, study_buddy
from .config import A_GRADE_MARK, NAME_COL, PASS_MARK, RISK_MARK


def run_analysis(csv_path, pass_mark=PASS_MARK, a_mark=A_GRADE_MARK,
                 group_size=3, impute="regression", risk_mark=RISK_MARK):
    """csv_path may be a file path or an open file object with a `name`."""
    # Phase 1 - Data setup
    raw = data_setup.load_data(csv_path)
    missing = data_setup.missing_value_report(raw)
    clean, imputation_log = data_setup.impute_missing(raw, strategy=impute)
    df = data_setup.add_derived_columns(clean)

    # Phase 2 - Statistics
    corr = statistics_analysis.attendance_correlation(df)

    # Phase 3 - Probability
    probs = probability.probability_table(df, pass_mark, a_mark)

    # Phase 4 - Linear algebra
    X = study_buddy.grade_vectors(df)
    Z = study_buddy.standardize(X)
    raw_sim = study_buddy.cosine_similarity_matrix(X)
    sim = study_buddy.cosine_similarity_matrix(Z)
    pairs = study_buddy.form_groups(sim, size=2)
    groups = study_buddy.form_groups(sim, size=group_size)
    coords, explained, loadings = study_buddy.pca_2d(Z)

    return {
        "settings": {"csv_path": str(getattr(csv_path, "name", csv_path)), "pass_mark": pass_mark,
                     "a_mark": a_mark, "group_size": group_size, "impute": impute,
                     "risk_mark": risk_mark},
        "raw_data": raw,
        "missing_report": missing,
        "imputation_log": imputation_log,
        "data": df,
        "class_averages": data_setup.class_averages(df),
        "summary_stats": statistics_analysis.describe_subjects(df),
        "attendance_correlation": corr,
        "correlation_matrix": statistics_analysis.correlation_matrix(df),
        "top_students": statistics_analysis.top_performers(df, 5),
        "bottom_students": statistics_analysis.top_performers(df, 5, ascending=True),
        "strongest_subjects": statistics_analysis.strongest_subject_counts(df),
        "probabilities": probs,
        "combined_events": probability.combined_events(df, pass_mark, a_mark),
        "conditional": probability.conditional_by_attendance(df, pass_mark, a_mark),
        "grade_distribution": probability.grade_distribution(df),
        # How much each Phase 2/3 figure could move in another class of the same size
        "inference": inference.inference_summary(df, pass_mark, a_mark),
        "linear_algebra": {
            "vectors": X,
            "z_vectors": Z,
            "raw_similarity": raw_sim,
            "similarity": sim,
            "raw_pair_values": study_buddy.off_diagonal(raw_sim),
            "std_pair_values": study_buddy.off_diagonal(sim),
            "pairs": pairs,
            "groups": groups,
            "pairs_table": study_buddy.group_table(pairs, sim, df, Z),
            "groups_table": study_buddy.group_table(groups, sim, df, Z),
            "top_matches": study_buddy.top_matches(sim, df, k=3),
            "mentor_pairs": study_buddy.mentor_pairs(sim, df, Z),
            "pair_score": study_buddy.average_group_similarity(pairs, sim),
            "pair_baseline": study_buddy.random_baseline(sim, 2),
            "group_score": study_buddy.average_group_similarity(groups, sim),
            "group_baseline": study_buddy.random_baseline(sim, group_size),
            "pca_coords": coords,
            "pca_explained": explained,
            "pca_loadings": loadings,
            "names": df[NAME_COL].tolist(),
        },
        # Phase 5 - Early warning
        "early_warning": prediction.build_early_warning(df, pass_mark, a_mark, risk_mark),
    }
