"""Plain-text class summary printed to the console and saved to a .txt file."""

import os
import textwrap

from .inference import format_p

WIDTH = 78


def _title(text):
    return ["", "=" * WIDTH, f" {text}", "=" * WIDTH]


def _pct(value):
    return f"{value:6.1%}"


def build_summary(results):
    cfg = results["settings"]
    df = results["data"]
    corr = results["attendance_correlation"]
    lin = results["linear_algebra"]
    lines = [
        "#" * WIDTH,
        "  GRADEVECTOR - STUDENT PERFORMANCE & STUDY GROUP ANALYZER",
        f"  Dataset: {os.path.basename(cfg['csv_path'])}   |   Students: {len(df)}   |   "
        f"Pass mark: {cfg['pass_mark']}   |   A grade: {cfg['a_mark']}+",
        "#" * WIDTH,
    ]

    # Phase 1
    lines += _title("PHASE 1 - DATA SETUP")
    missing = results["missing_report"]
    if missing.empty:
        lines.append("No missing values found.")
    else:
        lines.append(f"Missing values found: {int(missing.sum())} "
                     f"({', '.join(f'{c}: {n}' for c, n in missing.items())})")
        lines.append(f"Filled using '{cfg['impute']}' imputation:")
        for row in results["imputation_log"].itertuples():
            lines.append(f"   - {row.Student:<10} {row.Subject:<8} -> {row.Filled_Value}")
    lines.append("")
    lines.append("Class average per subject:")
    for subject, mean in results["class_averages"].items():
        bar = "#" * int(round(mean / 2))
        lines.append(f"   {subject:<8} {mean:6.2f}  {bar}")

    # Phase 2
    lines += _title("PHASE 2 - STATISTICAL ANALYSIS")
    stats = results["summary_stats"][["Mean", "Median", "Std_Dev", "Min", "Max", "CV_%"]]
    lines.append(stats.to_string(float_format=lambda v: f"{v:7.2f}"))
    lines.append("")
    lines.append(f"Attendance vs overall score: r = {corr['r']:.3f}, r^2 = {corr['r_squared']:.3f}")
    lines.append("Per subject: " + ", ".join(f"{s} r = {r:.2f}" for s, r in corr["per_subject"].items()))
    lines.append(f"Trend line: overall = {corr['slope']:.3f} x attendance + {corr['intercept']:.2f}")
    lines.append(_wrap(corr["interpretation"]))
    lines.append("")
    best = results["strongest_subjects"]
    lines.append("Students whose best subject is: " + ", ".join(f"{s} {n}" for s, n in best.items()))
    top = results["top_students"].iloc[0]
    lines.append(f"Top performer: {top['Student_Name']} (overall {top['Overall_Avg']:.2f})")

    # Phase 3
    lines += _title("PHASE 3 - PASS / FAIL ODDS (EMPIRICAL PROBABILITY)")
    probs = results["probabilities"]
    lines.append(f"{'Subject':<10}{'P(pass)':>10}{'P(fail)':>10}{'P(A)':>10}{'P(A) normal':>14}{'# A':>6}")
    for subject, row in probs.iterrows():
        lines.append(f"{subject:<10}{_pct(row['P(Pass)']):>10}{_pct(row['P(Fail)']):>10}"
                     f"{_pct(row['P(A grade)']):>10}{_pct(row['P(A) normal model']):>14}"
                     f"{int(row['Students with A']):>6}")
    lines.append("")
    for event, p in results["combined_events"].items():
        lines.append(f"   {event:<34} {_pct(p)}")
    lines.append("")
    lines.append("Conditional on attendance:")
    cond = results["conditional"]
    for band, row in cond.iterrows():
        lines.append(f"   {band:<17} n={int(row['Students']):<3} P(pass all) {_pct(row['P(pass all)'])}"
                     f"   P(>=1 A) {_pct(row['P(at least one A)'])}   avg {row['Avg overall']:.1f}")

    # Inference
    inf = results["inference"]
    lines += _title(f"HOW SURE ARE WE? 95% CONFIDENCE ({inf['n_resamples']:,} RESAMPLES)")
    lines.append("Class mean with bootstrap 95% interval:")
    for subject, row in inf["mean_ci"].iterrows():
        lines.append(f"   {subject:<8} {row['Mean']:6.2f}   [{row['CI_Low']:6.2f}, {row['CI_High']:6.2f}]"
                     f"   +/- {row['Margin']:.1f}")
    lines.append("")
    lines.append("P(A grade) with Wilson 95% interval:")
    for subject, row in inf["probability_ci"].iterrows():
        lines.append(f"   {subject:<8} {_pct(row['P(A grade)'])}   [{_pct(row['A_Low'])}, {_pct(row['A_High'])}]")
    lines.append("")
    c = inf["correlation"]
    lines.append(f"Attendance vs overall: r = {c['r']:.3f}, 95% CI [{c['ci_low']:.3f}, {c['ci_high']:.3f}], "
                 f"permutation test {c['p_text']}")
    band = inf["band_test"]
    if band:
        lines.append(_wrap(
            f"{band['high_band']} vs {band['low_band']} attendance: {band['mean_high']:.1f} vs "
            f"{band['mean_low']:.1f}, a gap of {band['difference']:.1f} marks (95% CI "
            f"{band['ci_low']:.1f} to {band['ci_high']:.1f}), {format_p(band['p_value'])}, "
            f"Cohen's d = {band['cohens_d']:.1f}.", indent=""))

    # Phase 4
    lines += _title("PHASE 4 - STUDY BUDDY MATCH (COSINE SIMILARITY)")
    raw_vals, std_vals = lin["raw_pair_values"], lin["std_pair_values"]
    lines.append(f"Raw grade vectors:          cosine range {raw_vals.min():.3f} to {raw_vals.max():.3f}"
                 "  (too similar to rank)")
    lines.append(f"Standardised (z-score):     cosine range {std_vals.min():.3f} to {std_vals.max():.3f}"
                 "  (used for matching)")
    lines.append(f"Average similarity in pairs: {lin['pair_score']:.3f}   "
                 f"(random pairing: {lin['pair_baseline']:.3f})")
    lines.append("")
    lines.append("STUDY BUDDY PAIRS")
    lines.append(f"{'#':>3}  {'Pair':<24}{'Cosine':>8}{'Avg':>7}   {'Strength':<9} Focus")
    for row in lin["pairs_table"].itertuples():
        lines.append(f"{row.Group:>3}  {row.Members:<24}{row.Cosine_Similarity:>8.3f}"
                     f"{row.Avg_Overall:>7.1f}   {row.Shared_Strength:<9} {row.Focus_Subject}")
    lines.append("")
    lines.append(f"STUDY GROUPS OF {cfg['group_size']}   (avg similarity {lin['group_score']:.3f}, "
                 f"random {lin['group_baseline']:.3f})")
    for row in lin["groups_table"].itertuples():
        lines.append(f"{row.Group:>3}  {row.Members:<32}{row.Cosine_Similarity:>7.3f}"
                     f"   focus: {row.Focus_Subject}")
    lines.append("")
    lines.append("BONUS - PEER MENTOR PAIRS (opposite profiles)")
    for row in lin["mentor_pairs"].itertuples():
        lines.append(f"   {row.Mentor:<9} mentors {row.Mentee:<9} in {row.Teach_Subject:<8}"
                     f"(+{row.Score_Gap} marks, cosine {row.Cosine_Similarity:.2f})")

    # Phase 5
    lines += _title("PHASE 5 - EARLY WARNING (LOGISTIC REGRESSION ON ATTENDANCE)")
    ew = results["early_warning"]
    for key, label in [("risk", f"Needs support (overall < {ew['risk_mark']} or a failed subject)"),
                       ("a_grade", "At least one A grade")]:
        fit = ew[key]
        if not fit:
            lines.append(f"{label}: too few students in one outcome group to fit a model.")
            continue
        m, model = fit["metrics"], fit["model"]
        lines.append(f"{label}: {fit['cases']} of {len(df)} students")
        lines.append(f"   P = 1 / (1 + e^-({model.intercept:.2f} {model.slope:+.4f} x attendance))")
        lines.append(f"   Odds x{model.odds_ratio():.2f} per +10 attendance points")
        lines.append(f"   Leave-one-out: accuracy {m['accuracy']:.0%} (majority guess {m['baseline_accuracy']:.0%}), "
                     f"sensitivity {m['sensitivity']:.0%}, AUC {m['auc']:.2f}")
    if "attendance_50" in ew:
        lines.append(_wrap(f"Below {ew['attendance_50']:.0f}% attendance a student is more likely than not to "
                           f"need support; risk falls under 20% from {ew['attendance_low_risk']:.0f}%.", indent=""))
    lines.append("")
    lines.append("What-if: expected overall score (95% prediction interval) by attendance")
    for att, row in ew["what_if"].iterrows():
        lines.append(f"   {att:>3}%   {row['Expected_Overall']:5.1f}  [{row['Overall_Low']:5.1f}, "
                     f"{row['Overall_High']:5.1f}]   P(needs support) {_pct(row['P_Needs_Support'])}"
                     f"   P(>=1 A) {_pct(row['P_At_Least_One_A'])}")
    register = ew["register"]
    if register is not None:
        flagged = register[register["Risk_Level"] != "Low"]
        lines.append("")
        lines.append(f"RISK REGISTER - {len(flagged)} students flagged")
        for row in flagged.itertuples():
            lines.append(f"   {row.Student:<9} {row.Risk_Level:<6} P = {row.P_Needs_Support:5.1%}   attendance "
                         f"{row.Attendance_Pct}% (+{row.Attendance_Gain_Needed} to low risk)   "
                         f"overall {row.Overall_Avg:.1f}")
    lines.append("")
    lines.append("#" * WIDTH)
    return "\n".join(lines)


def _wrap(text, indent="   "):
    return textwrap.fill(text, width=WIDTH, initial_indent=indent, subsequent_indent=indent)

