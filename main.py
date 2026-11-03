"""
GradeVector - Student Performance & Study Group Analyzer
=========================================================

Runs the whole project end to end:

    Phase 1  Data setup        load CSV, fix missing values, class-average bar chart
    Phase 2  Statistics        mean / median / std, attendance correlation
    Phase 3  Probability       P(pass), P(A grade), conditional odds
    Phase 4  Linear algebra    cosine similarity -> study buddy pairs and groups
    Phase 5  Early warning     logistic regression on attendance -> risk register
    + Inference                bootstrap / permutation tests -> 95% confidence intervals

For the interactive version of the same analysis:  streamlit run app.py

Usage:
    python main.py                                  # default settings
    python main.py --pass-mark 50 --group-size 4    # custom rules
    python main.py --data my_class.csv --impute median
"""

import argparse
import os
import sys

from analyzer import run_analysis
from analyzer.config import A_GRADE_MARK, PASS_MARK, RISK_MARK
from analyzer.report import build_report
from analyzer.summary import build_summary
from analyzer.visualize import make_all_charts

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Student Performance & Study Group Analyzer")
    parser.add_argument("--data", default=os.path.join(ROOT, "data", "student_data.csv"),
                        help="path to the student CSV (default: data/student_data.csv)")
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs"),
                        help="folder for charts, tables and the report (default: outputs/)")
    parser.add_argument("--pass-mark", type=int, default=PASS_MARK, help=f"pass mark (default {PASS_MARK})")
    parser.add_argument("--a-grade", type=int, default=A_GRADE_MARK,
                        help=f"minimum score for an A (default {A_GRADE_MARK})")
    parser.add_argument("--group-size", type=int, default=3, help="students per study group (default 3)")
    parser.add_argument("--impute", choices=["regression", "median"], default="regression",
                        help="how to fill missing scores (default regression)")
    parser.add_argument("--risk-mark", type=int, default=RISK_MARK,
                        help=f"overall average below this counts as needing support (default {RISK_MARK})")
    return parser.parse_args()


def save_tables(results, out_dir):
    lin = results["linear_algebra"]
    inf = results["inference"]
    ew = results["early_warning"]
    tables = {
        "cleaned_student_data.csv": (results["data"], False),
        "statistical_summary.csv": (results["summary_stats"], True),
        "probabilities.csv": (results["probabilities"], True),
        "study_buddy_pairs.csv": (lin["pairs_table"], False),
        "study_groups.csv": (lin["groups_table"], False),
        "top_matches.csv": (lin["top_matches"], False),
        "peer_mentor_pairs.csv": (lin["mentor_pairs"], False),
        "mean_confidence_intervals.csv": (inf["mean_ci"], True),
        "probability_confidence_intervals.csv": (inf["probability_ci"], True),
        "early_warning_models.csv": (ew["model_card"], True),
        "attendance_what_if.csv": (ew["what_if"], True),
    }
    if ew["register"] is not None:
        tables["risk_register.csv"] = (ew["register"], False)
    paths = []
    for name, (table, keep_index) in tables.items():
        path = os.path.join(out_dir, name)
        table.to_csv(path, index=keep_index)
        paths.append(path)
    return paths


def main():
    args = parse_args()
    if not os.path.exists(args.data):
        sys.exit(f"Data file not found: {args.data}\nRun `python generate_data.py` first.")

    results = run_analysis(args.data, pass_mark=args.pass_mark, a_mark=args.a_grade,
                           group_size=args.group_size, impute=args.impute, risk_mark=args.risk_mark)
    os.makedirs(args.out, exist_ok=True)

    summary = build_summary(results)
    print(summary)
    summary_path = os.path.join(args.out, "class_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")

    charts = make_all_charts(results, os.path.join(args.out, "charts"))
    tables = save_tables(results, args.out)
    report = build_report(results, charts, os.path.join(args.out, "report.html"))

    print(f"\nSaved {len(charts)} charts, {len(tables)} CSV tables and the text summary to {args.out}")
    print(f"Open the full report: {report}")


if __name__ == "__main__":
    main()
