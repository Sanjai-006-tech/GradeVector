"""GradeVector - Student Performance & Study Group Analyzer.

Phase 1  data_setup            load CSV, handle missing values, class averages
Phase 2  statistics_analysis   mean / median / std, attendance correlation
Phase 3  probability           empirical P(pass), P(A grade), conditional odds
         inference             bootstrap & Wilson confidence intervals, permutation tests
Phase 4  study_buddy           cosine similarity of grade vectors, group matching
Phase 5  prediction            logistic-regression early warning, attendance what-if
"""

from .pipeline import run_analysis

__version__ = "2.0.0"
__all__ = ["run_analysis"]
