"""Project-wide settings: column names, grading rules and chart colours."""

# ---- Dataset columns -------------------------------------------------------
NAME_COL = "Student_Name"
ATTENDANCE_COL = "Attendance_Pct"
SUBJECTS = ["Math_Score", "Science_Score", "English_Score"]
SUBJECT_LABELS = {"Math_Score": "Math", "Science_Score": "Science", "English_Score": "English"}
OVERALL_COL = "Overall_Avg"

# ---- Grading rules (override from the command line in main.py) -----------
PASS_MARK = 40        # minimum score to pass a subject
A_GRADE_MARK = 80     # "A" grade = scoring 80% or above

# Letter grade bands, checked top-down: first band whose cut-off is met wins.
GRADE_BANDS = [(80, "A"), (70, "B"), (60, "C"), (50, "D"), (40, "E"), (0, "F")]

# Attendance bands used for conditional probabilities.
ATTENDANCE_BANDS = [(90, "High (90%+)"), (75, "Medium (75-89%)"), (0, "Low (<75%)")]

# ---- Early warning (Phase 5) ----------------------------------------------
RISK_MARK = 60        # overall average below this (grade D or lower) = needs support
# Risk levels from the model's P(needs support), checked top-down.
RISK_LEVELS = [(0.5, "High"), (0.2, "Watch"), (0.0, "Low")]

# ---- Chart palette ---------------------------------------------------------
# Categorical slots 1-3 of a colour-blind-validated palette; each subject keeps
# its colour in every chart.
SUBJECT_COLORS = {"Math_Score": "#2a78d6", "Science_Score": "#eb6834", "English_Score": "#1baf7a"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
# Sequential blue ramp (light -> dark) for ordered data such as letter grades.
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#1c5cab", "#104281"]
# Diverging blue <-> grey <-> red for values in [-1, 1] (correlation, similarity).
DIVERGING = ["#d03b3b", "#ec9a90", "#f0efec", "#86b6ef", "#1c5cab"]
