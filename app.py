"""GradeVector interactive dashboard.

    streamlit run app.py

Runs the same analysis as main.py, but every setting is a live control: change
the pass mark, upload another class or pick a student, and the page updates.
All of the maths lives in the analyzer/ package; this file lays out the page
and dashboard/figures.py draws the interactive charts.
"""

import html
import io
import os
import tempfile

import pandas as pd
import streamlit as st

from analyzer import run_analysis
from analyzer.config import (A_GRADE_MARK, ATTENDANCE_COL, NAME_COL, OVERALL_COL, PASS_MARK, RISK_MARK,
                             SUBJECT_LABELS, SUBJECTS)
from analyzer.prediction import LOW_RISK, risk_level, what_if
from analyzer.report import build_report, key_findings
from analyzer.study_buddy import form_groups
from analyzer.visualize import make_all_charts
from dashboard import figures

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(ROOT, "data", "student_data.csv")
RISK_BADGE = {"High": "red", "Watch": "orange", "Low": "green"}

st.set_page_config(page_title="GradeVector", page_icon="🎓", layout="wide")
st.markdown("""<style>
.gv-hero { background: linear-gradient(135deg, #0d366b 0%, #1c5cab 100%); color: #fff;
  padding: 26px 30px 22px; border-radius: 14px; margin-bottom: 0.6rem; }
.gv-hero .eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: 12px; opacity: .75; }
.gv-hero .title { font-size: 34px; font-weight: 700; letter-spacing: -.01em; margin: 2px 0; }
.gv-hero .sub { opacity: .9; }
.gv-hero .meta { font-size: 13px; opacity: .7; margin-top: 10px; }
</style>""", unsafe_allow_html=True)


# ---- Analysis (cached: reruns only when the data or a setting changes) -----

@st.cache_data(show_spinner="Running the analysis...")
def analyse(data, name, pass_mark, a_mark, group_size, impute, risk_mark):
    buffer = io.BytesIO(data)
    buffer.name = name
    return run_analysis(buffer, pass_mark=pass_mark, a_mark=a_mark, group_size=group_size,
                        impute=impute, risk_mark=risk_mark)


@st.cache_data(show_spinner=False)
def html_report(data, name, *settings):
    """The same self-contained report main.py writes, built in a temporary folder."""
    results = analyse(data, name, *settings)
    with tempfile.TemporaryDirectory() as tmp:
        charts = make_all_charts(results, os.path.join(tmp, "charts"))
        path = build_report(results, charts, os.path.join(tmp, "report.html"))
        with open(path, "rb") as fh:
            return fh.read()


def plot(fig, key):
    st.plotly_chart(fig, theme=None, key=key, config={"displaylogo": False})


def csv_bytes(table, index=False):
    return table.to_csv(index=index).encode("utf-8")


def as_percent(table, columns, decimals=0):
    """Scale probability columns to 0-100 and give each a % format (keeps tables to whole percents)."""
    config = {c: st.column_config.NumberColumn(c.replace("_", " "), format=f"%.{decimals}f%%") for c in columns}
    return table.assign(**{c: table[c] * 100 for c in columns}), config


# ---- Sidebar ---------------------------------------------------------------

def sidebar():
    with open(SAMPLE, "rb") as fh:
        data, name = fh.read(), os.path.basename(SAMPLE)
    with st.sidebar:
        st.markdown("## 🎓 GradeVector")
        st.caption("Student performance & study group analyzer")
        source = st.segmented_control("Dataset", ["Sample class", "Upload CSV"], default="Sample class",
                                      required=True)
        if source == "Upload CSV":
            upload = st.file_uploader(
                "Class records", type="csv",
                help="Columns: Student_Name, Math_Score, Science_Score, English_Score, Attendance_Pct")
            st.download_button("Template (sample CSV)", data, file_name=name, mime="text/csv",
                               icon=":material/download:", type="tertiary")
            if upload is None:
                st.caption("Showing the sample class until a file is uploaded.")
            else:
                data, name = upload.getvalue(), upload.name
        try:
            students = len(pd.read_csv(io.BytesIO(data)))
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            st.error(f"Could not read {name}: {exc}")
            st.stop()

        st.markdown("**Grading rules**")
        pass_mark = st.slider("Pass mark", 20, 60, PASS_MARK)
        a_mark = st.slider("A grade from", 65, 95, A_GRADE_MARK)
        risk_mark = st.slider("Needs support if overall is below", 40, 75, RISK_MARK,
                              help="Also flagged: any subject below the pass mark. This is the outcome the "
                                   "early-warning model learns to predict.")
        st.markdown("**Study groups & cleaning**")
        if students >= 4:
            group_size = st.slider("Study group size", 2, min(6, students // 2), min(3, students // 2))
        else:
            group_size = 2
        impute = st.segmented_control("Fill missing scores by", ["regression", "median"], default="regression",
                                      required=True, format_func=str.capitalize)
    return data, name, (pass_mark, a_mark, group_size, impute, risk_mark)


def downloads(results, data, name, settings):
    lin, ew = results["linear_algebra"], results["early_warning"]
    with st.sidebar:
        st.divider()
        st.markdown("**Downloads**")
        st.download_button("Full HTML report", lambda: html_report(data, name, *settings),
                           file_name="gradevector_report.html", mime="text/html", type="primary",
                           icon=":material/description:", width="stretch",
                           help="All charts and tables in one file (takes a few seconds to build)")
        st.download_button("Study groups", csv_bytes(lin["groups_table"]), file_name="study_groups.csv",
                           icon=":material/groups:", width="stretch")
        if ew["register"] is not None:
            st.download_button("Risk register", csv_bytes(ew["register"]), file_name="risk_register.csv",
                               icon=":material/warning:", width="stretch")
        st.download_button("Cleaned data", csv_bytes(results["data"]), file_name="cleaned_student_data.csv",
                           icon=":material/table:", width="stretch")


# ---- Header ----------------------------------------------------------------

def header(results):
    cfg, df = results["settings"], results["data"]
    st.markdown(f"""<div class="gv-hero">
  <div class="eyebrow">Minor Project &middot; Data Analytics with Python</div>
  <div class="title">GradeVector</div>
  <div class="sub">Statistics, probability, linear algebra and an early-warning model for {len(df)} students</div>
  <div class="meta">{html.escape(os.path.basename(cfg['csv_path']))} &middot; pass mark {cfg['pass_mark']} &middot;
  A grade {cfg['a_mark']}+ &middot; needs support below {cfg['risk_mark']} &middot; {cfg['impute']} imputation</div>
</div>""", unsafe_allow_html=True)

    stats, events, corr = results["summary_stats"], results["combined_events"], results["attendance_correlation"]
    inf, lin, ew = results["inference"], results["linear_algebra"], results["early_warning"]
    ci = inf["mean_ci"].loc["Overall"]
    kpis = [
        ("Students", len(df), f"{int(results['missing_report'].sum())} values repaired",
         "Rows in the dataset after cleaning"),
        ("Class average", f"{stats.loc['Overall', 'Mean']:.1f}",
         f"95% CI {ci['CI_Low']:.1f} to {ci['CI_High']:.1f}", "Bootstrap confidence interval for the class mean"),
        ("Pass all subjects", f"{events['P(pass all three subjects)']:.0%}",
         f"pass mark {results['settings']['pass_mark']}", "Empirical probability"),
        ("At least one A", f"{events['P(at least one A)']:.0%}", f"A grade = {results['settings']['a_mark']}+",
         "Empirical probability"),
        ("Attendance link", f"r = {corr['r']:.2f}", inf["correlation"]["p_text"],
         "Pearson correlation with overall score; p-value from a permutation test"),
        ("Buddy-pair similarity", f"{lin['pair_score']:.2f}", f"random pairing {lin['pair_baseline']:.2f}",
         "Average cosine similarity inside study pairs"),
    ]
    if ew["risk"]:
        m = ew["risk"]["metrics"]
        kpis += [
            ("Flagged for support", int((ew["register"]["Risk_Level"] != "Low").sum()), "high or watch risk",
             "Students the attendance model flags"),
            ("Early-warning accuracy", f"{m['accuracy']:.0%}", f"AUC {m['auc']:.2f} · leave-one-out",
             "Accuracy on students the model did not see while training"),
        ]
    for start in range(0, len(kpis), 4):
        for col, (label, value, note, help_text) in zip(st.columns(4), kpis[start:start + 4]):
            col.metric(label, value, note, delta_color="off", delta_arrow="off", border=True, help=help_text)


# ---- Tabs ------------------------------------------------------------------

def overview_tab(results):
    cfg, df = results["settings"], results["data"]
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Key findings")
        st.markdown("\n".join(f"- {f.replace('<b>', '**').replace('</b>', '**')}" for f in key_findings(results)))
    with right:
        plot(figures.class_averages(df, cfg["pass_mark"], cfg["a_mark"]), "ov_averages")
        plot(figures.grade_mix(results["grade_distribution"]), "ov_grades")
    plot(figures.attendance_scatter(df, results["attendance_correlation"], results["early_warning"]["trend"]),
         "ov_scatter")


def data_tab(results):
    df, log, missing = results["data"], results["imputation_log"], results["missing_report"]
    a, b, c = st.columns(3)
    a.metric("Students", len(df), border=True)
    b.metric("Missing values found", int(missing.sum()), border=True)
    c.metric("Filled by", results["settings"]["impute"].capitalize(), border=True)
    if log.empty:
        st.success("No missing values: every score was present.")
    else:
        st.markdown("**Values repaired.** Regression predicts each gap from the student's other subjects and "
                    "attendance by least squares (solving *Xw ≈ y*), so the estimate fits that student's profile.")
        st.dataframe(log.rename(columns={"Filled_Value": "Filled value"}), hide_index=True, width="content")

    notes = log.groupby("Student")["Subject"].apply(lambda s: ", ".join(s) + " filled") if not log.empty else {}
    view = df.assign(Notes=df[NAME_COL].map(notes).fillna(""))
    score_cols = {s: st.column_config.ProgressColumn(SUBJECT_LABELS[s], min_value=0, max_value=100, format="%d")
                  for s in SUBJECTS}
    st.markdown("**Cleaned dataset**")
    st.dataframe(view, hide_index=True, height=420, column_order=[NAME_COL, "Notes", *df.columns[1:]],
                 column_config={
        NAME_COL: st.column_config.TextColumn("Student", pinned=True),
        "Notes": st.column_config.TextColumn("Repaired", width="small"),
        **score_cols,
        ATTENDANCE_COL: st.column_config.ProgressColumn("Attendance", min_value=0, max_value=100, format="%d%%"),
        OVERALL_COL: st.column_config.NumberColumn("Overall", format="%.1f"),
        "Overall_Grade": "Grade", "Attendance_Band": "Attendance band",
    })


def statistics_tab(results):
    cfg, df, inf = results["settings"], results["data"], results["inference"]
    st.subheader("Descriptive statistics")
    st.dataframe(results["summary_stats"].rename(columns=lambda c: c.replace("_", " ")), width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.2f")
                                for c in ["Mean", "Median", "Std Dev", "CV %"]})
    plot(figures.score_histograms(df, cfg["pass_mark"], cfg["a_mark"]), "st_hist")
    left, right = st.columns([1.3, 1])
    with left:
        plot(figures.attendance_scatter(df, results["attendance_correlation"], results["early_warning"]["trend"]),
             "st_scatter")
    with right:
        plot(figures.correlation_heatmap(results["correlation_matrix"]), "st_corr")
    st.info(results["attendance_correlation"]["interpretation"], icon=":material/lightbulb:")

    st.subheader("How sure can we be?")
    st.markdown(f"One class is a small sample, so every number above would shift a little for another class of "
                f"the same size. The **bootstrap** redraws the class with replacement {inf['n_resamples']:,} times "
                f"to see how far a statistic moves; a **permutation test** shuffles attendance to see what "
                f"correlation pure chance produces.")
    corr, band = inf["correlation"], inf["band_test"]
    cols = st.columns(4 if band else 2)
    cols[0].metric("Correlation r", f"{corr['r']:.3f}", f"95% CI {corr['ci_low']:.2f} to {corr['ci_high']:.2f}",
                   delta_color="off", delta_arrow="off", border=True)
    cols[1].metric("Chance of a link this strong", corr["p_text"], "permutation test", delta_color="off",
                   delta_arrow="off", border=True)
    if band:
        cols[2].metric("High vs low attendance gap", f"{band['difference']:.1f} marks",
                       f"95% CI {band['ci_low']:.1f} to {band['ci_high']:.1f}", delta_color="off",
                       delta_arrow="off", border=True)
        cols[3].metric("Effect size (Cohen's d)", f"{band['cohens_d']:.1f}", "0.8 already counts as large",
                       delta_color="off", delta_arrow="off", border=True)
    plot(figures.resampling_chart(inf), "st_resample")
    left, right = st.columns([1, 1])
    with left:
        plot(figures.mean_ci_chart(inf["mean_ci"]), "st_mean_ci")
    with right:
        st.markdown("**Reading the intervals**")
        st.markdown("Each interval is the range of class means seen in the middle 95% of resampled classes. "
                    "Where two subjects' intervals overlap a lot, this one class cannot show that one subject "
                    "really has a higher average than the other.")
        st.latex(r"r=\frac{\sum (x-\bar x)(y-\bar y)}{\sqrt{\sum (x-\bar x)^2\,\sum (y-\bar y)^2}}")


def probability_tab(results):
    inf = results["inference"]
    plot(figures.probability_chart(inf["probability_ci"], results["probabilities"]), "pr_main")
    st.markdown("Empirical probability is simply *students meeting the condition ÷ all students*. The diamonds "
                "show what a fitted normal curve predicts instead; where they disagree, the scores are not "
                "bell-shaped. Intervals use the **Wilson score** method, which stays sensible at 0% and 100%:")
    st.latex(r"\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}"
             r"{1 + \frac{z^2}{n}}, \quad z = 1.96")
    left, right = st.columns([1.2, 1])
    with left:
        plot(figures.conditional_chart(results["conditional"]), "pr_cond")
    with right:
        st.markdown("**Events across all subjects**")
        events, config = as_percent(pd.DataFrame({"Probability": results["combined_events"]}), ["Probability"], 1)
        st.dataframe(events, width="stretch", column_config=config)
        st.caption("If P(pass all) is close to the independent product, failing one subject says little "
                   "about failing another.")
    with st.expander("Full probability table"):
        table, config = as_percent(results["probabilities"],
                                   ["P(Pass)", "P(Fail)", "P(A grade)", "P(A) normal model"], 1)
        st.dataframe(table, width="stretch", column_config=config)


def study_groups_tab(results):
    lin, cfg = results["linear_algebra"], results["settings"]
    modes = {"Study pairs": (lin["pairs"], lin["pairs_table"], lin["pair_score"], lin["pair_baseline"]),
             f"Groups of {cfg['group_size']}": (lin["groups"], lin["groups_table"], lin["group_score"],
                                                lin["group_baseline"]),
             "Peer mentors": (form_groups(lin["similarity"], size=2, mode="complementary"), lin["mentor_pairs"],
                              None, None)}
    mode = st.segmented_control("Show", list(modes), default="Study pairs", required=True)
    groups, table, score, baseline = modes[mode]
    if score is not None:
        st.caption(f"Average cosine similarity inside {mode.lower()}: **{score:.3f}**, against {baseline:.3f} "
                   f"for 2,000 random groupings.")
    else:
        st.caption("Mentor pairs join the *most opposite* profiles (most negative cosine). The student ahead in a "
                   "subject coaches the other in it.")
    left, right = st.columns([1.25, 1], gap="medium")
    with left:
        plot(figures.buddy_map(results, groups), f"sg_map_{mode}")
    with right:
        st.dataframe(table, hide_index=True, height=540, column_config={
            "Group": st.column_config.NumberColumn("#", width=36),
            "Members": st.column_config.TextColumn("Members", width=120),
            "Cosine_Similarity": st.column_config.ProgressColumn("Cosine", min_value=-1, max_value=1,
                                                                 format="%.2f", width=90),
            "Avg_Overall": st.column_config.NumberColumn("Avg", format="%.1f", width=50),
            "Shared_Strength": st.column_config.TextColumn("Strength", width=70),
            "Focus_Subject": st.column_config.TextColumn("Focus", width=70),
            "Teach_Subject": st.column_config.TextColumn("Teaches", width=70),
            "Score_Gap": st.column_config.NumberColumn("Gap", format="+%d", width=50)})

    st.latex(r"\cos\theta = \frac{\mathbf a \cdot \mathbf b}{\lVert \mathbf a\rVert\,\lVert \mathbf b\rVert}"
             r"\qquad S = U U^{\mathsf T}")
    with st.expander("Why standardise the grade vectors first?"):
        left, right = st.columns([1.2, 1])
        with left:
            plot(figures.cosine_comparison(lin["raw_pair_values"], lin["std_pair_values"]), "sg_cos")
        with right:
            st.markdown("Raw scores are all positive, so every student's vector points into the same corner of "
                        "space and every pair looks almost identical. Converting each subject to **z-scores** "
                        "moves the origin to the average student; the angle between two vectors then shows who "
                        "is above or below average in which subjects.")
            a, b = lin["pairs"][0][:2]
            za, zb = lin["z_vectors"][a], lin["z_vectors"][b]
            st.markdown(f"Top pair: **{lin['names'][a]}** z = [{', '.join(f'{v:.2f}' for v in za)}] and "
                        f"**{lin['names'][b]}** z = [{', '.join(f'{v:.2f}' for v in zb)}] → "
                        f"cos θ = **{lin['similarity'][a, b]:.3f}**")
    with st.expander("Similarity matrix"):
        plot(figures.similarity_heatmap(lin["similarity"], lin["names"], lin["groups"]), "sg_heat")
    with st.expander("Top 3 matches for every student"):
        st.dataframe(lin["top_matches"], hide_index=True, width="stretch")


def early_warning_tab(results):
    ew, df = results["early_warning"], results["data"]
    if not ew["risk"]:
        st.warning("Too few students fall in one of the outcome groups to fit the early-warning model. "
                   "Try a different 'needs support' mark in the sidebar.")
        return
    st.markdown("Attendance is known weeks before exam results, so it can flag students while there is still "
                "time to help. A **logistic regression written from scratch** (Newton's method, one linear solve "
                "per step) turns attendance into a probability, and **leave-one-out cross-validation** checks it "
                "on students the model never saw.")
    st.latex(r"P(\text{needs support}\mid x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}\qquad "
             r"w \leftarrow w - H^{-1}g,\; H = X^{\mathsf T} W X + \lambda I")

    low, high = ew["observed_range"]
    attendance = st.slider("What if a student's attendance were...", 40, 100, 75, format="%d%%")
    w = what_if(ew, attendance)
    a, b, c = st.columns(3)
    a.metric("Expected overall score", f"{w['Expected_Overall']:.1f}",
             f"95% range {w['Overall_Low']:.0f} to {w['Overall_High']:.0f}", delta_color="off", delta_arrow="off",
             border=True)
    b.metric("P(needs support)", f"{w['P_Needs_Support']:.0%}", f"{w['Risk_Level']} risk", delta_color="off",
             delta_arrow="off", border=True)
    if ew["a_grade"]:
        c.metric("P(at least one A)", f"{w['P_At_Least_One_A']:.0%}", "same attendance", delta_color="off",
                 delta_arrow="off", border=True)
    if not low <= attendance <= high:
        st.caption(f"This class's attendance runs from {low:.0f}% to {high:.0f}%, so {attendance}% is an "
                   f"extrapolation; treat it with care.")
    left, right = st.columns(2)
    with left:
        plot(figures.logistic_chart(df, ew, "risk", attendance), "ew_risk")
    if ew["a_grade"]:
        with right:
            plot(figures.logistic_chart(df, ew, "a_grade", attendance), "ew_a")

    st.subheader("Model card")
    card, config = as_percent(ew["model_card"], ["LOO_Accuracy", "Majority_Baseline", "Sensitivity",
                                                 "Specificity"])
    st.dataframe(card, width="stretch", column_config={
        "Intercept": st.column_config.NumberColumn(format="%.2f"),
        "Slope": st.column_config.NumberColumn(format="%.4f"),
        "Odds_x_per_10pts": st.column_config.NumberColumn("Odds × per +10 pts", format="%.2f"),
        "AUC": st.column_config.NumberColumn(format="%.2f"),
        **config})
    if "attendance_low_risk" in ew:
        st.caption(f"The risk curve crosses 50% at {ew['attendance_50']:.0f}% attendance and drops below 20% from "
                   f"{ew['attendance_low_risk']:.0f}%: a concrete target for students on the register.")

    st.subheader("Risk register")
    levels = st.pills("Risk level", ["High", "Watch", "Low"], selection_mode="multi", default=["High", "Watch"])
    register = ew["register"][ew["register"]["Risk_Level"].isin(levels or [])]
    st.dataframe(register.assign(P_Needs_Support=register["P_Needs_Support"] * 100,
                                 P_At_Least_One_A=register["P_At_Least_One_A"] * 100),
                 hide_index=True, width="stretch", column_config={
        "Attendance_Pct": st.column_config.NumberColumn("Attendance", format="%d%%"),
        "Overall_Avg": st.column_config.NumberColumn("Overall", format="%.1f"),
        "Needs_Support": "Needs support now",
        "P_Needs_Support": st.column_config.ProgressColumn("P(needs support)", min_value=0, max_value=100,
                                                           format="%.0f%%", color="red"),
        "Risk_Level": "Risk",
        "Attendance_Gain_Needed": st.column_config.NumberColumn("Attendance points to low risk", format="+%d"),
        "P_At_Least_One_A": st.column_config.ProgressColumn("P(≥1 A)", min_value=0, max_value=100,
                                                            format="%.0f%%", color="green"),
    })
    st.info("Correlation is not causation, and the model learned from one class. Treat a flag as a reason to "
            "talk to a student, not as a verdict.", icon=":material/info:")


def student_tab(results):
    df, lin, ew = results["data"], results["linear_algebra"], results["early_warning"]
    names = df[NAME_COL].tolist()
    i = st.selectbox("Student", range(len(df)), format_func=lambda k: names[k])
    row = df.iloc[i]
    pair = next(g for g in lin["pairs"] if i in g)
    pair_no = lin["pairs"].index(pair)
    buddies = [j for j in pair if j != i]
    group = next(g for g in lin["groups"] if i in g)
    group_no = lin["groups"].index(group)
    rank = int(df[OVERALL_COL].rank(ascending=False, method="min").iloc[i])

    p_risk = float(ew["risk"]["model"].predict_proba(row[ATTENDANCE_COL])) if ew["risk"] else None
    level = risk_level(p_risk) if p_risk is not None else None
    st.markdown(f"### {row[NAME_COL]}")
    badges = st.container(horizontal=True)
    with badges:
        st.badge(f"Grade {row['Overall_Grade']}", color="blue")
        st.badge(row["Attendance_Band"], color="gray")
        if level:
            st.badge(f"{level} risk", color=RISK_BADGE[level], icon=":material/monitor_heart:")

    a, b, c, d = st.columns(4)
    a.metric("Overall average", f"{row[OVERALL_COL]:.1f}",
             f"{row[OVERALL_COL] - df[OVERALL_COL].mean():+.1f} vs class", border=True)
    b.metric("Class rank", f"#{rank} of {len(df)}", f"top {rank / len(df):.0%}", delta_color="off",
             delta_arrow="off", border=True)
    c.metric("Attendance", f"{row[ATTENDANCE_COL]:.0f}%",
             f"{row[ATTENDANCE_COL] - df[ATTENDANCE_COL].mean():+.1f} vs class", border=True)
    if p_risk is not None:
        d.metric("P(needs support)", f"{p_risk:.0%}", "from attendance alone", delta_color="off",
                 delta_arrow="off", border=True)

    left, middle, right = st.columns([1.1, 1, 1], gap="medium")
    with left:
        plot(figures.student_radar(df, i, buddies[0] if buddies else None), "sx_radar")
    with middle:
        plot(figures.student_strengths(lin["z_vectors"][i]), "sx_z")
        best, worst = lin["z_vectors"][i].argmax(), lin["z_vectors"][i].argmin()
        st.markdown(f"Relative to classmates, **{SUBJECT_LABELS[SUBJECTS[best]]}** is the strongest subject "
                    f"and **{SUBJECT_LABELS[SUBJECTS[worst]]}** the one to work on.")
    with right:
        with st.container(border=True):
            pair_row = lin["pairs_table"].iloc[pair_no]
            st.markdown(f"**:material/handshake: Study buddy**  \n{', '.join(names[j] for j in buddies)} "
                        f"(cos θ = {pair_row['Cosine_Similarity']:.2f})  \nWork together on "
                        f"**{pair_row['Focus_Subject']}**")
        with st.container(border=True):
            group_row = lin["groups_table"].iloc[group_no]
            st.markdown(f"**:material/groups: Study group {group_no + 1}**  \n{group_row['Members']}  \n"
                        f"Focus subject: **{group_row['Focus_Subject']}**")
        mentors = lin["mentor_pairs"]
        role = mentors[(mentors["Mentor"] == row[NAME_COL]) | (mentors["Mentee"] == row[NAME_COL])]
        if not role.empty:
            r = role.iloc[0]
            text = (f"Mentors **{r['Mentee']}** in {r['Teach_Subject']}" if r["Mentor"] == row[NAME_COL]
                    else f"Mentored by **{r['Mentor']}** in {r['Teach_Subject']}")
            with st.container(border=True):
                st.markdown(f"**:material/school: Peer mentoring**  \n{text}")
        matches = lin["top_matches"].iloc[i].drop("Student")
        st.caption("Closest profiles: " + " · ".join(matches))

    if level and level != "Low" and "attendance_low_risk" in ew:
        target = ew["attendance_low_risk"]
        st.warning(f"Raising attendance from {row[ATTENDANCE_COL]:.0f}% to about {target:.0f}% would bring "
                   f"{row[NAME_COL]}'s modelled risk below {LOW_RISK:.0%}.", icon=":material/trending_up:")
    left, right = st.columns([1.4, 1])
    with left:
        plot(figures.buddy_map(results, lin["groups"], highlight=i), "sx_map")
    with right:
        plot(figures.attendance_scatter(df, results["attendance_correlation"], ew["trend"], highlight=i),
             "sx_scatter")


# ---- Page ------------------------------------------------------------------

data, name, settings = sidebar()
try:
    results = analyse(data, name, *settings)
except (ValueError, KeyError, ZeroDivisionError) as exc:  # a malformed upload: explain instead of a traceback
    st.error(f"Could not analyse **{name}**: {exc}")
    st.stop()

downloads(results, data, name, settings)
header(results)
tabs = st.tabs([":material/insights: Overview", ":material/table_view: Data", ":material/functions: Statistics",
                ":material/casino: Probability", ":material/groups: Study groups",
                ":material/warning: Early warning", ":material/person_search: Student explorer"])
for tab, render in zip(tabs, [overview_tab, data_tab, statistics_tab, probability_tab, study_groups_tab,
                              early_warning_tab, student_tab]):
    with tab:
        render(results)
