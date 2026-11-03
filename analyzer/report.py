"""Builds a self-contained HTML report (charts embedded) from the analysis results."""

import base64
import html
import os
from datetime import date

import numpy as np
import pandas as pd

from .config import SUBJECT_LABELS, SUBJECTS
from .inference import format_p


def _img(path, alt, narrow=False):
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("ascii")
    css = ' class="narrow"' if narrow else ""
    return f'<figure{css}><img src="data:image/png;base64,{data}" alt="{html.escape(alt)}"></figure>'


def _cell(value, pct=False, decimals=False):
    """Format one table cell; numbers are right-aligned, text left-aligned."""
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        if pct:
            text = f"{value:.1%}"
        elif decimals:
            text = f"{value:.2f}"
        else:
            text = f"{int(value)}"
        return f'<td class="num">{text}</td>'
    return f"<td>{html.escape(str(value))}</td>"


def _table(df, pct_cols=(), index=True):
    numeric = [pd.api.types.is_numeric_dtype(df[c]) for c in df.columns]
    head = (["<th></th>"] if index else []) + [
        f'<th class="{"num" if is_num else ""}">{html.escape(str(c).replace("_", " "))}</th>'
        for c, is_num in zip(df.columns, numeric)]
    rows = []
    for label, row in df.iterrows():
        cells = [f"<th>{html.escape(str(label))}</th>"] if index else []
        cells += [_cell(v, pct=c in pct_cols, decimals=pd.api.types.is_float_dtype(df[c]))
                  for c, v in row.items()]
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (f'<table class="data"><thead><tr>{"".join(head)}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def key_findings(results):
    stats = results["summary_stats"].drop(index="Overall")
    probs = results["probabilities"].drop(index="Overall")
    corr = results["attendance_correlation"]
    cond = results["conditional"]
    lin = results["linear_algebra"]
    events = results["combined_events"]
    best, worst = stats["Mean"].idxmax(), stats["Mean"].idxmin()
    widest = stats["Std_Dev"].idxmax()
    easiest_a, hardest_a = probs["P(A grade)"].idxmax(), probs["P(A grade)"].idxmin()
    high, low = cond.index[0], cond.index[-1]
    findings = [
        f"<b>{best}</b> has the highest class average ({stats.loc[best, 'Mean']:.1f}) and "
        f"<b>{worst}</b> the lowest ({stats.loc[worst, 'Mean']:.1f}). {widest} has the widest spread "
        f"(SD {stats.loc[widest, 'Std_Dev']:.1f}), so ability in it varies most across the class.",
        f"Attendance is the strongest single signal in the data: <b>r = {corr['r']:.2f}</b> with overall "
        f"score, explaining about {corr['r_squared']:.0%} of the variation. Each extra 10 points of "
        f"attendance goes with roughly {corr['slope'] * 10:.1f} more marks.",
        f"A randomly selected student has a <b>{probs.loc[easiest_a, 'P(A grade)']:.0%}</b> chance of an "
        f"A in {easiest_a} but only {probs.loc[hardest_a, 'P(A grade)']:.0%} in {hardest_a}. "
        f"{events['P(pass all three subjects)']:.0%} of students pass all three subjects.",
        f"Attendance changes the odds sharply: in the {high.split(' (')[0].lower()} attendance band, "
        f"P(at least one A) = <b>{cond.loc[high, 'P(at least one A)']:.0%}</b>; in the "
        f"{low.split(' (')[0].lower()} band it is {cond.loc[low, 'P(at least one A)']:.0%}, and "
        f"P(pass all) falls to {cond.loc[low, 'P(pass all)']:.0%}.",
        f"Raw grade vectors are nearly parallel (cosine {lin['raw_pair_values'].min():.2f} to "
        f"{lin['raw_pair_values'].max():.2f}), so they cannot separate students. After standardising, "
        f"similarities span {lin['std_pair_values'].min():.2f} to {lin['std_pair_values'].max():.2f}, and "
        f"the matched pairs average <b>{lin['pair_score']:.2f}</b>, against {lin['pair_baseline']:.2f} "
        f"for random pairing.",
    ]
    inf = results["inference"]["correlation"]
    shuffles = results["inference"]["n_resamples"]
    findings.insert(2,
        f"The attendance link is not a fluke of this particular class. Across {shuffles:,} random shuffles of "
        f"attendance, chance alone never matched r = {inf['r']:.2f} (<b>{inf['p_text']}</b>), and "
        f"bootstrap resampling puts the true r between {inf['ci_low']:.2f} and {inf['ci_high']:.2f} "
        f"with 95% confidence.")
    ew = results["early_warning"]
    if ew["risk"] and "attendance_50" in ew:
        m = ew["risk"]["metrics"]
        flagged = int((ew["register"]["Risk_Level"] != "Low").sum())
        findings.append(
            f"Attendance alone gives an early warning before any exam is marked. Below "
            f"<b>{ew['attendance_50']:.0f}%</b> attendance, a student is more likely than not to need support. "
            f"Tested on students it had not seen (leave-one-out), the model is right {m['accuracy']:.0%} of the "
            f"time, against {m['baseline_accuracy']:.0%} for always guessing the majority (AUC {m['auc']:.2f}). "
            f"{flagged} students are flagged now.")
    return findings


def _inference_section(inf, charts, cfg, n):
    corr, band = inf["correlation"], inf["band_test"]
    mean_html = _table(inf["mean_ci"].rename(columns={"CI_Low": "95% low", "CI_High": "95% high",
                                                      "Margin": "± margin"}))
    prob_cols = {"Pass_Low": "Pass low", "Pass_High": "Pass high", "A_Low": "A low", "A_High": "A high"}
    prob_html = _table(inf["probability_ci"].rename(columns=prob_cols),
                       pct_cols=["P(Pass)", "Pass low", "Pass high", "P(A grade)", "A low", "A high"])
    band_text = ""
    if band:
        band_text = (f" Students in the {band['high_band'].split(' (')[0].lower()} attendance band average "
                     f"{band['mean_high']:.1f}, against {band['mean_low']:.1f} in the "
                     f"{band['low_band'].split(' (')[0].lower()} band: a gap of <b>{band['difference']:.1f} marks</b> "
                     f"(95% CI {band['ci_low']:.1f} to {band['ci_high']:.1f}, {format_p(band['p_value'])}). "
                     f"Cohen's d = {band['cohens_d']:.1f}, where 0.8 already counts as a large effect.")
    subjects = inf["probability_ci"].drop(index="Overall")
    if subjects["A_Low"].max() < subjects["A_High"].min():  # every pair of intervals overlaps
        overlap_note = (f"The P(A) intervals of all subjects overlap, so with one class of {n} the differences "
                        f"between subjects are suggestive rather than proven.")
    else:
        overlap_note = ("Some subjects' P(A) intervals do not overlap at all, so those differences are unlikely to "
                        "be chance alone.")
    return f"""<section>
  <span class="phase">Statistical inference &middot; Resampling</span>
  <h2>How sure can we be?</h2>
  <p class="lead">A class of {n} is a small sample, so each figure above would come out a little differently for
  another class of the same size. Resampling measures how much, using only NumPy and {inf['n_resamples']:,}
  repetitions. The <b>bootstrap</b> redraws the class with replacement and recomputes each statistic; the middle
  95% of the results is a 95% confidence interval. A <b>permutation test</b> shuffles attendance to break any
  real link and counts how often chance matches the real correlation; that share is the p-value.</p>
  {_img(charts['09'], 'Permutation test, bootstrap distribution of r, and P(A) with confidence intervals')}
  <div class="callout">Attendance vs overall score: r = {corr['r']:.3f}, 95% CI {corr['ci_low']:.3f} to
  {corr['ci_high']:.3f}, {corr['p_text']}.{band_text}</div>
  <div class="grid2">
    <div><h3>Class mean, bootstrap 95% CI</h3><div class="table-wrap">{mean_html}</div></div>
    <div><h3>Probabilities, Wilson 95% CI</h3><div class="table-wrap">{prob_html}</div></div>
  </div>
  <p>Proportions use the Wilson score interval instead of p &plusmn; 1.96&radic;(p(1&minus;p)/n). The simple
  formula gives a zero-width interval when every student passes, which overstates the certainty for 30 students:</p>
  <div class="formula">(p&#770; + z&sup2;/2n &plusmn; z&radic;[p&#770;(1&minus;p&#770;)/n + z&sup2;/4n&sup2;]) / (1 + z&sup2;/n),
  &nbsp; z = 1.96</div>
  <p class="lead">{overlap_note}</p>
</section>"""


def _early_warning_section(ew, charts, cfg):
    if not ew["risk"]:
        return """<section><span class="phase">Phase 5 &middot; Machine learning</span><h2>Early warning</h2>
  <p>Too few students fall in one of the outcome groups to fit the early-warning model for this class.</p></section>"""
    pct = ["LOO Accuracy", "Majority Baseline", "Sensitivity", "Specificity"]
    card = ew["model_card"].copy()
    card.columns = [c.replace("_x_per_10pts", " × per +10 pts").replace("_", " ") for c in card.columns]
    card_html = _table(card, pct_cols=pct)
    what_if = ew["what_if"].copy()
    what_if.index = [f"{a}% attendance" for a in what_if.index]
    what_if_html = _table(what_if.rename(columns={
        "Expected_Overall": "Expected overall", "Overall_Low": "95% low", "Overall_High": "95% high",
        "P_Needs_Support": "P(needs support)", "Risk_Level": "Risk", "P_At_Least_One_A": "P(≥1 A)"}),
        pct_cols=["P(needs support)", "P(≥1 A)"])
    register = ew["register"]
    flagged = register[register["Risk_Level"] != "Low"]
    columns = {"Attendance_Pct": "Attendance %", "Overall_Avg": "Overall", "Needs_Support": "Needs support now",
               "P_Needs_Support": "P(needs support)", "Risk_Level": "Risk",
               "Attendance_Gain_Needed": "Attendance points to low risk", "P_At_Least_One_A": "P(≥1 A)"}
    flagged_html = _table(flagged.rename(columns=columns), pct_cols=["P(needs support)", "P(≥1 A)"], index=False)
    register_html = _table(register.rename(columns=columns), pct_cols=["P(needs support)", "P(≥1 A)"],
                           index=False)
    threshold = ""
    if "attendance_50" in ew:
        threshold = (f" The fitted curve crosses 50% at <b>{ew['attendance_50']:.0f}% attendance</b> and falls "
                     f"below 20% from {ew['attendance_low_risk']:.0f}%, which gives a concrete attendance target.")
    chart = _img(charts["10"], "Logistic regression curves for needing support and for an A grade") \
        if "10" in charts else ""
    return f"""<section>
  <span class="phase">Phase 5 &middot; Machine learning</span>
  <h2>Early warning</h2>
  <p class="lead">Attendance is known weeks before results come out, so it can flag students while there is
  still time to help. A logistic regression, written from scratch, models the chance that a student
  <b>needs support</b> (overall below {ew['risk_mark']}, or any subject below the pass mark of
  {cfg['pass_mark']}):</p>
  <div class="formula">P(needs support | x) = 1 / (1 + e<sup>&minus;(&beta;<sub>0</sub> + &beta;<sub>1</sub>x)</sup>)</div>
  <p>The weights are fitted by maximum likelihood with Newton's method. Each step solves a linear system with the
  Hessian, w &larr; w &minus; H<sup>&minus;1</sup>g, where H = X<sup>T</sup>WX + &lambda;I and W holds
  p(1 &minus; p) on its diagonal. A small L2 penalty &lambda; keeps the slope finite if attendance separates
  the groups perfectly. A second model does the same for <b>at least one A grade</b>.</p>
  {chart}
  <div class="callout">Each model is validated by <b>leave-one-out cross-validation</b>: it is refitted {len(register)}
  times, each time without one student, and then predicts that student. The scores below therefore measure
  performance on unseen students, not on the data the model learned from.{threshold}</div>
  <h3>Model card</h3>
  <div class="table-wrap">{card_html}</div>
  <h3>What if attendance changed?</h3>
  <p class="lead">Expected overall score from the attendance trend line, with a 95% <i>prediction</i> interval for
  one student: s&radic;(1 + 1/n + (x&#8320; &minus; x&#772;)&sup2;/S<sub>xx</sub>) times the t critical value.
  The t value comes from a Cornish&ndash;Fisher expansion, so no statistics library is needed.</p>
  <div class="table-wrap">{what_if_html}</div>
  <h3>Risk register: {len(flagged)} students flagged</h3>
  <div class="table-wrap">{flagged_html}</div>
  <details><summary>Full register ({len(register)} students)</summary><div class="table-wrap">{register_html}</div></details>
  <div class="callout">A correlation does not prove that attendance <i>causes</i> results, and the model learned
  from one class of {len(register)}. Treat a flag as a reason to talk to a student, not as a verdict.</div>
</section>"""


CSS = """
:root {
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --line: #e1e0d9; --accent: #2a78d6; --accent-soft: #e8f1fc;
  --math: #2a78d6; --science: #eb6834; --english: #1baf7a;
  color-scheme: light;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 0 20px 64px; }
header { background: #0d366b; color: #fff; padding: 44px 0 36px; margin-bottom: 32px; }
header .eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: 12px; opacity: .75; }
header h1 { margin: 6px 0 4px; font-size: 34px; letter-spacing: -.01em; }
header p { margin: 0; opacity: .85; }
.meta { margin-top: 14px; font-size: 13px; opacity: .7; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: -58px 0 28px; }
.kpi { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px;
  box-shadow: 0 1px 2px rgba(0,0,0,.04); }
.kpi .label { font-size: 12px; color: var(--ink-2); }
.kpi .value { font-size: 26px; font-weight: 650; margin-top: 2px; }
.kpi .note { font-size: 12px; color: var(--muted); }
section { background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 26px 28px; margin-bottom: 22px; }
section h2 { margin: 0 0 4px; font-size: 22px; }
.phase { display: inline-block; font-size: 11.5px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  color: var(--accent); background: var(--accent-soft); border-radius: 999px; padding: 2px 10px; margin-bottom: 8px; }
h3 { font-size: 16px; margin: 26px 0 8px; }
.lead { color: var(--ink-2); margin-top: 4px; }
ul.findings { padding-left: 20px; } ul.findings li { margin: 8px 0; }
figure { margin: 18px 0; } figure img { max-width: 100%; height: auto; display: block; margin: 0 auto;
  border-radius: 8px; border: 1px solid var(--line); }
figure.narrow img { max-width: min(100%, 760px); }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }
.grid2 > * { min-width: 0; }  /* let wide tables scroll inside their column instead of widening the page */
@media (max-width: 820px) { .grid2 { grid-template-columns: 1fr; } section { padding: 20px 16px; } }
.table-wrap { overflow-x: auto; }
table.data { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
table.data th, table.data td { padding: 7px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
table.data .num { text-align: right; }
table.data tbody th { font-weight: 600; }
table.data thead th { color: var(--ink-2); font-weight: 600; border-bottom: 2px solid var(--line); }
table.data tbody tr:hover { background: var(--accent-soft); }
.formula { background: var(--page); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px;
  font-family: "Cambria Math", Cambria, Georgia, serif; font-size: 17px; text-align: center; margin: 12px 0; }
.callout { border-left: 3px solid var(--accent); background: var(--accent-soft); padding: 10px 14px;
  border-radius: 0 8px 8px 0; margin: 14px 0; color: var(--ink); }
details { margin-top: 12px; } summary { cursor: pointer; color: var(--accent); font-weight: 600; }
footer { text-align: center; color: var(--muted); font-size: 13px; margin-top: 30px; }
@media print { header { -webkit-print-color-adjust: exact; print-color-adjust: exact; } section { break-inside: avoid; } }
"""


def build_report(results, chart_paths, out_path):
    cfg = results["settings"]
    df = results["data"]
    stats = results["summary_stats"]
    corr = results["attendance_correlation"]
    probs = results["probabilities"]
    lin = results["linear_algebra"]
    events = results["combined_events"]
    charts = {os.path.basename(p)[:2]: p for p in chart_paths}
    inf = results["inference"]
    ew = results["early_warning"]
    mean_ci = inf["mean_ci"].loc["Overall"]

    kpis = [
        ("Students analysed", f"{len(df)}", f"{int(results['missing_report'].sum())} missing values repaired"),
        ("Class average", f"{stats.loc['Overall', 'Mean']:.1f}",
         f"95% CI {mean_ci['CI_Low']:.1f} to {mean_ci['CI_High']:.1f}"),
        ("Pass all subjects", f"{events['P(pass all three subjects)']:.0%}", f"pass mark {cfg['pass_mark']}"),
        ("P(A in any subject)", f"{events['P(at least one A)']:.0%}", f"A grade = {cfg['a_mark']}+"),
        ("Attendance link", f"r = {corr['r']:.2f}", f"{corr['strength']}, {inf['correlation']['p_text']}"),
        ("Buddy-pair similarity", f"{lin['pair_score']:.2f}", f"random: {lin['pair_baseline']:.2f}"),
    ]
    if ew["risk"]:
        m = ew["risk"]["metrics"]
        kpis += [
            ("Flagged for support", f"{int((ew['register']['Risk_Level'] != 'Low').sum())}",
             "high or watch risk from attendance"),
            ("Early-warning accuracy", f"{m['accuracy']:.0%}",
             f"leave-one-out, AUC {m['auc']:.2f}"),
        ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="label">{l}</div><div class="value">{v}</div><div class="note">{n}</div></div>'
        for l, v, n in kpis)
    findings = "".join(f"<li>{f}</li>" for f in key_findings(results))

    imputation = results["imputation_log"]
    missing_html = (_table(imputation, index=False) if not imputation.empty
                    else "<p>No missing values were found.</p>")
    stats_html = _table(stats)
    prob_html = _table(probs.assign(**{"Students with A": probs["Students with A"].astype(int)}),
                       pct_cols=["P(Pass)", "P(Fail)", "P(A grade)", "P(A) normal model"])
    events_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:.1%}</td></tr>" for k, v in events.items())
    cond = results["conditional"].assign(Students=lambda d: d["Students"].astype(int))
    cond_html = _table(cond, pct_cols=["P(pass all)", "P(at least one A)"])
    corr_rows = "".join(f"<tr><td>Attendance vs {s}</td><td>{r:.3f}</td></tr>"
                        for s, r in corr["per_subject"].items())

    pairs_html = _table(lin["pairs_table"], index=False)
    groups_html = _table(lin["groups_table"], index=False)
    matches_html = _table(lin["top_matches"], index=False)
    mentor_html = _table(lin["mentor_pairs"], index=False)
    data_cols = ["Student_Name", *SUBJECTS, "Attendance_Pct", "Overall_Avg", "Overall_Grade", "Attendance_Band"]
    data_html = _table(df[data_cols].rename(columns={**SUBJECT_LABELS, "Student_Name": "Student"}), index=False)

    inference_html = _inference_section(inf, charts, cfg, len(df))
    early_warning_html = _early_warning_section(ew, charts, cfg)

    example_a, example_b = lin["pairs"][0][:2]  # the top "pair" is a trio when the class size is odd
    names = lin["names"]
    za, zb = lin["z_vectors"][example_a], lin["z_vectors"][example_b]
    worked = (f"{names[example_a]} z = [{', '.join(f'{v:.2f}' for v in za)}], "
              f"{names[example_b]} z = [{', '.join(f'{v:.2f}' for v in zb)}]  &rarr;  "
              f"cos &theta; = {lin['similarity'][example_a, example_b]:.3f}")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GradeVector Report</title><style>{CSS}</style></head>
<body>
<header><div class="wrap">
  <div class="eyebrow">Minor Project &middot; Data Analytics with Python</div>
  <h1>GradeVector</h1>
  <p>Student Performance &amp; Study Group Analyzer: statistics, probability, linear algebra and an early-warning
  model on class records</p>
  <div class="meta">Generated {date.today():%d %B %Y} &middot; Dataset: {html.escape(os.path.basename(cfg['csv_path']))} &middot;
  Pass mark {cfg['pass_mark']} &middot; A grade {cfg['a_mark']}+ &middot; Needs support below {cfg['risk_mark']}
  &middot; Imputation: {cfg['impute']}</div>
</div></header>
<div class="wrap">
<div class="kpis">{kpi_html}</div>

<section>
  <h2>Key findings</h2>
  <ul class="findings">{findings}</ul>
</section>

<section>
  <span class="phase">Phase 1 &middot; Python &amp; Analytics</span>
  <h2>Data setup</h2>
  <p class="lead">The CSV is loaded with Pandas and its columns are checked. Any missing scores are then filled in.
  The <b>{cfg['impute']}</b> method predicts each missing score from the student's other subjects and attendance
  using least squares (solving <i>Xw &asymp; y</i>), so the filled value matches that student's profile
  instead of the class median.</p>
  <h3>Values repaired</h3>
  <div class="table-wrap">{missing_html}</div>
  {_img(charts['01'], narrow=True, alt='Bar chart of class average per subject')}
</section>

<section>
  <span class="phase">Phase 2 &middot; Statistics</span>
  <h2>Statistical analysis</h2>
  <p class="lead">Mean and median measure the centre of each subject, and the sample standard deviation measures its spread.
  The coefficient of variation (CV = SD / mean) puts spread on a common scale across subjects.</p>
  <div class="table-wrap">{stats_html}</div>
  {_img(charts['02'], 'Histograms of score distribution per subject')}
  <h3>Attendance and performance</h3>
  <div class="formula">r = &Sigma;(x &minus; x&#772;)(y &minus; y&#772;) / &radic;[&Sigma;(x &minus; x&#772;)&sup2; &middot; &Sigma;(y &minus; y&#772;)&sup2;]</div>
  <div class="callout">{html.escape(corr['interpretation'])}</div>
  <div class="grid2">
    <div>{_img(charts['03'], narrow=True, alt='Scatter of attendance versus overall score')}</div>
    <div>{_img(charts['04'], narrow=True, alt='Correlation heatmap')}
      <div class="table-wrap"><table class="data"><thead><tr><th>Pair</th><th>Pearson r</th></tr></thead>
      <tbody>{corr_rows}</tbody></table></div></div>
  </div>
</section>

<section>
  <span class="phase">Phase 3 &middot; Probability</span>
  <h2>Pass / fail odds</h2>
  <p class="lead">Empirical probability = (students meeting the condition) &divide; (all students). As a check,
  each subject is also fitted with a normal curve N(&mu;, &sigma;&sup2;), and P(score &ge; {cfg['a_mark']}) is read
  from it using a continuity correction.</p>
  <div class="table-wrap">{prob_html}</div>
  {_img(charts['05'], 'Probability charts')}
  <div class="grid2">
    <div><h3>Events across all subjects</h3><div class="table-wrap"><table class="data">
      <thead><tr><th>Event</th><th>Probability</th></tr></thead><tbody>{events_rows}</tbody></table></div></div>
    <div><h3>Conditional on attendance</h3><div class="table-wrap">{cond_html}</div></div>
  </div>
  <div class="callout">Where the observed and normal-curve values for P(A) disagree, the subject's scores are not
  bell-shaped. The empirical figure is then the more reliable one for this class.</div>
</section>

{inference_html}

<section>
  <span class="phase">Phase 4 &middot; Linear Algebra</span>
  <h2>Study buddy match</h2>
  <p class="lead">Each student becomes a vector <b>v</b> = [Math, Science, English]. Two students are similar when their
  vectors point the same way:</p>
  <div class="formula">cos &theta; = (<b>a</b> &middot; <b>b</b>) / (&Vert;<b>a</b>&Vert; &Vert;<b>b</b>&Vert;)</div>
  <p>Every pair is computed at once as a matrix product: normalise each row to unit length to get <i>U</i>,
  and <i>S = U U<sup>T</sup></i> is the 30 &times; 30 similarity matrix. Each score is first converted to a
  z-score (distance from the class average in standard deviations), because raw scores are all positive and
  every raw vector points the same way.</p>
  {_img(charts['06'], narrow=True, alt='Raw versus standardised cosine similarity distribution')}
  <p>Worked example (top pair): {worked}</p>
  <p>Groups are formed greedily: start from the most similar free pair, then add the student closest to the group.
  A swap-based local search then exchanges members between groups whenever that raises the total
  within-group similarity.</p>
  <h3>Study buddy pairs</h3>
  <div class="table-wrap">{pairs_html}</div>
  <p class="lead">Strength and focus subjects are relative to the class average: they show where the pair sits above or
  below their classmates, which is what a study session should target.</p>
  <h3>Study groups of {cfg['group_size']}</h3>
  <div class="table-wrap">{groups_html}</div>
  <p>Average within-group similarity <b>{lin['group_score']:.3f}</b> vs <b>{lin['group_baseline']:.3f}</b> for random groups.</p>
  {_img(charts['07'], 'Similarity heatmap ordered by study group')}
  {_img(charts['08'], 'Two-dimensional study buddy map from SVD')}
  <p class="lead">The map projects the 3-D standardised vectors onto their two main directions using the singular value
  decomposition (Z = U&Sigma;V<sup>T</sup>). Together these two directions keep
  {sum(lin['pca_explained']):.0%} of the information. Grey lines join buddy pairs.</p>
  <h3>Bonus: peer-mentor pairs</h3>
  <p>Matching on the <i>most negative</i> cosine pairs opposite profiles instead. In each pair, the student ahead
  in a subject coaches the other in it.</p>
  <div class="table-wrap">{mentor_html}</div>
  <details><summary>Top 3 matches for every student</summary><div class="table-wrap">{matches_html}</div></details>
</section>

{early_warning_html}

<section>
  <h2>Conclusion</h2>
  <p>The analysis meets all four phases of the project statement. The data is cleaned and charted (Phase 1);
  each subject gets descriptive statistics and the attendance link is quantified (Phase 2); empirical
  probabilities of passing and of an A are computed and checked against a normal model (Phase 3); and every
  student is placed in a study pair and a study group by cosine similarity of grade vectors (Phase 4).
  Two extensions go further: resampling puts 95% confidence intervals and p-values on the headline numbers,
  and a logistic-regression early-warning model (Phase 5) turns attendance into a risk score that can be
  read weeks before exams. The clearest practical message is that attendance tracks results closely. Students
  flagged by the risk register are the group to support first, and the peer-mentor pairs offer a direct way
  to do that.</p>
  <details><summary>Cleaned dataset ({len(df)} students)</summary><div class="table-wrap">{data_html}</div></details>
</section>

<footer>GradeVector &middot; generated by main.py &middot; Python, Pandas, NumPy, Matplotlib</footer>
</div></body></html>"""

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return out_path
