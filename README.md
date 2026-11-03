# GradeVector: Student Performance & Study Group Analyzer

A data analysis tool that evaluates student academic records using **Python, statistics, probability, linear algebra and machine learning**. It cleans a class dataset, summarises performance, estimates the odds of passing or scoring an A, and pairs students into study groups by treating each student's grades as a vector and comparing them with **cosine similarity**. It then goes two steps further. Resampling puts **95% confidence intervals and p-values** on every headline number, and a **logistic-regression early-warning model** flags students who are likely to need support from their attendance alone, before any exam is marked. Everything can be explored live in an **interactive Streamlit dashboard**.

> Minor Project · Data Analytics with Python

---

## Highlights

| | |
|---|---|
| **Interactive dashboard** | `streamlit run app.py`: live sliders for every rule, CSV upload, hover charts, a what-if attendance simulator and a per-student explorer with radar profiles |
| **Smart data cleaning** | Missing scores are predicted from each student's own profile with least-squares regression (`np.linalg.lstsq`); median filling is also available |
| **Full statistics** | Mean, median, sample standard deviation, coefficient of variation, and a Pearson correlation written out from its formula |
| **Statistical inference** | Bootstrap confidence intervals, permutation tests and Wilson score intervals, all hand-written in NumPy (no SciPy) |
| **Probability** | Empirical P(pass), P(A), joint and conditional probabilities, cross-checked against a fitted normal curve |
| **Linear algebra matching** | Cosine similarity on **standardised** grade vectors, greedy grouping refined by a swap-based local search, and an SVD projection to visualise the result |
| **Machine learning from scratch** | Logistic regression fitted by Newton's method (a Hessian solve per step) with L2 regularisation, validated by leave-one-out cross-validation (accuracy, sensitivity, AUC) |
| **Validated** | Matching quality is measured against 2,000 random groupings; 41 tests cover the maths and the dashboard |
| **Deliverables** | Interactive dashboard, console summary, 10 charts, 12 CSV tables, an executed Jupyter notebook and a self-contained HTML report |

---

## Project structure

```
minor project1/
├── generate_data.py              # Section 5 data script -> data/student_data.csv
├── main.py                       # runs every phase from the command line
├── app.py                        # interactive Streamlit dashboard
├── requirements.txt
├── pytest.ini
├── .streamlit/config.toml        # dashboard theme (same palette as the charts)
├── analyzer/                     # the analysis package (NumPy, Pandas, Matplotlib only)
│   ├── config.py                 # column names, pass mark, A-grade mark, risk mark, colours
│   ├── data_setup.py             # Phase 1: load, missing values, imputation
│   ├── statistics_analysis.py    # Phase 2: descriptive stats, correlation
│   ├── probability.py            # Phase 3: empirical & conditional probability
│   ├── inference.py              # bootstrap, permutation tests, Wilson intervals
│   ├── study_buddy.py            # Phase 4: vectors, cosine similarity, grouping, SVD
│   ├── prediction.py             # Phase 5: logistic regression, cross-validation, what-if
│   ├── visualize.py              # all static charts (Matplotlib)
│   ├── summary.py                # printed class summary
│   ├── report.py                 # self-contained HTML report
│   └── pipeline.py               # runs the phases in order
├── dashboard/
│   └── figures.py                # interactive Plotly charts for app.py
├── notebooks/
│   └── Student_Performance_Analysis.ipynb   # step-by-step walkthrough (executed)
├── tests/
│   ├── test_analyzer.py          # 39 tests of the maths
│   └── test_dashboard.py         # the dashboard renders and reacts to its controls
├── data/
│   └── student_data.csv          # generated dataset (30 students)
└── outputs/                      # everything main.py produces
    ├── report.html               # open in any browser
    ├── class_summary.txt
    ├── charts/  (01-10 *.png)
    ├── cleaned_student_data.csv
    ├── statistical_summary.csv
    ├── probabilities.csv
    ├── mean_confidence_intervals.csv
    ├── probability_confidence_intervals.csv
    ├── study_buddy_pairs.csv
    ├── study_groups.csv
    ├── top_matches.csv
    ├── peer_mentor_pairs.csv
    ├── early_warning_models.csv
    ├── attendance_what_if.csv
    └── risk_register.csv
```

---

## How to run

**Requirements:** Python 3.10 or newer.

```bash
# 1. Create and activate a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate
#    macOS / Linux: source .venv/bin/activate

# 2. Install the libraries
pip install -r requirements.txt

# 3. Generate the dataset (Section 5 script)
python generate_data.py

# 4. Run the full analysis
python main.py

# 5. Open the interactive dashboard (opens in your browser)
streamlit run app.py
```

Then open **`outputs/report.html`** in a browser, or the notebook in VS Code / Jupyter.

### The dashboard

| Tab | What you can do |
|---|---|
| Overview | Key findings and headline charts, updated live as the rules change |
| Data | See which values were missing and how they were filled; the cleaned table with inline bars |
| Statistics | Distributions, attendance trend with a 95% prediction band, and the resampling results |
| Probability | P(pass) and P(A) with Wilson intervals, observed vs normal model, conditional odds |
| Study groups | Switch between pairs, groups and peer mentors on the SVD map; similarity matrix |
| Early warning | Drag an attendance slider and watch the expected score and risk move; filter the risk register |
| Student explorer | Pick any student: radar profile, strengths as z-scores, buddy, group, mentor and an attendance target |

The sidebar changes the pass mark, A-grade cut-off, "needs support" mark, group size and imputation method, and accepts an uploaded CSV. It also offers the full HTML report and the CSV tables as downloads.

### Options

```bash
python main.py --pass-mark 50          # change the pass mark (default 40)
python main.py --a-grade 85            # change the A-grade cut-off (default 80)
python main.py --group-size 4          # study groups of 4 instead of 3
python main.py --impute median         # fill missing values with the median instead
python main.py --risk-mark 65          # "needs support" below an overall of 65 (default 60)
python main.py --data other_class.csv  # analyse a different CSV with the same columns
```

### Tests

```bash
python -m pytest
```

---

## Methodology

### Phase 1: Data setup (Python & Analytics)
* Load `student_data.csv` with Pandas and check the required columns (`Student_Name`, `Math_Score`, `Science_Score`, `English_Score`, `Attendance_Pct`).
* Report missing values. The generator drops about 5% of Math scores; with the fixed seed, 4 are missing.
* **Imputation by regression.** For each subject with gaps, fit `score ≈ w₀ + w₁·other_subject₁ + w₂·other_subject₂ + w₃·attendance` on the complete rows by least squares, then predict the missing values. A student with high attendance and strong grades elsewhere gets a fitting estimate instead of the class median.
* Bar chart of the class average per subject, with ±1 SD whiskers and pass/A-grade reference lines.

### Phase 2: Statistical analysis (Statistics)
* Mean, median, sample standard deviation (n − 1), min, max, range and coefficient of variation per subject and overall.
* Pearson correlation between attendance and overall average, written out from its definition and checked against Pandas:

  r = Σ(x − x̄)(y − ȳ) / √[Σ(x − x̄)² · Σ(y − ȳ)²]

* Least-squares trend line, r², a correlation matrix and the top and bottom performers.

### Phase 3: Pass/fail odds (Probability)
* **Empirical probability** = students meeting the condition ÷ total students.
* P(pass) for score ≥ 40, and P(A) for score ≥ 80 ("scoring above 80%"), per subject and overall.
* Joint events: P(pass all three), P(at least one A), P(A in all three), compared with the product rule for independent events.
* **Conditional probability** by attendance band: P(pass all | band) and P(at least one A | band).
* Normal-model cross-check: P(X ≥ 80) ≈ 1 − Φ((79.5 − μ)/σ), with a continuity correction for whole-number scores.

### Phase 4: Study buddy match (Linear Algebra)
* Each student is a vector **v** = [Math, Science, English], so the class is a 30 × 3 matrix.
* Cosine similarity: cos θ = (**a** · **b**) / (‖**a**‖ ‖**b**‖). All pairs are computed at once as S = U Uᵀ, where U holds the row vectors scaled to unit length.
* **Why standardise first:** raw scores are all positive, so every raw vector points the same way and every pair scores 0.93 to 1.00. That range is useless for ranking. Converting each subject to z-scores moves the origin to the average student, and similarities then spread across −1 to +1.
* **Grouping:** greedily seed each group with the most similar free pair and add the closest classmate. A local search then swaps students between groups whenever that raises total within-group similarity.
* Each pair or group gets a *shared strength* and a *focus subject* (the subjects where its average z-score is highest and lowest).
* **Bonus: peer mentoring.** Matching on the most *negative* cosine pairs opposite profiles. The student ahead in a subject mentors the other in it.
* **SVD map:** Z = U Σ Vᵀ projects the 3-D vectors onto the two directions that carry the most variation, so the pairs can be seen on a 2-D chart.

### How sure can we be? (Statistical inference)
One class of 30 is a small sample, so every figure would shift a little for another class. Each of these methods is written out in `analyzer/inference.py` using NumPy only:
* **Bootstrap confidence intervals.** Redraw the 30 students with replacement 10,000 times and recompute the statistic. The middle 95% of the results is the 95% interval, used for subject means, for r, and for the gap between attendance bands.
* **Permutation tests.** Shuffle attendance 10,000 times to destroy any real link and count how often chance matches the observed r (or band gap). That share is the p-value.
* **Wilson score intervals** for P(pass) and P(A): (p̂ + z²/2n ± z√[p̂(1−p̂)/n + z²/4n²]) / (1 + z²/n). Unlike p̂ ± 1.96√(p̂(1−p̂)/n), it does not collapse to zero width when every student passes.
* **Effect size**: Cohen's d, the band gap measured in pooled standard deviations.

### Phase 5: Early warning (Machine learning)
* Attendance is known weeks before results, so it can flag students while there is time to act. A student **needs support** if their overall average is below 60 or they fail any subject.
* **Logistic regression**, P(needs support | x) = 1 / (1 + e^−(β₀ + β₁x)), is fitted by maximum likelihood with **Newton's method**. Each step solves the linear system H d = g with H = XᵀWX + λI and W = diag(p(1 − p)), using a backtracking safeguard. A small L2 penalty λ keeps the slope finite if attendance separates the groups perfectly. A second model predicts "at least one A".
* **Leave-one-out cross-validation** refits the model 30 times, each time predicting the one student left out. It reports accuracy against a majority-class baseline, sensitivity, specificity and AUC.
* **What-if.** The expected overall score at any attendance, with a 95% *prediction* interval ŷ ± t·s·√(1 + 1/n + (x₀ − x̄)²/Sxx). The t critical value comes from a Cornish–Fisher expansion, so no statistics library is needed.
* **Risk register.** Every student gets a risk level (High ≥ 50%, Watch ≥ 20%, Low) and the number of attendance points they need to reach low risk.

---

## Results (seed 42 dataset)

| Measure | Value |
|---|---|
| Students / missing values repaired | 30 / 4 (all Math) |
| Class average (Math / Science / English) | 68.13 / 67.40 / 71.13 |
| Standard deviation (Math / Science / English) | 13.81 / 13.67 / 10.96 |
| Attendance vs overall score | **r = 0.943**, r² = 0.888 (strong positive) |
| P(pass) Math / Science / English | 96.7% / 93.3% / 100% |
| P(A, score ≥ 80) Math / Science / English | 13.3% / 16.7% / 33.3% |
| P(pass all three subjects) | 90.0% |
| P(at least one A): attendance 90%+ vs below 75% | 100% vs 12.5% |
| Raw vs standardised cosine range | 0.93–1.00 vs −1.00–1.00 |
| Average similarity: study pairs / groups of 3 | **0.903** / 0.794 (random: −0.01) |
| Attendance correlation, 95% bootstrap CI | r = 0.943, CI 0.880–0.978, permutation **p < 0.0001** |
| High vs low attendance band | 78.2 vs 54.8: gap 23.4 marks (95% CI 17.3–28.8), Cohen's d = 3.9 |
| P(A in English), Wilson 95% CI | 33.3% (19.2%–51.2%) |
| Early warning, leave-one-out | **93% accuracy** (baseline 80%), sensitivity 83%, **AUC 0.97** |
| Attendance at 50% risk / below 20% risk | 70% / 75% |
| Students flagged (High / Watch) | 6 / 2 |

**Takeaways**
1. Attendance is the strongest predictor of results. Each extra 10% of attendance goes with about 7.7 more marks, and the permutation test shows this is not chance.
2. English is the strongest subject: it has the highest average, the lowest spread and the most A grades. Math and Science have the widest spread. The P(A) intervals overlap, though, so one class cannot prove the subjects differ.
3. The low-attendance band (below 75%) is where all the failures are, so it should be the first focus for support.
4. Cosine matching on standardised vectors produces pairs far more alike than chance (0.90 vs about 0). Each pair also gets a clear subject to work on together.
5. The early-warning model needs nothing but attendance. On students it never saw, it picks out 5 of the 6 who ended up needing support, with one false alarm, and it gives each flagged student a concrete attendance target (about 75%).

Sample pairs: *Shaurya & Kriti* (0.994), *Sneha & Rashi* (0.991), *Aditya & Rahul* (0.989). The full list is in `outputs/study_buddy_pairs.csv`.

---

## Expected outcomes checklist

| Project statement requirement | Delivered in |
|---|---|
| Python script / Jupyter notebook that processes the CSV | `main.py`, `notebooks/Student_Performance_Analysis.ipynb` |
| Handle missing data | `analyzer/data_setup.py` (regression or median imputation) |
| Bar chart of class averages per subject | `outputs/charts/01_class_averages.png` |
| Mean, median, standard deviation per subject | `analyzer/statistics_analysis.py`, `outputs/statistical_summary.csv` |
| Correlation between attendance and performance | Phase 2 output, `03_attendance_vs_performance.png`, `04_correlation_heatmap.png` |
| Probability of passing / scoring an A (above 80%) | `analyzer/probability.py`, `outputs/probabilities.csv`, `05_probabilities.png` |
| Printed statistical summary including P(pass) | console output, `outputs/class_summary.txt` |
| Cosine similarity of grade vectors (dot product) | `analyzer/study_buddy.py`, `07_similarity_heatmap.png` |
| Study buddy pairings list | `outputs/study_buddy_pairs.csv`, `outputs/study_groups.csv` |
| Clean visualisations | 10 charts in `outputs/charts/` with a consistent, colour-blind-safe palette |
| *Extension:* confidence intervals and significance tests | `analyzer/inference.py`, `09_statistical_confidence.png`, `*_confidence_intervals.csv` |
| *Extension:* predictive early-warning model | `analyzer/prediction.py`, `10_early_warning.png`, `risk_register.csv` |
| *Extension:* interactive dashboard | `app.py` (`streamlit run app.py`) |

---

## Note on the data generator
`generate_data.py` is the Section 5 script with one addition: `random.seed(42)`. The original seeds NumPy but not Python's `random` module, which decides the missing values. Without the extra seed, a different set of Math scores went missing on every run. Everything else, including names, distributions and the 5% drop rate, is unchanged.

## Future scope
* Load real class records from a school's spreadsheet export.
* Add more subjects. The vector method works unchanged in any number of dimensions.
* Track performance across terms to spot improving or declining students, and retrain the early-warning model on each new term.
* Add more early signals (homework completion, quiz scores) as extra features in the logistic regression.
* Deploy the dashboard (for example on Streamlit Community Cloud) so teachers can use it without installing Python.

## Tech stack
Python 3 · NumPy · Pandas · Matplotlib · Streamlit · Plotly · Jupyter · pytest

The analysis package itself uses only NumPy, Pandas and Matplotlib. The statistics and machine learning are implemented from their formulas rather than imported from SciPy or scikit-learn.
