"""
Data Generation Script (Section 5 of the project statement).

Generates data/student_data.csv: 30 students with Math, Science and English
scores plus attendance. Grades are simulated to correlate with attendance,
and a few Math scores are dropped to give Phase 1 something to clean.

Only change from the original script: `random.seed(42)` is added next to
`np.random.seed(42)`. The original seeded NumPy but not Python's `random`
module, so the missing-value step produced different gaps on every run.
"""

import os
import random

import numpy as np
import pandas as pd

# Set a seed so the random data is the same every time you run it
np.random.seed(42)
random.seed(42)

# List of random student names
first_names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Ayaan", "Krishna", "Ishaan", "Shaurya",
               "Ananya", "Aadhya", "Diya", "Saanvi", "Priya", "Neha", "Riya", "Kriti", "Pooja", "Kavya",
               "Rahul", "Rohan", "Amit", "Karan", "Vikram", "Sneha", "Nidhi", "Tanvi", "Rashi", "Meera"]

# Generate data for 30 students
data = []
for name in first_names:
    attendance = np.random.randint(60, 101)

    # Base score is influenced by attendance to create a statistical correlation
    base_score = attendance * 0.8

    # Generate grades with some random noise (clamped between 0 and 100)
    math_grade = min(100, max(0, int(np.random.normal(base_score, 10))))
    science_grade = min(100, max(0, int(np.random.normal(base_score, 8))))
    english_grade = min(100, max(0, int(np.random.normal(base_score + 5, 7))))

    # Occasionally drop a grade to simulate missing data (for Phase 1 cleaning)
    if random.random() < 0.05:
        math_grade = np.nan

    data.append([name, math_grade, science_grade, english_grade, attendance])

# Create a Pandas DataFrame
df = pd.DataFrame(data, columns=["Student_Name", "Math_Score", "Science_Score", "English_Score", "Attendance_Pct"])

# Save to CSV
os.makedirs("data", exist_ok=True)
filename = os.path.join("data", "student_data.csv")
df.to_csv(filename, index=False)
print(f"Saved {len(df)} student records to {filename} "
      f"({int(df.isna().sum().sum())} missing values)")
