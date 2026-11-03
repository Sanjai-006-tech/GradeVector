"""Phase 4 - Study Buddy Match: linear algebra on grade vectors.

Each student is a vector in 3-D space: [Math, Science, English].
Cosine similarity measures the angle between two vectors:

    cos(theta) = (a . b) / (|a| * |b|)

1 means "pointing the same way" (same profile), 0 means unrelated and
-1 means opposite profiles.

Why we standardise first: raw scores are all positive (roughly 40-90), so
every raw vector points into the same corner of space and every pair scores
about 0.98. Converting each subject to z-scores (how far above or below the
class average, in standard deviations) moves the origin to the "average
student". The angle then reflects who is above/below average in which
subjects, which is the profile we actually want to match on.
"""

import itertools

import numpy as np
import pandas as pd

from .config import NAME_COL, OVERALL_COL, SUBJECTS, SUBJECT_LABELS


def grade_vectors(df):
    """Matrix with one row per student: [Math, Science, English]."""
    return df[SUBJECTS].to_numpy(dtype=float)


def standardize(X):
    """Column-wise z-scores: (x - column mean) / column std."""
    return (X - X.mean(axis=0)) / X.std(axis=0)


def cosine_similarity(a, b):
    """Cosine similarity between two vectors, straight from the definition."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_similarity_matrix(X):
    """All pairwise cosine similarities in one matrix product.

    Normalise every row to length 1; then entry (i, j) of X_unit @ X_unit.T is
    the dot product of two unit vectors, which is exactly cos(theta).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # a student exactly at the class average has no direction
    unit = X / norms
    return np.clip(unit @ unit.T, -1.0, 1.0)


def off_diagonal(sim):
    """All similarities between two different students (each pair once)."""
    i, j = np.triu_indices(len(sim), k=1)
    return sim[i, j]


# ---- Grouping --------------------------------------------------------------

def _within_score(group, S):
    """Total similarity over every pair inside one group."""
    return sum(S[a, b] for a, b in itertools.combinations(group, 2))


def _greedy_groups(S, size):
    """Seed each group with the most similar free pair, then grow it with the
    student who has the highest average similarity to the current members."""
    free = set(range(len(S)))
    groups = []
    while len(free) >= size:
        idx = sorted(free)
        sub = S[np.ix_(idx, idx)].copy()
        np.fill_diagonal(sub, -np.inf)
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        group = [idx[i], idx[j]]
        while len(group) < size:
            candidates = free - set(group)
            group.append(max(candidates, key=lambda c: S[c, group].mean()))
        groups.append(group)
        free -= set(group)
    # Students left over (class size not divisible by group size) join the
    # group they fit best.
    for student in sorted(free):
        best = max(groups, key=lambda g: S[student, g].mean())
        best.append(student)
    return groups


def _improve_by_swaps(groups, S):
    """Local search: swap two students between groups whenever that raises
    the total within-group similarity. Repeats until no swap helps."""
    improved = True
    while improved:
        improved = False
        for g1, g2 in itertools.combinations(range(len(groups)), 2):
            for a, b in itertools.product(range(len(groups[g1])), range(len(groups[g2]))):
                A, B = groups[g1], groups[g2]
                before = _within_score(A, S) + _within_score(B, S)
                new_a = A[:a] + [B[b]] + A[a + 1:]
                new_b = B[:b] + [A[a]] + B[b + 1:]
                if _within_score(new_a, S) + _within_score(new_b, S) > before + 1e-9:
                    groups[g1], groups[g2] = new_a, new_b
                    improved = True
    return groups


def form_groups(sim, size=2, mode="similar"):
    """Split students into groups of `size`.

    mode="similar"       -> maximise similarity inside groups (study buddies).
    mode="complementary" -> maximise *dissimilarity* (strong + weak profiles,
                            for peer mentoring).
    Returns a list of groups, each a list of row indices.
    """
    if size < 2:
        raise ValueError("Group size must be at least 2")
    if size > len(sim):
        raise ValueError("Group size cannot exceed the number of students")
    S = sim if mode == "similar" else -sim
    groups = _improve_by_swaps(_greedy_groups(S, size), S)
    return sorted((sorted(g) for g in groups), key=lambda g: -_mean_similarity(g, S))


def _mean_similarity(group, S):
    pairs = list(itertools.combinations(group, 2))
    return sum(S[a, b] for a, b in pairs) / len(pairs)


def average_group_similarity(groups, sim):
    return float(np.mean([_mean_similarity(g, sim) for g in groups]))


def random_baseline(sim, size, trials=2000, seed=0):
    """Average within-group similarity if groups were drawn at random.

    Used to show the matching does much better than chance.
    """
    rng = np.random.default_rng(seed)
    n = len(sim)
    scores = []
    for _ in range(trials):
        order = rng.permutation(n)
        groups = [order[k:k + size] for k in range(0, n - n % size, size)]
        scores.append(average_group_similarity(groups, sim))
    return float(np.mean(scores))


# ---- Tables for the report -------------------------------------------------

def group_table(groups, sim, df, Z):
    """One row per group with members, similarity and a suggested focus subject."""
    names = df[NAME_COL].to_numpy()
    labels = [SUBJECT_LABELS[s] for s in SUBJECTS]
    rows = []
    for number, g in enumerate(groups, start=1):
        profile = Z[g].mean(axis=0)  # group's average z-score per subject
        rows.append({
            "Group": number,
            "Members": ", ".join(names[g]),
            "Cosine_Similarity": round(_mean_similarity(g, sim), 3),
            "Avg_Overall": round(float(df.iloc[g][OVERALL_COL].mean()), 1),
            "Shared_Strength": labels[int(np.argmax(profile))],
            "Focus_Subject": labels[int(np.argmin(profile))],
        })
    return pd.DataFrame(rows)


def top_matches(sim, df, k=3):
    """Each student's k most similar classmates."""
    names = df[NAME_COL].to_numpy()
    rows = []
    for i, name in enumerate(names):
        order = [j for j in np.argsort(-sim[i]) if j != i][:k]
        row = {"Student": name}
        for rank, j in enumerate(order, start=1):
            row[f"Match_{rank}"] = f"{names[j]} ({sim[i, j]:.2f})"
        rows.append(row)
    return pd.DataFrame(rows)


def mentor_pairs(sim, df, Z):
    """Pair opposite profiles; the student ahead in a subject mentors the other in it.

    In an odd-sized class one group has three members; its strongest student
    mentors the other two.
    """
    names = df[NAME_COL].to_numpy()
    overall = df[OVERALL_COL].to_numpy()
    labels = [SUBJECT_LABELS[s] for s in SUBJECTS]
    rows = []
    for group in form_groups(sim, size=2, mode="complementary"):
        mentor = max(group, key=lambda i: overall[i])
        for mentee in group:
            if mentee == mentor:
                continue
            subject = int(np.argmax(Z[mentor] - Z[mentee]))
            rows.append({
                "Mentor": names[mentor],
                "Mentee": names[mentee],
                "Cosine_Similarity": round(float(sim[mentor, mentee]), 3),
                "Teach_Subject": labels[subject],
                "Score_Gap": int(df.iloc[mentor][SUBJECTS[subject]] - df.iloc[mentee][SUBJECTS[subject]]),
            })
    return pd.DataFrame(rows)


def pca_2d(Z):
    """Project the standardised 3-D grade vectors onto their 2 main directions.

    Uses the singular value decomposition Z = U S V^T: the rows of V^T are the
    principal directions, and S^2 tells how much variance each one carries.
    """
    centered = Z - Z.mean(axis=0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    explained = s ** 2 / np.sum(s ** 2)
    # Fix the sign so the first axis increases with overall performance
    if vt[0].sum() < 0:
        coords[:, 0] *= -1
        vt[0] *= -1
    return coords, explained[:2], vt[:2]
