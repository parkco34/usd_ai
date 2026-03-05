#!/usr/bin/env pythonS
"""
Refernce: 
    https://medium.com/@curryrowan/understanding-simpsons-paradox-d97c0a5e68be
Simpson's Paradox Verification Script (UC Berkeley 1973 admissions-style table)

GOAL
-----
Verify two things using the table you provided:

1) The *marginal* (overall) admission rate can favor men:
      P(admit | man)  >  P(admit | woman)

2) The *conditional* (within-department/major) admission rates can favor women
   in most (or even all) departments:
      P(admit | woman, dept=j)  >  P(admit | man, dept=j)   for many j

This is Simpson's Paradox:
- Aggregating across a confounder (department) reverses the direction of association.

WHY THIS HAPPENS (conceptual)
-----------------------------
Let departments be j = A,...,F.

Overall (marginal) admission rate for men is a weighted average:
    P(admit | man) = Σ_j P(admit | man, j) * P(j | man)

Overall (marginal) admission rate for women is:
    P(admit | woman) = Σ_j P(admit | woman, j) * P(j | woman)

Even if P(admit | woman, j) >= P(admit | man, j) within each department,
the weights P(j | woman) and P(j | man) can differ a lot.
If women apply more to low-admit departments (small acceptance),
their weighted average can be lower overall.

This script computes:
- admitted counts per dept (from %)
- overall rates
- within-dept comparisons
- "natural sciences vs social sciences" grouping to mirror the article's narrative
"""

import numpy as np
import pandas as pd
from textwrap import dedent

# ============================================================
# 1) Enter the data table as given
# ============================================================

df = pd.DataFrame(
    {
        "Dept": ["A", "B", "C", "D", "E", "F"],
        "Men_Applicants": [825, 560, 325, 417, 191, 373],
        "Men_Pct_Admit": [62, 63, 37, 33, 28, 6],
        "Women_Applicants": [108, 25, 593, 375, 393, 341],
        "Women_Pct_Admit": [82, 68, 34, 35, 24, 7],
    }
)

# ============================================================
# 2) Convert percentages to admitted counts (approx)
#    NOTE: The table gives rounded percentages, so counts
#    can be off by a few compared to the original raw data.
# ============================================================

df["Men_Admitted"] = np.round(df["Men_Applicants"] * df["Men_Pct_Admit"] / 100).astype(int)
df["Women_Admitted"] = np.round(df["Women_Applicants"] * df["Women_Pct_Admit"] / 100).astype(int)

df["Men_Rejected"] = df["Men_Applicants"] - df["Men_Admitted"]
df["Women_Rejected"] = df["Women_Applicants"] - df["Women_Admitted"]

# Helpful: compute admission rates from the reconstructed counts
df["Men_Rate_From_Counts"] = df["Men_Admitted"] / df["Men_Applicants"]
df["Women_Rate_From_Counts"] = df["Women_Admitted"] / df["Women_Applicants"]
df["Women_minus_Men_Rate"] = df["Women_Rate_From_Counts"] - df["Men_Rate_From_Counts"]

# ============================================================
# 3) Overall (marginal) admission rates
# ============================================================

men_total_apps = df["Men_Applicants"].sum()
women_total_apps = df["Women_Applicants"].sum()

men_total_adm = df["Men_Admitted"].sum()
women_total_adm = df["Women_Admitted"].sum()

men_overall_rate = men_total_adm / men_total_apps
women_overall_rate = women_total_adm / women_total_apps

# ============================================================
# 4) Within-department comparison (partial associations)
# ============================================================

df["Dept_Winner"] = np.where(
    df["Women_Rate_From_Counts"] > df["Men_Rate_From_Counts"],
    "Women higher",
    np.where(
        df["Women_Rate_From_Counts"] < df["Men_Rate_From_Counts"],
        "Men higher",
        "Tie"
    )
)

# ============================================================
# 5) Demonstrate the WEIGHTING mechanism explicitly
#    These are the weights in the marginal averages:
#       P(dept=j | gender)
# ============================================================

df["Weight_Men_P(dept|man)"] = df["Men_Applicants"] / men_total_apps
df["Weight_Women_P(dept|woman)"] = df["Women_Applicants"] / women_total_apps

df["Men_Weighted_Contribution"] = df["Men_Rate_From_Counts"] * df["Weight_Men_P(dept|man)"]
df["Women_Weighted_Contribution"] = df["Women_Rate_From_Counts"] * df["Weight_Women_P(dept|woman)"]

# ============================================================
# 6) Optional: "Natural sciences vs social sciences" grouping
#    The Medium article talks about two buckets.
#    The original Berkeley example is often summarized as:
#        A,B = higher-admit (less competitive) bucket
#        C,D,E,F = lower-admit (more competitive) bucket
#    This is a common pedagogical grouping.
# ============================================================

df["Bucket"] = np.where(df["Dept"].isin(["A", "B"]), "High-admit bucket (A,B)", "Low-admit bucket (C,D,E,F)")

bucket = (
    df.groupby("Bucket", as_index=False)
      .agg(
          Men_Applicants=("Men_Applicants", "sum"),
          Men_Admitted=("Men_Admitted", "sum"),
          Women_Applicants=("Women_Applicants", "sum"),
          Women_Admitted=("Women_Admitted", "sum"),
      )
)

bucket["Men_Rate"] = bucket["Men_Admitted"] / bucket["Men_Applicants"]
bucket["Women_Rate"] = bucket["Women_Admitted"] / bucket["Women_Applicants"]
bucket["Women_minus_Men_Rate"] = bucket["Women_Rate"] - bucket["Men_Rate"]

# ============================================================
# 7) Print results in a note-taking friendly way
# ============================================================

print("=" * 72)
print("UC BERKELEY ADMISSIONS (reconstructed from dept-level % table)")
print("=" * 72)

print("\n--- Department-level reconstructed table ---")
show_cols = [
    "Dept",
    "Men_Applicants", "Men_Pct_Admit", "Men_Admitted", "Men_Rate_From_Counts",
    "Women_Applicants", "Women_Pct_Admit", "Women_Admitted", "Women_Rate_From_Counts",
    "Women_minus_Men_Rate",
    "Dept_Winner",
]
print(df[show_cols].to_string(index=False))

print("\n--- Overall (marginal) totals ---")
print(dedent(f"""
    Men:   admitted = {men_total_adm} / applicants = {men_total_apps}
           overall admission rate = {men_overall_rate:.4f}  ({men_overall_rate*100:.2f}%)

    Women: admitted = {women_total_adm} / applicants = {women_total_apps}
           overall admission rate = {women_overall_rate:.4f}  ({women_overall_rate*100:.2f}%)

    Difference (Men - Women) = {men_overall_rate - women_overall_rate:.4f}
""").strip())

print("\n--- Within-department comparison summary ---")
counts = df["Dept_Winner"].value_counts()
print(counts.to_string())

print("\n--- Weighting explanation (how the reversal happens) ---")
print(dedent("""
Overall rate is a weighted average:
    P(admit | man)   = Σ_j P(admit | man, j)   * P(j | man)
    P(admit | woman) = Σ_j P(admit | woman, j) * P(j | woman)

Below are the weights P(dept=j | gender) and each dept's contribution.
""").strip())

weight_cols = [
    "Dept",
    "Men_Rate_From_Counts", "Weight_Men_P(dept|man)", "Men_Weighted_Contribution",
    "Women_Rate_From_Counts", "Weight_Women_P(dept|woman)", "Women_Weighted_Contribution",
]
print(df[weight_cols].to_string(index=False))

print(dedent(f"""
Check (sum of contributions):
    Σ men contributions   = {df["Men_Weighted_Contribution"].sum():.6f}  (should match {men_overall_rate:.6f})
    Σ women contributions = {df["Women_Weighted_Contribution"].sum():.6f}  (should match {women_overall_rate:.6f})
""").strip())

print("\n--- 2-bucket aggregation (mirrors the article narrative) ---")
print(bucket.to_string(index=False))

print("\n" + "=" * 72)
print("INTERPRETATION (What you should observe)")
print("=" * 72)
print(dedent(f"""
1) Marginal (overall):
   Men overall admission rate   = {men_overall_rate*100:.2f}%
   Women overall admission rate = {women_overall_rate*100:.2f}%

   With this table, the overall rate is typically higher for men, matching the
   common claim (about ~44% vs ~35% in the classic story). Because we reconstructed
   counts from rounded percentages, your exact totals may differ slightly.

2) Conditional (within departments):
   Look at 'Dept_Winner' per row.
   In this table, women have higher admit rates in A and B, men have slightly
   higher in C and E, and women slightly higher in D and F (depending on rounding).
   The key point is: the within-dept differences are small and mixed, while the
   overall difference is large in the opposite direction.

3) Why the paradox appears:
   Compare the weights:
       P(dept|woman) vs P(dept|man).
   Women apply much more heavily to the low-admit departments (C-F),
   so their weighted average gets pulled down even if they do as well or
   better within some departments.

This is Simpson's Paradox: aggregated association reverses after conditioning
on a confounder (department).
""").strip())

# ============================================================
# 8) Optional: simple visualization (stacked bar of application mix)
#    Uncomment if you want plots.
# ============================================================

import matplotlib.pyplot as plt

plt.figure()
plt.bar(df["Dept"], df["Weight_Men_P(dept|man)"])
plt.title("Application mix by department (Men)")
plt.ylabel("P(dept | man)")
plt.xlabel("Department")
plt.show()

plt.figure()
plt.bar(df["Dept"], df["Weight_Women_P(dept|woman)"])
plt.title("Application mix by department (Women)")
plt.ylabel("P(dept | woman)")
plt.xlabel("Department")
plt.show()

plt.figure()
plt.scatter(df["Men_Rate_From_Counts"], df["Women_Rate_From_Counts"])
for _, r in df.iterrows():
    plt.text(r["Men_Rate_From_Counts"], r["Women_Rate_From_Counts"], r["Dept"])
    plt.plot([0, 1], [0, 1])
    plt.title("Within-dept admit rates: Women vs Men (each point is a dept)")
    plt.xlabel("Men admit rate")
    plt.ylabel("Women admit rate")
    plt.show()
