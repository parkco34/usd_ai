#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy import stats
from textwrap import dedent

# ============================================================
# Problem 5.23: Sentiment Toward AI and Gender
# ============================================================

# Dataset creation (given)
np.random.seed(2024)
n = 300

sentiments = ["Positive", "Neutral", "Negative"]
genders = ["Male", "Female", "Other"]
usage_levels = ["Daily", "Weekly", "Rarely", "Never"]

df_ai = pd.DataFrame({
    "sentiment": np.random.choice(sentiments, size=n, p=[0.44, 0.33, 0.23]),
    "gender": np.random.choice(genders, size=n, p=[0.49, 0.48, 0.03]),
    "age": np.random.randint(18, 75, size=n),
    "ai_usage_frequency": np.random.choice(usage_levels, size=n),
    "trust_in_ai": np.random.randint(1, 6, size=n),
})

# ============================================================
# Part (a): Contingency Table
# ============================================================
print("="*60)
print("Part (a): Contingency Table")
print("="*60)

# Cross-tabulation of sentiment by gender
contingency = pd.crosstab(df_ai["sentiment"], df_ai["gender"], margins=True, margins_name="Total")
print("\nContingency Table (Observed Counts):")
print(contingency)

# Row percentages (sentiment distribution within each gender)
row_pct = pd.crosstab(df_ai["sentiment"], df_ai["gender"], normalize="columns") * 100
print("\nColumn Percentages (% sentiment within each gender):")
print(row_pct.round(1))

# ============================================================
# Part (b): Chi-Squared Test of Independence
# ============================================================
print("\n" + "="*60)
print("Part (b): Chi-Squared Test of Independence")
print("="*60)

print("\nHypotheses:")
print("  H₀: Sentiment toward AI is INDEPENDENT of gender")
print("  Hₐ: Sentiment toward AI DEPENDS on gender")

# Observed counts (without margins)
observed = pd.crosstab(df_ai["sentiment"], df_ai["gender"])

# Get dimensions
r, c = observed.shape
print(f"\nTable dimensions: {r} rows × {c} columns")

# Perform chi-squared test
chi2, p_value, dof, expected = stats.chi2_contingency(observed)

# Convert expected to DataFrame
expected_df = pd.DataFrame(expected, index=observed.index, columns=observed.columns)

print("\nObserved Counts (O):")
print(observed)

print("\nExpected Counts (E) under H₀:")
print(expected_df.round(2))

# Show calculation for one cell as example
row_totals = observed.sum(axis=1)
col_totals = observed.sum(axis=0)
print(dedent(f"""
Example calculation for E_Positive,Male:
  E = (Row total × Column total) / n
  E = ({row_totals['Positive']} × {col_totals['Male']}) / {n} = {(row_totals['Positive'] * col_totals['Male']) / n:.2f}
"""))

# Residuals
residuals = observed - expected_df
print("Residuals (O - E):")
print(residuals.round(2))

# Chi-squared contributions from each cell
contributions = (observed - expected_df)**2 / expected_df
print("\nChi-squared contributions (O - E)² / E:")
print(contributions.round(3))

print(f"\nχ² statistic = Σ[(O - E)² / E] = {chi2:.4f}")
print(f"Degrees of freedom: df = ({r} - 1)({c} - 1) = {dof}")
print(f"P-value: P(χ²_{dof} ≥ {chi2:.4f}) = {p_value:.4f}")

# ============================================================
# Part (c): Interpretation
# ============================================================
print("\n" + "="*60)
print("Part (c): Interpretation")
print("="*60)

alpha = 0.05

print(dedent(f"""
The chi-squared statistic χ² = {chi2:.4f} measures the total discrepancy
between the observed counts and what we would expect if sentiment were
independent of gender.

The p-value of {p_value:.4f} represents the probability of observing a
χ² value of {chi2:.4f} or larger if the null hypothesis (independence)
were true.

Since p = {p_value:.4f} {"<" if p_value < alpha else "≥"} α = {alpha}:
  → We {"REJECT" if p_value < alpha else "FAIL TO REJECT"} H₀
"""))

if p_value < alpha:
    print(dedent("""
  → There IS statistically significant evidence that sentiment toward AI
    depends on gender.
  → Different genders have different distributions of sentiment toward AI.
    """))
else:
    print(dedent("""
  → There is NO statistically significant evidence that sentiment toward
    AI depends on gender.
  → The observed differences in sentiment across genders could plausibly
    be due to random sampling variation alone.
  → Based on this data, gender does not appear to be a significant factor
    in determining attitudes toward AI.
    """))

# Check assumptions
min_expected = expected_df.min().min()
print(dedent(f"""
ASSUMPTIONS CHECK:
{"-"*50}
1. INDEPENDENCE: Each respondent is counted once and responses are independent.
   IMPORTANCE: Fundamental for valid inference.

2. RANDOM SAMPLING: The sample is representative of the population of interest.
   IMPORTANCE: Required for generalization.

3. EXPECTED COUNTS ≥ 5: All expected cell counts should be at least 5.
   Minimum expected count: {min_expected:.2f}
   {"✓ Assumption satisfied." if min_expected >= 5 else "⚠ Some cells < 5. Consider combining categories or using Fisher's exact test."}
   IMPORTANCE: Required for χ² approximation to be valid.

NOTE: This is simulated data where sentiment and gender were generated
INDEPENDENTLY by design. Therefore, we would expect to find NO significant
association, which is consistent with our result.
"""))

