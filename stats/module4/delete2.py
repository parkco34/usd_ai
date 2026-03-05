#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# ============================================================
# Problem 5.23 — Sentiment Toward AI and Gender (Chi-Square Test)
# ============================================================

# -----------------------------
# 1) Generate the mock dataset
# -----------------------------
np.random.seed(2024)

n = 300

sentiments = ["Positive", "Neutral", "Negative"]
genders = ["Male", "Female", "Other"]
usage_levels = ["Daily", "Weekly", "Rarely", "Never"]

df_ai = pd.DataFrame(
    {
        "sentiment": np.random.choice(sentiments, size=n, p=[0.44, 0.33, 0.23]),
        "gender": np.random.choice(genders, size=n, p=[0.49, 0.48, 0.03]),
        "age": np.random.randint(18, 75, size=n),
        "ai_usage_frequency": np.random.choice(usage_levels, size=n),
        "trust_in_ai": np.random.randint(1, 6, size=n),
    }
)

print("Head of generated dataset:")
print(df_ai.head(), "\n")


# ---------------------------------------------------
# (a) Contingency table: sentiment x gender (counts)
# ---------------------------------------------------
# Rows = sentiment categories
# Columns = gender categories
# Cell O_ij = observed count in (sentiment=i, gender=j)

ct = pd.crosstab(df_ai["sentiment"], df_ai["gender"])

print("=" * 60)
print("(a) Contingency Table: Sentiment by Gender (Observed Counts)")
print("=" * 60)
print(ct, "\n")

# Helpful: row/column proportions (optional but good for interpretation)
row_props = pd.crosstab(df_ai["sentiment"], df_ai["gender"], normalize="index")
col_props = pd.crosstab(df_ai["sentiment"], df_ai["gender"], normalize="columns")

print("Row proportions P(gender | sentiment):")
print(row_props.round(3), "\n")

print("Column proportions P(sentiment | gender):")
print(col_props.round(3), "\n")


# ----------------------------------------------------------
# (b) Chi-squared test of independence (no continuity corr.)
# ----------------------------------------------------------
# H0: sentiment ⟂ gender  (independent)
# HA: sentiment depends on gender (association)

# chi2_contingency returns:
#   chi2 = Σ Σ (O_ij - E_ij)^2 / E_ij
#   p    = P(ChiSq_df >= chi2_obs) under H0
#   dof  = (r-1)(c-1)
#   expected = E_ij = (row_total_i * col_total_j) / n

chi2, p_value, dof, expected = chi2_contingency(ct, correction=False)

expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)

print("=" * 60)
print("(b) Chi-Squared Test of Independence")
print("=" * 60)
print(f"chi2 statistic = {chi2:.4f}")
print(f"degrees of freedom = {dof}")
print(f"p-value = {p_value:.6f}\n")

print("Expected counts under H0 (independence):")
print(expected_df.round(2), "\n")

# Rule-of-thumb check for validity (expected counts typically >= 5)
min_expected = expected_df.to_numpy().min()
print(f"Min expected cell count = {min_expected:.2f}")
if min_expected < 5:
    print("WARNING: Some expected counts are < 5. Chi-square approximation may be weak.\n")
else:
    print("Expected counts condition looks fine (rule of thumb: all E_ij >= ~5).\n")


# -----------------------------
# Effect size: Cramér's V
# -----------------------------
# Chi-square significance tells you whether association exists (statistical).
# Cramér's V estimates strength of association (practical size), on [0, 1].
#
# V = sqrt( chi2 / ( n * (min(r-1, c-1)) ) )

r, c = ct.shape
k = min(r - 1, c - 1)
cramers_v = np.sqrt(chi2 / (n * k))

print("=" * 60)
print("Effect Size")
print("=" * 60)
print(f"Cramér's V = {cramers_v:.4f}\n")


# ----------------------------------------------------------
# (Optional) Standardized residuals (where differences occur)
# ----------------------------------------------------------
# Residual for each cell:
#   (O_ij - E_ij) / sqrt(E_ij)
# Large magnitude cells contribute most to chi-square.

std_resid = (ct - expected_df) / np.sqrt(expected_df)

print("=" * 60)
print("Standardized Residuals (diagnostic: which cells drive chi-square?)")
print("=" * 60)
print(std_resid.round(2), "\n")


# ----------------------------------------------------------
# (c) Interpretation in context
# ----------------------------------------------------------
alpha = 0.05

print("=" * 60)
print("(c) Interpretation")
print("=" * 60)

if p_value <= alpha:
    decision = "REJECT H0"
    conclusion = (
        "There is statistically significant evidence that sentiment toward AI "
        "is associated with gender (i.e., the distribution of sentiment differs across genders)."
    )
else:
    decision = "FAIL TO REJECT H0"
    conclusion = (
        "There is not sufficient statistical evidence to conclude that sentiment toward AI "
        "depends on gender. Observed differences are consistent with sampling variability."
    )

print(f"Significance level alpha = {alpha}")
print(f"Decision: {decision}")
print(conclusion, "\n")

print("Notes for interpretation:")
print(
    f"- The chi-square test answers: 'Is there evidence of *any* association?' (p-value)\n"
    f"- Cramér's V answers: 'How strong is the association?' (effect size)\n"
    f"- Standardized residuals show which (sentiment, gender) cells deviate most from independence.\n"
    f"  As a rule of thumb, |residual| ≳ 2 is often considered a noticeable deviation.\n"
)


