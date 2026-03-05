#!/usr/bin/env python
import numpy as np
from scipy.stats import t
import pandas as pd
from textwrap import dedent
from assign4_11 import ci_construction

# ===================== Load data =====================
df = pd.read_csv("./students.dat", sep=r"\s+")

# ===================== Part (a): One-sample t CI for mean =====================
tv_hrs = df["tv"].dropna()

results = ci_construction(tv_hrs)
mew_hat, s, se, dfree, tcrit, me, ci = results

print("95% Confidence Interval for Mean Weekly TV Hours")
print("=" * 55)
print(f"Sample mean (ȳ):                {mew_hat:.2f} hours/week")
print(f"Sample standard deviation (s):  {s:.2f}")
print(f"Standard error (SE = s/√n):     {se:.2f}")
print(f"Degrees of freedom (df):        {dfree}")
print(f"t-critical value (95%):         {tcrit:.3f}")
print(f"Margin of error (ME):           {me:.2f}")
print(f"95% CI for μ:                   ({ci[0]:.2f}, {ci[1]:.2f}) hours/week\n")

print("INTERPRETATION:")
print("-" * 55)
print(dedent(f"""
We are 95% confident that the true population mean weekly number of hours spent
watching TV lies between {ci[0]:.2f} and {ci[1]:.2f} hours.

In repeated random sampling, about 95% of intervals constructed by this same
method would contain the true population mean μ.
"""))

print(dedent("""
ASSUMPTIONS AND THEIR IMPORTANCE:
1) Random sampling / representativeness: This is the key assumption for generalizing
   from the sample to the population of all students. If the data are not representative,
   the CI may not describe the population mean.
2) Independence: Needed so the standard error formula s/√n is valid.
3) Population shape: The t-interval is exact under normality, but with moderate n the
   Central Limit Theorem implies the sampling distribution of ȳ is approximately normal
   even if TV hours are right-skewed.
"""))

# ===================== Part (b): Two-sample pooled t CI (male vs female) =====================
tv_gender = df[["tv", "gender"]].dropna()

# Assumes gender coding: 0 = male, 1 = female (as in your code)
male = tv_gender[tv_gender["gender"] == 0]
female = tv_gender[tv_gender["gender"] == 1]

male_results = ci_construction(male["tv"])
female_results = ci_construction(female["tv"])

# Sample sizes
n_male = len(male)
n_female = len(female)

# Group summaries (from your helper)
m_mew, m_s, m_se, m_df, m_tcrit, m_me, m_ci = male_results
f_mew, f_s, f_se, f_df, f_tcrit, f_me, f_ci = female_results

# Point estimate for difference (male - female)
diff_hat = m_mew - f_mew

# ====== Pooled SD and pooled SE (equal-variance model) ======
pool_est = np.sqrt(((n_male - 1) * (m_s ** 2) + (n_female - 1) * (f_s ** 2)) / (n_male + n_female - 2))
pool_se = pool_est * np.sqrt((1 / n_male) + (1 / n_female))

# Degrees of freedom
dof = n_male + n_female - 2

# t critical
alpha = 0.025
t_crit = t.ppf(q=1 - alpha, df=dof)

# Margin of error + CI
me_pooled = t_crit * pool_se
ci_diff = (diff_hat - me_pooled, diff_hat + me_pooled)

# ====== Equal-variance guideline check (Section 4.5.2 rule-of-thumb) ======
sd_ratio = max(m_s, f_s) / min(m_s, f_s)
passes_50pct_rule = sd_ratio < 1.50

# ====== Practical importance rule (commit to a judgment) ======
# Here: treat differences above ~2 hours/week as practically meaningful for habits.
practical_threshold = 2.0
if ci_diff[1] < -practical_threshold:
    practical_stmt = (f"The entire CI is below -{practical_threshold:.1f}, suggesting a practically meaningful "
                      f"difference (males watch at least ~{practical_threshold:.1f} fewer hours/week).")
elif ci_diff[0] > practical_threshold:
    practical_stmt = (f"The entire CI is above {practical_threshold:.1f}, suggesting a practically meaningful "
                      f"difference (males watch at least ~{practical_threshold:.1f} more hours/week).")
elif (ci_diff[0] <= -practical_threshold) or (ci_diff[1] >= practical_threshold):
    practical_stmt = (f"The CI includes differences at least as large as ±{practical_threshold:.1f} hours/week, "
                      f"so practically meaningful differences are plausible given the data.")
else:
    practical_stmt = (f"The entire CI lies within ±{practical_threshold:.1f} hours/week, so any difference in means "
                      f"appears practically small even if a tiny difference exists.")

# ===================== Output =====================
print("\n================ SAMPLE STATISTICS =================\n")

print("Male group:")
print(f"  Sample size (n_m):            {n_male}")
print(f"  Sample mean (x̄_m):            {m_mew:.3f}")
print(f"  Sample SD (s_m):               {m_s:.3f}")
print(f"  95% CI for μ_m:                ({m_ci[0]:.3f}, {m_ci[1]:.3f})\n")

print("Female group:")
print(f"  Sample size (n_f):            {n_female}")
print(f"  Sample mean (x̄_f):            {f_mew:.3f}")
print(f"  Sample SD (s_f):               {f_s:.3f}")
print(f"  95% CI for μ_f:                ({f_ci[0]:.3f}, {f_ci[1]:.3f})\n")

print("============= POOLED (EQUAL-VARIANCE) WORK =============")
print(f"Point estimate (x̄_m - x̄_f):     {diff_hat:.3f}")
print(f"Pooled SD (s_p):                 {pool_est:.3f}")
print(f"Pooled SE:                       {pool_se:.3f}")
print(f"Degrees of freedom (df):         {dof}")
print(f"t-critical (95%):                {t_crit:.3f}")
print(f"Margin of error (ME):            {me_pooled:.3f}")
print(f"95% CI for (μ_m - μ_f):          ({ci_diff[0]:.3f}, {ci_diff[1]:.3f})\n")

print("EQUAL-VARIANCE CHECK (rule of thumb):")
print(f"  SD ratio = max(s_m, s_f) / min(s_m, s_f) = {sd_ratio:.3f}")
print(f"  50% rule satisfied (< 1.50)?   {passes_50pct_rule}\n")

print("INTERPRETATION:")
print(dedent(f"""
Estimating the difference in mean weekly TV watching hours between men and women,
the 95% confidence interval is ({ci_diff[0]:.3f}, {ci_diff[1]:.3f}).

Since this interval {"includes 0, we do not have evidence of a difference in the population means"
if ci_diff[0] <= 0 <= ci_diff[1]
else "does not include 0, we have evidence of a difference in the population means"}
at the 95% confidence level.

Practical importance (using ±{practical_threshold:.1f} hours/week as a meaningful habit-size difference):
{practical_stmt}
"""))

print(dedent(f"""
ASSUMPTIONS FOR THE TWO-SAMPLE (POOLED) t-INTERVAL AND THEIR IMPORTANCE:
1) Random sampling / representativeness within each gender group: Needed to generalize
   to the population of students in each group.
2) Independence: Within and between groups; needed for the pooled SE formula.
3) Equal population variances (σ_m^2 = σ_f^2): Required for the pooled t interval.
   Here, the SD ratio is {sd_ratio:.3f}. Using the textbook guideline ("worry when one SD is at least 50% larger"),
   this {"does NOT raise concern" if passes_50pct_rule else "raises concern"} for the equal-variance assumption.
4) Population shape / outliers: TV hours may be right-skewed, but with moderate group sample sizes,
   the CLT supports approximate normality of the group means.
"""))
