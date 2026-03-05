#!/usr/bin/env python
"""
Using the Students data file, for the corresponding population, construct a 95% confidence interval **(a)** for the mean weekly number of hours spent watching TV; **(b)** to compare females and
males on the mean weekly number of hours spent watching TV. In each case, state assumptions,
including the practical importance of each, and interpret results.
"""
import numpy as np
from scipy.stats import t
import pandas as pd
from textwrap import dedent
from assign4_11 import ci_construction

df = pd.read_csv("./students.dat", sep="\s+")

# a) variables in quetion
tv_hrs = df["tv"]

# Calculate confidence interval w/ all relevant statistical values
results = ci_construction(tv_hrs)
mew_hat, s, se, dfree, tcrit, me, ci = results

print("95% Confidence Interval for Mean Weekly TV Hours")
print("=" * 20)
print(f"Sample mean (ȳ): {mew_hat:.2f} hours/week")
print(f"Sample standard deviation (s): {s:.2f}")
print(f"Standard error (SE = s/√n): {se:.2f}")
print(f"Degrees of freedom: {dfree}")
print(f"t-critical value (95%): {tcrit:.3f}")
print(f"Margin of error: {me:.2f}\n")

print(f"95% Confidence Interval:")
print(f"({ci[0]:.2f}, {ci[1]:.2f}) hours/week\n")

print("INTERPRETATION:")
print("-" * 20)
print(dedent(f"""
We can be 95% confident that the true population mean (weekly number of hours
wathcing tv) lies between {ci[0]:.2f} and {ci[1]:.2f}
hours.\nMeaning, if were to iteratively take random samples of
students and develop the confidence intervals, approximately 95%
of those would contain the true population mean.
             """))

print(dedent(f"""
ASSUMPTIONS AND THERE IMPORTANCE:
The t-distribution assumed independent observations from a random sample from the population, where if the data weren't randomly collected, the
sampling wouldn't reflect the population and the interval would
describe the uncertainty about the sample and not the population.
It also assumes that the population is approximately normal.

We would probably expect the distribution of the hours of tv being watched to
be right-skewed, since there's no negative time, but 
thanks to the Central Limit Theorem, even if the population distribution is
skewed, the sampling distribution of the mean becomes
approximately normal for moderate n.
             """))

# For part b) Comparing females and males
tv_gender = df[["tv", "gender"]]
male = tv_gender[tv_gender["gender"] == 0]
female = tv_gender[tv_gender["gender"] == 1]
male_results = ci_construction(male["tv"])
female_results = ci_construction(female["tv"])

# Sample sizes
n_male = len(male)
n_female = len(female)

# Sample Statistics
m_mew, m_s, m_se, m_df, m_tcrit, m_me, m_ci = male_results
f_mew, f_s, f_se, f_df, f_tcrit, f_me, f_ci = female_results

# Pooled sample mean
diff_hat = m_mew - f_mew

# Pooled estimate (sample standard deviation) and assumes equal variance !!
pool_est = np.sqrt(((n_male - 1) * (m_s)**2 + (n_female - 1) * (f_s)**2) /
                   ((n_male - 1) + (n_female - 1)))

# Standard error
pool_se = pool_est * np.sqrt((1/n_male) + (1/n_female))

# Degrees of Freedom
dof = n_male + n_female - 2

# t quantile having probability a in the right-tail
alpha = 0.025
# t-critical value, where 1-alpha = cumulative probability
t_crit = t.ppf(q=1-alpha, df=dof)

# Marginal Error
me = t_crit * pool_se
ci_diff = (diff_hat - me_pooled, diff_hat + me_pooled)

# Confidence Intervals
ci = (diff_hat - me, diff_hat + me)

# treat differences above ~2 hours/week as practically meaningful for habits.
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



# Output
print("\n================ SAMPLE STATISTICS =================\n")

print("Male group:")
print(f"  Sample mean (x̄_m):        {m_mew:.3f}")
print(f"  Sample SD (s_m):           {m_s:.3f}")
print(f"  Standard Error (SE_m):     {m_se:.3f}")
print(f"  Degrees of Freedom:        {m_df}")
print(f"  t-critical value:          {m_tcrit:.3f}")
print(f"  Margin of Error:           {m_me:.3f}")
print(f"  95% CI for μ_m:            ({m_ci[0]:.3f}, {m_ci[1]:.3f})\n")

print("Female group:")
print(f"  Sample mean (x̄_f):        {f_mew:.3f}")
print(f"  Sample SD (s_f):           {f_s:.3f}")
print(f"  Standard Error (SE_f):     {f_se:.3f}")
print(f"  Degrees of Freedom:        {f_df}")
print(f"  t-critical value:          {f_tcrit:.3f}")
print(f"  Margin of Error:           {f_me:.3f}")
print(f"  95% CI for μ_f:            ({f_ci[0]:.3f}, {f_ci[1]:.3f})\n")

print("============= POOLED (EQUAL-VARIANCE) WORK =============")
print(f"Point estimate (x̄_m - x̄_f):     {diff_hat:.3f}")
print(f"Pooled SD (s_p):                 {pool_est:.3f}")
print(f"Pooled SE:                       {pool_se:.3f}")
print(f"Degrees of freedom (df):         {dof}")
print(f"t-critical (95%):                {t_crit:.3f}")
print(f"Margin of error (ME):            {me_pooled:.3f}")
print(f"95% CI for (μ_m - μ_f):          ({ci_diff[0]:.3f}, {ci_diff[1]:.3f})\n")

print("\nINTERPRETATION:")
print(dedent(f"""
Estimating the difference in mean weekly TV watching hours between men and women,
the 95% confidence interval includes 0, indicating no statistically detectable
difference between the population means at the 95% confidence level.

The 95% confidence interval is ({ci[0]:.3f}, {ci[1]:.3f}). Since this interval
{"includes 0, we do not have evidence of a difference in the population means"
 if ci[0] <= 0 <= ci[1]
 else "does not include 0, we have evidence of a difference in the population means"}
at the 95% confidence level.

Practical Importance: 
The data is consistent with males watching from {ci[0]:.2f} to {ci[1]:.2f} hours/week more TV than females (negative means fewer).
If the difference is 
"""))


print(dedent(f"""
ASSUMPTIONS FOR THE TWO-SAMPLE (POOLED) t-INTERVAL AND THEIR IMPORTANCE:
1) Random sampling / representativeness within each gender group: Needed to generalize
   to the population of students in each group.
2) Independence: Within and between groups; needed for the pooled SE formula.
3) Equal population variances (σ_m^2 = σ_f^2): This is required for the pooled t interval.
   If badly violated, the pooled interval can be inaccurate (too wide or too narrow).
4) Population shape / outliers: Each group's TV-hours distribution may be right-skewed,
   but with moderate n the CLT supports approximate normality of the group means.
             """))







