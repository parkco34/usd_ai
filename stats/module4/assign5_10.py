#!/usr/bin/env python
"""
A study of sheep mentioned in Exercise 1.27 analyzed whether the sheep survived for a year
from the original observation time (1 = yes, 0 = no) as a function of their weight (*kg*) at the
original observation. Stating any assumptions including the conceptual population of interest,
use a *t* test with the data in the Sheep data file at the text website to compare mean weights
of the sheep that survived and did not survive. Interpret the *P*-value.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from textwrap import dedent

df = pd.read_csv("sheep.dat", sep="\s+")

# Splitting survival status
live = df[df["survival"] == 1]["weight"]
died = df[df["survival"] == 0]["weight"]

# Sample sizes
n_surv = len(live)
n_died = len(died)

# Sample Statistics
mean_surv = np.mean(live)
mean_died = np.mean(died)
std_surv = np.std(live, ddof=1)
std_died = np.std(died, ddof=1)

# Point estimate of difference
diff_hat = mean_surv - mean_died

# Pooled standard deviation (assumes equal variance)
s_pooled = np.sqrt(((n_surv - 1) * std_surv**2 + (n_died - 1) * std_died**2) /
                   (n_surv + n_died - 2))

# Standard error of the difference
se_diff = s_pooled * np.sqrt(1/n_surv + 1/n_died)

# Degrees of freedom
dof = n_surv + n_died - 2

# t-statistic
t_stat = diff_hat / se_diff

# P-value (two-tailed)
p_value = 2 * (1 - t.cdf(abs(t_stat), df=dof))

# Verify with scipy
t_check, p_check = stats.ttest_ind(live, died, equal_var=True)

print(f"\nSurvived group (n = {n_surv}):")
print(f"  Sample mean: {mean_surv:.3f} kg")
print(f"  Sample SD: {std_surv:.3f} kg")

print(f"\nDied group (n = {n_died}):")
print(f"  Sample mean: {mean_died:.3f} kg")
print(f"  Sample SD: {std_died:.3f} kg")

print(f"\nPoint estimate (ȳ_surv - ȳ_died): {diff_hat:.3f} kg")
print(f"Pooled standard deviation: {s_pooled:.3f} kg")
print(f"Standard error of difference: {se_diff:.3f} kg")
print(f"Degrees of freedom: df = {n_surv} + {n_died} - 2 = {dof}")
print(f"t-statistic: t = {diff_hat:.3f} / {se_diff:.3f} = {t_stat:.3f}")
print(f"P-value (two-tailed): {p_value:.4f}")

print(f"\n[scipy verification: t = {t_check:.3f}, p = {p_check:.4f}]")

# Boxplot comparison
plt.figure(figsize=(8, 5))
plt.boxplot([died, live], tick_labels=['Did Not Survive (0)', 'Survived (1)'])
plt.ylabel("Weight (kg)")
plt.xlabel("Survival Status")
plt.title("Sheep Weight by Survival Status")
plt.show()

# Interpretation
print(dedent(f"""
The p-value {p_value:.4f} is the probability of observing a difference in sample means of
{abs(diff_hat):.3f} kg or more extreme, given that the null hypothesis is true.

Since p = {p_value:.4f} {"<" if p_value < alpha else "≥"} alpha = {alpha},
we {"reject" if p_value < alpha else "fail to reject"} the null hypothesis.

{"There is statistically significant evidence that mean weight differs between sheep that survived and those that did not."
 if p_value < alpha else
 "There is insufficient evidence that mean weight differs by survival status."}

{"Sheep that survived had a higher mean weight "
 f"({mean_surv:.2f} kg) compared to those that died "
 f"({mean_died:.2f} kg), a difference of {diff_hat:.2f} kg."
 if diff_hat > 0 else ""}
"""))

print(dedent(f"""
\nASSUMPTIONS
1) Independence
2) Normality
3) Random sampling
4) Equal variances
             """))
