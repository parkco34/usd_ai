#!/usr/bin/env python
"""
For the `Students` data file at the text website, analyze political ideology.
(b) Construct the 95% confidence interval for $\mu$. Explain how results relate to those of the
test in (a).
"""
import pandas as pd
from textwrap import dedent
import numpy as np
from scipy.stats import t

def ci_construction(y, cl=0.95):
    """
    Constructs a Confidence interval for the given sample with quantitative
    (discrete) values in order to infer the unknown population mean.
    ------------------------------------------------
    INPUT:
        y: (list, np.array, pd.DataFrame, pd.Series) Sample of values (not a large sample).
        cl: (float) Confidence Level (default 0.95)
    OUTPUT:
        tuple of values: (sample mean, sample standard deviation, standard error,
                          degrees of freedom, t-critical value, margin of error,
                          confidence interval)
    """
    if isinstance(y, pd.DataFrame):
        y = y.loc[:, 0]
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    # Sample size
    n = len(y)
    # Confidence Level check
    if cl == 0.95:
        # t quantile having probability alpha in the right-tail
        alpha = 0.025
    else:
        alpha = (1 - cl) / 2
    # Sample Statistics
    mew_hat = sum(y) / n  # sample mean
    std = np.std(y, ddof=1)  # Sample standard deviation
    se = std / np.sqrt(n)  # Standard error
    # t-critical value
    df = n - 1  # Degrees of freedom
    t_crit = t.ppf(1 - alpha, df=df)
    # Margin of error
    me = t_crit * se
    # Confidence Interval for confidence level 95%
    ci = (mew_hat - me, mew_hat + me)
    return (mew_hat, std, se, df, t_crit, me, ci)

df = pd.read_csv("./students.dat", sep="\s+")

# a) 
# Isolate ideology variable
ideology = df["ideol"]

# Stats
x_bar, s, se, df, t_critical, margin_of_error, ci = ci_construction(ideology, cl=0.95)

# Output
print(f"Sample size n = {len(ideology)}")
print(f"Sample mean x̄ = {x_bar:.3f}")
print(f"Sample standard deviation s = {s:.3f}")
print(f"Standard error SE = s/√n = {se:.3f}\n")

# b) CONFIDENCE INTERVAL (95%)
print("\nCONFIDENCE INTERVAL FOR POPULATION MEAN (POLITICAL IDEOLOGY)\n")

print(f"Degrees of freedom = {df}")
print(f"Critical t value (95%) = {t_critical:.3f}")
print(f"Margin of error = {margin_of_error:.3f}\n")

print(f"95% confidence interval for μ: ({ci[0]:.3f}, {ci[1]:.3f})\n")

# Interpretation]
print(dedent(f"""
We can be 95% confident the true population mean for political ideology lies between {ci[0]:.2f} and {ci[1]:.2f} on the 1-y scale.
             """))

mu_null = 4

if ci[0] <= mu_null <= ci[1]:
    print(dedent(f""" 
\nSince the null hypothesis of the population mean is {mu_null}, it's inside
                 the confidence interval and we fail to reject the nully
                 hypothesis at the significance level of 0.05.
The data doesn't provide enough evidence that the population mean ideology
                 differs from 'moderate.'
                 """))

else:
    print(dedent(f"""
Since the null hypothesis of the population mean is {mu_null}, it's not inside in the confidence interval so we reject the null
hypothesis at the significance level 0.05. 
The results indicate the population mean ideology differs from 'moderate' and is lower, on average, meaning more liberal.
                 """))
