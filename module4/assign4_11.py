#!/usr/bin/env python
"""
The observations on number of hours of daily TV watching for the 10 subjects in the 2018 GSS
who identified themselves as Islamic were 0, 0, 1, 1, 1, 2, 2, 3, 3, 4.
(a)  Construct and interpret a 95% confidence interval for the population mean.
(b)  Suppose the observation of 4 was incorrectly recorded as 24. What would you obtain for
the 95% confidence interval? What does this suggest about potential effects of outliers on
confidence intervals for means?
------------------------------------------------------------
STEPS:
    1) Assumptions:
        - independent/random sample
        - Apporximately Normal distribution w/ small n

    2) Sample Stats
    
    3) t-critical value

    4) 95% CI

    5) Interpret
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from textwrap import dedent

def ci_construction(y, cl=0.95):
    """
    Constructs a Confidence interval for the given sample with quantitative (discrete) values in order to infer the unknown population mean.
    ------------------------------------------------
    INPUT:
        y: (list, np.array, pd.DataFrame, pd.Series) Sample of values (not a large sample).
        cl: (float) Confindence Level (default 0.95)

    OUTPUT:
        tuple of values: (sample mean, sample standard deviation, standard error, degrees of freedom, t-critical value, margine of error, confidence interval)
    """
    if isinstance(y, pd.DataFrame):
        y = y.loc[:, 0]

    if isinstance(y, pd.Series):
        y = y.to_numpy()

    # Sample size
    n = len(y)

    # Confidence Level check
    if cl == 0.95:
        # t quantile having probability a in the right-tail
        alpha = 0.025

    else:
        alpha = (1 - cl) / 2

    # Sample Statistics
    mew_hat = sum(y) / n # sample meann
    std = np.std(y, ddof=1) # Sample standard deviatition
    se = std / np.sqrt(n) # Standard error

    # t-critical value
    df = n - 1 # Degrees of freedom
    t_crit = t.ppf(1 - alpha, df=df)

    # Margin of error
    me = t_crit * se

    # Confidence Interval for confidence level 95%
    ci = (mew_hat - me, mew_hat + me)

    return (mew_hat, std, se, df, t_crit, me, ci)

if __name__ == "__main__":
    # a)
    y = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    n = len(y) # Sample size
    mew_hat, std, se, df, t_crit, me, ci = ci_construction(y)

    # b)
    y_24 = [0, 0, 1, 1, 1, 2, 2, 3, 3, 24]
    mew_hat_24, std_24, se_24, df_24, t_crit_24, me_24, ci_24 = ci_construction(y_24)


    print(f"Fore the sample: {y}")
    print(f"Sample Mean: {mew_hat:.3f}")
    print(f"Sampe Standard Deviation: {std:.3f}")
    print(f"Standard Error: {se:.3f}")
    print(f"Degrees of Freedom: df = {n} - 1 = {df}")
    print(f"t-critical value for a 95% CI: {t_crit:.3f}")
    print(f"Margin of Error: {me:.3f}")
    print(f"Confidence interval: ({ci[0]:.3f}, {ci[1]:.3f})")

    # Interpretation
    print(dedent(f"""a)
        Assumptions: Normal distribution of the population.
        We can be 95% confident that the true population for the daily hours spent watching tv is lies between {ci[0]:.3f} and {ci[1]:.3f} hours per day.
        Confidence interval made by using a t-distribution since the population std dev is uknown, with a small sample size of {n}.  the interval shows the sample mean {mew_hat:.3f} and sampling variabislity of the mean as the standard error {se:.3f}.

        Interpreted probabilistically, if we were to repeatedly take many rand samples of size 10 from this population mean and construct a 95% confidence interval in the same way each time, approximately 95% of those intervals would contain the true population mean.
        """))

    # b)
    print(f"Fore the sample: {y_24}")
    print(f"Sample Mean: {mew_hat_24:.3f}")
    print(f"Sampe Standard Deviation: {std_24:.3f}")
    print(f"Standard Error: {se_24:.3f}")
    print(f"Degrees of Freedom: df = {n} - 1 = {df_24}")
    print(f"t-critical value for a 95% CI: {t_crit_24:.3f}")
    print(f"Margin of Error: {me_24:.3f}")
    print(f"Confidence interval: ({ci_24[0]:.3f}, {ci_24[1]:.3f})")

    # Interprtation
    print(dedent(f"""
        b)
        Using data where the value 24 was incorrectly recorded, we get a 95% confidence
        interval for the population mean for the daily number of hours of tv
        watching as ({ci_24[0]:.3f}, {ci_24[1]:.3f}),
        which is a lot wider than in part a) since the outlier throws off the
        sample mean {mew_hat_24:.3f} and inflates the sample standard
        error {std_24:.3f}, which increases the standard error
        {se_24:.3f} along with the margin of error
        {me_24:.3f}
        """))


