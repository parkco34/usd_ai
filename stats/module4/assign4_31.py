#!/usr/bin/env python
"""
The `Houses` data file at the book’s website lists, for 100 home sales in Gainesville, Florida,
several variables, including the selling price in thousands of dollars and whether the house
is new (1 = yes, 0 = no). Prepare a short report in which, stating all assumptions including
the relative importance of each, you conduct descriptive and inferential statistical analyses to
compare the selling prices for new and older homes.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ci_construct_two_stats(y1, y0, ci=0.95):
    """
    Constructs a Confidence interval to compare two population means using two independent samples, and assumes equal variance.
    ------------------------------------------------
    INPUT:
        y1: (list, np.array, pd.DataFrame, pd.Series) Sample of values for group 1 (e.g., new homes).
        y0: (list, np.array, pd.DataFrame, pd.Series) Sample of values for group 0 (e.g., older homes).
        cl: (float) Confindence Level (default 0.95)

    OUTPUT:
        tuple of values:
            (mew_hat_1, mew_hat_0, diff_hat,
             std_1, std_0, se_diff,
             df, t_crit, me, ci)
    """
    if isinstance(y1, pd.DataFrame) and isinstance(y0, pd.DataFrame):
        y1, y0 = y1.loc[:, 0], y0.loc[:, 0]

    if isinstance(y1, pd.Series) and isinstance(y0, pd.DataFrame):
        y1, y0 = y1.loc[:, 0].to_numpy(), y0.loc[:, 0].to_numpy()

    # Sample means
    n1, n0 = len(y1), len(y0)

    # Confidence level check
    if cl == 0.95:
        alpha = 0.025

    else:
        alpha = (1 - cl) / 2 # ??

    # Sample stats
    mew_hat_1 = sum(y1) / n1
    mew_hat_0 = sum(y0) / n0
    diff_hat = mew_hat_1 - mew_hat_0

    std_1 = np.std(y1, ddof=1)
    std_0 = np.std(y0, ddof=1)

    # Standard error and degrees of freedom
    s_pooled = np.sqrt(((n1 - 1)*(std_1**2) + (n0 - 1)*(std_0**2)) / (n1 + n0 - 2))
    se_diff = s_pooled * np.sqrt((1/n1) + (1/n0))
    df = n1 + n0 - 2

    # t-critical value
    t_crit = t.ppf(1 - alpha, df=df)

    # Margin of error + CI for (mu1 - mu0)
    me = t_crit * se_diff
    ci = (diff_hat - me, diff_hat + me)

    return (mew_hat_1, mew_hat_0, diff_hat, std_1, std_0, se_diff, df, t_crit, me, ci)
    

if __name__ == "__main__":
    df = pd.read_csv("./house.dat", sep="\s+")    

    breakpoint()




