#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy.stats import t

def ci_construction_two_means(y1, y0, cl=0.95, var_equal=False):
    """
    Constructs a Confidence interval to compare two population means using two independent samples.
    By default, uses Welch's method (does NOT assume equal variances).
    ------------------------------------------------
    INPUT:
        y1: (list, np.array, pd.DataFrame, pd.Series) Sample of values for group 1 (e.g., new homes).
        y0: (list, np.array, pd.DataFrame, pd.Series) Sample of values for group 0 (e.g., older homes).
        cl: (float) Confindence Level (default 0.95)
        var_equal: (bool) If True, uses pooled two-sample t interval (assumes equal variances).
                          If False, uses Welch two-sample t interval (recommended).

    OUTPUT:
        tuple of values:
            (mew_hat_1, mew_hat_0, diff_hat,
             std_1, std_0, se_diff,
             df, t_crit, me, ci)
    """
    # ----- type handling (same style as your ci_construction) -----
    if isinstance(y1, pd.DataFrame):
        y1 = y1.loc[:, 0]
    if isinstance(y0, pd.DataFrame):
        y0 = y0.loc[:, 0]

    if isinstance(y1, pd.Series):
        y1 = y1.to_numpy()
    if isinstance(y0, pd.Series):
        y0 = y0.to_numpy()

    # Sample sizes
    n1 = len(y1)
    n0 = len(y0)

    # Confidence Level check
    if cl == 0.95:
        alpha = 0.025
    else:
        alpha = (1 - cl) / 2

    # Sample Statistics
    mew_hat_1 = sum(y1) / n1
    mew_hat_0 = sum(y0) / n0
    diff_hat = mew_hat_1 - mew_hat_0

    std_1 = np.std(y1, ddof=1)
    std_0 = np.std(y0, ddof=1)

    # ----- Standard Error + df (Welch default, pooled optional) -----
    if var_equal:
        # Pooled (assumes equal population variances)
        s_pooled = np.sqrt(((n1 - 1)*(std_1**2) + (n0 - 1)*(std_0**2)) / (n1 + n0 - 2))
        se_diff = s_pooled * np.sqrt((1/n1) + (1/n0))
        df = n1 + n0 - 2
    else:
        # Welch (does not assume equal variances)
        se_diff = np.sqrt((std_1**2)/n1 + (std_0**2)/n0)
        df_num = ((std_1**2)/n1 + (std_0**2)/n0) ** 2
        df_den = (((std_1**2)/n1) ** 2) / (n1 - 1) + (((std_0**2)/n0) ** 2) / (n0 - 1)
        df = df_num / df_den

    # t-critical value
    t_crit = t.ppf(1 - alpha, df=df)

    # Margin of error + CI for (mu1 - mu0)
    me = t_crit * se_diff
    ci = (diff_hat - me, diff_hat + me)

    return (mew_hat_1, mew_hat_0, diff_hat, std_1, std_0, se_diff, df, t_crit, me, ci)


