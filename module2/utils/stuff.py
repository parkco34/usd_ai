#!/usr/bin/env python
import numpy as np

def bernoulli_trial(prob, rng=None):
    """
    Performs a Bernoulli trial.
    ------------------------
    INPUT:
        prob: (float) Probability of what ?

    OUTPUT:
        (1 for Success OR 0 for failure): (int)
    """
    if rng is None:
        rng = np.random.default_rng()

    return int(rng.random() < p)

def mean_std(x):
    """
    Returns sample mean and standard deviation
    ------------------------------------------
    INPUT:
        x: (np.array) Values

    OUTPUT:
        mean, std_dev: (float, float)
    """
    x = np.asarray(x)

    return x.mean(), x.std()

def normalize(x):
    """
    Z-Score normalization.
    ---------------------
    INPUT:
        x: (np.array)

    OUTPUT:
        zscore: (float)
    """
    x = np.asarray(x)

    return (x - x.mean() / x.std(ddof=1))

def bayesian():
    pass





