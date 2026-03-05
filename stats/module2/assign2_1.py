#!/usr/bin/env python
"""
For the rain simulation example in Section 2.1.1, but with probability of rain 0.30 on any
given day, simulate the outcome (a) on the next day, (b) the next 10 days. (c) Simulate the
proportion of days of rain for the next (i) 100 days, (ii) 10,000 days, (iii) 1,000,000 days. Use
the simulation to explain the long-run relative frequency definition of probability.
"""
import pandas as pd
import numpy as np

# random seed for consistent results
np.random.seed(73)

def rainy_days(n, p, size=None):
    """
    Probability it rains in the next 'size' days.
    ---------------------------------------------
    INPUT:
        n: (int) NUmber of trials
        p: (float) Probability
        size: (int) Number of days to consider

    OUTPUT:
        prob: (np.ndarray) Array of values between 0 (doesn't rain) and 1 (rains)
    """ 
    # Probability
    prob = np.random.binomial(n, p, size)

    # If single day, output single number, otherwise output array
    if size > 10:
        # Proportion
        proportion = prob.mean()

        print(f"\nThe proportion of rainy days for the next {size} days is {proportion:.3f}") 
        return 

    elif size == 1:
        print(f"\nSimulation of possible {size} rainy days is {prob[0]}") 
        return prob[0]

    print(f"\nSimulation for the next {size} possible rainy days is {prob}")

    return prob

# Constants
N, P = 1, 0.30

# List of days to consider for n
DAYS = [1, 10, 100, int(1e4), int(1e6)]

# Iterate thru the values for binomial calculation
for days in DAYS:
    rainy_days(N, P, days)

# ======================== Interpretation ========================
print("FREQUENSIST VIEW:\nThe probability of an event is the value that the proportion of times the event occurs approaches as the number of trials becomes large.")





