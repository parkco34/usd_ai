#!/usr/bin/env python
"""
For the rain simulation example in Section 2.1.1, but with probability of rain 0.30 on any
given day, simulate the outcome (a) on the next day, (b) the next 10 days. (c) Simulate the
proportion of days of rain for the next (i) 100 days, (ii) 10,000 days, (iii) 1,000,000 days. Use
the simulation to explain the long-run relative frequency definition of probability.
"""
import numpy as np
import matplotlib.pyplot as plt
import debug_hook # Automatically go into full debug env

"""
R -> PYTHON
-----------
rbinom = (# of simulations, trials, probability)
binomial(trials, probability, # of simulations)
"""

# Generator for random numbers
rng = np.random.default_rng()

# a) Next 100 days



# b) Next 10,000 days



# c) Next 1,000,000 days


