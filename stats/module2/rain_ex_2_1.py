#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import debug_hook # Automatically go into full debug env

"""
R -> PYTHON
-----------
rbinom = (# of simulations, trials, probability)
binomial(trials, probability, # of simulations)
"""

# Random number generator
rng = np.random.default_rng()

# Random digit between 0 and 9 (inclusive)
# Weather forcasting, where a 0 or 1 = Rain, otherwise no rain for the NEXT DAY
rand_dig = rng.integers(low=0, high=10)

# Simulating for a week
#days = rng.integers(low=0, high=10, size=7)
#
## 1 simluation for 7 'unfair' coin flips, where 20% it rains
#sample_20_1 = rng.binomial(n=7, p=0.20, size=1) #
## 7 simulations for one coin flip
#sample_20_2 = rng.binomial(n=1, p=0.20, size=7)
#
## Check output
#print(f"Single week simulation (count of rain days): {sample_20_1}")
#print(f"Seven individual day outcomes: {sample_20_2}")
#print(f"Shape of sample_20_1: {sample_20_1.shape}")
#print(f"Shape of sample_20_2: {sample_20_2.shape}")

# Long-run relative frequency
"""
As n increases, the proportion converges toward the true probability of 0.20. This is the Law of Large Numbers in action
"""
# Claude's help with the plot
# More points, linear scale, cleaner look
trial_sequence = np.arange(1, 1001)  # 1 to 1000
proportions = [rng.binomial(n, 0.20) / n for n in trial_sequence]

plt.figure(figsize=(10, 6))
plt.plot(trial_sequence, proportions, 'b.', markersize=3)
plt.axhline(y=0.20, color="r", linestyle="--")
plt.xlabel("n", fontsize=12)
plt.ylabel("proportion", fontsize=12)
plt.ylim(0, 0.5)
plt.xlim(0, 1000)
plt.tight_layout()
plt.show()


#breakpoint()

