#!/usr/bin/env python
"""
Conditional Probabilities Exercises
------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Instance of numpy's Generator class
rng = np.random.default_rng()

# Discrete/Continuous random number generators
# ============================Generating UNIFORM random generator ========================================
#  Bernoulli random variable X, for Uniform random variable, U
def bernoulli(p):
    U = rng.random()
    print(f"Uniform random variable: U = {U}")

    return (U < p)


# =========================== Coin toss Simulation  ========================================
"""
You can make the coin biased via: toss = (U < 0.6), where probability of heads
equalling 0.6
"""
n = 1000
U = rng.random(n)
toss = (U < 0.5)
avg = [sum(toss[:i]) / i for i in range(1, n+1)]

# plotting
plt.xlabel("Coin Toss NUmber")
plt.ylabel("Proportion of Heads")
plt.axis([0, n, 0, 1])
plt.plot(range(1, n+1), avg)
plt.show()



breakpoint()


