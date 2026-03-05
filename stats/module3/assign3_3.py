#!/usr/bin/env python
"""
The 49 students in a class at the University of Florida made blinded evaluations of pairs of
cola drinks. For the 49 comparisons of Coke and Pepsi, Coke was preferred 29 times. In the
population that this sample represents, is this strong evidence that a majority prefers Coke?
Use a simulation of a sampling distribution to answer.
"""
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

# Random seed
rng = np.random.default_rng(73)

# Sample space
n = 49

# Observed proportion for preferring coke
pi_hat_observed = 29 / n

# Assume: Binomial dist. and 50/50 for preferring coke
pi = 1/2

# Theoretical standard error
se = np.sqrt(pi * (1 - pi) / n)

# simulation
Y = rng.binomial(n, pi, size=int(1e6))

# Sample proportion
pi_hat = Y / n

# Sample proportion mean (simulated)
pi_hat_mew = pi_hat.mean()

# Simulated standard error
se_simulated = pi_hat.std(ddof=1)

# Simulation-based tail probabilities
p_one = np.mean(pi_hat >= pi_hat_observed)
p_two = np.mean(np.abs(pi_hat - pi) >= abs(pi_hat_observed - pi))

# Z-score method standard error
z = (pi_hat_observed - pi) / se

# Output values
print(f"""\n
For a sample size of {n}, the presumed population proportion {pi},
observed sample proportion {pi_hat_observed:.3f}, and 
simulated sample proportion {pi_hat_mew:.3f}, 
we get a (theoretical) standard error of {se:.3f},
and a simulated standard error of {se_simulated:.3f}.
      """)

# Interpretation
print(f"""
We observed {pi_hat_observed:.3f} preferring COKE.
To assess whether this is strong evidence for the majority of the population
preferring coke, we assumed the 'no-majority' (50/50) scenario, where pi = {pi} and
simulated many random samples of size {n}.
This produced a sampling distribution for pi_hat under pi = {pi}. 
The simulated standard error of pi_hat was close to the theoretical value 
SE = {se:.3f}, confirming the expected variability from sample to sampple.
      """)
print(f"""
How unusual the observed result is under pi={pi}, I calculated the proportion
of simulations with pi_hat >= {pi_hat_observed:.3f}.
The tail proportion was {p_one:.3f}, meaning that if the population was
actually 50/50, results at least this favorable to coke would occur about
{p_one*100:.2f}% of the time via random sampling.
than half of population prefers coke.
This happens about {p_one*100:.2f}% of the time under pi = {pi}, which is not
rare.\n
Also, the Z-score metho shows that it's about {z:.3f} standard erros above
{pi}.
      """)


# Plotting histogram
bins = ceil(1 + np.log2(len(Y)))
plt.hist(pi_hat, bins=bins, density=True, edgecolor="black")
plt.axvline(pi_hat_observed, linestyle="--", label=f"Observed pi_hat = {pi_hat_observed:.3f}")
plt.axvline(pi, linestyle=":", label="Assumed pi = 0.50")
plt.xlabel("Simulated sample proportion (pi_hat)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()




