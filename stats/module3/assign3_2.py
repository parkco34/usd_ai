#!/usr/bin/env python
"""
REDO ?!
In an exit poll of 1648 voters in the 2020 Senatorial election in Arizona, 51.5% said they voted
for Mark Kelly and 48.5% said they voted for Martha McSally
a) Suppose that actually 50% of the population voted for Kelly. If this exit poll had the
properties of a simple random sample, find the standard error of the sample proportion
voting for him.
b) Under the 50% presumption, are the results of the exit poll surprising? Why? Would you
be willing to predict the election outcome? Explain by (i) conducting a simulation; (ii)
using the value found in (a) for the standard error.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate random seed
rng = np.random.default_rng(73) # rng seed

# Sample size
n = 1648
# True population proportion
pi = 1/2

# a) =========== Find Standard error ===========
# Observed sample proportion
pi_hat_observed = 0.515

# Standard error (theoretical)
se = np.sqrt(pi* (1 - pi) / n)

# b) =========== Simulation and Interpretation ===========
# Number of voters in sample who voted for MArk Kelly (simulated counts)
Y = rng.binomial(n, pi, size=int(2e5))

# Sample proportions
pi_hat = Y / n

# OUtput
print("a) Find Standard Error:")
print(f"Standard Error (theoretical): {se:.5f}\nAbout {se*100:.3f} percentage points")

print("\nb) INTERPRETATION:")
# Standard Error (Simulated)
se_simulation = pi_hat.std(ddof=1)

# Simulation-based tail probabilities
p_one_sided = np.mean(pi_hat >= pi_hat_observed)
p_two_sided = np.mean(np.abs(pi_hat - pi) >= abs(pi_hat_observed - pi))

print(42*"=")
print(f"Standard Error (simulated): {se:.5f}\nAbout {se_simulation*100:.3f} percentage points")
print("Thus, theoretical and simulated are approximately the same.")
print("Simulation correctly approximating the sample distribution.")
print(
    f"\nIf the true proportion is {pi}, then random sampling with n={n} makes the \n"
    f"sample proportion vary by about {se:.5f}.\n"
    f"The observed pi_hat = {pi_hat_observed:.3f} is\n"
    f"{pi_hat_observed - pi:.3f} above 0.50, which is\n"
    f"{((pi_hat_observed - pi) / se):.3f} standard errors.\n"
    "This is not surprising."
)
print(f"One-sided P(pi_hat >= 0.515 | pi=0.50) ≈ {p_one_sided:.3f}")
print(f"Two-sided P(|pi_hat-0.50| >= 0.015)    ≈ {p_two_sided:.3f}")
print("Conclusion: Not surprising under pi=0.50; weak evidence to predict winner from this alone.")
print(42*"=")
print("""In summary, If that fraction is large (say 10-15%), 
then observing 0.515 isn't surprising—it could easily happen by chance even in a tied race. That means you can't confidently predict Kelly wins.""")

# Plotting results
plt.hist(pi_hat, bins=50, density=True, edgecolor="black")
plt.axvline(pi_hat_observed, linestyle="--", label="Observed pi_hat = 0.515")
plt.axvline(pi, linestyle=":", label="Assumed pi = 0.50")
plt.xlabel("Simulated sample proportion (pi_hat)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()


