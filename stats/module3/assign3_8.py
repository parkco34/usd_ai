#!/usr/bin/env python
"""
Construct the sampling distribution of the sample proportion of heads, for flipping a balanced
coin (a) once; (b) twice; (c) three times; (d) four times. Describe how the shape changes
as the number of flips $n$ increases. What would happen if $n$ kept growing? Why?
"""
from scipy.stats import binom, norm
from scipy.special import comb
import numpy as np
import matplotlib.pyplot as plt

def sampling_dist_phat(n, p=0.5):
    """
    Returns:
      phat_vals: array of possible p-hat values (0, 1/n, ..., 1)
      probs:     array of probabilities P(p-hat = k/n)
    """
    k = np.arange(n + 1)              # k = number of heads
    phat_vals = k / n                 # p-hat = k/n
    probs = np.array([comb(n, ki) * (p**ki) * ((1 - p)**(n - ki)) for ki in k])
    return phat_vals, probs


# a) - d) Sampling dists
ns = [1, 2, 3, 4]
p = 0.5

for n in ns:
    phat_vals, probs = sampling_dist_phat(n, p=p)

    # Print the distribution in a clean way
    print("\n" + "=" * 60)
    print(f"n = {n} flips (balanced coin, p = {p})")
    print("-" * 60)
    print("p-hat values:", phat_vals)
    print("probabilities:", np.round(probs, 4))

    # Theoretical mean and standard deviation of p-hat
    mu = p
    se = np.sqrt(p * (1 - p) / n)
    print(f"Mean of p-hat: {mu:.2f}")
    print(f"SD (SE) of p-hat: {se:.4f}")

    # Plot (bar chart)
    plt.figure()
    plt.bar(phat_vals, probs, width=0.8 * (1 / n), edgecolor="black")
    plt.xlabel("Sample proportion of heads, $\hat{p}$")
    plt.ylabel("Probability")
    plt.title(f"Sampling distribution of $\hat{{p}}$ for n = {n}")
    plt.xticks(phat_vals)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

# Summary
print("\n" + "=" * 60)
print("How variability changes with n (SE = sqrt(p(1-p)/n))")
print("-" * 60)
for n in ns:
    se = np.sqrt(p * (1 - p) / n)
    print(f"n = {n:>2}: SE = {se:.4f}")


# Interpretation
print("""
For larger n, the shape of the sampling distribution becomes more bell shaped (Normal distribution) via the Central Limit Theorem.
      """)





