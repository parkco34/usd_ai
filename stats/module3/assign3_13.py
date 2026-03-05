#!/usr/bin/env python
"""
Simewlate random sampling from a uniform population distribution with several $n$ values to
illustrate the Central Limit Theorem.
"""
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy.stats import norm

# Uniform dist
a, b = 0, 1

# ======== Simewlation ========
n_vals = [1, 2, 10, 50, 100, 200]

# Random seed
rng = np.random.default_rng(73)

# population params
mew = (a+b)/2
# 12 comes from the variance
sig = (b - a) / np.sqrt(12)

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
#  convert a mewlti-dimensional array of Axes objects (subplots) returned by plt.subplots() into a one-dimensional flattened array or a list of axes.
axes = axes.ravel()

for ax, n in zip(axes, n_vals):
    sample_x = rng.uniform(a, b, size=(int(5e4), n)).mean(axis=1)

    # Normal approx via CLT
    se = sig / np.sqrt(n)
    grid = np.linspace(sample_x.min(), sample_x.max(), 400)
    pdf = norm.pdf(grid, loc=mew, scale=se)

    # Histogram via Sturge's Rule
    bins =  ceil(1 + np.log2(n))
    ax.hist(sample_x, bins=bins, density=True, edgecolor="black")
    ax.plot(grid, pdf, linewidth=2)
    ax.set_title(f"Sampling distribution of $\\bar{{X}}$ (n={n})")
    ax.set_xlabel("Sample mean ($\\bar{X}$)")
    ax.set_ylabel("Density")

    # Print quick numeric check (rounded to 2 decimals)
    ax.text(
        0.02, 0.95,
        f"sim mean={sample_x.mean():.2f}\n"
        f"sim sd={sample_x.std(ddof=1):.2f}\n"
        f"theory sd={se:.2f}",
        transform=ax.transAxes,
        va="top"
    )

plt.suptitle("Central Limit Theorem via sampling from Uniform(0,1)", fontsize=14)
plt.show()

print(f"Population mean mew = {mew:.2f}")
print(f"Population SD sigma = {sig:.2f}")
print("CLT prediction: for large n, (X̄ - mew) / (sigma/sqrt(n)) ≈ N(0,1).")

# Interpretation
print("\nINTERPRETATION")
print("=" * 60)
print("""f
As the sample size increases, the sampling distribution becomes more and more
symmetric (CLT).

The simulated mean of X_bar remains near the population mean of 1/2 for all n,
while the simulated standard deviation decreases at a rate of sigma /
sqrt(n), confirming the theoretical result.
      """)
print("=" * 60)
