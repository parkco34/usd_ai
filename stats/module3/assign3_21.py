#!/usr/bin/env python
"""
In your school, suppose that GPA has an approximate normal distribution with $\mu = 3.0, \sigma = 0.40.$ Not knowing $\mu$, you randomly sample $n = 25$ students to estimate it. Using simulation for this application, illustrate the difference between a sample data distribution and the sampling distribution of Y .
"""
import numpy as np
import matplotlib.pyplot as plt

mew = 3
sig = 0.4
n = 25

rng = np.random.default_rng(73)

# 1) ONe sample
sample = rng.normal(mew, sig, size=n)

# Plotting
plt.hist(sample, bins=8, edgecolor="black", density=True)
plt.xlabel("GPA")
plt.ylabel("Density")
plt.title("Sample Data Distn (n=25)")
plt.tight_layout()
plt.show()

print(f"Sample Mean: {sample.mean():.3f}")
print(f"Sample Standard Deviation: {sample.std(ddof=1):.3f}")

# 2) Many Samples: Sampling dist of mean
sims = int(1e4)

sample_mew = rng.normal(mew, sig, size=(sims, n)).mean(axis=1)

# Plotting
plt.hist(sample_mew, bins=30, edgecolor="black", density=True)
plt.xlabel("Sample Mean GPA")
plt.ylabel("Density")
plt.title("Sampling Distribution of the Sample Mean")
plt.tight_layout()
plt.show()

print(f"Mean of sample means: {sample_mew.mean():.3f}")
print(f"SD of sample means: {sample_mew.std(ddof=1):.3f}")
print(f"Theoretical SE: {sig/np.sqrt(n):.3f}")

# Interpretation
print(f"""
1. The sample data distribution (first histogram) is wide and noisy, and its center
varies substantially from sample to sample. This distribution reflects individual
GPA variability in the population, not the precision of the estimator.\n

2. The sampling distribution of the sample mean (second histogram) shows the
distribution of the mean GPA computed from repeated samples of size n = {n}.
This distribution is approximately normal, much narrower, and centered near the
population mean μ = {mew}. The standard deviation of the sample mean is
SD(Ȳ) ≈ {sample_mew.std(ddof=1):.3f}, which is close to the theoretical value
σ/√n = {sigma/np.sqrt(n):.3f}.
""")

