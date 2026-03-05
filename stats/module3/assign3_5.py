#!/usr/bin/env python
"""
The example in Section 3.1.4 simulated sampling distributions of the sample mean to determine how precise $\bar{Y}$ for $n=25$ may estimate a population mean $\mu.$
a) Find the theoretical standard error of $\bar{Y}$ for the scenario values of $\sigma = 5$ and 8. How do they compare to the standard deviations of the 100,000 sample means in the simulations?
b) In the first scenario, we chose $\sigma = 5$ under the belief that if $\mu = 20$, about 2/3 of the sample values would fall between `$`15 and `$`25. For the gamma distribution with $(\mu, \sigma) = (20,5),$ show that the actual probability between 15 and 25 is 0.688.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# x range
x = np.arange(0, 60.01, 0.01)

# Sample size
n = 25

# a) Standard errors (theoretical)
mew = 20
sig1, sig2 = 5, 8
se1, se2 = sig1/np.sqrt(n), sig2/np.sqrt(n)

# Output
print(f"Theoretical Standard Errors are")
print(f"se1 = {se1:.3f}")
print(f"se2 = {se2:.3f}")

# Gamma parameters
shape1 = (mew / sig1) ** 2
scale1 = (sig1 ** 2) / mew
shape2 = (mew / sig2) ** 2
scale2 = (sig2 ** 2) / mew

print("Gamma parameters:")
print(f"Scenario 1: shape={shape1:.2f}, scale={scale1:.2f}")
print(f"Scenario 2: shape={shape2:.2f}, scale={scale2:.2f}\n")

# ---- Simulation of sampling distribution of the mean ----
# Verifying P(15 =< Y <= 25) ~ 0.688
# P(15 <= Y <= 25) for Scenario 1
p = gamma.cdf(25, a=shape1, scale=scale1) - gamma.cdf(15, a=shape1, scale=scale1)
print(f"P(15 <= Y <= 25) = {p:.3f}")

# Random seed
rng = np.random.default_rng(73)
reps = int(1e5)

Y1 = rng.gamma(shape=shape1, scale=scale1, size=(reps, n))
Y2 = rng.gamma(shape=shape2, scale=scale2, size=(reps, n))

Y1mean = Y1.mean(axis=1)
Y2mean = Y2.mean(axis=1)

print("Simulation results (100,000 sample means):")
print(f"Scenario 1 mean(Ȳ): {Y1mean.mean():.3f}   SD(Ȳ): {Y1mean.std(ddof=1):.3f}")
print(f"Scenario 2 mean(Ȳ): {Y2mean.mean():.3f}   SD(Ȳ): {Y2mean.std(ddof=1):.3f}\n")

print("Comparison (theory vs simulation):")
print(f"Scenario 1: theoretical SE={se1:.3f} vs simulated SD(Ȳ)={Y1mean.std(ddof=1):.3f}")
print(f"Scenario 2: theoretical SE={se2:.3f} vs simulated SD(Ȳ)={Y2mean.std(ddof=1):.3f}")

# Interpretation
print(f"""
INTERPRETATION:\n
The simulated SDs of Y_bar are close to the theoretical SEs, {se1}, {se2},
confirming the samplling distribution variablilty.
Scenario 2 has a larger spread because the standard deviation is larger, so
SE(Y) is larger.
      """)

# Gamma PDFs
plt.plot(x, gamma.pdf(x, a=shape1, scale=scale1),
         label=f"Scenario 1: Gamma(shape={shape1:.2f}, scale={scale1:.2f})")
plt.plot(x, gamma.pdf(x, a=shape2, scale=scale2),
         label=f"Scenario 2: Gamma(shape={shape2:.2f}, scale={scale2:.2f})")
plt.xlabel("y (sales)")
plt.ylabel("Density")
plt.title("Gamma Population Distributions")
plt.legend()
plt.tight_layout()
plt.show()

# Histograms of sampling distributions (means)
plt.hist(Y1mean, bins=50, density=True, alpha=0.6, label="Scenario 1 (sigma=5)")
plt.hist(Y2mean, bins=50, density=True, alpha=0.6, label="Scenario 2 (sigma=8)")
plt.xlabel("Sample mean (Ȳ), n=25")
plt.ylabel("Density")
plt.title("Sampling Distributions of the Mean (100,000 simulations)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
