#!/usr/bin/env python
"""
For a point estimate of the mean of a population that is assumed to have a normal distribution,
a data scientist decides to use the average of the sample lower and upper quartiles for the $n = 100$
observations, since unlike the sample mean $\bar{Y}$, the quartiles are not affected by outliers. Evaluate
the precision of this estimator compared to $\bar{Y}$ by randomly generating 100,000 samples of size
100 each from a $N(0, 1)$ distribution and comparing the standard deviation of the 100,000
estimates with the theoretical standard error of $\bar{Y}$.
"""
import numpy as np

# Sample Size
n = 100
SAMPLES = int(1e5)

# Random seed for reproducibility
rng = np.random.default_rng(73)

# Normal dist params
sig = 1
mew = 0

# Critical values
Z_68 = 1
Z_95 = 1.96
Z_99 = 2.576

# Generating 100_000 samples of size 100 w/ avg of lower/upper quartiles
y = []
for samp in range(SAMPLES):
    # Generating random variates
    x = rng.normal(size=n)
    # Average of 3rd and 1st quartile
    y_bar_i = (np.quantile(x, 0.75) + np.quantile(x, 0.25)) / 2
    y.append(y_bar_i)

# Estimate standard deviation (not sample, population estimate)
est_std = np.std(y, ddof=1)
y_bar = np.mean(y)

# Theoretical standard error
se = sig / np.sqrt(n)

print(f"Estimated standard deviation: {est_std:.3f}")
print(f"Theoretical standard error: {se:.3f}")
print(f"Mean of estimator (sampling mean): {y_bar:.3f}\n")

# Confidence Intervals to evaluate precision
# Confidence Levels: 68, 95, 99 % w/ critical values: 1, 1.96, 2.576
ci_68_q = (mew - Z_68 * est_std, mew + Z_68 * est_std)
ci_95_q = (mew - Z_95 * est_std, mew + Z_95 * est_std)
ci_99_q = (mew - Z_99 * est_std, mew + Z_99 * est_std)

# Confidence Intervals for the sample mean
ci_68 = (mew - Z_68 * se, mew + Z_68 * se)
ci_95 = (mew - Z_95 * se, mew + Z_95 * se)
ci_99 = (mew - Z_99 * se, mew + Z_99 * se)

print("Typical Error bands for estimator (quartile):")
print(f"68% CI: ({ci_68_q[0]:.3f}, {ci_68_q[1]:.3f})")
print(f"95% CI: ({ci_95_q[0]:.3f}, {ci_95_q[1]:.3f})")
print("Confidence INtervals for sample mean:")
print(f"68% CI: ({ci_68[0]:.3f}, {ci_68[1]:.3f})")
print(f"95% CI: ({ci_95[0]:.3f}, {ci_95[1]:.3f})")
print(f"99% CI: ({ci_99[0]:.3f}, {ci_99[1]:.3f})")

# Interpretation
print("\nINTERPRETATION:")
print(60*"+")
print(f"""
Quartile-based estimator is approximately unbiased, since the sampling mean ({y_bar:.3f}) which is almost equal to the true population mean ({mew}).
sampling variability is higher than the sample mean, where the estimated SD (quartile) is {est_std:.3f} and the theoretical standard deviation is {se:.3f}.
This shows an increase in variabiliity of about {((est_std)/se - 1)*100:.1f}% relative to sample mean.
The confidence intervals based on quartiles are wider for each, meaning lowercprecision.

When the data contains no outliers (as with our N(0,1) simulation), the midhinge's robustness provides no benefit, yet we still pay the cost of increased variance."
      """)
print(60*"+")



