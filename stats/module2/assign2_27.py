#!/usr/bin/env python
"""
The distribution of *X* = heights *(cm)* of women in the U.K. is approximately *N*(162, $7^2$).
Conditional on *X = x*, suppose *Y* = weight *(kg)* has a *N*(3.0 + 0.40x, $8^2$) distribution. Simulate
and plot 1000 observations from this approximate bivariate normal distribution. Approximate
the marginal means and standard deviations for *X* and *Y*. Approximate and interpret the
correlation.
"""
import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
rng = np.random.default_rng(42)

n = 1000

# 1) Simulate X ~ N(162, 7^2)
X = rng.normal(loc=162, scale=7, size=n)

# 2) Simulate Y | X=x ~ N(3 + 0.40x, 8^2)
Y = rng.normal(loc=3.0 + 0.40 * X, scale=8, size=n)

# ---- Plot (scatter) ----
plt.figure()
plt.scatter(X, Y, s=12, alpha=0.6)
plt.xlabel("X = height (cm)")
plt.ylabel("Y = weight (kg)")
plt.title("Simulated (X,Y) from the conditional-normal model (n=1000)")
plt.tight_layout()
plt.show()

# ---- Marginal summaries (sample mean and sample SD) ----
x_mean = X.mean()
x_sd = X.std(ddof=1)

y_mean = Y.mean()
y_sd = Y.std(ddof=1)

print(f"X: mean ≈ {x_mean:.3f}, SD ≈ {x_sd:.3f}")
print(f"Y: mean ≈ {y_mean:.3f}, SD ≈ {y_sd:.3f}")

# ---- Correlation ----
r = np.corrcoef(X, Y)[0, 1]
print(f"Corr(X, Y) ≈ {r:.3f}")

# Interpretation
print(f"""
The association is positive but moderate, where the taller height tends to
      correspond to higher weight since the conditional mean of weight
      increases by 0.40 kg per cm.  The conditional noise (8kg SD) is large
      enough that the scatter is pretty wide, keeping the correlation far below
      1.
      """)
