#!/usr/bin/env python
"""
2.48 A probability distribution has a scale parameter θ if, when you multiply θ by a constant c, all values in the distribution multiply by c. It has a location parameter θ if, when you increase θ by a constant c, all values in the distribution increase by c.

(a) For a scale parameter θ, the distribution of Y / θ does not depend on θ. Show that for the gamma distribution (2.10) with θ = 1 / λ, Y / θ has mean and variance not dependent on θ.

(b) For a location parameter θ, the pdf is a function of y − θ and the distribution of Y − θ does not depend on θ. For a normal distribution, show that μ is a location parameter.
------------------------------------------------------------------------------------------
Plot the gamma distribution by fixing the shape parameter *k* = 3 and setting the scale parameter
= 0.5, 1, 2, 3, 4, 5. What is the effect of increasing the scale parameter? (See also Exercise 2.48.)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

"""
Problem:
Plot the gamma distribution by fixing the shape parameter k = 3 and setting the
scale parameter theta = 0.5, 1, 2, 3, 4, 5. What is the effect of increasing theta?
"""

# Fixed shape parameter
k = 3

# Scale parameters to compare
scales = [0.5, 1, 2, 3, 4, 5]

# Choose an x-range wide enough to show the largest-scale distribution
x = np.linspace(0, 40, 1000)

# Plot gamma pdfs for each scale
plt.figure()
for theta in scales:
    y = gamma.pdf(x, a=k, scale=theta)
    plt.plot(x, y, label=f"scale = {theta}")

plt.xlabel("y")
plt.ylabel("Probability Density")
plt.title("Gamma Distributions with k = 3 and Varying Scale")
plt.legend()
plt.tight_layout()
plt.show()


# Interpretation
print(f"""
Increasing theta stretches the distribution to the right, where
the MODE: (k-1)theta = 2*theta and the MEAN is 3*theta.
Hence, the SCALE PARAMETER.

As theta increases the distribution gets wider, the peak height decreases and the right tail grows longer.

The SHAPE PARAMETER (k) controls the form, or skewness.
The SCALE PARAMETER (theta) controls the units and magnitude.
      """)
