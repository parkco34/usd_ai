#!/usr/bin/env python
"""
For a sequence of observations of a binary random variable, you observe the geometric random
variable (Section 2.2.2) outcome of the first success on observation number $y = 3$. Find and plot
the likelihood function.
"""
import numpy as np
import matplotlib.pyplot as plt

# First success @ y = 3
y = 3

# Array of possible pi values, arbitrarily chosen number of samples
pi_pts = np.linspace(0.001, 0.999, int(1e4))

# Likelihood for geometric dist
L = ((1 - pi_pts)**(y-1)) * pi_pts

# MLE
pi_hat = 1 / y

# Likelihood @ MLE for marking in plot
L_hat = ((1 - pi_hat)**(y-1)) * pi_hat

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(pi_pts, L, "b-", label="Likelihood: L(\u03C0) = P(Y=3 | \u03C0)"
)
plt.axvline(pi_hat, color="red", linestyle="--", label=f"MLE: pi_hat = 1 / {y} = {pi_hat:.3f}")
plt.scatter([pi_hat], [L_hat], color="red")
plt.xlabel("\u03C0 success probability")
plt.ylabel("L(\u03C0|y=3)")
plt.title("Likelihood Function for Geometric Distribution\nObservation: First Success on Trial y=3")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting the log-likelihood
logL = np.log(L)

plt.figure(figsize=(10, 6))
plt.plot(pi_pts, logL, "k-", label=r"$\ell(\pi)=\log L(\pi)$")
plt.axvline(pi_hat, color="red", linestyle="--", label=f"MLE: {pi_hat:.3f}")
plt.xlabel(r"$\pi$ (success probability per trial)")
plt.ylabel(r"log-likelihood")
plt.title(f"Log-Likelihood for Geometric Model\nObservation: first success on trial y={y}")
plt.grid(True)
plt.tight_layout()
plt.show()

# Interpretation
print(f"""
The likelihood function is L(\u03C0) = (1 - \u03C0)^(y-1), where \u03C0 is the trial of each success probbability, and L(\u03C0) measures how compatible each \u03C0 s with the observed data y = {y}.  

IT's maximized at \u03C0\u0302 = 1 / y = {pi_hat:.3f}, which is the maximium likelihood estimate of the success probability, which shows that two failures followed by a success.
      """)


