#!/usr/bin/env python
"""
Before a Presidential election, polls are taken in two swing states. The Republican candidate
was preferred by 59 of the 100 people sampled in state A and by 525 of 1000 sampled in state
B. Treat these as independent binomial samples, where the parameter $\pi$ is the population
proportion voting Republican in the state.
(a) If we can treat these polls as if the samples were random, use significance tests of $H_0$:
$\pi  = 0.50$ against $H_a:  \pi > 0.50$ to determine which state has greater evidence supporting a
Republican victory. Explain your reasoning.
(a) If we can treat these polls as if the samples were random, use significance tests of $H_0$:
$\pi  = 0.50$ against $H_a:  \pi > 0.50$ to determine which state has greater evidence supporting a
Republican victory. Explain your reasoning.
"""
import numpy as np
from scipy import stats
from textwrap import dedent

# State A
n_a, x_a = 100, 59
p_hat_a = x_a / n_a

# State B
n_b, x_b = 1000, 525
p_hat_b = x_b / n_b

# null hypothesis value
pi_0 = 1/2

print(dedent(f"""
\nHYPOTHESIS TEST FOR PROPORTIONS:
Null: pi = {pi_0} (population evenly split)
pi > {pi_0} (Rep. majority - one-tailed)
             """))

# Test statistics: numerator = how far sample is fro null hypothesis
# Denominator = standard error under H_0 (expecte variability)
se_a, se_b = np.sqrt(pi_0 * (1 - pi_0) / n_a), np.sqrt(pi_0 * (1 - pi_0) / n_b)

# Z-scores
z_a = (p_hat_a - pi_0) / se_a
z_b = (p_hat_b - pi_0) / se_b

# p-values (one-tailed)
p_a = 1 - stats.norm.cdf(z_a)
p_b = 1 - stats.norm.cdf(z_b)

# OUtput results
print("\nState A")
print(f"Sample size (n):        {n_a}")
print(f"Sample proportion (p̂):  {p_hat_a:.3f}")
print(f"Standard error:         {se_a:.4f}")
print(f"z-statistic:            {z_a:.3f}")
print(f"p-value (one-tailed):   {p_a:.4f}")

print("\nState b")
print(f"Sample size (n):        {n_b}")
print(f"Sample proportion (p̂):  {p_hat_b:.3f}")
print(f"Standard error:         {se_b:.4f}")
print(f"z-statistic:            {z_b:.3f}")
print(f"p-value (one-tailed):   {p_b:.4f}")

# Comparison
print("\nComparison")
if p_a < p_b:
    stronger = "State a"
    weaker = "State b"
else:
    stronger = "State b"
    weaker = "State a"

print(f"\n{stronger} has STRONGER evidence for Republican victory")
print(f"(smaller p-value = more evidence against H₀)")

# Interpretation
print(dedent(f"""
State A shows stronger evidence because:
Even though state B has a larger sample size, state A's EFFECT SIZE, dividing y1_hat - y2_hat by the pooled standard deviation estimate (deviation from 1/2), which is much larger:

State A deviates by {(p_hat_a - pi_0)*100:.1f} percentage points.
State B deviates by {(p_hat_b - pi_0)*100:.1f} percentage points.

Z-stat balances both:
1) Effect size
2) Precision, how reliable the estimate its

Thus, State A's large effect size outdoes State B's precision advantage.
             """))




