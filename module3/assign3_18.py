#!/usr/bin/env python
"""
Sunshine City, which attracts primarily retired people, has 90,000 residents with a mean age of 72 years and a standard deviation of 12 years. The age distribution is skewed to the left. A random sample of 100 residents of Sunshine City has $\bar{y} = 70$ and $s = 11.$ a) Describe the center and spread of the (i) population distribution, (ii) sample data distribution. What shape does the sample data distribution probably have? Why? b) Find the center and spread of the sampling distribution of $\bar{Y}$ for $n = 100.$ What shape does it have and what does it describe? c) Explain why it would not be unusual to sample a person of age 60 in Sunshine City, but it would be highly unusual for the sample mean to be 60, for a random sample of 100 residents. d) Describe the sampling distribution of $\bar{Y}$ : (i) for a random sample of size $n = 1$; (ii) if you sample all 90,000 residents.
"""
import numpy as np
from scipy.stats import norm
from math import sqrt

N = int(9e4)
mew = 72
sigma = 12
ybar = 70
sd = 11
n = 100

# a) center/spread
print("(a) CENTER & SPREAD")
print("(i) Population distribution:")
print(f"  Center (mean): {mew:.2f} years")
print(f"  Spread (SD):   {sigma:.2f} years")
print("  Shape: left-skewed (given)\n")

print(f"""(ii) Sample data distribution:
  Center (sample mean ybar): {ybar:.2f} years
  Spread (sample SD s):      {sd:.2f} years
""")

print("""Probable shape of the sample *data* distribution:
- Likely still left-skewed (sample tends to resemble population).
- But may look less skewed due to n=100.
- CLT makes the *sample mean* approx normal; it does not force raw data to be normal.
""")

# b) sampling distribution of Ybar for n=100
se = sigma / sqrt(n)

print("(b) SAMPLING DISTRIBUTION OF Ybar (n=100)")
print(f"Center: E(Ybar) = mew = {mew:.2f}")
print(f"Spread: SD(Ybar) = sigma/sqrt(n) = {se:.2f}")
print("Shape: Approximately normal by CLT.")
print(f"Describes: distribution of sample means from many random samples of size {n}.\n")

# c) individual 60 vs sample mean 60
z_ind = (60 - mew) / sigma
z_mean = (60 - mew) / se
p_ybar = norm.cdf(60, loc=mew, scale=se)

print("(c) 60 YEARS OLD vs SAMPLE MEAN 60")
print(f"Individual age 60: z = (60 - {mew})/{sigma} = {z_ind:.2f}")
print("Interpretation: about 1 SD below the mean -> not unusual for an individual.\n")

print(f"Sample mean 60 (n=100): z = (60 - {mew})/{se:.2f} = {z_mean:.2f}")
print(f"P(Ybar <= 60) ≈ {p_ybar:.3e}")
print("Interpretation: averaging 100 shrinks variability, so a mean of 60 is extremely unlikely.\n")

# d) extremes
print("(d) SAMPLING DISTRIBUTION OF Ybar IN EXTREMES")
print("(i) n=1:")
print(f"  Same as population distribution: mean={mew:.2f}, SD={sigma:.2f}, left-skewed.\n")

print("(ii) n=90,000 (census):")
print("  Ybar equals the population mean exactly.")
print("  Mean=72.00, SD=0.00, degenerate spike at 72.\n")

