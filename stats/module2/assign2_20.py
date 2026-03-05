#!/usr/bin/env python
"""
(a) Construct a histogram or a smooth-curve approximation for the *pdf* of income in the
corresponding population by plotting results using the density function in R (explained
in Exercise 1.18).
(b)  Of the probability distributions studied in this chapter, which do you think might be
most appropriate for these data? Why? Plot the probability function of that distribution
having the same mean and standard deviation as the income values. Does it seem to
describe the income distribution well?
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde, skew

df = pd.read_csv("./data/income.dat", sep="\s+")

# Isolate variable
x = df["income"]

# a) Histogram
plt.figure()
plt.hist(x, bins="sturges", density=True, edgecolor="black")
plt.xlabel("Income (thousands of dollars)")
plt.ylabel("Estimated density")
plt.title("Income histogram (density-scaled)")
plt.tight_layout()
plt.show()

# KDE (smooth curve approximation)
xs = np.linspace(x.min(), x.max(), 400)
kde_default = gaussian_kde(x)  # SciPy chooses a default bandwidth rule
plt.figure()
plt.hist(x, bins="sturges", density=True, edgecolor="black")
plt.plot(xs, kde_default(xs))
plt.xlabel("Income (thousands of dollars)")
plt.ylabel("Estimated density")
plt.title("Income: histogram + KDE (default bandwidth)")
plt.tight_layout()
plt.show()

# b) Descriptive/inferential stats
n = x.size
mean = x.mean()
median = x.median()
std = x.std(ddof=1)
var = x.var(ddof=1)
q1 = x.quantile(0.25)
q3 = x.quantile(0.75)
iqr = q3 - q1
xmin, xmax = x.min(), x.max()
sk = skew(x, bias=False)

p90 = x.quantile(0.90)
p95 = x.quantile(0.95)

# Boxplot outlier rule (1.5*IQR)
lb = q1 - 1.5 * iqr
ub = q3 + 1.5 * iqr
outliers = x[(x < lb) | (x > ub)].sort_values()

print("\n--- Overall descriptive statistics (income, in $1000s) ---")
print(f"n = {n}")
print(f"mean = {mean:.3f}")
print(f"median = {median:.3f}")
print(f"std dev = {std:.3f}")
print(f"variance = {var:.3f}")
print(f"min = {xmin:.3f}, Q1 = {q1:.3f}, median = {median:.3f}, Q3 = {q3:.3f}, max = {xmax:.3f}")
print(f"IQR = {iqr:.3f}")
print(f"skewness = {sk:.3f}")
print(f"90th percentile = {p90:.3f}, 95th percentile = {p95:.3f}")
print(f"Outlier fences: [{lb:.3f}, {ub:.3f}]")
print(f"Outliers (by 1.5*IQR rule): {outliers.to_list()}")

# c) Badnwidth in KDE
kde_small = gaussian_kde(x, bw_method=kde_default.factor * 0.5)
kde_large = gaussian_kde(x, bw_method=kde_default.factor * 2.0)

plt.figure()
plt.hist(x, bins="sturges", density=True, edgecolor="black")
plt.plot(xs, kde_small(xs), label="KDE: smaller bandwidth (0.5x)")
plt.plot(xs, kde_default(xs), label="KDE: default bandwidth")
plt.plot(xs, kde_large(xs), label="KDE: larger bandwidth (2.0x)")
plt.xlabel("Income (thousands of dollars)")
plt.ylabel("Estimated density")
plt.title("Income KDE with different bandwidths")
plt.legend()
plt.tight_layout()
plt.show()

# d) Box plots
plt.figure()
df.boxplot(column="income", by="race")
plt.xlabel("Race")
plt.ylabel("Income (thousands of dollars)")
plt.title("Income by race (box plots)")
plt.suptitle("")  # removes pandas' automatic extra title line
plt.tight_layout()
plt.show()

group_stats = df.groupby("race")["income"].describe()
print("\n--- Income by race: describe() ---")
print(group_stats)
