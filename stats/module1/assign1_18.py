#!/usr/bin/env python
"""
The `Income` data file ([http://stat4ds.rwth-aachen.de/data/Income.dat](http://stat4ds.rwth-aachen.de/data/Income.dat)) at the book’s website reports annual income values in the U.S., in thousands of dollars.

(a) Using software, construct a histogram. Describe its shape.
(b) Find descriptive statistics to summarize the data. Interpret them.
(c) The kernel density estimation method finds a smooth-curve approximation for a histogram. At each value, it takes into account how many observations are nearby and their distance, with more weight given those closer. Increasing the bandwidth increases the influence of observations further away. Plot a smooth-curve approximation for the histogram of income values. Summarize the impact of increasing and of decreasing the bandwidth substantially from the default value.
(d) Construct and interpret side-by-side box plots of income by race (B = Black, H = Hispanic, W = White).  Compare the incomes using numerical descriptive statistics
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil, log2
from scipy.stats import gaussian_kde

df = pd.read_csv("./income.dat", sep="\s+")

# a) Histogram
# Isolate income variable
x = df["income"]
# Bins using Sturge's Rule
k = ceil(1 + log2(df.shape[0]))

# Plotting histogram w/ density=True so the area under histogram =1
plt.hist(x, bins=k, edgecolor="black", density=True)
plt.xlabel("Income (in thousands $)")
plt.ylabel("Probaility Density")
plt.title("How likely is an observation to fall near this income range?")
plt.tight_layout()
plt.show()

# SHape
print("Histogram is right-skewed so the distribution is not Normal")

# b) ============ Descriptive Stats =============
n = x.shape[0]
mew = x.mean()
med = x.median()

# Sample std dev and variance
std_dev = x.std(ddof=1)
var = x.var(ddof=1)

min_val = x.min()
max_val = x.max()
q1 = x.quantile(0.25)
q3 = x.quantile(0.75)
iqr = q3 - q1

# Help from Claude for how to do this nicely instead of print statements
summary = pd.Series(
    {
        "n": n,
        "mean": mew,
        "median": med,
        "std_dev": std_dev,
        "variance": var,
        "min": min_val,
        "Q1 (25%)": q1,
        "Q3 (75%)": q3,
        "IQR": iqr,
        "max": max_val,
    }
)

print("\nDescriptive Stats for Income (in thousands $)")
print(summary.round(2))


# ======== Interpretations =============
print(f"Mean, {mew:.2f} and  the median, {med:.2f}")
print("Thus, the distribution is right-skewed as we saw in the histogram.")
print(f"""The mean value gets pulled up by the outliers, where 50% of income
      lies w/in Q1 {q1:.2f} and Q3 {q3:.2f}, w/ and IQR of {iqr:.2f}.
      The spread is resistent to outliers.""")
print("\nRange:")
print(f"Range from min {min_val:.2f} to max {max_val:.2f}")
print("This shows high-income values relative to center.")
print("\nStandard Deviations:")
print(f"""
std_dev = {std_dev:.2f},  which is large due to distribution being skewed via
      outliers, so using the approximately normal distribution won't work here.
      """)

# c) ======= Kernel Desnity Estimation ==================
# Income values for smooth curve
x_grid = np.linspace(min_val, max_val, 500)

# KDE with default bandwidth
kde = gaussian_kde(x)

plt.hist(x, bins=k, density=True, edgecolor="black")
# Had help from Claude to unsderstand how to plot this without it looking so
# ugly ʘ‿ʘ
plt.plot(x_grid, kde(x_grid), linewidth=2, label="Default bandwidth")
plt.xlabel("Income (in thousands $)")
plt.ylabel("Probability Density")
plt.title("Kernel Density Estimate of Income (Default Bandwidth)")
plt.legend()
plt.tight_layout()
plt.show()

# for undersmoothing
kde_small = gaussian_kde(x, bw_method=0.3)

plt.hist(x, bins=k, density=True, edgecolor="black")
plt.plot(x_grid, kde_small(x_grid), linewidth=2, label="Small bandwidth")
plt.xlabel("Income (in thousands $)")
plt.ylabel("Probability Density")
plt.title("KDE with Small Bandwidth (Undersmoothing)")
plt.legend()
plt.tight_layout()
plt.show()

# For oversmoothing (larger bandwidth)
kde_large = gaussian_kde(x, bw_method=2.0)

plt.hist(x, bins=k, density=True, edgecolor="black", alpha=0.4)
plt.plot(x_grid, kde_large(x_grid), linewidth=2, label="Large bandwidth")
plt.xlabel("Income (in thousands $)")
plt.ylabel("Probability Density")
plt.title("KDE with Large Bandwidth (Oversmoothing)")
plt.legend()
plt.tight_layout()
plt.show()

# Summary of increasing/decreasing bandwidth from default value
print("\nDefault")
print("""The kernel density estimate gives us a smooth approx. of income
      distribution, a unimodal, right skewed shape which matches with my
      histogram.""")
print("\nDecreasing Bandwidth")
print("""When bandwidth decreases quite a bit, the curve gets chaotic and
      irregular as a result of the sensitivity to local noise; undersmoothing.""")
print("\nIncreasing Bandwidth")
print("""Bandwitdh controls the BIAS-VARIANCE tradeoff w/ smaller bandwidth
      reducing the BIAS at the cost of increase in VARIANCE, and larger
      bandidths increase in BIAS but reduces VARIANCE.""")

# d) =========== Side by side box plots ==============
# Dataframe with relevant variables
dframe = df[["income", "race"]]

# Plotting
dframe.boxplot(column="income", by="race")
plt.suptitle("")  # removes the automatic pandas suptitle
plt.title("Income by Race (B=Black, H=Hispanic, W=White)")
plt.xlabel("Race")
plt.ylabel("Income (in thousands $)")
plt.tight_layout()
plt.show()

# NumericalDescriptive Stats w/ help from Claude (I take notes tho)
# Compare groups using mean/median/spread + quartiles and IQR
group_stats = (
    dframe.groupby("race")["income"]
    .agg(
        n="count",
        mean="mean",
        median="median",
        std_dev=lambda s: s.std(ddof=1),
        variance=lambda s: s.var(ddof=1),
        min="min",
        q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75),
        max="max",
    )
)

# IQR
group_stats["IQR"] = group_stats["q3"] - group_stats["q1"]

print("\nDescriptive statistics for Income (in thousands $) by Race:")
print(group_stats.round(2))

# Comapring statements for medians and means
medians = group_stats["median"].sort_values(ascending=False)
means = group_stats["mean"].sort_values(ascending=False)

print("\nComparison using medians:")
print(medians.round(2))

print("\nComparison using means, but  sensitive to high income outliers:")
print(means.round(2))

# Simple interpretation helper text (you can keep or edit)
top_med_race = medians.index[0]
low_med_race = medians.index[-1]

# Interpretations
print("\nInterpreation:")
print(f"""Median incomes differ by RACE, w/ the highest for {top_med_race} and
      lowest for {ow_med_race}.""")
print("""Since incomes are right skewed, the medians and IQRs more informative
      than means and std devs.""")
print("""Boxplots reveal variations in the center (mean), and the spread, as
      well as the high-income outliers.""")






