#!/usr/bin/env python
from math import floor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("house.dat", sep="\s+")
# Selling prices to be analyzed
prices = df["price"]

n = len(df)

# Stats
rng = max(prices) - min(prices)
mew = prices.mean()
med = prices.median()
stdev = prices.std()

# Sturges Rule for getting the bins
k = int(np.log2(n) + 1)
bin_width = int(rng / k)
# Stop value
stop = floor(max(prices) + bin_width)
# Create bins
bins = np.arange(min(prices),stop, bin_width)

print("n =", n)
print(f"Mean = {mew:.3f}")
print(f"Median = {med:.3f}")
print(f"Std dev = {stdev:.3f}\n")

# Plotting histogram
# Create the frequency distribution
# histogram(a, bins=10, range=None, density=None, weights=None)
counts, edges = np.histogram(prices, bins=bins)

# Display the frequency table with relative frequencies
print("Frequency Distribution (with Relative Frequencies):")
for i in range(len(counts)):
    rel_freq = counts[i] / n
    print(f"[{edges[i]:.1f} - {edges[i+1]:.1f}): {counts[i]}  (Rel freq = {rel_freq:.3f})")


plt.hist(prices, bins=bins, edgecolor='black')
# Explanatory variable
plt.xlabel("House Prices")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of House Prices")
plt.show()

# Get bin that contains the price range with the most houses
max_idx = np.argmax(counts) # ?
lower = edges[max_idx]
upper = edges[max_idx + 1]
perc = counts[max_idx] / n * 100

print(f"Max bin index: {max_idx}: [{lower:.1f}, {upper:.1f}) "
      f"with {counts[max_idx]} homes = {perc:.1f}% of {n}.")

print(f"""The distribution of house prices is unimodal and right-skewed. Most houses ({perc:.1f}%) are priced between ${lower:.1f}K and ${upper:.1f}K, with a few outliers pulling the mean upward to ${mew:.1f}K, which higher than the median of ${med:.2f}K.""")

"""
Percentages of observations that fall within one std dev of the mean. Why not close to 68% ?
"""
# mean +- 1*std dev
lower_bound = round(mew - stdev, 3)
upper_bound = round(mew + stdev, 3)
# Housing prices between lower/upper bound
inrange = sum((prices >= lower_bound) & (prices <= upper_bound))
# % within the bounds
perc_in = inrange / n * 100

print(f"\nLower bound (mean - 1 std): ${lower_bound:.1f}K")
print(f"Upper bound (mean + 1 std): ${upper_bound:.1f}K")
print(f"Percentage of observations within bounds: {perc_in:.2f}%")
print(f"This is a result of the fact that the distribution is right-skewed and far from 'Normal': mean {mew:.2f} > median {med:.2f}")

# Box PLot
plt.figure(figsize=(15, 6))
plt.boxplot(prices, vert=False)
plt.xlabel("Price (in thousands)")
plt.title("Boxplot of House Prices - Notice the Right-Skewed Distribution")
plt.ylabel("Housing Prices")
plt.show()

# Find outliers
Q1 = prices.quantile(0.25)
Q3 = prices.quantile(0.75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

outliers = prices[(prices < lower_fence) | (prices > upper_fence)]

print(f"Lower fence = {lower_fence:.1f}")
print(f"Upper fence = {upper_fence:.1f}")
print("Outliers:")
print(outliers)

# Interpret
print("""The whiskers are longer on the right, 9 outlier dots on the right side and the median line is not to the left like I would expect a for aright-skewed distribution but, all-in-all this indicates the righ-skewness of the distribution, confirming my hypothesis.""")

# Comparing selling prices to new houses
mask = df["new"] == 1
new = df[mask]["price"]
old =  df[~mask]["price"]

new_n = len(new)
old_n = len(old)

# Explanatory variable: new (categorical)
# Response variable: price (continuous)
# Descriptive Stats: Five Number SUmmary
new_mean = new.mean()
old_mean = old.mean()
new_med = new.median()
old_med = old.median()
new_std = new.std()
old_std = old.std()
# Ranges
new_rng = new.max() - new.min()
old_rng =old.max() - old.min()

# Changes
mean_dif = abs(new_mean - old_mean)
med_df = abs(new_med - old_med)
std_dif = abs(new_std - old_std)

# Output stats
print(f"\nDescriptive Statistics Comparison:")
print(f"Old houses (n={len(old)}):")
print(f"  Mean: ${old_mean:.2f}K")
print(f"  Median: ${old_med:.2f}K")
print(f"  Std Dev: ${old_std:.2f}K")
print(f"\nNew houses (n={len(new)}):")

print(f"\nNew houses (n={len(new)}):")
print(f"  Mean: ${new_mean:.2f}K")
print(f"  Median: ${new_med:.2f}K")
print(f"  Std Dev: ${new_std:.2f}K")

print(f"\nDifferences:")
print(f"  Mean difference: ${mean_dif:.2f}K")
print(f"  Median difference: ${med_df:.2f}K")

# Plotting Old houses and pricing
plt.figure(figsize=(10, 5))
plt.boxplot([old, new], tick_labels=['Old', 'New'])
plt.ylabel("Housing Prices (thousands)")
plt.title("Old v New Housing Prices")
plt.tight_layout()
plt.show()

print(f"""\nBased on the comparison, new houses n={new_n:.2f} have higher
      prices than old houses n={old_n}, with medians of ${new_med:.2f}K and
      ${old_med:.2f}K, respectively.  New houses show more price variability
      (std dev: ${new_std:.2f}K v ${old_std:.2f}K). Both are right-skewed
      distributions, but the old houses have outliers.""")
print("The sample size for the new houses are very small, therefore less reliable")


