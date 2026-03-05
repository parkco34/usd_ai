#!/usr/bin/env python
"""
The `Houses` data file at the book’s website lists, for 100 home sales in Gainesville, Florida,
several variables, including the selling price in thousands of dollars and whether the house
is new (1 = yes, 0 = no). Prepare a short report in which, stating all assumptions including
the relative importance of each, you conduct descriptive and inferential statistical analyses to
compare the selling prices for new and older homes.

"""
from textwrap import dedent
from math import floor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def ci_construct_two_stats(y1, y0, ci=0.95):
    """
    Constructs a Confidence interval to compare two population means
    using two independent samples, and assumes equal variance.
    ------------------------------------------------
    INPUT:
        y1: (list, np.array, pd.DataFrame, pd.Series)
            Sample of values for group 1 (e.g., new homes).
        y0: (list, np.array, pd.DataFrame, pd.Series)
            Sample of values for group 0 (e.g., older homes).
        ci: (float)
            Confidence Level (default 0.95)
    OUTPUT:
        tuple of values:
            (mew_hat_1, mew_hat_0, diff_hat, std_1, std_0, se_diff, df, t_crit, me, conf_interval)
    """
    if isinstance(y1, pd.DataFrame) and isinstance(y0, pd.DataFrame):
        y1, y0 = y1.iloc[:, 0], y0.iloc[:, 0]
    if isinstance(y1, pd.Series) and isinstance(y0, pd.Series):
        y1, y0 = y1.to_numpy(), y0.to_numpy()

    # Sample sizes
    n1, n0 = len(y1), len(y0)

    # Alpha for two-tailed CI
    alpha = (1 - ci) / 2

    # Sample stats
    mew_hat_1 = sum(y1) / n1
    mew_hat_0 = sum(y0) / n0
    diff_hat = mew_hat_1 - mew_hat_0
    std_1 = np.std(y1, ddof=1)
    std_0 = np.std(y0, ddof=1)

    # Standard error and degrees of freedom
    s_pooled = np.sqrt(((n1 - 1)*(std_1**2) + (n0 - 1)*(std_0**2)) / (n1 + n0 - 2))
    se_diff = s_pooled * np.sqrt((1/n1) + (1/n0))
    df = n1 + n0 - 2

    # t-critical value
    t_crit = stats.t.ppf(1 - alpha, df=df)

    # Margin of error + CI for (mu1 - mu0)
    me = t_crit * se_diff
    conf_interval = (diff_hat - me, diff_hat + me)

    return (mew_hat_1, mew_hat_0, diff_hat, std_1, std_0, se_diff, df, t_crit, me, conf_interval)


df = pd.read_csv("./house.dat", sep="\s+")

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
    print(f"[{edges[i]:.1f} - {edges[i+1]:.1f}): {counts[i]} (Rel freq = {rel_freq:.3f})")

plt.hist(prices, bins=bins, edgecolor='black')
# Explanatory variable
plt.xlabel("House Prices")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of House Prices")
plt.show()

# Get bin that contains the price range with the most houses
max_idx = np.argmax(counts)
lower = edges[max_idx]
upper = edges[max_idx + 1]
perc = counts[max_idx] / n * 100
print(f"Max bin index: {max_idx}: [{lower:.1f}, {upper:.1f}) "
      f"with {counts[max_idx]} homes = {perc:.1f}% of {n}.")

print(f"""The distribution of house prices is unimodal and right-skewed.
Most houses ({perc:.1f}%) are priced between ${lower:.1f}K and ${upper:.1f}K,
with a few outliers pulling the mean upward to ${mew:.1f}K, which higher than the median of ${med:.2f}K.""")


"""
Percentages of observations that fall within one std dev of the mean.
Why not close to 68% ?
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
print("""The whiskers are longer on the right, 9 outlier dots on the right side
and the median line is not to Q1 like I would expect a for aright-skewed distribution but,
all-in-all this indicates the righ-skewness of the distribution, confirming my hypothesis.""")

# Comparing selling prices to new houses
mask = df["new"] == 1
new = df[mask]["price"]
old = df[~mask]["price"]

new_n = len(new)
old_n = len(old)

# Explanatory variable: new (categorical)
# Response variable: price (continuous)

# Descriptive Stats: Five Number SUmmary
new_mew = new.mean()
old_mew = old.mean()
new_med = new.median()
old_med = old.median()
new_std = new.std()
old_std = old.std()

# Ranges
new_rng = new.max() - new.min()
old_rng = old.max() - old.min()

# Changes
mew_dif = abs(new_mew - old_mew)
med_df = abs(new_med - old_med)
std_dif = abs(new_std - old_std)

# Output stats
print(f"\nDescriptive Statistics Comparison:")
print(f"Old houses (n={len(old)}):")
print(f"  Mean: ${old_mew:.2f}K")
print(f"  Median: ${old_med:.2f}K")
print(f"  Std Dev: ${old_std:.2f}K")

print(f"\nNew houses (n={len(new)}):")
print(f"  Mean: ${new_mew:.2f}K")
print(f"  Median: ${new_med:.2f}K")
print(f"  Std Dev: ${new_std:.2f}K")

print(f"\nDifferences:")
print(f"  Mean difference: ${mew_dif:.2f}K")
print(f"  Median difference: ${med_df:.2f}K")

# Plotting Old houses and pricing
plt.figure(figsize=(10, 5))
plt.boxplot([old, new], tick_labels=['Old', 'New'])
plt.ylabel("Housing Prices (thousands)")
plt.show()

"""
Extending code for:
The Houses data file at the book's website lists, for 100 home sales in Gainesville, Florida,
several variables, including the selling price in thousands of dollars and whether the house
is new (1 = yes, 0 = no). Prepare a short report in which, stating all assumptions including
the relative importance of each, you conduct descriptive and inferential statistical analyses to
compare the selling prices for new and older homes.
"""

# INFERENTIAL ANALYSIS
print("\nINFERENTIAL ANALYSIS")
# Variance comparison (assumption check)
var_ratio = max(new_std, old_std) / min(new_std, old_std)
print(f"\nNew house std dev: ${new_std:.2f}K")
print(f"Old house std dev: ${old_std:.2f}K")
print(f"Ratio of standard deviations: {var_ratio:.2f}")
print("We proceed under the equal-variance assumption (pooled method), as specified.")

# W/in group CIs (descriptive)
print("\nWithin-Group Confidence Intervals (Descriptive)")
conf_level = 0.95

# New houses CI
new_se = new_std / np.sqrt(new_n)
t_value_new = stats.t.ppf((1 + conf_level) / 2, df=new_n-1)
margin_new = t_value_new * new_se
lower_new = new_mew - margin_new
upper_new = new_mew + margin_new
print(f"New houses (n={new_n}): mean = ${new_mew:.2f}K")
print(f"  95% CI for mew_new: (${lower_new:.2f}K, ${upper_new:.2f}K)")

# Old houses CI
old_se = old_std / np.sqrt(old_n)
t_value_old = stats.t.ppf((1 + conf_level) / 2, df=old_n-1)
margin_old = t_value_old * old_se
lower_old = old_mew - margin_old
upper_old = old_mew + margin_old
print(f"Old houses (n={old_n}): mean = ${old_mew:.2f}K")
print(f"  95% CI for mew_old: (${lower_old:.2f}K, ${upper_old:.2f}K)")

# INferential comparison
print("Comparison")
results = ci_construct_two_stats(new, old, ci=0.95)
mew_hat_1, mew_hat_0, diff_hat, std_1, std_0, se_diff, df_val, t_crit, me, conf_interval = results

print(f"Point estimate (x_new - x_old): ${diff_hat:.2f}K")
print(f"Pooled standard error: ${se_diff:.2f}K")
print(f"Degrees of freedom: {df_val}")
print(f"t-critical (two-tailed): {t_crit:.4f}")
print(f"Margin of error: ${me:.2f}K")
print(f"95% CI for (mew_new - mew_old): (${conf_interval[0]:.2f}K, ${conf_interval[1]:.2f}K)")

# Interpretation
print("\nInterpretation")
if conf_interval[0] > 0:
    print(f"Since the entire CI is above 0, we have evidence at the 95% level")
    print(f"that new homes sell for more than older homes.")
elif conf_interval[1] < 0:
    print(f"Since the entire CI is below 0, we have evidence at the 95% level")
    print(f"that new homes sell for less than older homes.")
else:
    print(f"Since the CI contains 0, we do not have sufficient evidence at the 95% level")
    print(f"to conclude a difference in mean selling prices.")

# Two-sample t-test (for p-value, consistent with CI)
t_stat, p_value = stats.ttest_ind(new, old)
print(f"\nTwo-sample t-test (equal variance assumed):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject null: Significant difference in means.")
else:
    print("No significant difference.")

# ASSUMPTIONS 
print("\nASSUMPTIONS:")
print(dedent(f"""
1) INDEPENDENCE
Each home sale is independent of the others and new and old home samples are
independent groups.

2) NORMALITY
Population distribution approximately normal, where CLT useful in the
comparisons as well.

3) EQUAL VARIANCES
             """))
print(f"\nThe sample mean difference of ${diff_hat:.2f}K is the point estimate for (mew_new - mew_old).")
print(f"""Assuming independence, approximate normality (supported by CLT given sample sizes),
and equal variances (as specified), the t-distribution represents uncertainty in our estimate.""")



