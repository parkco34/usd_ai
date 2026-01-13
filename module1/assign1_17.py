#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("murder.dat", sep="\s+")
murder = df["murder"]

# Exclude D.C.
no_dc = df[df["state"] != "DC"]["murder"]

mean_no_dc = no_dc.mean()
stdev1 = no_dc.std()
mean_all = murder.mean()
stdev2 = murder.std()

# Find outliers
q11 = no_dc.quantile(0.25)
q31 = no_dc.quantile(0.75)
q12 = murder.quantile(0.25)
q32 = murder.quantile(0.75)
iqr1 = q31 - q11
iqr2 = q32 - q12

lower_bound1 = q11 - 1.5 * iqr1
lower_bound2 = q12 - 1.5 * iqr2
upper_bound1 = q31 + 1.5 * iqr1
upper_bound2 = q32 + 1.5 * iqr2

outliers1 = no_dc[(no_dc < lower_bound1) | (no_dc > upper_bound1)]
outliers2 = murder[(murder < lower_bound2) | (murder > upper_bound2)]

# Outlier tables (only change this block)
high_outliers1 = df.loc[(df["state"] != "DC") & (df["murder"] > upper_bound1), ["state","murder"]].sort_values("murder")
high_outliers2 = df.loc[df["murder"] > upper_bound2, ["state","murder"]].sort_values("murder")

# outliers values for standard deviations from mean
no_dc_out = max(high_outliers1["murder"])
dc_out = max(high_outliers2["murder"])

print(f"============== No D.C. ==============")
print(f"High Outliers:\n{high_outliers1}")
print(f"\nMean: {mean_no_dc:.2f} murders per 100,000")
print(f"Median: {no_dc.median()}")
print(f"""Standard Deviation describes the variability: {stdev1:.2f}, where some states are safer than others""")
print(f"Minimum: {no_dc.min()}\nMaximum: {no_dc.max()}")
# z-score
no_dc_zscore = (no_dc_out - mean_no_dc) / stdev1
print(f"{no_dc_zscore:.1f} standard deviations from the mean")

print(f"\n============== With D.C.=============")
print(f"High nOutliers: {high_outliers2}")
print(f"\nMean: {mean_all:.2f} murders per 100,000\nStandard Deviation: {stdev2:.2f}")
print(f"Median: {murder.median()}")
print(f"\nVariation: {stdev2/mean_all*100:.1f}%")
print(f"Minimum: {murder.min()}\nMaximum: {murder.max()}")
# z-score
dc_zscore = (dc_out - mean_all) / stdev2
print(f"{dc_zscore:.1f} standard deviations from the mean!")

# Five-number summary
print("\n===== Five-Number Summary (No DC) =====")
print(f"Min: {no_dc.min():.1f}")
print(f"Q1: {q11:.1f}")
print(f"Median: {no_dc.median():.1f}")
print(f"Q3: {q31:.1f}")
print(f"Max: {no_dc.max():.1f}")
print(f"IQR: {iqr1:.1f}")

# Boxplot
# Create box plots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].boxplot(no_dc)
axes[0].set_ylabel('Murder Rate per 100,000')
axes[0].set_title('Without DC')
axes[0].set_xticklabels(['States'])

axes[1].boxplot(murder)
axes[1].set_ylabel('Murder Rate per 100,000')
axes[1].set_title('With DC')
axes[1].set_xticklabels(['States + DC'])

plt.suptitle('Murder Rates Box Plot Comparison')
plt.tight_layout()
plt.show()

# Median change
median_no_dc = no_dc.median()
median_dc = murder.median()
range_no_dc = no_dc.max() - no_dc.min()
range_dc = murder.max() - murder.min()

print(f"Mean change: {mean_no_dc:2f} to {mean_all:.2f}: {abs(mean_all - mean_no_dc):.2f}")


print(f"Median change: {median_no_dc:.2f} to {median_dc:.2f}: {abs(median_dc - median_no_dc):.2f}")
print("The Mean is affected more")

print(f"\nRange change: {range_no_dc:.2f} to {range_dc:.2f}: {range_dc - range_no_dc:.2f}")
print(f"IQR change: {iqr1:.2f} → {iqr2:.2f} (change: {abs(iqr2-iqr1):.2f})")
print(f"The range is affected more than the IQR")
print(f"\nThe median {median_no_dc} shows half of the states w/ murder rates below {median_no_dc} per 100,000")
print(f"IQR of {iqr1:.2f} indicates middle 50% of states vary by {iqr1:.2f} murders per 100,000")

# Conclusion
print("\nD.C. is an extreme outlier and should be analyzed separately.")


#breakpoint()
