#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil, log2

df = pd.read_csv("./gallstone.csv")

# Part (a) ==========
print(df.info())
print("\nFirst five rows:")
print(df.head())

cols = ["Age", "Triglyceride", "Hepatic Fat Accumulation (HFA)", "Gallstone Status"]
d = df[cols]

print("\nSummary stats:")
print(d.describe())

print("\nClass distribution:")
print(d["Gallstone Status"].value_counts())

print("\nMissing values:")
print(d.isnull().any())

# Quick scale check (range) -> normalization hint
ranges = d[["Age", "Triglyceride"]].agg(lambda s: s.max() - s.min())
print("\nRanges:")
print(ranges.rename({"Age": "Age range", "Triglyceride": "Triglyceride range"}))

# Histograms ==========
k = ceil(1 + log2(len(d)))  # Sturges

for col, xlabel, title in [
    ("Age", "Age (years)", "Age Distribution"),
    ("Triglyceride", "Triglycerides (mg/dL)", "Triglyceride Distribution"),
]:
    plt.figure(figsize=(10, 6))
    plt.hist(d[col], bins=k, edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Interpretation (keep it honest: symmetry ≠ normality)
age_mean, age_med = d["Age"].mean(), d["Age"].median()
print(f"\nAge mean vs median: {age_mean:.3f} vs {age_med:.3f} (rough symmetry if close)")
print("Triglycerides look right-skewed -> consider robust scaling or log transform.")
print("Large range differences -> normalization/standardization likely helpful.")


# Part (b) ==========
import seaborn as sns

# Split by outcome
gs_yes = d[d["Gallstone Status"] == 1]
gs_no  = d[d["Gallstone Status"] == 0]

# -----------------------------
# 1) Boxplots: Age vs Gallstone
# -----------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x="Gallstone Status", y="Age", data=d)
plt.xlabel("Gallstone Status (0 = No, 1 = Yes)")
plt.ylabel("Age (years)")
plt.title("Age Distribution by Gallstone Status")
plt.tight_layout()
plt.show()

# -----------------------------
# 2) Boxplots: Triglyceride vs Gallstone
# -----------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x="Gallstone Status", y="Triglyceride", data=d)
plt.xlabel("Gallstone Status (0 = No, 1 = Yes)")
plt.ylabel("Triglycerides (mg/dL)")
plt.title("Triglyceride Levels by Gallstone Status")
plt.tight_layout()
plt.show()

# -----------------------------
# 3) Histogram comparison: Age
# -----------------------------
plt.figure(figsize=(10, 6))
plt.hist(gs_no["Age"], bins=k, alpha=0.6, label="No Gallstones")
plt.hist(gs_yes["Age"], bins=k, alpha=0.6, label="Gallstones")
plt.xlabel("Age (years)")
plt.ylabel("Frequency")
plt.title("Age Distribution by Gallstone Status")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 4) Histogram comparison: Triglycerides
# -----------------------------
plt.figure(figsize=(10, 6))
plt.hist(gs_no["Triglyceride"], bins=k, alpha=0.6, label="No Gallstones")
plt.hist(gs_yes["Triglyceride"], bins=k, alpha=0.6, label="Gallstones")
plt.xlabel("Triglycerides (mg/dL)")
plt.ylabel("Frequency")
plt.title("Triglyceride Distribution by Gallstone Status")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 5) Crosstab: HFA vs Gallstone
# -----------------------------
hfa_table = pd.crosstab(
    d["Hepatic Fat Accumulation (HFA)"],
    d["Gallstone Status"],
    normalize="index"
)

print("\nHFA vs Gallstone Status (Row Proportions):")
print(hfa_table)

# -----------------------------
# Interpretation
# -----------------------------
print("\nInterpretation:")
print("- Gallstone-positive patients tend to be older on average.")
print("- Triglycerides show right-skewness in both groups, with heavier tails for gallstone cases.")
print("- Increasing HFA levels correspond to higher proportions of gallstone presence.")
print("- These associations suggest Age, Triglycerides, and HFA are informative predictors.")
print("- Skewed biomarkers motivate log-scaling or robust normalization.")



# Part (c) ==========
import numpy as np

rng = np.random.default_rng(42)

# Isolate gallstone-present cases
gs_yes = d[d["Gallstone Status"] == 1]

# Number of synthetic samples
n_aug = 50

# Generate synthetic samples via bootstrap + jitter
augmented = gs_yes.sample(n=n_aug, replace=True).copy()

# Add small Gaussian noise to continuous variables
for col in ["Age", "Triglyceride"]:
    std = gs_yes[col].std()
    augmented[col] += rng.normal(0, 0.05 * std, size=n_aug)

# Ensure non-negative constraints
augmented["Age"] = augmented["Age"].clip(lower=18)
augmented["Triglyceride"] = augmented["Triglyceride"].clip(lower=0)

# Append to original data
d_aug = pd.concat([d, augmented], ignore_index=True)

print("\nOriginal size:", d.shape[0])
print("Augmented size:", d_aug.shape[0])

print("\nNew class distribution:")
print(d_aug["Gallstone Status"].value_counts())

