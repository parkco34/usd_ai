#!/usr/bin/env python
"""
AI in healthcare thrives on clean, insightful data prep: models like random forests predict gallstones from features like BMI and age, but garbage in = garbage out. Here, you'll perform EDA on the Gallstone-1 dataset to spot patterns, directly informing AI steps like feature scaling or class balancing. This is your first taste of how stats prevent biased predictions in tools saving lives, like FDA-approved AI diagnostics flagging risks pre-surgery.

### Dataset
UCI's Gallstone-1 [https://www.kaggle.com/datasets/xixama/gallstone-dataset-uci?resource=download]. Load gallstone.xlsx (~320 rows).

### Tasks

(a) Feature Exploration
Load data. Show first 5 rows and info. Summarize key features and their distributions. Explain what these findings reveal about the data's central tendencies and variability, and how they might inform initial AI preprocessing decisions, such as normalization.

(b) Exploratory Data Analysis
Conduct a deeper analysis on key features and their relationship to `gallstone status`. Prepare a brief report on your findings. Include at least 5 clean, well-formatted visualizations.

(c) Data Augmentation
Generate 50 synthetic "augmented" rows for gallstone-present cases. Append, re-run your analysis in part (b). How does augmentation smooth variances, mimicking real GANs for underrepresented classes?
"""
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil, log2

df = pd.read_csv("./gallstone.csv")

# Select only what you need
features = ['Age', 'Triglyceride', 'Gallstone Status']
df_subset = df[features]

# Part (a) - Basic exploration
print(df_subset.info())
print(df_subset.head())
print(df_subset.describe())
print("\nClass distribution:")
print(df['Gallstone Status'].value_counts())

# Interpretation ==========
# Features
age = df_subset["Age"]
tri = df_subset["Triglyceride"]
# ------ Stats ------
# Age range
rang_age = age.max() - age.min()
# Triglycerides range
rang_tri = tri.max() - tri.min()
age_mew = age.mean()
age_med = age.median()
age_std = age.std(ddof=1) # Sample!

# Plot histogram
# Sturge's Rule for number of bins
k = ceil(1 + log2(df_subset.shape[0]))

# Age 
plt.figure(figsize=(10, 6))
plt.hist(age, bins=k, edgecolor="black")
plt.xlabel("Age of Patient (in years)")
plt.ylabel("Frequency")
plt.title("Age Distribution")
#plt.show()

# Triglyceride 
plt.figure(figsize=(10, 6))
plt.hist(age, bins=k, edgecolor="black")
plt.xlabel("Triglyceride concentration in blood (mg/dL)")
plt.ylabel("Frequency")
plt.title("Triglyceride Distribution")
#plt.show()

# Stats
mew_age = age.mean()
mew_tri = tri.mean()
med_age = age.median()
med_tri = tri.median()
std_age = age.std(ddof=1)
std_tri = tri.std(ddof=1)

# Class Balance plot (excuse to use to practice chaining)
plt.figure(figsize=(6, 4))
df_subset["Gallstone Status"].value_counts().sort_index().plot(kind="bar", edgecolor="black")
plt.xlabel("Gallstone Status (0 = no, 1 = yes)")
plt.ylabel("Count")
plt.title("Class Balance")
plt.tight_layout()
#plt.show()

# Interpretation
print("\nInterpretation")
print("Data:")
print("""
Features = 'Trigrlyceride' (concentration in blood (mg/dL)) and
  'Age' (in years)
Target variable = 'Gallstones' (BOOLEAN)
      """)

print("\nInterpretation")
print(f"""
Age:
      The mean age ({mew_age:.2f}) and the median age({med_age:.2f}) are close
      enough to be approximately NORMAL, as is also seen by the histogram.
      The standard deviation ({std_age:.2f}) is mild, and the range
      ({rang_age}) imply a low (relatively) variabliity, so there doesn't seem
      to be any profound outliers.
      """)

print(f"""\n
Triglyceride:
      Levels show more variability ({std_tri:.2f}) and massive range
      ({rang_tri:}), with some extreme outliers influencing  the mean. Since
      the mean ({mew_tri:.2f}) is more than the median ({med_tri:.2f}), and the
      distribution is right-skewed as seen by the histogram.
""")

print(f"""
Gallstone Status:
      Has a balanced class distribution of a gallstone is present (1) or not
      (0).  No need for class-balancing techniques.
      """)

print(f"""
Ai Preprocessing:
    Since the ranges are such different sizes, normalization would be required
      because the Triglyceride feature would be most prominent compared to the
      Age data, sabotaging an AI pipeline.
      """)

# PArt b) ==================

grouped_stats = df_subset.groupby("Gallstone Status").describe()
print(grouped_stats)

# Boxplot: Age vs Gallstone Status
plt.figure(figsize=(8, 5))
df_subset.boxplot(column="Age", by="Gallstone Status")
plt.xlabel("Gallstone Status (0 = no, 1 = yes)")
plt.ylabel("Age (years)")
plt.title("Age by Gallstone Status")
plt.suptitle("")
plt.tight_layout()
#plt.show()

# Boxplot: Triglyceride vs Gallstone Status
plt.figure(figsize=(8, 5))
df_subset.boxplot(column="Triglyceride", by="Gallstone Status")
plt.xlabel("Gallstone Status (0 = no, 1 = yes)")
plt.ylabel("Triglyceride (mg/dL)")
plt.title("Triglyceride by Gallstone Status")
plt.suptitle("")
plt.tight_layout()
#plt.show()

# Overlaid histogram for Age
plt.figure(figsize=(10, 6))
plt.hist(df_subset[df_subset["Gallstone Status"] == 0]["Age"],
         bins=k, alpha=0.6, label="No Gallstones", edgecolor="black")
plt.hist(df_subset[df_subset["Gallstone Status"] == 1]["Age"],
         bins=k, alpha=0.6, label="Gallstones", edgecolor="black")
plt.xlabel("Age (years)")
plt.ylabel("Frequency")
plt.title("Age Distribution by Gallstone Status")
plt.legend()
plt.tight_layout()
#plt.show()

# Overlaid histogram for Triglyceride
plt.figure(figsize=(10, 6))
plt.hist(df_subset[df_subset["Gallstone Status"] == 0]["Triglyceride"],
         bins=k, alpha=0.6, label="No Gallstones", edgecolor="black")
plt.hist(df_subset[df_subset["Gallstone Status"] == 1]["Triglyceride"],
         bins=k, alpha=0.6, label="Gallstones", edgecolor="black")
plt.xlabel("Triglyceride (mg/dL)")
plt.ylabel("Frequency")
plt.title("Triglyceride Distribution by Gallstone Status")
plt.legend()
plt.tight_layout()
#plt.show()

# Scatter plot: Age vs Triglyceride colored by class
plt.figure(figsize=(10, 6))
plt.scatter(
    df_subset["Age"],
    df_subset["Triglyceride"],
    c=df_subset["Gallstone Status"],
    edgecolor="black"
)
plt.xlabel("Age (years)")
plt.ylabel("Triglyceride (mg/dL)")
plt.title("Age vs Triglyceride Colored by Gallstone Status")
plt.tight_layout()
#plt.show()

# Interpretation
print("""Interpretation""")
print("""
Patients with gallstones typically have higher Triglyceride levels in the blood
      with higher variability and extreme outliers, making a strong predictor. 

Age shows problems in its classes, where the gallstone-positive patients skew
      the towards older patients.

The scatter plot shows there isn't one culprit but the interaction of age AND
      triglyceride.  Multivariate models (linear regression, random forests,
      etc.) would be better suited.
      """)

breakpoint()
