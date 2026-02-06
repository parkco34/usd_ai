#!/usr/bin/env python
"""
REFERENCES:
    https://medium.com/@ugursavci/complete-exploratory-data-analysis-using-python-9f685d67d1e4
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import dedent


df = pd.read_csv("heart_failure_prediction.csv")

# 1) EDA =========================
# Cholesterol shouldn't be zero: Replace zeros with median value
df["Cholesterol"] = df["Cholesterol"].replace(0, df["Cholesterol"].median())
# Resting blood pressure cannot be zero
df["RestingBP"] = df["RestingBP"].replace(0, df["RestingBP"].median())

# Segmant and sort data values into bins
# Bins for Age (since continuous, numeric data)
df["Age_bin"] = pd.cut(
    df["Age"], 
    bins=[0, 39, 59, 100], 
    labels=["<40", "40-59", "60+"])

# Oldpeak bins
df["Oldpeak_bin"] = pd.cut(
    df["Oldpeak"], 
    bins=[-np.inf, 1, 2, np.inf], 
    labels=["Normal", "Moderate", "High"])

# Cholesterol bins
df["Cholesterol_bin"] = pd.cut(
    df["Cholesterol"],
    bins=[0, 200, 240, np.inf],
    labels=["Normal", "Borderline", "High"]
)

# Max heart rate bins
q1, q2 = df["MaxHR"].quantile([0.33, 0.66])
df["MaxHR_bin"] = pd.cut(
    df["MaxHR"],
    bins=[-np.inf, q1, q2, np.inf],
    labels=["Low", "Medium", "High"]
)

# Dataset Overview and Descriptive Statistics (What am I working with)
print("======== Preview of Data ======== ")
print(f"\nSample of the data:\n{df.head()}")
print(f"\n(Rows, Columns) = {df.shape}")
df.info()
print(f"\nDescribe (numeric data): {df.describe()}")
print("The top is the most common value. ")
print("The freq is the most common value’s frequency.")
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
print(f"\nNumber of duplicated rows: {df.duplicated().sum()}")

# Plotting
plt.figure(figsize=(10 , 6))

# 1) Heart disease by age group
plt.subplot(2, 2, 1) # (n_rows, n_columns, first index)
# Computing cross tabulation of 'age_bin' grouping by 'HeartDisease'
# Normalize to get probabilities
pd.crosstab(df["Age_bin"], df["HeartDisease"],
            normalize="index")[1].plot(kind="bar")
plt.title("P(Heart Disease | Age Group)")
plt.ylabel("Proability")

# 2) ST Slope vs Heart Disease
plt.subplot(2, 2, 2)
pd.crosstab(df["ST_Slope"], df["HeartDisease"],
            normalize="index")[1].plot(kind="bar")
plt.title("P(Heart Disease | ST Slope)")

# 3) Exercise Angina
plt.subplot(2, 2, 3)
pd.crosstab(df["ExerciseAngina"], df["HeartDisease"],
            normalize="index")[1].plot(kind="bar")
plt.title("P(Heart Disease | Exercise Angina)")

# 4 Choleterol
plt.subplot(2, 2, 4)
pd.crosstab(df["Cholesterol_bin"], df["HeartDisease"],
            normalize="index")[1].ploit(kind="bar")


# Execute plot
plt.tight_layout()
plt.show()



