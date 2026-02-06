#!/usr/bin/env python
"""
1. Exploratory Data Analysis
    * Produce at least four insightful plots
    * Write 3–5 sentences summarising the most important risk factors you observe

2. Bayesian Network Structure Learning
    * Using pgmpy, try at least two different structure-learning algorithms and visualize both resulting graphs
    * Choose the one that makes the most clinical sense, use it for the rest of the project, and briefly justify your final choice (2-4 sentences)

3. Parameter Learning & Clinical Inference
    * Fit the conditional probability tables (CPTs) on the full data, then perform and interpret these five queries (show the numerical result + 2–3 sentence interpretation for each):

        **a) P(HeartDisease=1 | Age_bin='60+', ST_Slope='Flat'**
      
        **b) Same patient but with ExerciseAngina=1**
      
        **c) P(HeartDisease=1 | Cholesterol_bin='High', MaxHR_bin='Low')**
      
        **d) P(HeartDisease=1 | ChestPainType='ATA', ExerciseAngina=0)**
      
        **e) Full diagnostic: P(HeartDisease=1 | Age_bin='60+', ST_Slope='Flat', ExerciseAngina=1, Oldpeak_bin='High')**

4. Turn your Bayesian Network into a probabilistic classifier
    * For each patient, compute P(HeartDisease=1 | all observed features)
    * Predict 1 if probability > 0.5
    * Report accuracy and AUC (optional but encouraged: use a 70/30 train/test split so numbers are realistic)

5. Answer both questions clearly:
    * Why might a hospital or cardiologist prefer your Bayesian Network over a neural network or XGBoost that has 3–5% higher accuracy?
    * Name and briefly describe one real-world medical system or company in 2025 that actually uses Bayesian Networks or Bayesian deep learning in clinical practice (a 30-second Google is allowed – just cite your source).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import dedent


df = pd.read_csv("heart_failure_prediction.csv")

# 1) EDA =========================
# Dataset Overview and Descriptive Statistics (What am I working with)
print("======== Preview of Data ======== ")
print(f"\nSample of the data:\n{df.head()}")
print(f"\n(Rows, Columns) = {df.shape}")
print(f"\nData types:\n {df.dtypes}")
df.info()
print(f"\nDescribe (numeric data): {df.describe()}")
print("The top is the most common value. ")
print("The freq is the most common value’s frequency.")
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
print(f"\nNumber of duplicated rows: {df.duplicated().sum()}")

# Cholesterol shouldn't be zero: Replace zeros with median value
df["Cholesterol"] = df["Cholesterol"].replace(0, df["Cholesterol"].median())
# Resting blood pressure cannot be zero
df["RestingBP"] = df["RestingBP"].replace(0, df["RestingBP"].median())

# Segmant and sort data values into bins
# Bins for Age (since continuous, numeric data)
df["Age_bin"] = pd.cut(
    df["Age"], 
    bins=[0, 39, 50, 120], 
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

# Plotting

plt.figure(figsize=(10, 6))

# 1) Heart Disease by Age group
plt.subplot(2, 2, 1) # (n_rows, n_columns, ?)
pd.crosstab(df["Age_bin"], df["HeartDisease"], normalize="index")[1].plot(kind="bar")
plt.title("P(Heart Disease | Age Group)")
plt.ylabel("Probability")

# 2) ST slope vs Heart Disease
# ST Slope 
plt.subplot(2, 2, 2)
pd.crosstab(df["ST_Slope"], df["HeartDisease"], normalize="index")[1].plot(kind="bar")
plt.title("P(Heart Disease | ST Slope)")

# 3) Exercise Angina
plt.subplot(2, 2, 3)
pd.crosstab(df["ExerciseAngina"], df["HeartDisease"], normalize="index")[1].plot(kind="bar")
plt.title("P(Heart Disease | Exercise Angina)")

# 4) Cholesterol vs MaxHR heatmap
plt.subplot(2, 2, 4)
heatmap = pd.crosstab(
    df["Cholesterol_bin"], df["MaxHR_bin"],
    values=df["HeartDisease"],
    aggfunc="mean"
)
plt.imshow(heatmap, cmap="Reds")
plt.xticks(range(len(heatmap.columns)), heatmap.columns)
plt.yticks(range(len(heatmap.index)), heatmap.index)
plt.colorbar(label="P(Heart Disease)")
plt.title("Risk Heatmap for \nCholesterol and Max Heart Rate")

plt.tight_layout()
plt.show()


# Interpretation
print(dedent(f"""
Older patients (60 and up) show a much higher probabaility of heartdisease compared to younger age groups. 
A flat or down-sloping ST segment indicates an increase risk in heart disease, which is consistent with observations in ECGs.

Exercise Angina profoundly increase the risk, which suggests stress-related caridac problems.

Patients with both high cholesterol and low max hear rate show the highest estimated risk.
             """))




