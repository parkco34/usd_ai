#!/usr/bin/env python
from datetime import datetime
from eda_toolkit import flex_corr_matrix
import kagglehub
import math
from matplotlib import gridspec
import matplotlib.pyplot as plt
from model_metrics import summarize_model_performance
from model_metrics import show_roc_curve
from model_metrics import show_confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, chi2_contingency
from textwrap import dedent
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
timestamp_alltimes_start = datetime.now()
path = kagglehub.dataset_download("dhrubangtalukdar/lung-cancer-prediction-dataset")
df = pd.read_csv(path + '/lung_cancer.csv', sep=',', na_values=" ?", skipinitialspace=True)
TARGET = "lung_cancer_risk"

# =========== EDA =============
print("======== Preview of Data ======== ")
print(f"Sample of the data:\n{df.head()}")
print(f"\n(Rows, Columns) = {df.shape}\n")
df.info()
print(f"\nDescribe (numeric data): {df.describe()}")
print(f"\nNumber of duplicated rows: {df.duplicated().sum()}")
print(f"\nNumber of Null Values: {df.isnull().sum().sum()}")
print(f"\nProportion of high-risk patients: {(df[TARGET].sum() / len(df[TARGET])) * 100}%")

# Categorizing columns by type using Claude
BINARY_COLS = [
    "gender", "smoker", "passive_smoking", "occupational_exposure",
    "radon_exposure", "family_history_cancer", "copd", "asthma",
    "previous_tb", "chronic_cough", "chest_pain", "shortness_of_breath",
    "fatigue", "xray_abnormal"
]

# Ordinal columns (documented for reference; not used in modeling)
ORDINAL_COLS = ["income_level", "diet_quality", "healthcare_access"]

CONTINUOUS_COLS = [
    "age", "education_years", "smoking_years", "cigarettes_per_day",
    "pack_years", "air_pollution_index", "bmi", "oxygen_saturation",
    "fev1_x10", "crp_level", "exercise_hours_per_week",
    "alcohol_units_per_week"
]

# =========== Additional Descriptive Stats =============
print("\n======== Descriptive Stats by Risk Class ========")
grouped = df.groupby(TARGET)[CONTINUOUS_COLS].agg(["mean", "std", "median"])
grouped_stacked = grouped.stack(level=0)
grouped_stacked.index.names = ["risk_class", "feature"]
print(grouped_stacked.round(2).to_string())

print("\n======== Skewness & Kurtosis (Continuous Features) ========")
skew_kurt = pd.DataFrame({
    "Skewness": df[CONTINUOUS_COLS].skew(),
    "Kurtosis": df[CONTINUOUS_COLS].kurtosis()
}).round(3)
print(skew_kurt.to_string())

# Color constants
purple = "#723658"
orange = "#ed7952"
EDGE = "black"

# Set seaborn style once for all plots
sns.set(font_scale=1.2, style="dark")
custom_palette = {0: "skyblue", 1: "navy"}

# =========== Plots =============
# ---- Target Distribution ----
fig, ax = plt.subplots(figsize=(6, 6))
sns.countplot(data=df, x='lung_cancer_risk', palette="rocket", ax=ax)
total = len(df)
for p in ax.patches:
    count = int(p.get_height())
    percent = 100 * count / total

    ax.text(
        p.get_x() + p.get_width()/2,
        count + total*0.01,
        f"{count}\n({percent:.1f}%)",
        ha="center"
    )

ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Low Risk (0)", "High Risk (1)"])
ax.set_xlabel("Lung Cancer Risk")
ax.set_title("Target Variable Distribution")
plt.tight_layout()
plt.show()
print("\n\n\n")

# ---- Proportion Mean Plots ----
# Continuous features use the mean to show the average value per risk group.
# Binary (0/1) features use the mean to show the proportion of 1s in each group.

# Adding grid to make it look like one figure
# Sorting by demo, cont and bin (makes it look like 3 figures but its one)
DEMO = ["age"]
CONT = ["pack_years", "oxygen_saturation", "fev1_x10"]
BIN  = ["gender", "asthma", "copd", "chronic_cough", "shortness_of_breath", "smoker"]

# keep only columns that exist
DEMO = [c for c in DEMO if c in df.columns]
CONT = [c for c in CONT if c in df.columns]
BIN  = [c for c in BIN  if c in df.columns]

def panel(ax, cols, title, palette, ylim=None, ylabel="Mean", alpha=0.7):
    # seaborn computes the mean per TARGET internally
    sub = df.melt(id_vars=TARGET, value_vars=cols, var_name="feature", value_name="value")
    sns.barplot(data=sub, x="feature", y="value", hue=TARGET,
                palette=palette, errorbar=None, ax=ax, alpha=alpha, width=0.5)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(title="Lung Cancer Risk", labels=["Low (0)", "High (1)"])

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

panel(axes[0], DEMO, "Demographics", custom_palette, ylabel="Mean")
panel(axes[1], CONT, "Continuous Features", custom_palette, ylabel="Mean")
panel(axes[2], BIN,  "Binary Features", "rocket", ylim=(0, 1),
      ylabel="Proportion")

plt.suptitle("Selected Features by Risk Class", fontsize=15, fontweight="bold")
plt.tight_layout()
# Add border around entire figure
fig.patch.set_edgecolor("black")
fig.patch.set_linewidth(2)
plt.show()
print("\n\n\n")

# ---- Continuous Distributions By Class ----
def features_by_target(df, column_plot, num_cols, w, h, title):
    custom_palette = {0: 'skyblue', 1: 'navy'}

    keys_list = list(column_plot.keys())
    values_list = list(column_plot.values())

    n_plots = len(column_plot)
    n_cols = num_cols
    n_rows = int(math.ceil(n_plots / n_cols))

    gs = gridspec.GridSpec(n_rows, n_cols)
    fig = plt.figure(figsize=(w, h))

    for i in range(n_plots):
        ax = fig.add_subplot(gs[i])
        if values_list[i] == 'hist':
            sns.histplot(
                data=df, x=keys_list[i], hue='lung_cancer_risk',
                palette=custom_palette, bins=8, alpha=0.6, ax=ax
            )
        elif values_list[i] == 'count':
            sns.countplot(
                data=df, x=keys_list[i], hue='lung_cancer_risk',
                palette="rocket", ax=ax
            )
        elif values_list[i] == 'box':
            sns.boxplot(
                data=df, x=keys_list[i], hue='lung_cancer_risk',
                palette="crest", ax=ax, gap=0.2
            )
        else:
            print('Incorrect plot type, please check if '
                  'column_plot has hist, count or box plots only.')
        ax.set_xlabel(keys_list[i])
        ax.set_title(f"{keys_list[i]} By Lung Cancer Risk")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

#Print Count and Histogram Plots
column_plot = {'copd': 'count','chronic_cough':'count','shortness_of_breath':'count',
               'family_history_cancer':'count','age':'hist', 'pack_years':'hist', 'oxygen_saturation': 'hist'}
features_by_target(df, column_plot, 3, 20, 20, 'Distributions by Risk Class')
plt.show()
print("\n\n\n")

#Print Box Plots
column_plot = {'oxygen_saturation':'box', 'pack_years':'box', 'age':'box'}
features_by_target(df, column_plot, 3, 14, 6, 'Outlier Observation by Risk Class')
plt.show()
print("\n\n\n")

# ---- Correlation Heatmap ----
#(Shpaner & Gil, 2024)
df_num = df.select_dtypes(np.number)
flex_corr_matrix(
    df=df,
    cols=df_num.columns.to_list(),
    annot=True,
    cmap="coolwarm",
    figsize=(18, 14),
    title="Lung Cancer Correlation Matrix",
    xlabel_alignment="right",
    label_fontsize=10,
    tick_fontsize=10,
    xlabel_rot=45,
    ylabel_rot=0,
    text_wrap=50,
    vmin=-1,
    vmax=1,
    cbar_label="Correlation Index",
    triangular=True,
)
print("\n\n\n")

# ---- Narrowed Down Correlations with Target ----
narrowed_df = df[["age", "pack_years", "copd", "family_history_cancer",
                  "chronic_cough", "shortness_of_breath", "oxygen_saturation", "lung_cancer_risk"]]
corr = narrowed_df.corr()
fig, ax = plt.subplots(figsize=(8, 6))
target_corr = corr[TARGET].drop(TARGET).sort_values()
colors = [purple if v > 0 else orange for v in target_corr.values]
ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor=EDGE)
ax.set_xlabel("Pearson Correlation with lung_cancer_risk")
ax.set_title("Feature Correlations with Target",
             fontsize=14, fontweight="bold")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.show()
print("\n\n\n")

timestamp_alltimes_end = datetime.now()
print(f"\033[1;32mTotal Execution time for EDA: {timestamp_alltimes_end - timestamp_alltimes_start}\n")

timestamp_logistic_start = datetime.now()
# ====== Significance Testing ========
print("# ====== Significance Testing ========")
# Continuous/discrete features
FEATURES_CONT = ["age", "pack_years", "oxygen_saturation"]
FEATURES_DISC  = ["copd", "family_history_cancer", "chronic_cough", "shortness_of_breath"]

# Low/High risk data
low = df[df[TARGET] == 0]
hi = df[df[TARGET] == 1]

# Bonferroni correction: alpha / number of tests
N_TESTS = len(FEATURES_CONT) + len(FEATURES_DISC)
ALPHA_CORRECTED = 0.05 / N_TESTS

# Iterate thru continuous/discrete features applying relevant tests
for col in FEATURES_CONT:
    x0 = low[col]
    x1 = hi[col]

    t, p = ttest_ind(x0, x1, equal_var=False)

    print(dedent(f"""
{col}\np = {p:.3f}
mean0 = {x0.mean():.3f}
mean1 = {x1.mean():.3f}
                 """))

for col in FEATURES_DISC:
    # build contingency table
    table = pd.crosstab(df[TARGET], df[col])
    table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    # Independence test (Yates' correction disabled; n=5000 is large enough)
    chi2, p_val, dof, expect = chi2_contingency(table, correction=False)

    # Proportions
    lows = table.loc[0, 1] / table.loc[0].sum()
    highs = table.loc[1, 1] / table.loc[1].sum()

    print(dedent(f"""
{col}\np = {p_val:.3f}
{'Significant' if p_val < ALPHA_CORRECTED else 'Not significant'} (Bonferroni alpha = {ALPHA_CORRECTED:.4f})
P({col}=1 | Low) = {lows:.3f}
P({col}=1 | High) = {highs:.3f}
                 """))

print(dedent(f"""
INTERPRETATION
---------------
For the CONTINUOUS features:

AGE:
High risk patients are, on average, older
(mean_high = {hi['age'].mean():.3f} vs
mean_low = {low['age'].mean():.3f})
p < 0.001, meaning a significant difference.

PACK_YEARS:
High risk patients typically have more years smoking.
mean_high = {hi['pack_years'].mean():.3f}
mean_low = {low['pack_years'].mean():.3f}
p < 0.001, so smoking history has a strong association with risk.

OXYGEN SATURATION:
Lower oxygen saturation levels for high risk patients.
mean_high = {hi['oxygen_saturation'].mean():.3f}
mean_low = {low['oxygen_saturation'].mean():.3f}
p < 0.001, which implies many things like, exercise matters.
                 """))

timestamp_alltimes_end = datetime.now()
print(f"\033[1;32mTotal Execution time for Signigicance Testing: {timestamp_alltimes_end - timestamp_alltimes_start}\n")


# =========== Modeling =============
timestamp_logistic_start = datetime.now()

#Define the independent and dependent variables
# Both models use the same 7 features for a fair comparison
X = df[['age', 'pack_years', 'copd', 'family_history_cancer',
        'chronic_cough', 'shortness_of_breath', 'oxygen_saturation']]
y = df[TARGET]

#Split the data into train and test sets: 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Generate a pipeline to scale the data using a standard scaler and fit the logistic regression model
pipe = Pipeline([("scaler", StandardScaler()),("logreg", LogisticRegression())])
model1 = pipe.fit(X_train, y_train)

#Calculate the prediction and probability of the dependent variable
y_pred1 = pipe.predict(X_test)
y_prob1 = pipe.predict_proba(X_test)[:, 1]

#Print model metrics
print("Accuracy:", accuracy_score(y_test, y_pred1))

timestamp_logistic_end = datetime.now()
print(f"\033[1;32mTotal Execution time for Logistic Regression Model: {timestamp_logistic_end - timestamp_logistic_start}\n")

# Sample patient: 30 yo, pack_years=20, no COPD or family history
# or shortness of breath, existing chronic cough, O2 sat of 98.
patient = pd.DataFrame([{
    'age': 30,
    'pack_years': 20,
    'copd': 0,
    'family_history_cancer': 0,
    'chronic_cough': 1,
    'shortness_of_breath': 0,
    'oxygen_saturation': 98
}])

risk_prob = pipe.predict_proba(patient)[0][1]
print("Predicted risk (Logistic Regression):", round(risk_prob, 2))

timestamp_forest_start = datetime.now()

def rf_model(random_state=73):
    """
    Baseline Random Forest model
    """
    return RandomForestClassifier(
        # Number of Decision trees in forest (500)
        n_estimators=500,
        # n_samples / (n_classes * np.bincount(y))
        class_weight="balanced",
        # NO depth limit
        max_depth=None,
        # Subset of features to consider for best split: sqrt(n_features)
        max_features="sqrt",
        # Minimum number of samples to split (internal node)
        min_samples_split=2,
        # Samples on right/left branch required for each leaf node
        min_samples_leaf=5,  # 5 was chosen to reduce variance (default=1)
        # Random seed for reproducibility
        random_state=random_state,
        # Out-of-bag score for estimates generalization w/out using test data
        oob_score=True,
        # Use all CPU cores
        n_jobs=-1
    )

# Fitting Random Forest Classifier
# Unlike Logistic Regression, scaling is NOT required because trees split on thresholds so feature magnitudes don't affect the splits.
forest = rf_model()
model2 = forest.fit(X_train, y_train)
y_pred2 = forest.predict(X_test)
y_prob2 = forest.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("OOB Score:", round(forest.oob_score_, 4))

timestamp_forest_end = datetime.now()
print(f"\033[1;32mTotal Execution time for Random Forest Model: {timestamp_forest_end - timestamp_forest_start}\n")

# Reuse same patient for RF prediction
risk_prob = forest.predict_proba(patient)[0][1]
print("Predicted risk (Random Forest):", round(risk_prob, 2))

#(Shpaner, 2025)
model_titles = ["Logistic Regression", "Random Forest"]

model_performance = summarize_model_performance(
    model=[model1, model2],
    model_title=model_titles,
    X=X_test,
    y=y_test,
    model_type="classification",
    return_df=True
)

print(model_performance)

#Confusion matrix
#(Shpaner, 2025)
show_confusion_matrix(
    model=[model1, model2],
    X=X_test,
    y=y_test,
    model_title=model_titles,
    cmap="Blues",
    text_wrap=20,
    subplots=True,
    n_cols=2,
    n_rows=1,
    figsize=(6, 6),
)

#ROC Curve
#(Shpaner, 2025)
show_roc_curve(
    model=[model1, model2],
    X=X_test,
    y=y_test,
    model_title=model_titles,
    decimal_places=2,
    n_cols=2,
    n_rows=1,
    curve_kwgs={
        "Logistic Regression": {"color": "blue", "linewidth": 2},
        "Random Forest": {"color": "black", "linewidth": 2},
    },
    linestyle_kwgs={"color": "red", "linestyle": "--"},
    subplots=True,
)

with pm.Model(coords=coords) as model:
    b0 = pm.Normal("b0", 0, 1)

    # Link coefficients to named dimensions
    bc = pm.Normal("bc", 0, 1, dims="cont")
    bb = pm.Normal("bb", 0, 1, dims="bin")

    p = pm.math.sigmoid(
        b0
        + pm.math.dot(Xc, bc)
        + pm.math.dot(Xb, bb)
    )

    pm.Bernoulli("y_obs", p=p, observed=y)

    idata = pm.sample(
        2000,
        tune=1000,
        target_accept=0.9,
        return_inferencedata=True
    )

# Summary
az.summary(idata, hdi_prob=0.95)

# Forest plot with variable labels
az.plot_forest(
    idata,
    var_names=["bc", "bb"],
    combined=True,
    hdi_prob=0.95
)
