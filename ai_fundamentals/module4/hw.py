#!/usr/bin/env python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import dedent

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from xgboost import XGBClassifier

# Global settings
STATE = 73
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Load cancer dataset w/ Features and Target
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, name="target")

# 0 = Malignant, 1 = Benign
target_names = cancer.target_names

print("Dataset description:")
print(cancer.DESCR)

X.head()
y.head()

# Combining into one dataframe for some reason
df = X.copy()
df["target"] = y
# Assigning target names to the target values instead of 0/1
df["diagnosis"] = df["target"].map(lambda x: target_names[x])

# Problem context and Data summary
print(dedent("""\n
PROBLEM CONTEXT
===============
The breast cancer Wisconsin dataset has numerical features computed via images of fine needle aspirate (FNA) samples of breast masses.
It originated from the University of Wisconsin.  

The classification problem is to predict whether a tumor is malignant or benign based on the measured cell-nuclei characteristics.
This is an important problem to sovle because accurate early classification supports medical decision-making, reducing unnecessary procedures, and help finding dangerous tumors quickly.
             """))

print(dedent(f"""
DATASET OVERVIEW
================
"Feature matrix shape:", {X.shape}
"Target shape:", {y.shape}
"\nFirst five rows:"
{df.head()}
"\nClass counts:"
{df["diagnosis"].value_counts()}
"\nMissing values per column:"
{df.isnull().sum()
}"\nSummary statistics:"
{X.describe().T}
             """))

# ++++++++++ EDA ++++++++++
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="diagnosis")
plt.title("Class Distribution: Benign vs Malignant")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Histogram stuffz
chosen_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean concavity",
    "worst radius"
]

df[chosen_features].hist(bins=20, figsize=(14, 10))
plt.suptitle("Histogram of Selected Breast Cancer Features")
plt.tight_layout()
plt.show()

# Pairplot for subsets
pairplot_feats = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "diagnosis"
]

sns.pairplot(df[pairplot_feats], hue="diagnosis", diag_kind="hist")
plt.suptitle("Pairplot of Selected Features by Diagnosis")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
corr = X.corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

print(dedent(f"""
EXPLATORY DATA ANALYSIS
======================
Dataset has {X.shape[0]} observations and {X.shape[1]} numerica predictor variables.
NO missing data.
Target classes are imbalanced, with benign cases more common than malignant ones.
Histograms and pairplots imply multiple features related to tumor size and shape, like radius, perimeter, area, and concavity, 
differ profoundly between the two dignosis groups. 
The correlation heatmap also shows strong multicolinearity among size-related predictors, especially radius, perimeter, and area measurements.
One important modeling idea is that the predictors include overlapping info, which can increase model complexity and the risk of overfitting.
Also, considering class, since even moderate imbalance can affect how evaluation metrics should be interpreted.
             """))

# Training/Testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # 80/20 rule
    test_size=.2,
    random_state=STATE,
    stratify=y
)

print("TRAIN/TEST SPLIT")
print(dedent(
f"""
X_train shape: {X_train.shape}
X_test shape: {X_test.shape}
Training Class Proportions:
{y_train.value_counts(normalize=True)}
Test Class Proportions:
{y_test.value_counts(normalize=True)}
"""
))

print(dedent(f"""
ROC-AUC measures how well the model differentiates between benign cases as the positive class.
             """))

# HYperparameter selection
print(dedent(f"""
HYPERPARAMETER SELECTION
========================
1) max_depth:
COntrols the maximum depth of each tree in the XGBoost, useful since the dataset has {X.shape[0]} numerical features and the EDA showed multicolinearity among radius, perimeter, and area measurements.

2) learning_rate (step size parameter):
Controls how much each new  tree contributes to the overall boosted model.
Since the dataset is pretty small, this is useful for the learning rate generally makes learning more gradual and improves generalization.
             """))

# Initial model trainging
initial_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.10,
    subsample=1.0,
    colsample_bytree=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    importance_type="gain",
    random_state=STATE
)

initial_model.fit(X_train, y_train)

y_pred_initial = initial_model.predict(X_test)
y_prob_initial = initial_model.predict_proba(X_test)[:, 1]

initial_accuracy = accuracy_score(y_test, y_pred_initial)
initial_auc = roc_auc_score(y_test, y_prob_initial)

print(dedent(f"""
INITIAL MODEL PERFORMANCE
=========================
Initial accuracy: {initial_accuracy:.3f}
Initial ROC-AUC: {initial_auc:.3f}

Classification Report
{classification_report(y_test, y_pred_initial, target_names=target_names)}
             """))

cm_initial = confusion_matrix(y_test, y_pred_initial)
disp_initial = ConfusionMatrixDisplay(
    confusion_matrix=cm_initial,
    display_labels=target_names
)

disp_initial.plot()
plt.title("Initial Model Confusion Matrix")
plt.tight_layout()
plt.show()

RocCurveDisplay.from_estimator(initial_model, X_test, y_test)
plt.title("Initial Model Roc Curve")
plt.tight_layout()
plt.show()

print(dedent(f"""
INITIAL MODEL TRAININIG
======================
The initial XGBoost model was trained on the training set using the chosen hyperparameter values.
I selected accuracy and ROC-AUC as the two evaluation metrics, since accuracy provides a simple metric for overall classification, while ROC-AUC evaluates how well the model separates the classes across possible decision thresholds.
Initial model achieved an accuracy of {initial_accuracy:.3f} and a ROC-AUC
             score of {initial_auc:.3f}.
             """))

# HYperparameter Tuning
param_dist = {
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.10, 0.20, 0.30],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
}

xgb_base = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    importance_type="gain",
    random_state=STATE
)

random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=20,
    scoring="roc_auc",
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=STATE
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

y_pred_tuned = best_model.predict(X_test)
y_prob_tuned = best_model.predict_proba(X_test)[:, 1]

tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
tuned_auc = roc_auc_score(y_test, y_prob_tuned)

print(dedent(f"""
HYPERPARAMETER TUNING
=====================
Results:
Best parameters are {random_search.best_params_}
Best Cross-validation:
{random_search.best_score_:.3f}
Tuned Accuracy: {tuned_accuracy:.3f}
Tuned ROC-AUC: {tuned_auc:.3f}

Tuned Model Classification Report:
{classification_report(y_test, y_pred_tuned, target_names=target_names)}
             """))

cm_tuned = confusion_matrix(y_test, y_pred_tuned)
disp_tuned = ConfusionMatrixDisplay(
    confusion_matrix=cm_tuned,
    display_labels=target_names
)
disp_tuned.plot()
plt.title("Tuned Model Confusion Matrix")
plt.tight_layout()
plt.show()

RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.title("Tuned Model ROC Curve")
plt.tight_layout()
plt.show()

if tuned_accuracy > initial_accuracy or tuned_auc > initial_auc:
    tuning_interpretation = (
        "The tuned model improved at least one of the evaluation metrics relative "
        "to the initial model, suggesting that the hyperparameter search found a "
        "better balance between model complexity and generalization."
    )
else:
    tuning_interpretation = (
        "The tuned model performed similarly to the initial model, suggesting that "
        "the original parameter settings were already strong for this dataset."
    )

print(dedent(f"""
HYPERPARAMETER TUNING (continued)
======================
I used RandomizedSeaerchCV to search over multiple combinations of
             n_estimators, max_depth, learning_rate, subsample, and
             colsample_bytree.
The best combo of hyperparameters I found was {random_search.best_params_}
The initial model achieved an accuracy of {initial_accuracy:.3f} and ROC-AUC of {initial_auc:.3f}.
             Th tuned model achieved accuracy {tuned_accuracy:.3f} and a ROC-AUC of {tuned_auc:.3f}.
{tuning_interpretation}
             """))

# Feature IMportance and INterpretation
# Because importance_type="gain" was specified, these importances
# reflect gain-based feature importance.
importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

top_n = 10
top_importances = importances.head(top_n)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_importances.values, y=top_importances.index)
plt.title("Top 10 XGBoost Feature Importances (Gain)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

top_features = list(top_importances.index[:5])
top_scores = list(top_importances.values[:5])

print(dedent(f"""
FEATURE IMPORTANCE AND INTERPRETATION
====================================
{top_importances}

             The final tuned XGBoost model ranked {top_features[0]}
             ({top_scores[0]:.3f}) and {top_features[1]} ({top_scores[1]:.3f}),
             and {top_features[2]} ({top_scores[2]:.3f}) 
as its three most important predictors. These were followed by {top_features[3]}
({top_scores[3]:.4f}) and {top_features[4]} ({top_scores[4]:.4f}). This suggests
that the model relied most strongly on measurements related to tumor size,
boundary irregularity, and structural complexity, which is medically plausible
because malignant tumors tend to show more irregular and invasive cellular
patterns than benign tumors. The importance values here are gain-based
XGBoost importances, so they reflect how much each feature improved the model's
splits on average rather than implying causation. In a medical context, this
matters because it helps identify which observed tumor characteristics were most
informative for the classification decision.
             """))

# Summary
summary_df = pd.DataFrame({
    "Model": ["Initial XGBoost", "Tuned XGBoost"],
    "Accuracy": [initial_accuracy, tuned_accuracy],
    "ROC-AUC": [initial_auc, tuned_auc]
})

print(dedent(f"""
MODEL SUMMARY
=============
{summary_df}
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
AI Disclosure:
Claude was used to outsource the tedium of print statements, some visualization portions, but mostly to elaborate on the concepts learned for the sake of implementing the code myself.
             """))




