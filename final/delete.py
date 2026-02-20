#!/usr/bin/env python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)
from sklearn.inspection import permutation_importance
from textwrap import dedent


def read_file(path):
    """
    Reads file
    ----------------------------
    INPUT:
        path: (str)

    OUTPUT:
        (pd.DataFrame)
    """
    try:
        if path.endswith(".csv"):
            return pd.read_csv(path)

        elif path.endswith(".dat"):
            return pd.read_csv(path, sep=r"\s+")

        else:
            raise ValueError("File must end with .csv or .dat")

    except Exception as err:
        print(f"\nOOPS! -> {err}")
        return None


# ===== MAIN ======
df = read_file("data/lung_cancer.csv")

# Safety checks (small + non-invasive)
if df is None:
    raise FileNotFoundError("Could not load 'data/lung_cancer.csv'.")

if "lung_cancer_risk" not in df.columns:
    raise KeyError("Target column 'lung_cancer_risk' not found in dataset columns.")

if df.isna().sum().sum() > 0:
    print("\nWARNING: Missing values detected. sklearn RandomForest does NOT accept NaNs.")
    print("Missing values per column (non-zero only):")
    miss = df.isna().sum()
    print(miss[miss > 0].sort_values(ascending=False))

# Target and features
X_rf = df.drop("lung_cancer_risk", axis=1)
y_rf = df["lung_cancer_risk"]

# Splitting data into train/test sets
Xrf_train, Xrf_test, yrf_train, yrf_test = train_test_split(
    X_rf,
    y_rf,
    test_size=0.3,     # 30/70 test/train split
    random_state=73,   # random seed for reproducibility
    stratify=y_rf      # preserves class proportions in train/test
)

print(f"Training set: {Xrf_train.shape[0]} samples")
print(f"Test set: {Xrf_test.shape[0]} samples")

# Fitting Random Forest Classifier
# Unlike Logistic Regression, scaling is NOT required because trees split on thresholds so feature magnitudes don't affect splits.
forest = RandomForestClassifier(
    n_estimators=100,          # number of trees
    class_weight="balanced",   # handles class imbalance via inverse-frequency weighting
    max_depth=None,            # no depth limit
    max_features="sqrt",       # sqrt(n_features) per split (decorrelates trees)
    min_samples_split=2,       # min samples required to split an internal node
    min_samples_leaf=5,        # min samples required at a leaf node (variance control)
    random_state=73,
    oob_score=True,            # out-of-bag estimate of generalization
    n_jobs=-1                  # use all CPU cores
)

# Build trees using training data
forest.fit(Xrf_train, yrf_train)

# OOB score (you turned it on; include it in your write-up)
print(f"\nOOB Score: {forest.oob_score_:.3f}")

# Predict class labels
y_hat = forest.predict(Xrf_test)

# Predicted class probabilities, shape: (n_samples, n_classes); we take P(class=1)
y_prob = forest.predict_proba(Xrf_test)[:, 1]

# ====== Evaluation Metrics =======
accuracy = accuracy_score(yrf_test, y_hat)
f1 = f1_score(yrf_test, y_hat)
precision = precision_score(yrf_test, y_hat)
recall = recall_score(yrf_test, y_hat)
roc_auc = roc_auc_score(yrf_test, y_prob)

print(f"\nAccuracy:  {accuracy:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"ROC AUC:   {roc_auc:.3f}")

print("\nClassification Report:")
print(classification_report(yrf_test, y_hat))

# Confusion Matrix
conf_mat = confusion_matrix(yrf_test, y_hat)

df_mat = pd.DataFrame(
    conf_mat,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

print(f"\nConfusion Matrix:\n{df_mat}")

# ===== ROC Curve (plot) =====
# ROC curve is TPR vs FPR as the decision threshold varies.
# TPR = TP/(TP+FN), FPR = FP/(FP+TN)
fpr, tpr, thresholds = roc_curve(yrf_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve - Random Forest (Lung Cancer Risk)")
plt.show()

# ===== Feature Importance (Gini) =====
importance = pd.DataFrame({
    "Feature": X_rf.columns,
    "Importance": forest.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFeature Importance (Gini Impurity Reduction):")
print(importance.head(10))

print("\nTop 3 Features (Gini):")
print(importance.head(3)["Feature"].tolist())

# ===== Permutation Importance (more trustworthy than Gini) =====
# Idea: shuffle a feature column -> break its association -> see how much performance drops.
perm = permutation_importance(
    forest,
    Xrf_test,
    yrf_test,
    n_repeats=10,
    random_state=73,
    scoring="roc_auc"
)

perm_df = pd.DataFrame({
    "Feature": X_rf.columns,
    "Importance": perm.importances_mean,
    "Std": perm.importances_std
}).sort_values("Importance", ascending=False)

print("\nPermutation Importance (ROC AUC drop):")
print(perm_df.head(10))

print("\nTop 3 Features (Permutation):")
print(perm_df.head(3)["Feature"].tolist())

# ===== Short interpretation block (drop into report) =====
print(dedent(f"""
\nRandom Forest Interpretation (for write-up)
-------------------------------------------
- Random Forest = average of many decision trees built on bootstrap samples.
- At each split, only a random subset of features is considered (max_features="sqrt"),
  which decorrelates trees and reduces variance.
- OOB Score (~generalization estimate): {forest.oob_score_:.3f}
- Test ROC AUC: {roc_auc:.3f}

Top predictors (Permutation Importance):
{perm_df.head(5)["Feature"].tolist()}
"""))

