#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    roc_curve)
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
            return pd.read_csv(path, sep="\s+")

        else:
            raise ValueError("File must end with .csv or .dat")

    except Exception as err:
        print(f"\nOOPS! -> {err}")

        return None

# ===== MAIN ======
df = read_file("data/lung_cancer.csv")

# Target and features
X_rf = df.drop("lung_cancer_risk", axis=1) # Need to separate Target from features
y_rf = df["lung_cancer_risk"]

# Splitting data into train/test sets
Xrf_train, Xrf_test, yrf_train, yrf_test = train_test_split(
    X_rf,
    y_rf,
    test_size=0.3, # for 30/70 test/train split
    random_state=73, # Random seed for reproducibility
    stratify=y_rf # deals with class imbalance
)

print(f"Training set: {Xrf_train.shape[0]} samples")
print(f"Test set: {Xrf_test.shape[0]} samples")

# Fitting Random Forest Classifier
# Unlike Logistic Regression, scaling is NOT required because trees split on thresholds so feature magnitudes don't affect the splits.
forest = RandomForestClassifier(
    # Number of trees in forest
    n_estimators=100,
    # n_samples / (n_classes * np.bincount(y))
    class_weight="balanced",
    # NO depth limit
    max_depth=None,
    # Features to consider for best split: sqrt(n_features)
    max_features="sqrt",
    # Mininum number of samples to split (internal node)
    min_samples_split = 2, # Samples @ leaf node
    # Samples on right/left branch required for each leaf node
    min_samples_leaf=5, # 5 was chosen to reduce variance (default=1)
    # Random seed for reproducibility
    random_state=73,
    # Out-of-bag score for estimates generalization w/out using test data
    oob_score=True,
    # Use all CPU cores
    n_jobs=-1
)

# Build trees using training data
forest.fit(Xrf_train, yrf_train)

# Output Out-of-bag score
print(f"\nOOB Score: {forest.oob_score:.3f}")

# Predict class of input sample, where the predicted class is one w/ highest mean probability estimate across trees
y_hat = forest.predict(Xrf_test)
# Predict class probabilities for X_test

# Predicted class prbabilities, returns an array of shape: (n_samples, n_classes), where we care about the CLASS, hence the [:,1]
y_prob = forest.predict_proba(Xrf_test)[:, 1]

# ======  Evaluation MEtrics =======
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

# Confusion MAtrix
conf_mat = confusion_matrix(yrf_test, y_hat)
# Convert to pandas DataFrame
df_mat = pd.DataFrame(
    conf_mat, 
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

# Output Confusion matrix
print(f"COnfusion Matrix:\n{df_mat}")

# Feature importance via Gini Impurity and sorting for importance
importance = pd.DataFrame({
    "Feature": X_rf.columns,
    "Importance": forest.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFeature IMportance (Gini Impurity Reduction):")
print(importance.head(10))
print("Top 3:")
print(importance.head(3)["Feature"].tolist())

# Permutation Importance more reliable than Gini
# Shufflea feature, break association, evaluate performance declines
permute = permutation_importance(
    forest,
    Xrf_test, 
    yrf_test,
    # duh ( ͡° ͜ʖ ͡°)
    scoring="roc_auc",
    # Permutations
    n_repeats=13,
    # Random seed for reproducibility
    random_state=73
)

# Convert to dataframe
perm_df = pd.DataFrame({
    "Feature": X_rf.columns,
    "Importance": permute.importances_mean,
    "Std": permute.importances_std
}).sort_values("Importance", ascending=False)

print("\nPermutation Importance (ROC AUC drop):")
print(perm_df.head(10))

print("\nTop 3 Features (Permutation):")
print(perm_df.head(3)["Feature"].tolist())

# =============== Check for Leakage ===============
# Not sure what to do with this ??
print("\nCheck for leakage:")
print(df.corr(numeric_only=True)["lung_cancer_risk"].sort_values(ascending=False))

# =========== PLOT ==============
# Plotting ROC curve: TPR vs FPR
# TPR = TP/(TP+FN), FPR = FP/(FP+TN)
fpr, tpr, thresholds = roc_curve(yrf_test, y_prob)

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve - Random Forest (Lung Cancer Risk)")
plt.show()










