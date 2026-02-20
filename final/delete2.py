import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, classification_report,
                             confusion_matrix)
from textwrap import dedent

# ======= MAIN ===========

# 1. Load the dataset
df = pd.read_csv("/Users/whitney/usd_ai/final/data/lung_cancer.csv")

# Target and features
X = df.drop("lung_cancer_risk", axis=1)
y = df["lung_cancer_risk"]

print(f"Dataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Class distribution:\n{y.value_counts()}")
print(f"Class balance: {y.mean():.2%} positive (risk=1)")
print()

# 2. Split data into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 3. Fit Random Forest
# NOTE: Random forests are ensemble methods that build many decision trees
# and aggregate their predictions (majority vote for classification).
# Unlike logistic regression, scaling is NOT required bc trees split on
# thresholds, so feature magnitudes don't affect the splits.
rf = RandomForestClassifier(
    n_estimators=100,       # number of trees in the forest
    max_depth=None,         # let trees grow fully (no depth limit)
    min_samples_split=2,    # minimum samples to split an internal node
    min_samples_leaf=1,     # minimum samples at a leaf node
    random_state=42,
    n_jobs=-1               # use all CPU cores for parallel training
)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# 4. Evaluate
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("RANDOM FOREST CLASSIFIER EVALUATION")
print("=" * 60)
print(f"\nAccuracy:  {accuracy:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"],
                         columns=["Pred 0", "Pred 1"])
print("Confusion Matrix:")
print(cm_df)

# Feature importance (Gini importance from the forest)
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFeature Importances (Gini / Mean Decrease in Impurity):")
print(importance_df.to_string(index=False))

# 5. Reflection
top3 = importance_df.head(3)["Feature"].tolist()

print(dedent(f"""
REFLECTION:
Does a Random Forest seem like a reasonable AI model for this problem?

Random Forest achieves an accuracy of {accuracy:.2f} and ROC AUC of {roc_auc:.2f}
on the lung cancer risk dataset. {"These are strong results." if accuracy > 0.80 else "These are decent results." if accuracy > 0.65 else "These results are modest."}

WHY RANDOM FOREST FOR THIS PROBLEM:
------------------------------------
The dataset has {X.shape[1]} features including both continuous variables
(age, bmi, pack_years, crp_level, etc.) and binary indicators
(smoker, family_history_cancer, copd, asthma, etc.).

Random Forests handle this mixed-type data naturally because decision
trees split on thresholds, so they don't care about scale or distribution.
No need for StandardScaler or one-hot encoding here.

STRENGTHS:
- Handles nonlinear relationships between features and lung cancer risk.
  For example, smoking_years and pack_years likely have a nonlinear,
  threshold-like effect on risk, which trees capture naturally.
- Built-in feature importance via Gini impurity, telling us which
  variables matter most. The top 3 features are: {', '.join(top3)}.
- Resistant to overfitting (compared to a single decision tree) because
  each tree sees a bootstrap sample and a random subset of features.
  This is the "wisdom of the crowd" effect, also called bagging.
- No assumptions about the data distribution (unlike logistic regression
  which assumes linearity in the log-odds).

LIMITATIONS:
- Less interpretable than logistic regression. We get feature importances
  but NOT coefficients that tell us the direction/magnitude of effect.
  For medical applications, a doctor might want to know "how much does
  10 more pack_years increase risk?" which logistic regression answers
  more directly.
- Can overfit on noisy features if max_depth is not controlled.
  With max_depth=None, trees grow fully which risks memorizing noise,
  but the ensemble averaging mitigates this.
- Gini importance can be biased toward high-cardinality features
  (continuous variables get more split opportunities). Permutation
  importance would be a more robust alternative.

CONCLUSION:
Random Forest is a STRONG choice for predicting lung cancer risk.
The ensemble approach handles the complex interactions between risk
factors (smoking + age + family history) that a linear model might miss.
For a clinical decision support system, we might combine this with
logistic regression: use RF for prediction accuracy, and logistic
regression for interpretability and explaining risk factors to patients.
             """))

