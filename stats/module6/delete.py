#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, classification_report)
from textwrap import dedent

# ======= MAIN ===========

# 1. Load the dataset
# NOTE: sklearn's load_diabetes is a REGRESSION dataset (continuous target).
# For logistic regression (binary classification), we need to convert the target
# to binary. We'll use the median as the threshold:
# above median => 1 (higher disease progression), below median => 0
data = load_diabetes()

X = pd.DataFrame(data.data, columns=data.feature_names)
y_continuous = pd.Series(data.target, name="target")

# Convert continuous target to binary using the median as threshold
median_target = y_continuous.median()
y = (y_continuous > median_target).astype(int)

print(f"Dataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target median: {median_target}")
print(f"Class distribution:\n{y.value_counts()}")
print()

# 2. Split data into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 3. Fit logistic regression (scale features first)
# Scaling is important bc logistic regression is sensitive to feature magnitudes
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Predictions
y_pred = log_reg.predict(X_test_scaled)
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

# 4. Evaluate
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION EVALUATION")
print("=" * 60)
print(f"\nAccuracy:  {accuracy:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance (coefficients)
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": log_reg.coef_[0]
}).sort_values("Coefficient", ascending=False)

print("\nFeature Coefficients (scaled):")
print(coef_df.to_string(index=False))

# 5. Reflection
print(dedent(f"""
REFLECTION:
Does logistic regression seem like a reasonable AI model for this problem?

Logistic regression achieves an accuracy of {accuracy:.2f} and ROC AUC of {roc_auc:.2f}
on this dataset. {"These are decent results." if accuracy > 0.65 else "These results are modest."}

STRENGTHS for this problem:
- Interpretable: each coefficient tells us how a feature affects the log-odds
  of higher disease progression. For example, bmi likely has a positive
  coefficient, meaning higher BMI increases the predicted odds.
- The binary outcome (above/below median progression) naturally fits the
  logistic regression framework.
- With 10 features and 442 samples, logistic regression is appropriate
  since it doesn't need huge amounts of data like deep learning would.
- Outputs probabilities in [0, 1], unlike OLS which can predict outside bounds.

LIMITATIONS:
- The original target is continuous, so we LOSE information by binarizing it.
  A linear regression on the continuous target might actually be more
  informative for predicting disease progression.
- Logistic regression assumes linearity in the log-odds, which may not hold.
  If the true relationship between features and diabetes progression is
  nonlinear, methods like random forests or gradient boosting could perform
  better.
- Multicollinearity among features (age, bmi, blood pressure, etc.) could
  inflate coefficient variance and reduce interpretability.

CONCLUSION:
Logistic regression is a REASONABLE starting point for this binary classification
task. It provides a solid baseline with interpretable results. However, since
the original data is continuous, a regression model might be more appropriate.
For a strictly binary prediction task, more flexible models could potentially
improve performance, but logistic regression's simplicity and interpretability
make it a good choice for an AI system where understanding the "why" matters.
             """))
