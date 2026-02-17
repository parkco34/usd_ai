#!/usr/bin/env python
"""
In this exercise, you will build a simple AI classification model using logistic regression to predict whether a patient has diabetes. You will use the Diabetes dataset available directly from sklearn.datasets.### Task

Load the diabetes dataset, explore the features, and build a logistic regression model that predicts the presence of diabetes.
Then, discuss whether logistic regression is appropriate for this AI task based on what we have covered in class.
### Your Steps

1. Load the diabetes dataset from sklearn.

2. Split the data into training and test sets.

3. Fit a logistic regression model. *Hint: Scale the features if necessary.*

4. Evaluate accuracy, f1 score, precision, recall, and roc auc on the test set.

5. Write a short reflection: does logistic regression seem like a reasonable AI model for this problem? Why or why not?
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, classification_report)
from textwrap import dedent

# 1. Load the dataset
data = load_diabetes()

# Variables
X = pd.DataFrame(data.data, columns=data.feature_names)
# Continuous target
y_cont = pd.Series(data.target, name="target")

# Convert continuous target ot binary using median threshold
median_target = y_cont.median()
y = (y_cont > median_target).astype(int)

print(f"Dataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target median: {median_target}")
print(f"Class distribution:\n{y.value_counts()}")

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

# Outputs
print(dedent(f"""
\nAccuracy: {accuracy:.3f}
F1 SCore: {f1:.3f}
Precision: {precision:.3f}
Recall: {recall:.3f}
roc auc: {roc_auc:.3f}
             """))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance (coefficients)
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": log_reg.coef_[0]
}).sort_values("Coefficient", ascending=False)

print("\nFeature Coefficients (scaled):")
print(coef_df.to_string(index=False))

# 5. Refection
print(dedent(f"""
\nREFLECTION:
             Logistic Regression has an accuracy of {accuracy:.3f} and ROC AUC
             of {roc_auc:.3f}.
{"These are reasonable results " if accuracy > .65 else "Results not so profound." }

ADVANTAGES FOR THIS PARTICULAR PROBLEM INCLUDE:
-------------------------------------------------
INTERPRETABILITY:
    Each coefficient tells us how a feature affects the log-odds of higher disease progression.

Binary outcomes naturally fit the logstic regression model, and with 10
             features and 442 samples, it's proper to use since it doesn't
             require a ton of data like deep learning.

LIMITATIONS:
-------------
Original targetn is continuous and thus lose information when we do the
             encoding.
Linear Refression would be better for a continuous target.
Muticollinaerity among features could inflate the coefficient variance,
             actually reducing interpretability.
             """))





