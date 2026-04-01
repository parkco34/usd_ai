#!/usr/bin/env python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')
target_names = iris.target_names

print("Dataset Description:")
print(iris.DESCR)

X.head()

y.head()

#Combine into one DataFrame for easier analysis
df = X.copy()
df['target'] = y
df['species'] = df['target'].map(lambda x: target_names[x])

sns.pairplot(df, hue="species")
plt.suptitle("Pairplot of Iris Features", y=1.02)
#plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

coeff_df = pd.DataFrame(model.coef_, columns=iris.feature_names)
coeff_df['class'] = target_names
print("\nFeature Coefficients per Class:")
print(coeff_df)

breakpoint()
