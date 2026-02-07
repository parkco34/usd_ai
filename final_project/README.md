# Lung Cancer Risk Analysis  
### Final Team Project – Probability and Statistics for Artificial Intelligence (AAI-500)

---

## Project Overview
This project applies statistical analysis and predictive modeling to a lung cancer dataset in order to understand how behavioral and physiological risk factors relate to lung cancer diagnosis. Using probability, inferential statistics, and classification models, we aim to identify which features are most strongly associated with lung cancer risk and assess model performance under uncertainty.

This analysis demonstrates how foundational statistics support AI-driven healthcare decision making, particularly in early risk screening contexts.

---

## Team Members
| Name | Contributions |
|-----|--------------|
| Member 1 | Data cleaning, EDA |
| Member 2 | Modeling and evaluation |
| Member 3 | Report writing and presentation |

---

## Dataset Description
- **Dataset name:** Lung Cancer Prediction Dataset  
- **Source:** Kaggle  
  https://www.kaggle.com/datasets/dhrubangtalukdar/lung-cancer-prediction-dataset
- **Observations:** ~300 records  
- **Features:** 15 input features + 1 target variable  
- **Target Variable:**  
  - `LUNG_CANCER` (Binary: Yes / No)

### Feature Summary
The dataset includes demographic, behavioral, and symptom-based variables such as:
- Age
- Gender
- Smoking
- Alcohol consumption
- Chronic disease
- Fatigue
- Wheezing
- Chest pain
- Shortness of breath
- Anxiety
- Peer pressure

Most predictors are binary (Yes/No), with `Age` as a continuous variable.

### Inclusion Criteria
- Binary classification task appropriate for logistic regression
- Sufficient feature count (>10)
- Healthcare relevance
- Appropriate sample size for statistical inference

---

## Project Structure

