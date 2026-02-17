#!/usr/bin/env python
"""
The `Students` data file shows responses on variables summarized in Exercise 1.2.
(a) Fit the linear model using *hsgpa* = high school GPA, *tv* = weekly hours watching TV, and *sport* = weekly hours participating in sports as predictors of *cogpa* = college GPA. Report the prediction equation. What do the *P*-values suggest?
(b)  Summarize the estimated effect of *hsgpa.*
(c) Report and interpret $R^2$, adjusted $R^2$, and the multiple correlation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
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

# ====== MAIN =======
df = read_file("./Students.dat")

# Predictors
X = df[["hsgpa", "tv", "sport"]]
# Target
y = df["cogpa"]

# a) ======= Fit the linear model ========
# Add constant (intercept) for statsmodels OLS
X_const = sm.add_constant(X)

# Fit OLS
model = sm.OLS(y, X_const).fit()

# Summary
print(model.summary())

# Pull out the coefficients
b0 = model.params["const"]
b_hsgpa = model.params["hsgpa"]
b_tv = model.params["tv"]
b_sport = model.params["sport"]

print(dedent(f"""
Prediction Equation: 
cogpa_hat = {b0:.3f} + {b_hsgpa:.3f} * hsgpa + {b_tv:.3f} * tv + {b_sport:.3f} * sport

P-values:
p_hsgpa = {model.pvalues["hsgpa"]:.3f}
p_tv = {model.pvalues["tv"]:.3f}
p_sport = {model.pvalues["sport"]:.3f}

INTERPRETATION
--------------
             The p-value for hsgpa is {model.pvalues["hsgpa"]:.3f}, which is
 {"< 0.05 (statistically significant)" if model.pvalues['hsgpa'] < 0.05 else ">= 0.05 (not statistically significant)"}.
This suggests high school GPA {"IS" if model.pvalues['hsgpa'] < 0.05 else "is NOT"} a significant predictor of college GPA.

The p-value for tv is {model.pvalues['tv']:.3f}, which is {"< 0.05 (statistically significant)" if model.pvalues['tv'] < 0.05 else ">= 0.05 (not statistically significant)"}.
The p-value for sport is {model.pvalues['sport']:.3f}, which is {"< 0.05 (statistically significant)" if model.pvalues['sport'] < 0.05 else ">= 0.05 (not statistically significant)"}.

So hsgpa seems to be the main driver, while tv and sport
{"have" if model.pvalues['tv'] < 0.05 or model.pvalues['sport'] < 0.05 else "do NOT have"} strong evidence of being useful predictors
when hsgpa is already in the model.
             """))

# b) ========= Estimated efect of hsgpa ======
print(dedent(f"""
Estimated effect of hsgpa:
b_hsgpa = {b_hsgpa:.3f}.

INTERPRETATION
---------------
For every unit increase in highschool gpa, the predicted college gpa increases
             by {model.conf_int().loc["hsgpa"][0]:.3f}, 
The 95% confidence interval for the hsgpa coefficient is:
[{model.conf_int().loc['hsgpa'][0]:.3f}, {model.conf_int().loc['hsgpa'][1]:.3f}]

Since this interval {"does NOT contain 0" if model.conf_int().loc['hsgpa'][0] > 0 or model.conf_int().loc['hsgpa'][1] < 0 else "contains 0"},
we {"can" if model.conf_int().loc['hsgpa'][0] > 0 or model.conf_int().loc['hsgpa'][1] < 0 else "cannot"} conclude the effect is statistically significant at alpha = 0.05.
             """))

# c) ======== R^2 and Multiple Correlation =======
R2 = model.rsquared
adj_R2 = model.rsquared_adj
multiple_corr = np.sqrt(R2)

print(dedent(f"""
R^2 = {R2:.3f}
Adjusted R^2 = {adj_R2:.4f}
Multiple Correlation (R) = sqrt(R^2) = {multiple_corr:.3f}

INTERPRETATION
--------------
             R^2 = {R2:.3f} means that {R2*100:.1f}% of the variability in college gpa is explained the linear model with hsgpa, tv, and sport, as predictors.
The remaining {(1-R2)*100:.1f}% is UNEXPLAINED.

Adjusted R^2 = {adj_R2:.3f} penalizes for the number of predictors.
It's {"close to" if abs(R2 - adj_R2) < 0.05 else "noticeably lower than"} R^2, which suggests
{"the extra predictors aren't adding much noise" if abs(R2 - adj_R2) < 0.05 else "some predictors may not be contributing much"}.

The multiple correlation R = {multiple_corr:.4f} is the correlation between
the observed college GPA values and the fitted (predicted) values from the model.
             
             """))







