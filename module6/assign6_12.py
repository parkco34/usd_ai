#!/usr/bin/env python
"""
For the `UN` data file at the book’s website (see Exercise 1.24), construct a multiple regression model predicting Internet using all the other variables. Use the concept of multicollinearity to explain why adjusted $R^2$ is not dramatically greater than when GDP is the sole predictor. Compare the estimated GDP effect in the bivariate model and the multiple regression model and explain why it is so much weaker in the multiple regression model.
"""
import numpy as np
import pandas as pd
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

# ======= MAIN ===========
df = read_file("./UN.dat")

# Target var
y = df["Internet"]

# Numeric predictors
preds = [c for c in df.columns if c not in ["Nation", "Internet"]]
X = df[preds]

# Bivariate model
X_gdp = sm.add_constant(df[["GDP"]])
model_gdp = sm.OLS(y, X_gdp).fit()

print(model_gdp.summary())

# ====== Multiple Regression ======
X_all_const = sm.add_constant(X)
model_all = sm.OLS(y, X_all_const).fit()

print(model_all.summary())

# Comparing
b_gdp_bivar = model_gdp.params["GDP"]
b_gdp_multi = model_all.params["GDP"]

print(dedent(f"""
Bivariate:
    b_gdp = {b_gdp_bivar:.3f}
    R^2 = {model_gdp.rsquared:.3f}
    Adjusted R^2 = {model_gdp.rsquared:.3f}

Multiple (all predictors)a:
    b_GDP = {b_gdp_multi:.4f}
    R^2 = {model_all.rsquared:.4f}
    Adjusted R^2 = {model_all.rsquared_adj:.4f}

The adjusted R^2 isn't dramatically larger with all predictors because of Multicollinearity.
Predictors are highly correlated with HDI, negatively correlated with GII and
             fertility.  Becaause these predictors share quite a bit of
             overlapping information, adding them doesn't explain new
             variability in internet outside what GDP already explains.

Adjusted R^2 penalizes for each additional predictor so if the additional ones
             don't add much in terms of explanatory power, adjusted R^2 doesn't improve much.  It could even decrease!

The GDP effect is weaker in multiple regression because, 
in the bivariate model, b_gdp = {b_gdp_bivar:.3f}, which means each unit increase in GDP predicts a {b_gdp_bivar:.3f} increase in internet usage.

In the multiple regression model, the other variables are included, so GDP only
             deals with its unique partial effect on internet....  And since
             GDP shares a lot of information with those variables
             (multicollinearity), its unique contributions decreases.
             """))

# Correlation matrix and stuff
print("\nCorrelation matrix of predictors:")
print(X.corr().round(3))


#breakpoint()



