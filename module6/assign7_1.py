#!/usr/bin/env python
"""
For the `Houses` data file described in Section 7.1.3, consider $Y$ = selling price, $x_1$ = tax bill (in dollars), and $x_2$ = whether the house is new.
(a) Form the scatterplot of $y$ and $x_1$. Does the normal GLM structure of constant variability in $y$ seem approproate? If not, how does it seem to be violated?
(b) Using the identity link function, fit the (i) normal GLM, (ii) gamma GLM. For each model, interpret the effect of $x_2$.
(c) For each model, describe how the estimated variability in selling prices varies as the mean selling price varies from 100 thousand to 500 thousand dollars.
(d) Which model is preferred according to AIC?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import dedent
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
df = read_file("./Houses.dat")

# a) ======= Scatterplot ========
sns.set_style("whitegrid", {'grid.color': 'black'})
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="taxes", y="price")
plt.title("Selling Price vs Tax Bill")
plt.xlabel("Tax Bill (dollars)")
plt.ylabel("Selling Price (thousands $)")
plt.show()

print(dedent(f"""
The scatter plot shows the variability in selling price increase sa the tax bill increases.  The spread of y values expands out to the right, which violates the constant variabililty assumptions of the normal GLM.

The variance seems to increase with the mean, implying there might be a model,
             where the variance is proprtional to a function of the mean (Gamma
            dist.).
"""))

# b) =========== GLMs w/ Identity link ========
# Normal GLM
normal_glm = smf.glm(
    "price ~ taxes + new",
    data=df,
    family=sm.families.Gaussian()).fit()

print(normal_glm.summary())

# Gamma GLM
gamma_glm = smf.glm(
    "price ~ taxes + new",
    data=df,
    family=sm.families.Gamma(link=sm.families.links.Identity())).fit()

print(gamma_glm.summary())

# Coefficients
b_new_normal = normal_glm.params["new"]
b_new_gamma = gamma_glm.params["new"]

print(dedent(f"""
Normal GLM:
    b_new = {b_new_normal:.2f}
    A new house has a predicted selling price that is ${b_new_normal:.2f} thousand
  {"higher" if b_new_normal > 0 else "lower"} than a non-new house, controlling for tax bill.

Gamma GLM:
  b_new = {b_new_gamma:.2f}
  A new house has a predicted selling price that is ${b_new_gamma:.2f} thousand
  {"higher" if b_new_gamma > 0 else "lower"} than a non-new house, controlling for tax bill.

Both use the identity link, so the interpretation is additive in both cases.
The coefficient represents the expected change in price (thousands $) for
new vs. not-new houses, holding taxes constant.
    
             """))

# c) ======== Variabliilty at different mean values
# Normal GLM
# Estimated sigma^2 from the normal GLM
sig2_norm = normal_glm.scale

# Gamma GLM: Var(Y) = phi * mu^2, where phi is the scale parameter
phi_gamma = gamma_glm.scale

mu_100 = 100
mu_500 = 500

# Normal GLM variance is constant
var_normal_100 = sig2_norm
var_normal_500 = sig2_norm

# Gamma GLM variance: Var(Y) = phi * mu^2
var_gamma_100 = phi_gamma * mu_100**2
var_gamma_500 = phi_gamma * mu_500**2

print(dedent(f"""
Estiamted variabililty @ different mean selling prices

Normal GLM:
    var(Y) = sig^2 = {sig2_norm:.2f} 
    SD at mu=100K: {np.sqrt(var_normal_100):.2f} thousand
    SD at mu=500K: {np.sqrt(var_normal_500):.2f} thousand

Gamma GLM:
    Var(Y) = phi * mu^2, where phi = {phi_gamma:.6f}
    At mu = 100K: Var = {phi_gamma:.6f} * 100^2 = {var_gamma_100:.2f}, SD = {np.sqrt(var_gamma_100):.2f} thousand
    At mu = 500K: Var = {phi_gamma:.6f} * 500^2 = {var_gamma_500:.2f}, SD = {np.sqrt(var_gamma_500):.2f} thousand

The Gamma GLM predicts a lot more variability at higher mean prices, and the SD
             @ mu=500K is 5 times the SD @ mu=100K, matching the spread's shape
             in the scatterplot.  
The normal GLM completey misses this. 
             """))

# d) ================= AIC Comapring =========
aic_normal = normal_glm.aic
aic_gamma = gamma_glm.aic

print(dedent(f"""
d) AIC comparison:
Normal GLM AIC = {aic_normal:.1f}
Gamma GLM AIC  = {aic_gamma:.1f}

The {"Gamma" if aic_gamma < aic_normal else "Normal"} GLM has the lower AIC
({min(aic_normal, aic_gamma):.1f} vs {max(aic_normal, aic_gamma):.1f}),
so it is PREFERRED.

This makes sense given the heteroscedasticity we observed in part (a).
The Gamma GLM better captures the increasing variance with the mean,
which the normal GLM's constant variance assumption fails to model.
             """))







