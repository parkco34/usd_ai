#!/usr/bin/env python
"""
Refer to the example in Section 6.2.5 of the crime rate in Florida counties.
(a) Explain what it means when we say these data exhibit *Simpson’s paradox*. What could cause this change in the direction of the association between crime rate and education
when we adjust for urbanization?
(b) Using the Florida data file, construct the scatterplot between *x* = education (HS) and *y* = income (Income), for which the correlation is 0.79. If we had data at the individual
level as well as aggregated for a county, sketch a scatterplot to show that at that level the correlation could be much weaker. So, predictions about individuals based on the
behavior of aggregate groups, known as the *ecological fallacy*, can be quite misleading.
(c) Refer to (b), in which *x* falls between 54.5 and 84.9. Is it sensible to use the least squares line to predict a county’s median income if *x* = 0? Sketch a hypothetical true relationship between *x* and *E(Y)* to show the danger of *extrapolation*, using a fitted line to predict *E(Y)* far from the observed *x* values.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def sample_mean(var):
    """
    Returns the mean of the input variable (sample mean in this case).
    """
    # INput validation
    if not isinstance(var, pd.Series):
        raise TypeError("Input must be a pandas Series!")

    return var.mean()

def sample_stats(x, y):
    """
    Sums of squares/cross-products
    ---------------------------------------------
    INPUT:
        x: (pd.Series)
        y: (pd.Series)
    
    OUTPUT:
        (x_bar, y_bar, sxx, sxy, syy): (tuple)
    """
    # Sample means
    x_bar, y_bar = sample_mean(x), sample_mean(y)

    # Sample standard deviations
    sxx = np.sum((x - x_bar)**2)
    sxy = np.sum((x - x_bar) * (y - y_bar))
    syy = np.sum((y - y_bar)**2)

    return x_bar, y_bar, sxx, sxy, syy

def linear_fit(x_bar, y_bar, sxx, sxy):
    """
    Fits the model using least squares.
    
    WHY THE MATH, MATHS:
    -------------------
    b1_hat = sxy / sxx
        = (sum((xi - x_bar) * (yi - y_bar)) / sum((xi - x_bar)^2)
    b0_hat = y_bar - b1_hat * x_bar
    --------------------------------------------------------
    INPUT:
        x_bar: (np.float)
        y_bar: (np.float)
        sxx: (np.float)
        sxy: (np.float)

    OUTPUT:
        (b0_hat, b1_hat): (tuple)
    """
    # Estimate parameters
    b1_hat = sxy / sxx
    b0_hat = y_bar - b1_hat * x_bar

    return b0_hat, b1_hat

def correlation(Sxx, Sxy, Syy):
    """
    Computes sample correlation (r).

    r = Sxy / sqrt(Sxx * Syy)
    ----------------------------------
    INPUT:
        Sxx: (np.float)
        Sxy: (np.float)
        syy: (np.float)
        
    OUTPUT:
        r: (np.float) Correlation
    """
    # Division by zero case
    if np.sqrt(Sxx * Syy) != 0:
        return Sxy / np.sqrt(Sxx * Syy)

    else:
        print("Cannot divide by zero!")
        return None

# ======= Main =======
# a)======= Simpson's Paradox =========
print(dedent(f"""
Simpson's Paradox is when the direction of an association reverses after adjusting for a confounding variable, which is a variable that is correlated with both the dependent variable and the independent variable.
Here, the marginal association between crime rate and education is positive,
meaning counties with higher education seem to exhibit higher crime rates.
When we adjust for urbanization, the association flips sign!  In this example, urbanization is the confounding variable.
This makes sense since more people in urban areas, more education opportunities and thus more crime.
             """))

df = read_file("./Florida.dat")

# Variables
X_ed = df["HS"]
X_urb = df["Urban"]
y_income = df["Income"]
y = df["Crime"]

# Marginal association
x_bar_a, y_bar_a, sxx_a, sxy_a, syy_a = sample_stats(X_ed, y)

# Correlation
corr_a = correlation(sxx_a, sxy_a, syy_a)

print(f"Marginal Association: r(HS, Crime) = {corr_a:.2f}")

# Stratefy Urban into two groups, splitting via median value
urb_cut = X_urb.median()
df["UrbanGroup"] = np.where(df["Urban"] >= urb_cut, "High Urban", "Low Urban")

# Correlations within strata
corr_by_group = {}
for g in ["Low Urban", "High Urban"]:
    sub = df[df["UrbanGroup"] == g]
    xb, yb, Sxx, Sxy, Syy = sample_stats(sub["HS"], sub["Crime"])
    corr_by_group[g] = correlation(Sxx, Sxy, Syy)

print(dedent(f"""
W/in group (Ubran Stratification):
r(HS, Crime | LowUrban) = {corr_by_group["Low Urban"]:.2f}
             r(HS, Crime | Highurban) = {corr_by_group["High Urban"]:.2f}

INTERPRETATION
--------------
Simpson's Paradox exists if the marginal correlation differs from in sign from the w/in group correlations.
             """))

# Plot: HS vs Crime colored by UrbanGroup + marginal LS line
sns.set_style("whitegrid", {'grid.color': 'black'})
sns.scatterplot(data=df, x="HS", y="Crime", hue="UrbanGroup")

# Marginal least squares line for HS -> Crime
b0_a, b1_a = linear_fit(x_bar_a, y_bar_a, sxx_a, sxy_a)
x_line = np.linspace(X_ed.min(), X_ed.max(), 200)
y_line = b0_a + b1_a * x_line
plt.plot(x_line, y_line, color="black", label="Marginal LS Line")

plt.xlabel("Education (HS %)")
plt.ylabel("Crime Rate")
plt.title("Florida Counties: HS vs Crime (Stratified by Urban)")
plt.legend()
plt.tight_layout()
plt.show()

# b) ======= Scatter plot and stats: ECOLOGICAL FALLACY ===========
# Sample stats for HS vs Income
x_bar_b, y_bar_b, sxx_b, sxy_b, syy_b = sample_stats(X_ed, y_income)
corr_b = correlation(sxx_b, sxy_b, syy_b)
b0_b, b1_b = linear_fit(x_bar_b, y_bar_b, sxx_b, sxy_b)

print(dedent(f"""\n
Sample mean of HS: {x_bar_b:.2f}
Sample mean of Income: {y_bar_b:.2f}
Sxx = {sxx_b:.2f}
Sxy = {sxy_b:.2f}
Syy = {syy_b:.2f}

Correlation: {corr_b:.2f}

Least Squares Line:
y_hat = {b0_b:.2f} + {b1_b:.2f}x
             """))

# Plotting: HS vs Income
sns.set_style("whitegrid", {'grid.color': 'black'})
sns.scatterplot(data=df, x="HS", y="Income")

# regression line (your existing code is fine)
x_line = np.linspace(X_ed.min(), X_ed.max(), 200)
y_line = b0_b + b1_b * x_line
plt.plot(x_line, y_line, color="red", label="LS Line")

plt.xlabel("EDUCATION (HS %)")
plt.ylabel("MEDIAN INCOME")
plt.title("FLORIDA: EDUCATION VS INCOME")
plt.tight_layout()
plt.legend()
plt.show()

print(dedent(f"""\n
INTERPRETATION
--------------
The correlation {corr_b:.2f} shows a strong positive linear association between county level education and median income.
Counties with more high school graduates tend to have larger incomes.  

Individual data would likely show a weaker correlation resulting from w/in-county  variation, as this is aggregated data.
Hence, ECOLOGICAL FALLACY, which is when information on individuals is falsely deduced from inference for the group they belong to.
             """))

# c) ========== Extrapolation ===========
print(f"Observed range: {X_ed.min()} and {X_ed.max()}")

# x-value outside of range for extrapolation
x_out = 0
# y-extrapolation
y_ext = b0_b + b1_b * x_out
# Center of data
x_mid = (X_ed.min() + X_ed.max()) / 2

print(dedent(f"""
Predicted income when HS = 0: {y_ext:.2f}.

Since 0 is well outside the observed range of data, this prediction is EXTRAPOLATION.

The Least Squares model's linear assumptions only hold up for the observed data.  Outside, the true relationship could be non-linear.
             """))

# Hypothetical non-linearity
sns.set_style("whitegrid", {'grid.color': 'black'})
sns.scatterplot(data=df, x="HS", y="Income")

# Nonlinear curve
x_curve = np.linspace(x_out, X_ed.max(), 200)
y_curve = b0_b + b1_b * x_curve

# Quadratic additive w/ negative constant so the curve is downward at the extremes
c = -1/7 # Arbitrarily chosen value to make divergence visible
y_curve_true = y_curve + c * (x_curve - x_mid)**2

# plot
plt.plot(x_curve, y_curve, color="red", label="Fitted Line")
plt.plot(
    x_curve, 
    y_curve_true, 
    color="purple", 
    linestyle="--",
    label="Hypothetical True E(Y|X)"
)

# Boundary lines for visualization
plt.axvline(
    X_ed.min(), 
    linestyle=":",
    color="gray"
)
plt.axvline(
    X_ed.max(),
    linestyle=":",
    color="gray"
)

# Extrapolatd prediction @ x=0
sns.scatterplot(
    x=[x_out],
    y=[y_ext],
    color="red", 
    marker="X",
    s=100,
    label=f"Extrapolation Prediction @ HS=0: {y_ext:.2f}"
               )

plt.xlabel("Education (HS %)")
plt.ylabel("Median Income")
plt.title(dedent("""
Extrapolation Danger: Fitted Line vs Hypothetical True Relationship
                 """))
plt.legend()
plt.tight_layout()
plt.show()
