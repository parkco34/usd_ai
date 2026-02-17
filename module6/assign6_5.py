#!/usr/bin/env python
"""
Using the covid19.dat file:
(a) Construct the two scatterplots shown in Figure 6.3.
(b) Find and interpret the correlation between time and (i) cases, (ii) log(cases).
(c)  Fit the linear model for the log-transformed counts and report the prediction equation.$^{29}.$ Explain why the predicted count at day $x+1$ equals the predicted count at day $x$ multiplied by $\text{exp}(\hat\beta_1)=1.36.$
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
        raise TypeError("Input not valid\nNot a pandas Series")

    return var.mean()

def sample_stats(x, y):
    """
    Computes the sample means and samples standard deviations.
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
    return Sxy / np.sqrt(Sxx * Syy)

# ======== Main =========
df = read_file("./covid19.dat")

# Explanatory/Target
X = df["day"]
y = df["cases"]

# Log transformation
log_y = np.log(y)

# a) ==== Scatter plots ====
# Add gridlines cuz i want to  ( ͡° ͜ʖ ͡°)
sns.set_style("darkgrid") # Must be BEFORE the figure 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1
# Using pre-existing axes for plot
sns.scatterplot(data=df, x="day", y="cases", ax=ax1)
ax1.set(title="Number of Covid-19 cases in U.S.", xlabel="Day", ylabel="Number of Cases")

# Plot 2
sns.scatterplot(x=X, y=log_y, ax=ax2)
ax2.set(title="Number of Covid-19 cases in U.S.", xlabel="Day", ylabel="log(Number of Cases)")

plt.tight_layout()
plt.show()

# b) ====== Correlation and interpretation ========
# Sample stats
x_bar, y_bar, sxx, sxy, syy  = sample_stats(X, y)
x_bar_log, y_bar_log, sxx_log, sxy_log, syy_log  = sample_stats(X, log_y)

# Correlation for each case
corr1 = correlation(sxx, sxy, syy)
corr2 = correlation(sxx_log, sxy_log, syy_log)


print(dedent(f"""
---------------
INTERPRETATION
---------------
The correlation of original y: {corr1:.3f} and the correlation for
the log(y) is {corr2:.3f}.
This shows that the linear correlation underestimates the true strength of the
association between the number of days Covid has been actively circulating
through the given population and the number of cases reported.

The correlation is much closer to 1 after doing a log-transformation to the
exponential function, getting a linear relationship as shown by the scatter
plots.
             """))

# c) Linear Fit and prediction equation
b0_log, b1_log = linear_fit(x_bar_log, y_bar_log, sxx_log, sxy_log)

# Set white background with black grid
sns.set_style("whitegrid", {'grid.color': 'black'})
sns.scatterplot(x=X, y=log_y)

# Fitted line: y_hat = b0 + b1 * x (over 200 evenly spaced values)
x_line = np.linspace(X.min(), X.max(), 200)
# Line to go on scatterplot
y_line_log = b0_log + b1_log * x_line

# Plotting
plt.plot(x_line, y_line_log, linewidth=2)
plt.title("Log(Number of Cases) by Day")
plt.xlabel("Day")
plt.ylabel("log(Cases)")
plt.show()

# Growth factor (exponential)
growth_factor = np.exp(b1_log)
print(f"Each additional day multiplies predicted cases by {growth_factor:.2f}.")

# Prediction equation
print(f"log(y_hat) = {b0_log:.2f} + {b1_log:.4f}x")
print(f"exp(beta1) = {np.exp(b1_log):.2f}")

pct_increase = (np.exp(b1_log) - 1) * 100
print(dedent(f"""
Thus, the model predicts a {pct_increase:.2f}% increase in cases per day
during the initial exponential growth phase.
"""))

