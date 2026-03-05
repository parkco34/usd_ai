#!/usr/bin/env python
"""
For advanced industrialized nations, the Firearms data file at the text website shows annual homicide rates (per million population) and the number of firearms (per 100 people), with data taken from Wikipedia and [smallarmssurvey.org](smallarmssurvey.org)." (a) Construct a scatterplot and highlight any observations that fall apart from the general trend. (b) Find the correlation with and without the outlying observation. Why is it so different in the two cases? (c) Fit the linear regression model with and without the outlying observation, and note how influential an outlier can be on the fit.
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
    Computes the sample means and sample standard deviations, sample
    covariance, 
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

# ======= MAIN ===========
df = read_file("./firearms.dat")
# Proper column name
df["murder_rates"] = df["deaths"]
# Drop redundant column
df.drop(labels="deaths", axis=1, inplace=True)

# a) Scatterplot and outliers; Two CONTINUOUS datatypes
X = df["firearms"] # per 100 people
y = df["murder_rates"] # per 1E6 population

# Sample means and sample standard deviations
x_bar, y_bar, sxx, sxy, syy  = sample_stats(X, y)

# Estimating parameters
b0, b1 = linear_fit(x_bar, y_bar, sxx, sxy)

# Set white background with black grid
sns.set_style("whitegrid", {'grid.color': 'black'})
sns.scatterplot(data=df, x="firearms", y="murder_rates")

# Fitted line: y_hat = b0 + b1 * x (over 200 evenly spaced values)
x_line = np.linspace(X.min(), X.max(), 200)
# Line to go on scatterplot
y_line = b0 + b1 * x_line

# Highlight outlier (largest firearms value)
outlier = df.loc[df["firearms"].idxmax()]

plt.scatter(outlier["firearms"],
            outlier["murder_rates"],
            color="red",
            s=200)

# Plotting
plt.plot(x_line, y_line, linewidth=2)

plt.title("Murder Rates Based on Number of Firearms")
plt.xlabel("Number of Firearms (per 100 people)")
plt.ylabel("Murder Rates (per million population)")
plt.show()

# Outlier observation
print(dedent(f"""
The {outlier["Nation"]} is the outlier of the {df.shape[0]} 
advanced industrial nations with {X.max()} firearms per 
100 people and {outlier["murder_rates"]} murders for a population of a million people.

WIth a sample size of only {len(df)}, a single outlier like the US
will strongly influence the correlation and regression fit.
             """))

# b) Correlation with and without outlier
# With outlier
corr = correlation(sxx, sxy, syy)

# Outlier narrative
x, y = outlier["firearms"], outlier["murder_rates"]

# Interpretation
print(dedent(f"""
INTERPRETATION
--------------
With the outlier ({outlier["Nation"]}),
the correlation is {corr:.3f}.
This is interesting, given that the outlier is so far outside the majority of
observations. You'd think it would be A LOT closer to 1.

Since, for the US, the firearms (per 100 people) {x} and 
the murder rate (per million population) {y} are relatively so large,
the term in the numerator (Covariance) raises the correlation to such an extent that
it influences the analysis drastically.

Geometrcially, the correlation can be shown as r = cos(theta), where theta is
the angle between xi - x_bar and yi - y_bar.  The US rotates the direction of
the x-vector profoundly.
             """))

# Dataframe without outlier
no_us = df[df["Nation"] != "US"]

# Sample statistics
X2 = no_us["firearms"]
y2 = no_us["murder_rates"]

x2_bar, y2_bar, sxx2, sxy2, syy2 = sample_stats(X2, y2)

# Estimating parameters without US
b0_nous, b1_nous = linear_fit(x2_bar, y2_bar, sxx2, sxy2)

# Correlation without US
corr_nous = correlation(sxx2, sxy2, syy2)

# c) Fitting the linear regression model without the outlier
X_line2 = np.linspace(X2.min(), X2.max(), 200)
# Line
y_line2 = b0_nous + b1_nous * X_line2

# Scatterplot without the US
sns.set_style("whitegrid", {'grid.color': 'black'})
sns.scatterplot(no_us, x="firearms", y="murder_rates")

# Plotting
plt.plot(X_line2, y_line2, linewidth=2)

plt.title("Murder Rates Based on Number of Firearms w/out US")
plt.xlabel("Number of Firearms (per 100 people)")
plt.ylabel("Murder Rates (per million population)")
plt.show()

# Comparison of outlier influence
delt_b0 = b0 - b0_nous
delt_b1 = b1 - b1_nous

# Interpretation via the difference between w/ outlier and without
print(dedent(f"""
WITH outlier (US included):
    y_hat = b0 + b1 * x
    y_hat = {b0:.2f} + {b1:.2f} * x

WITHOUT outlier (US removed):
    y_hat = b0_noUS + b1_noUS * x
    y_hat = {b0_nous:.2f} + {b1_nous:.2f} * x

Without the outlier {outlier["Nation"]}, the correlation
is now {corr_nous:.2f}!
y_hat = {b0:.2f}

Change in intercept:  delta_b0 = b0 - b0_noUS = {delt_b0:.2f}
Change in slope:      delta_b1 = b1 - b1_noUS = {delt_b1:.2f}

That is, the correlation without the US is NEGATIVE, which means
that the more firearms advanced nation's (otherthan the US) citizens possess,
the lower the murder rate!!      (>▽<)

An observation is influential if the removing it largely changes the slope,
intercept, or correlation.
The US in this case has an extreme x value and large cross-product Sxy.

Thus, correlation and least squares regression do not seem to be robust, given
this problem.
             """))
