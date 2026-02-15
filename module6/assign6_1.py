#!/usr/bin/env python
"""
For the Scottish hill Races data, a linear model can predict men’s record times from women’s record times.
(a)  Show the scatterplot and report the prediction equation. Predict the men’s record time for the Highland Fling, for which timeW = 490.05 minutes.
(b) Find and interpret the correlation.
(c) We could impose the natural constraint that when timeW = 0, then timeM = 0. Fit the model $E(Y_i)=\beta x_i.$ Interpret the estimated slope.
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
        raise("Input not valid\nNot a pandas Series")

    return var.mean()

def sample_stats(x, y):
    """
    Computes the sample means and samples Standard Errors.
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

def line_thru_origin(x, y):
    """
    Fits the no-intercept line through the origin.
    yi = b_1 * xi

    With b_0 = 0, we minimize LEAST SQUARES: sum(yi - b_1 * xi)**2,
    by taking derivative: -2*sum(xi*yi) + 2*b_1*sum(xi**2)
    -------------------------------------------------
    INPUT:
        x: (pd.Series) Expanatory variables
        y: (pd.Series) Target

    OUTPUT:
        b_hat: (np.float)
    """
    b_hat = np.sum(x * y) / sum(x**2)

    return b_hat

# =========== Main ==============
# Load data
df = read_file("./ScotsRaces.dat")

# Explanatory/Target variables
X = df["timeW"] # Women's time in minutes
y = df["timeM"] # Men's time in minutes

# Sample means and sample standard deviations
x_bar, y_bar, sxx, sxy, syy  = sample_stats(X, y)

# a) ======= Fit model and prediction equation ========
# Estimating parameters
b0_hat, b1_hat = linear_fit(x_bar, y_bar, sxx, sxy)

# Set white background with black grid
sns.set_style("whitegrid", {'grid.color': 'black'})
sns.scatterplot(df, x="timeW", y="timeM")

# Fitted line: y_hat = b0_hat + b1_hat * x (over 200 evenly spaced values)
x_line = np.linspace(X.min(), X.max(), 200)
# Line to go on scatterplot
y_line = b0_hat + b1_hat * x_line

# Plotting
plt.plot(x_line, y_line, linewidth=2)

plt.title("ScotsRaces: Men's vs Women's Record Times")
plt.xlabel("timeW (minutes)")
plt.ylabel("timeM (minutes)")
#plt.show()

# Predict men's time for HighlandFling while timeW=490.05
x0 = 490.05

# Equation
y_hat = b0_hat + b1_hat * x0

# Output
print(f"""
The predicted equations is
y_hat = b0_hat + b1_hat * x0 = {y_hat:.2f}
w/ b0_hat = {b0_hat:.2f},  b1_hat = {b1_hat:.2f} and x0={x0}
      """)

# b) ======= Correlation ===========
# Correlation
r = correlation(sxx, sxy, syy)

# C) ======= Interpreting the estimated slope  =======
# Intercept set to zero imposing the constraint timeW = 0 = timeM
b_hat = line_thru_origin(X, y)

# a) Results
print(dedent(f"""\n
a) Prediction equation (least squares):
timeM_hat = {b0_hat:.2f} + {b1_hat:.2f} * timeW (x)
\nPredicted mens's record time when timeW = {x0}
timeM_hat = {y_hat:.2f} minutes.
A {(y_hat - x0):.2f} minute difference.
             """))
# b) results
print(dedent(f"""
b) Correlation:
r = {r:.2f}

INTERPRETATION:
Very strong linear association between timeW and timeM.
In fact, it's nearly perfectly associated (almost 1) with sign (+), meaning
when women take longer in a race, so do men.

The COEFFICIENT OF DETERMINATION (r^2) ~ {r**2:.2f} is approx.
{(r**2) * 100:.1f}%, which is the proportion of variaibility in men's record
times EXPLAINED by the linear relationship with women's times, where only 
about {(1 - r**2)*100:.1f}% of is UNEXPLAINED (SSE).
             """))

# c) Through origin
print(dedent(f"""\n
c) No intercept, where we minimize the least squares, which becomes:
sum((yi - b_1 * xi)**2). E(Y) = b1_hat*x forces the regression line through the
origin. This makes sense, physically, since a running at 0 m/s would imply
you're not moving at all and if two people are racing, if one
isn't moving neither is the other.

Taking the derivative: 
-2*xi [sum(xi * yi) - b_1 * sum(xi**2)]

Setting to zero and solving for b_hat:
b_hat = sum(xi * yi) / sum(xi**2)

Therefore, b_hat = {b_hat:.2f}

INTERPRETATION:
--------------
b_hat, where there's no intercept so the baseline is adjusted and b_hat is
basically a scaling factor, where men's time is a FIXED proportion
of women's time.  Analogous to solving for acceleration in into physics, where
you start with the equation v = v0 + (1/2)*a*t^2 nad v0 = 0
(initially at rest).

Men are approximately {(1-b_hat)*100:.1f}% faster than women.
If b_hat = {b_hat:.2f}, men complete races about {b_hat*100:.1f}%
of the time it takes women.

In part a) With b_hat = {b_hat:.2f}, the model predicts men's
record times are approx. {b_hat:.2f} of women's times, across
all races.  
Also, men complete the races nearly {(1-b_hat)*100:.1f}% faster than women.

Unlike in the model in a), where the intercept b0_hat = {b0_hat:.2f} causes the
men-to-women time ratio to vary with the length of the race, and the one
without the intercept has a singe scaling factor, which is a stronger
assumption.
             """))
