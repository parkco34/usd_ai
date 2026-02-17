#!/usr/bin/env python
"""
A study of sheep mentioned in Exercise 1.27 analyzed whether the sheep survived for a year from the original observation time (1 = yes, 0 = no) as a function of their weight (*kg*) at the original observation.
(a) Does the survival of the sheep seem to depend on their weight? If so, how does the weight of a sheep affect its probability of survival? Answer by fitting a generalized linear model for the survival probability.
(b)  For what weight values do the sheep have more than a 50% chance of survival?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

# ======= MAIN ======
df = read_file("./Sheep.dat")

# a) ======= Fit logistic regression  =====
model = smf.glm(
    "survival ~ weight",
    data=df,
    family=sm.families.Binomial()).fit()

print(model.summary())

b0 = model.params["Intercept"]
b1 = model.params["weight"]

# Plotting the fitted logistic curve
sns.set_style("whitegrid", {'grid.color': 'black'})
sns.scatterplot(data=df, x="weight", y="survival")

# Fitted probability curve
x_range = np.linspace(df["weight"].min() - 2, df["weight"].max() + 2, 300)
# Logistic function: P(Y=1) = exp(b0 + b1*x) / (1 + exp(b0 + b1*x))
logit_vals = b0 + b1 * x_range
prob_vals = np.exp(logit_vals) / (1 + np.exp(logit_vals))

plt.plot(x_range, prob_vals, color="orange", label="Fitted Logistic Curve")
plt.axhline(
    y=0.5,
    color="red",
    linestyle="--",
    label="P = 0.50"
)

plt.title("Sheep Survival Probility vs Weight")
plt.xlabel("Weight (kg)")
plt.ylabel("P(Survival = 1)")
plt.tight_layout()
plt.legend()
plt.show()

# Odds ratio
odds_ratio = np.exp(b1)

print(dedent(f"""
Logistic regression model:
log(p / (1 - p)) = {b0:.2f} + {b1:.2f} *weight

P-value for weight: {model.pvalues["weight"]:.3f},
which is {"< 0.05 (statistically significant)" if model.pvalues['weight'] < 0.05 else ">= 0.05"}.

So, survival does depend on weight.

The coefficient for weight is b1 = {b1:.4f} ({"positive" if b1 > 0 else "negative"}).
This means that for each 1 kg increase in weight, the LOG-ODDS of survival
{"increase" if b1 > 0 else "decrease"} by {abs(b1):.4f}.

The odds ratio: exp(b1) = exp({b1:.4f}) = {odds_ratio:.4f}
This means for each additional kg, the ODDS of survival are multiplied by {odds_ratio:.4f}.
{"Heavier sheep are more likely to survive." if b1 > 0 else "Heavier sheep are less likely to survive."}

             """))

# b) ====== Weight for > 50% survival ==========
# weight > -b0 / b1
weight_50 = -b0 / b1

print(dedent(f"""
At P = 0.50, the logit = 0: b0 + b1 * weight = 0
{b0:.3f} + {b1:.3f} * weight = 0
weight = {-b0:.3f} / {b1:.3f} = {weight_50:.2f} kg

Since the coefficient for weight is {"positive" if b1 > 0 else "negative"},
sheep with weight {">" if b1 > 0 else "<"} {weight_50:.2f} kg have MORE than a 50% chance
of survival.

So a sheep needs to weigh at least {weight_50:.2f} kg to have a better than
even chance of making it through the year.
             """))









