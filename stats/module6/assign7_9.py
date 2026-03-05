#!/usr/bin/env python
"""
The `SoreThroat` data file at the book’s website contains data from from a study$^{23}$ about *Y* = whether a patient having surgery had a sore throat on waking (1 = yes, 0 = no) as a function of *D* = duration of the surgery (in minutes) and *T* = type of device used to secure the airway (1 = tracheal tube, 0 = laryngeal mask airway).
(a) Fit a GLM using both explanatory variables as main effects. Interpret effects.
(b) Fit a GLM permitting interaction between the explanatory variables. Interpret the effect of *D* at each category of *T*.
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
df = read_file("./SoreThroat.dat")

# a) ======= Main effects model ========
model_main = smf.glm("Y ~ D + T",
                      data=df,
                      family=sm.families.Binomial()).fit()

print(model_main.summary())

b0 = model_main.params["Intercept"]
b_D = model_main.params["D"]
b_T = model_main.params["T"]

# Odds ratios
or_D = np.exp(b_D)
or_T = np.exp(b_T)

print(dedent(f"""
logit(P(sore throat)) = {b0:.4f} + {b_D:.4f} * D + {b_T:.4f} * T

Effect of Duration (D):
             b_D = {b_D:.3f} odds ratio = exp({b_D:.3f}) = {or_D:.3f}.
for each additional minute of surgery, the odds of a sore throat are multiplied
by {or_D:.3f}.
P-value = {model_main.pvalues["D"]:.3f}.

Effect of Device Type (T):
  b_T = {b_T:.4f}, odds ratio = exp({b_T:.4f}) = {or_T:.4f}
  Using a tracheal tube (T=1) vs laryngeal mask (T=0), the odds of
  a sore throat are multiplied by {or_T:.4f}, holding duration constant.
  {"Tracheal tube is associated with HIGHER odds of sore throat." if b_T > 0 else "Tracheal tube is associated with LOWER odds of sore throat."}
  p-value = {model_main.pvalues['T']:.4f} {"(significant)" if model_main.pvalues['T'] < 0.05 else "(not significant)"}


             """))

# b)======= interaction model ======
odel_inter = smf.glm("Y ~ D + T + D:T",
                       data=df,
                       family=sm.families.Binomial()).fit()

print(model_inter.summary())

b0_i = model_inter.params["Intercept"]
b_D_i = model_inter.params["D"]
b_T_i = model_inter.params["T"]
b_DT_i = model_inter.params["D:T"]

print(dedent(f"""
logit(P(sore throat)) = {b0_i:.4f} + {b_D_i:.4f} * D + {b_T_i:.4f} * T + {b_DT_i:.4f} * D*T

Effect of Duration (D) at each category of T:

When T = 0 (laryngeal mask airway):
  logit(P) = {b0_i:.4f} + {b_D_i:.4f} * D
  Effect of D: b_D = {b_D_i:.4f}
  Odds ratio per minute = exp({b_D_i:.4f}) = {np.exp(b_D_i):.4f}
  For each extra minute, odds of sore throat {"increase" if b_D_i > 0 else "decrease"}
  by {abs(np.exp(b_D_i) - 1)*100:.1f}% with the laryngeal mask.

When T = 1 (tracheal tube):
  logit(P) = ({b0_i:.4f} + {b_T_i:.4f}) + ({b_D_i:.4f} + {b_DT_i:.4f}) * D
           = {b0_i + b_T_i:.4f} + {b_D_i + b_DT_i:.4f} * D
  Effect of D: b_D + b_DT = {b_D_i:.4f} + {b_DT_i:.4f} = {b_D_i + b_DT_i:.4f}
  Odds ratio per minute = exp({b_D_i + b_DT_i:.4f}) = {np.exp(b_D_i + b_DT_i):.4f}
  For each extra minute, odds of sore throat {"increase" if (b_D_i + b_DT_i) > 0 else "decrease"}
  by {abs(np.exp(b_D_i + b_DT_i) - 1)*100:.1f}% with the tracheal tube.

The interaction term (p = {model_inter.pvalues['D:T']:.4f}) suggests that
the effect of surgery duration on sore throat probability
{"DOES" if model_inter.pvalues['D:T'] < 0.05 else "does NOT significantly"} differ between the two device types.

             """))

# PLotting
sns.set_style("whitegrid", {'grid.color': 'black'})

d_range = np.linspace(df["D"].min(), df["D"].max(), 200)

# T = 0 (laryngeal mask)
logit_T0 = b0_i + b_D_i * d_range
prob_T0 = np.exp(logit_T0) / (1 + np.exp(logit_T0))

# T = 1 (tracheal tube)
logit_T1 = (b0_i + b_T_i) + (b_D_i + b_DT_i) * d_range
prob_T1 = np.exp(logit_T1) / (1 + np.exp(logit_T1))

plt.plot(d_range, prob_T0, linewidth=2, label="T=0 (Laryngeal Mask)")
plt.plot(d_range, prob_T1, linewidth=2, label="T=1 (Tracheal Tube)")

# Scatter the observed data
mask_T0 = df["T"] == 0
mask_T1 = df["T"] == 1
plt.scatter(df.loc[mask_T0, "D"], df.loc[mask_T0, "Y"], alpha=0.4, marker='o', label="Observed T=0")
plt.scatter(df.loc[mask_T1, "D"], df.loc[mask_T1, "Y"], alpha=0.4, marker='s', label="Observed T=1")

plt.title("Sore Throat Probability vs Surgery Duration (Interaction Model)")
plt.xlabel("Duration (minutes)")
plt.ylabel("P(Sore Throat)")
plt.legend()
plt.show()



