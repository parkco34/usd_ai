#!/usr/bin/env python
"""
Consider the mammogram diagnostic example in Section 2.1.4.

(a) Show that the joint probability distribution of diagnosis and disease status is as shown in
Table 2.6. Given that a diagnostic test result is positive, explain how this joint distribution
shows that the 12% of incorrect diagnoses for the 99% of women not having breast cancer
swamp the 86% of correct diagnoses for the 1% of women actually having breast cancer.
(b) The first test for detecting HIV-positive status had a sensitivity of 0.999 and specificity
of 0.9999. Explain what these mean. If at that time 1 in 10,000 men were truly HIVpositive, find the positive predictive value. Based on this example, explain the potential
disadvantage of routine diagnostic screening of a population for a rare disease.

**TABLE 2.6** Joint probability distribution for disease status and diagnosis of breast cancer
mammogram, based on conditional probabilities in Table 2.1

| Disease Status | Positive (+) | Negative (-) | Total |
|----------------|--------------|--------------|-------|
| Yes (D)        | 0.0086       | 0.0014       | 0.01  |
| No ($D^c$)        | 0.1188       | 0.8712       | 0.99  |
"""
import pandas as pd

# Given values (from the mammogram example)
p_D = 0.01
p_notD = 1 - p_D

sens = 0.86           # P(+ | D)
fnr = 1 - sens        # P(- | D)

fpr = 0.12            # P(+ | D^c)
spec = 1 - fpr        # P(- | D^c)

# Joint probabilities
joint = {
    ("Yes (D)", "Positive (+)"): p_D * sens,
    ("Yes (D)", "Negative (-)"): p_D * fnr,
    ("No (D^c)", "Positive (+)"): p_notD * fpr,
    ("No (D^c)", "Negative (-)"): p_notD * spec,
}

table = pd.DataFrame(
    {
        "Positive (+)": [joint[("Yes (D)", "Positive (+)")], joint[("No (D^c)", "Positive (+)")] ],
        "Negative (-)": [joint[("Yes (D)", "Negative (-)")], joint[("No (D^c)", "Negative (-)")] ],
    },
    index=["Yes (D)", "No (D^c)"]
)

table["Total"] = table["Positive (+)"] + table["Negative (-)"]
table.loc["Total"] = table.sum(axis=0)

print(table)

# Positive Predictive Value: P(D | +)
ppv = joint[("Yes (D)", "Positive (+)")] / table.loc["Total", "Positive (+)"]
print(f"\nPPV = P(D | +) = {ppv:.4f} = {ppv*100:.2f}%")

# Interpretation
print(f"""
Even though the test is pretty good at detecting cancer when it's there (86%
Sensitivity), the disease is rare with 1% chance of actually having it.  The
      larger group (without cancer) with 99% generates enough false positives
      (12%) that it overwhelms the smaller group of true positives.
      """)
