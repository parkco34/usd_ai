#!/usr/bin/env python
"""
You will use a mock survey dataset that measures public sentiment toward artificial intelligence. Your task is to generate the dataset, construct a contingency table, and test whether sentiment depends on gender. ### Dataset creation Run the starter code below to generate the dataset as a pandas DataFrame named df_ai. import pandas as pd import numpy as np # Set seed for reproducibility np.random.seed(2024) # Sample size n = 300 # Categories sentiments = ["Positive", "Neutral", "Negative"] genders = ["Male", "Female", "Other"] usage_levels = ["Daily", "Weekly", "Rarely", "Never"] # Generate the DataFrame df_ai = pd.DataFrame( { "sentiment": np.random.choice(sentiments, size=n, p=[0.44, 0.33, 0.23]), "gender": np.random.choice(genders, size=n, p=[0.49, 0.48, 0.03]), "age": np.random.randint(18, 75, size=n), "ai_usage_frequency": np.random.choice(usage_levels, size=n), "trust_in_ai": np.random.randint(1, 6, size=n), } ) df_ai.head() Using the dataset you just created: (a) Form a contingency table that cross classifies sentiment by gender. (b) For the hypothesis $H_0:$ sentiment toward AI is independent of gender, conduct a chi squared test of independence. (c) Interpret the results in the context of attitudes toward AI.
"""
import numpy as np
import pandas as pd
from textwrap import dedent
from scipy.stats import chi2_contingency

import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(2024)

# Sample size
n = 300

# Categories
sentiments = ["Positive", "Neutral", "Negative"]
genders = ["Male", "Female", "Other"]
usage_levels = ["Daily", "Weekly", "Rarely", "Never"]

# Generate the DataFrame
df_ai = pd.DataFrame(
    {
        "sentiment": np.random.choice(sentiments, size=n, p=[0.44, 0.33, 0.23]),
        "gender": np.random.choice(genders, size=n, p=[0.49, 0.48, 0.03]),
        "age": np.random.randint(18, 75, size=n),
        "ai_usage_frequency": np.random.choice(usage_levels, size=n),
        "trust_in_ai": np.random.randint(1, 6, size=n),
    }
)

df_ai.head()

# a) Contingency table
ct = pd.crosstab(df_ai["sentiment"], df_ai["gender"])

print("Contingency Table: Sentiment by Gender")
print(ct, "\n")

# b) Chi-squared test of independence
chi2, p_val, dof, expected = chi2_contingency(ct, correction=False)
expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)

print(dedent(f"""\n
Chi-sqaured test of independence
chi-stat: {chi2:.4f}
Degrees of Freedom: {dof}
p-value: {p_val:.3f}
             """))

print("Expected counts under H0 (independence):")
print(expected_df.round(2), "\n")

# Check for validity via finding the smallest cell count of the contingency table
# Converts it to a numpy array (better than using .values for the DataFrame)
min_expected = expected_df.to_numpy().min()
print(f"Min expected cell count = {min_expected}")

if min_expected < 5:
    print("Chi-square approx could be weak!")

else:
    print("Expected counts condition looks good since all cell counts are approx >= 5")

# Effect Size
r, c = ct.shape
k = min(r - 1, c - 1)
cramers_v = np.sqrt(chi2 / (n * k))
# From section 4.5.2 of the text
print("\nEffect Size")
print(f"Cramér's V = {cramers_v:.4f}\n")

# c) INterpretation
# Significance level
alpha = 0.05

if p_val <= alpha:
    decision = "reject null"
    print(f"Significance level alpha = {alpha}")
    print(f"Decision: {decision}")
    print(dedent("""
There is statistically significant evidence the sentiment towards AI is associated with gender.
                 """))

else:
    decision = "fail to reject null"
    print(f"Significance level alpha = {alpha}")
    print(f"Decision: {decision}")
    print(dedent("""
There isn't enough evidence that the sentiment towards AI has to do with gender.
                 """))


print(
    f"- The chi-square test answers: 'Is there evidence of *any* association?' (p-value)\n"
    f"- Cramér's V answers: 'How strong is the association?' (effect size)\n"
    f"- Standardized residuals show which (sentiment, gender) cells deviate most from independence.\n"
    f"  As a rule of thumb, |residual| ≳ 2 is often considered a noticeable deviation.\n"
)





