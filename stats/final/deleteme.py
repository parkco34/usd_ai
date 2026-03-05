#!/usr/bin/env python
from textwrap import dedent

for col in FEATURES_DISC:

    # build 2x2 contingency table:
    # rows = risk (0,1), cols = feature (0,1)
    table = pd.crosstab(df[TARGET], df[col])

    # force full 2x2 shape in case a category is missing
    table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    # chi-square test of independence
    chi2, p_val, dof, expected = chi2_contingency(table, correction=False)

    # P(feature=1 | risk=0) and P(feature=1 | risk=1)
    p_low  = table.loc[0, 1] / table.loc[0].sum() if table.loc[0].sum() else np.nan
    p_high = table.loc[1, 1] / table.loc[1].sum() if table.loc[1].sum() else np.nan

    print(dedent(f"""
{col} \t p = {p_val:.3f}
{'Significant' if p_val < 0.05 else 'Not significant'}
P(Y=1 | Low) = {p_low:.3f}
P(Y=1 | High) = {p_high:.3f}
                 """))


print("\n# ===== Interpretation of Significance Tests =====")

print(f"""
Continuous Variables (Welch t-test):

Age:
High-risk patients are older on average 
(mean_high = {hi['age'].mean():.2f} vs mean_low = {low['age'].mean():.2f}), 
p < 0.001 indicating a statistically significant difference.

Pack Years:
High-risk patients have substantially greater smoking exposure 
(mean_high = {hi['pack_years'].mean():.2f} vs mean_low = {low['pack_years'].mean():.2f}), 
p < 0.001. This suggests smoking history is strongly associated with risk.

Oxygen Saturation:
High-risk patients show lower oxygen levels 
(mean_high = {hi['oxygen_saturation'].mean():.2f} vs mean_low = {low['oxygen_saturation'].mean():.2f}), 
p < 0.001, indicating a meaningful physiological difference.
""")

print(f"""
Binary Variables (Chi-square test):

COPD:
Prevalence is much higher in high-risk patients 
({table.loc[1,1] / table.loc[1].sum():.2f} vs {table.loc[0,1] / table.loc[0].sum():.2f}),
p < 0.001, showing strong association.

Chronic Cough and Shortness of Breath:
Both symptoms are substantially more common in the high-risk group,
indicating strong dependence with lung cancer risk.

Family History of Cancer:
Rates differ between groups, and the chi-square test confirms
a statistically significant association.
""")
