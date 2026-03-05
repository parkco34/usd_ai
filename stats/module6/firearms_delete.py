#!/usr/bin/env python
# ===============================
# Compare models WITH and WITHOUT outlier
# ===============================

print(dedent(f"""
PART (c): LINEAR REGRESSION COMPARISON
--------------------------------------

WITH OUTLIER (US included):
y_hat = {b0:.3f} + {b1:.3f}x

WITHOUT OUTLIER (US removed):
y_hat = {b0_nous:.3f} + {b1_nous:.3f}x

MATHEMATICAL INTERPRETATION
---------------------------

Recall that the least squares slope is:

    b1_hat = Sxy / Sxx
            = sum((xi - x_bar)(yi - y_bar)) / sum((xi - x_bar)^2)

The US has:
    • Extremely large x value (high leverage)
    • Extremely large y value

This dramatically increases Sxy (covariance numerator),
which in turn increases the slope.

When the US is INCLUDED:
    The slope is POSITIVE ({b1:.3f}),
    suggesting that more firearms are associated with higher murder rates.

When the US is REMOVED:
    The slope becomes {b1_nous:.3f},
    which is NEGATIVE.

This shows that a single high-leverage observation
can completely change the direction of association.

WHY THIS HAPPENS (Conceptually):
--------------------------------

Least squares minimizes:

    sum((yi - y_hat_i)^2)

Because residuals are squared, extreme observations
have disproportionately large influence.

The US has both:
    1) Large leverage  (far from x_bar)
    2) Large residual contribution

Thus it PULLS the regression line toward itself.

CONCLUSION:
-----------

The regression fit is highly sensitive to influential points.

This dataset demonstrates:

    • Correlation is not robust.
    • Least squares regression is not robust.
    • High-leverage observations can reverse conclusions.

In practical AI modeling terms,
this is why influence diagnostics (Cook's Distance,
leverage scores) and robust regression methods
are important when fitting linear models.
"""))


