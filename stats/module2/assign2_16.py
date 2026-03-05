#!/usr/bin/env python
"""
(a)  In the first week, the observations from Sunday to Saturday are 10, 8, 14, 7, 21, 44, 60.
Do you think that the Poisson distribution might describe the random variability of this
phenomenon adequately. Why or why not?
(b) Would you expect the Poisson distribution to better describe, or more poorly describe,
the number of weekly admissions to the hospital for a rare disease? Why?
"""
# a)
import numpy as np

y = np.array([10, 8, 14, 7, 21, 44, 60])

mean_y = y.mean()
var_y = y.var(ddof=1)          # sample variance
dispersion = var_y / mean_y

print(f"Mean = {mean_y:.2f}")
print(f"Sample variance = {var_y:.2f}")
print(f"Variance/Mean (dispersion index) = {dispersion:.2f}")

if dispersion > 1.5:
    print("Conclusion: Overdispersion (Var >> Mean) → single-rate Poisson is not adequate.")
else:
    print("Conclusion: Variability is close to Poisson-like.")


# Interpretation
print(f"""
The Poisson model is not adequate for these ER counts since,
the week shows system changes, where the rate is not constant.
Also, the data is definitely overdispersed relative to Poisson, meaning the
      variance is larger than the mean.
      """)

# b)
print("""
The Poisson model would likely work better with weekly admissions since,
- Rare events in the fixed interval (weekly)
- Indepence of occurrences
- Constant risk over the week (for the most part).

However, if there's seasonality or clustering, etc. then I think the negative
      binomial dist. would be more appropriate.
      """)
