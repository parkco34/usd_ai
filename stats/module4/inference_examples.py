#!/usr/bin/env python
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

#true_prob = 0.7
#
#np.random.seed(0)
#
## Generate binary data with known probability
#data = np.random.choice([0, 1], size=100, p=[1-true_prob, true_prob])
#
## Perform MLE
#est_prob = max(np.mean(data), 1 - np.mean(data))
#print(f"True probability is {true_prob}")
#print(f"MLE = {est_prob}")

# ==> CI FOR MEANS OF NORMAL POPULATIONS
#np.random.seed(0)
#data2 = np.random.normal(loc=5, scale=2, size=100)
#
## Sample statistics
#sample_mew = np.mean(data2)
## ddof=1 for unbiased estimate
#sample_std = np.std(data2, ddof=1)
#sample_size = len(data2)
#
## Set confidence level
#conf_lvl = 0.95
#
## Calculate CRITICAL VALUE (based on Normal dist)
## (1 - confidence_level) / 2 =---=>> probability divided by 2 for a two-tailed
## test ! ?
#z_crit = norm.ppf(1 - (1 - conf_lvl) / 2)
#
## Standard error
#se = sample_std / np.sqrt(sample_size)
#
## Margin of error - extent to which the sample mean can vary from the
## population mean
#marg_err = z_crit * se
#
## Confidence Interval
#ci = (sample_mew - marg_err,
#     sample_mew + marg_err)
#
#print(f"Condience Interval: {ci}")

# ==> BOOTSTRAP SAMPLING
#np.random.seed(42)
#data3 = np.random.normal(loc=19, scale=2, size=100)
#
## B.S. iterations
#n_bs = 200
#
#bs_means = []
#for _ in range(n_bs):
#    bs_sample = np.random.choice(data3, size=len(data3), replace=True)
#    bs_means.append(np.mean(bs_sample))
#
## Histogram
#plt.hist(bs_means, bins=10)
#plt.xlabel("Bootstrap Sample Mean")
#plt.ylabel("Frequency")
#plt.title("Distribution of Bootstrap Sample Means")
#plt.show()

# ===> Bayesian Approach to Inference: PROBABILITY OF GETTING HEADS
from scipy.stats import beta

# Priors
a_prior = 2
b_prior = 2

# data
heads, tails = 7, 3

# Posterior params
a_post = a_prior + heads
b_post = b_prior + tails

# Compute posterior dist
post = beta(a_post, b_post)

# Calculate point estimates
mew_est = post.mean()
med_est =  post.median()

# Credible interval
cred_interval = post.interval(0.95)

print(f"Posterior dist params: {a_post}, {b_post}")
print(f"Mean/Median estimate: {mew_est:.3f}, {med_est:.3f}")
print(f"95% Credible Interval: [{cred_interval[0]:.3f}, {cred_interval[1]:.3f}]")







