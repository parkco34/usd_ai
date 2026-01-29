#!/usr/bin/env python
"""
In real-world AI deployments, such as banking systems for detecting fraudulent transactions (where fraud cases represent <1% of data, leading to imbalanced classes), model performance is often evaluated using precision on limited validation sets. A baseline model like logistic regression might show 0.90 precision on a small validation sample of n=100, outranking a more sophisticated ensemble method at 0.60, yet on full production data, the ensemble excels. Use Python to simulate the phenomenon. Assume "true" population precisions for the logistic model (mean ≈0.85, σ=0.10) and ensemble (mean ≈0.88, σ=0.08) follow a lognormal distribution to reflect the positive-skewed nature of precision scores in imbalanced settings. Generate populations of size 10,000 for each model, then perform 1,000 bootstrap resamples at n=50 (small validation) and n=500 (realistic production scale). Compute and plot the sampling distributions of the bootstrap means for both models in each scenario. Calculate the empirical SE for the small-n case. Explain how CLT predicts the distribution shapes, why extreme rankings emerge at low n, and how this variability impacts AI decisions. Propose a mitigation for imbalanced data to prevent errant decisions.
"""
"""
Sampling Distributions Simulation: Precision on Imbalanced Fraud Detection

Goal:
- Show how a small validation sample (n=50) can produce volatile precision estimates that
  mis-rank models, even when the ensemble is truly better in the population.
- Compare with a larger sample (n=500), where rankings stabilize.

Assumptions (given):
- "True" population precision values are positively skewed -> model as Lognormal.
- Logistic population: mean ≈ 0.85, sd = 0.10
- Ensemble population: mean ≈ 0.88, sd = 0.08
- Population size per model: 10,000
- Bootstrap resamples: B = 1,000
- Resample sizes: n=50 and n=500

Notes:
- We treat the 10,000 simulated values as the "population" of possible precision outcomes.
- Then we bootstrap sample means from that population to approximate the sampling distribution
  of the mean precision.
"""

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Helpers
# =========================
def lognormal_params_from_mean_sd(mean: float, sd: float):
    """
    Convert desired Lognormal mean/sd on the original scale
    into Normal(mu, sigma^2) parameters on the log scale.

    If X ~ LogNormal(mu, sigma^2), then:
      E[X] = exp(mu + sigma^2/2)
      Var[X] = (exp(sigma^2) - 1) * exp(2mu + sigma^2)

    Solve:
      sigma^2 = ln(1 + (sd^2 / mean^2))
      mu      = ln(mean) - sigma^2 / 2
    """
    sigma2 = np.log(1.0 + (sd**2) / (mean**2))
    mu = np.log(mean) - 0.5 * sigma2
    sigma = np.sqrt(sigma2)
    return mu, sigma


def bootstrap_means(pop: np.ndarray, n: int, B: int, rng: np.random.Generator):
    """
    Bootstrap sampling distribution of the sample mean from a fixed population array.
    """
    # indices shape: (B, n)
    idx = rng.integers(0, pop.size, size=(B, n))
    return pop[idx].mean(axis=1)


def summarize_sampling(means: np.ndarray, label: str):
    """
    Small summary for reporting.
    """
    return {
        "label": label,
        "boot_mean": float(np.mean(means)),
        "boot_sd": float(np.std(means, ddof=1)),
        "p05": float(np.quantile(means, 0.05)),
        "p95": float(np.quantile(means, 0.95)),
    }


# =========================
# Simulation setup
# =========================
rng = np.random.default_rng(73)

POP_SIZE = 10_000
B = 1_000
n_small = 50
n_large = 500

# Given target population moments (original scale)
logistic_mean_target, logistic_sd_target = 0.85, 0.10
ensemble_mean_target, ensemble_sd_target = 0.88, 0.08

# Convert to log-scale parameters
mu_L, sig_L = lognormal_params_from_mean_sd(logistic_mean_target, logistic_sd_target)
mu_E, sig_E = lognormal_params_from_mean_sd(ensemble_mean_target, ensemble_sd_target)

# Simulate "population" of precision values for each model
pop_logistic = rng.lognormal(mean=mu_L, sigma=sig_L, size=POP_SIZE)
pop_ensemble = rng.lognormal(mean=mu_E, sigma=sig_E, size=POP_SIZE)

# Precision is in [0,1] in reality; lognormal can exceed 1.
# For a clean interpretation as "precision-like", clip to [0, 1].
pop_logistic = np.clip(pop_logistic, 0.0, 1.0)
pop_ensemble = np.clip(pop_ensemble, 0.0, 1.0)

# Population summaries (empirical)
pop_L_mean = pop_logistic.mean()
pop_L_sd = pop_logistic.std(ddof=1)
pop_E_mean = pop_ensemble.mean()
pop_E_sd = pop_ensemble.std(ddof=1)

print("POPULATION (simulated) summaries")
print("=" * 40)
print(f"Logistic  : mean={pop_L_mean:.3f}, sd={pop_L_sd:.3f}")
print(f"Ensemble  : mean={pop_E_mean:.3f}, sd={pop_E_sd:.3f}")
print("\nInterpretation:")
print(
    "These 10,000-point arrays represent the 'production-scale' variability in precision that can occur\n"
    "across different samples/slices of fraud data. The ensemble is slightly better on average, and both\n"
    "models have right-skewed outcomes (rare, very high precision events can happen)."
)

# =========================
# Bootstrap sampling distributions of mean precision
# =========================
means_L_small = bootstrap_means(pop_logistic, n=n_small, B=B, rng=rng)
means_E_small = bootstrap_means(pop_ensemble, n=n_small, B=B, rng=rng)

means_L_large = bootstrap_means(pop_logistic, n=n_large, B=B, rng=rng)
means_E_large = bootstrap_means(pop_ensemble, n=n_large, B=B, rng=rng)

# Empirical SE for small-n (sd of the sampling distribution of the mean)
emp_se_L_small = np.std(means_L_small, ddof=1)
emp_se_E_small = np.std(means_E_small, ddof=1)

# CLT-predicted SE (using empirical population sd)
clt_se_L_small = pop_L_sd / np.sqrt(n_small)
clt_se_E_small = pop_E_sd / np.sqrt(n_small)

# Mis-ranking probability: logistic > ensemble (based on mean precision estimates)
p_wrong_small = float(np.mean(means_L_small > means_E_small))
p_wrong_large = float(np.mean(means_L_large > means_E_large))

print("\nSAMPLING DISTRIBUTIONS (bootstrap means) summaries")
print("=" * 55)
s1 = summarize_sampling(means_L_small, "Logistic (n=50)")
s2 = summarize_sampling(means_E_small, "Ensemble (n=50)")
s3 = summarize_sampling(means_L_large, "Logistic (n=500)")
s4 = summarize_sampling(means_E_large, "Ensemble (n=500)")

for s in [s1, s2, s3, s4]:
    print(
        f"{s['label']:<16}  mean={s['boot_mean']:.3f}  sd(SE)={s['boot_sd']:.3f}  "
        f"5th-95th=[{s['p05']:.3f}, {s['p95']:.3f}]"
    )

print("\nEmpirical SE for small-n (n=50):")
print(f"  SE(Logistic) empirical = {emp_se_L_small:.4f}   | CLT approx = {clt_se_L_small:.4f}")
print(f"  SE(Ensemble) empirical = {emp_se_E_small:.4f}   | CLT approx = {clt_se_E_small:.4f}")

print("\nRanking volatility (probability of wrong ranking due to sampling noise):")
print(f"  P( Logistic mean > Ensemble mean ) when n=50  ≈ {p_wrong_small:.3f}")
print(f"  P( Logistic mean > Ensemble mean ) when n=500 ≈ {p_wrong_large:.3f}")

print("\nInterpretation:")
print(
    "At n=50, the sampling distributions of the mean precision are much wider (larger SE), so it is not rare\n"
    "to see the worse model (logistic) appear better purely by chance. At n=500, SE shrinks by ~1/sqrt(n),\n"
    "the distributions tighten, and wrong rankings become far less frequent."
)

# =========================
# Plotting: sampling distributions
# =========================
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

# (1) n=50
axes[0, 0].hist(means_L_small, bins=30, density=True, alpha=0.7, label="Logistic")
axes[0, 0].hist(means_E_small, bins=30, density=True, alpha=0.7, label="Ensemble")
axes[0, 0].set_title("Sampling distribution of mean precision (n=50, B=1000)")
axes[0, 0].set_xlabel("Bootstrap mean precision")
axes[0, 0].set_ylabel("Density")
axes[0, 0].legend()

# (2) n=500
axes[0, 1].hist(means_L_large, bins=30, density=True, alpha=0.7, label="Logistic")
axes[0, 1].hist(means_E_large, bins=30, density=True, alpha=0.7, label="Ensemble")
axes[0, 1].set_title("Sampling distribution of mean precision (n=500, B=1000)")
axes[0, 1].set_xlabel("Bootstrap mean precision")
axes[0, 1].set_ylabel("Density")
axes[0, 1].legend()

# (3) Show tightening directly: logistic only
axes[1, 0].hist(means_L_small, bins=30, density=True, alpha=0.7, label="n=50")
axes[1, 0].hist(means_L_large, bins=30, density=True, alpha=0.7, label="n=500")
axes[1, 0].set_title("Logistic: mean precision tightens as n increases")
axes[1, 0].set_xlabel("Bootstrap mean precision")
axes[1, 0].set_ylabel("Density")
axes[1, 0].legend()

# (4) Show tightening directly: ensemble only
axes[1, 1].hist(means_E_small, bins=30, density=True, alpha=0.7, label="n=50")
axes[1, 1].hist(means_E_large, bins=30, density=True, alpha=0.7, label="n=500")
axes[1, 1].set_title("Ensemble: mean precision tightens as n increases")
axes[1, 1].set_xlabel("Bootstrap mean precision")
axes[1, 1].set_ylabel("Density")
axes[1, 1].legend()

plt.show()

print("\nCLT connection (what to say in words):")
print("=" * 40)
print(
    "Even if individual precision outcomes are skewed (not Normal), the Central Limit Theorem says the\n"
    "sample mean (or bootstrap mean) tends toward an approximately Normal distribution as n grows.\n"
    "The key scaling is:\n"
    "    SE(Ȳ) ≈ σ / sqrt(n)\n"
    "So increasing n from 50 to 500 shrinks SE by sqrt(50/500)=sqrt(0.1)≈0.316 (about a 68% reduction).\n"
    "That tightening is exactly why rankings stabilize with larger validation/production evaluation."
)

print("\nMitigation proposal (to prevent errant AI deployment decisions):")
print("=" * 60)
print(
    "1) Use uncertainty-aware model selection:\n"
    "   - Instead of ranking by a single precision point estimate, compute bootstrap (or CV) confidence\n"
    "     intervals for precision (or PR-AUC) and prefer models that are better with high probability.\n"
    "\n"
    "2) Evaluate with bigger, stratified validation (especially for fraud <1%):\n"
    "   - Ensure enough positive (fraud) cases by stratified sampling, longer collection windows,\n"
    "     or targeted labeling. Small-n precision is dominated by whether you happened to sample a few\n"
    "     hard/easy fraud examples.\n"
    "\n"
    "3) Use metrics appropriate for imbalance:\n"
    "   - Precision alone can be misleading; include recall, PR-AUC, and cost-weighted utility.\n"
    "     Then tune thresholds for business objectives (e.g., maximize recall at a minimum precision).\n"
    "\n"
    "4) Training-side imbalance handling:\n"
    "   - Class-weighted loss, focal loss, or careful resampling (e.g., undersample majority + keep a\n"
    "     clean holdout) reduces sensitivity to imbalance and can improve stability in production."
)

# Citation marker requested by the environment
print("\nReference marker for course text context: :contentReference[oaicite:0]{index=0}")

