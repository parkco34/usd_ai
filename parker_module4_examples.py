"""
Assignment 4.1 - Example Solutions for Learning
================================================
Parker - These 5 solutions demonstrate key patterns you'll reuse.
Pay attention to the STRUCTURE of each solution, not just the code.

Pattern to internalize:
1. State what you're computing (in comments)
2. Set up your data/parameters
3. Perform the computation
4. Visualize and interpret

Remember to cite AI assistance per your syllabus!
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Set a consistent style for all plots
plt.style.use('seaborn-v0_8-whitegrid')

# OUTPUT DIRECTORY - saves plots to same folder as this script
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_plot(filename):
    """Helper to save plots to the script's directory."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=150)
    print(f"Plot saved to: {filepath}")
    return filepath


# =============================================================================
# PROBLEM 4.1: Comparing Estimators via Monte Carlo Simulation
# =============================================================================
# CONCEPT: We want to compare two ways of estimating the population mean:
#   1. Sample mean (y_bar) - the usual average
#   2. Quartile average - (Q1 + Q3) / 2 - supposedly more robust to outliers
#
# APPROACH: Generate many samples, compute both estimators for each,
#           then compare their standard deviations (precision)
#
# WHY THIS MATTERS: In AI/ML, you'll often need to evaluate estimators
#                   through simulation when theory is intractable

def problem_4_1():
    """Compare precision of sample mean vs quartile-based estimator."""

    print("=" * 60)
    print("PROBLEM 4.1: Comparing Estimators via Monte Carlo")
    print("=" * 60)

    # -------------------------------------------
    # STEP 1: Define simulation parameters
    # -------------------------------------------
    num_simulations = 100_000  # How many samples to generate
    sample_size = 100          # n = 100 as specified

    # We'll store our estimates here
    # Think of these as "buckets" to collect results
    sample_means = np.zeros(num_simulations)
    quartile_averages = np.zeros(num_simulations)

    # -------------------------------------------
    # STEP 2: Run the simulation
    # -------------------------------------------
    # Set seed for reproducibility (good practice!)
    np.random.seed(42)

    for i in range(num_simulations):
        # Generate one sample from N(0,1)
        sample = np.random.standard_normal(sample_size)

        # Compute both estimators for this sample
        sample_means[i] = np.mean(sample)

        # Quartile average: (Q1 + Q3) / 2
        q1 = np.percentile(sample, 25)
        q3 = np.percentile(sample, 75)
        quartile_averages[i] = (q1 + q3) / 2

    # -------------------------------------------
    # STEP 3: Analyze the results
    # -------------------------------------------
    # Standard deviation of the estimates = empirical standard error
    se_sample_mean = np.std(sample_means)
    se_quartile_avg = np.std(quartile_averages)

    # Theoretical SE for sample mean from N(0,1): sigma/sqrt(n) = 1/sqrt(100) = 0.1
    theoretical_se = 1 / np.sqrt(sample_size)

    # -------------------------------------------
    # STEP 4: Report findings
    # -------------------------------------------
    print(f"\nResults from {num_simulations:,} simulations (n={sample_size}):")
    print(f"  Sample Mean - Empirical SE:     {se_sample_mean:.4f}")
    print(f"  Sample Mean - Theoretical SE:   {theoretical_se:.4f}")
    print(f"  Quartile Average - Empirical SE: {se_quartile_avg:.4f}")
    print(f"\nPrecision Ratio (Quartile/Mean): {se_quartile_avg/se_sample_mean:.2f}")

    # INTERPRETATION
    print("\n--- INTERPRETATION ---")
    print(f"The quartile-based estimator has SE that is {se_quartile_avg/se_sample_mean:.2f}x")
    print("larger than the sample mean's SE.")
    print("This means it's LESS precise - you trade precision for robustness to outliers.")

    # -------------------------------------------
    # STEP 5: Visualize
    # -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of sample means
    axes[0].hist(sample_means, bins=50, density=True, alpha=0.7, color='steelblue')
    axes[0].axvline(0, color='red', linestyle='--', label='True mean (μ=0)')
    axes[0].set_xlabel('Sample Mean')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Distribution of Sample Means\n(SE = {se_sample_mean:.4f})')
    axes[0].legend()

    # Histogram of quartile averages
    axes[1].hist(quartile_averages, bins=50, density=True, alpha=0.7, color='coral')
    axes[1].axvline(0, color='red', linestyle='--', label='True mean (μ=0)')
    axes[1].set_xlabel('Quartile Average')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Distribution of Quartile Averages\n(SE = {se_quartile_avg:.4f})')
    axes[1].legend()

    plt.tight_layout()
    save_plot('problem_4_1_plot.png')
    plt.close()
    return se_sample_mean, se_quartile_avg


# =============================================================================
# PROBLEM 4.2: Likelihood Function for Geometric Distribution
# =============================================================================
# CONCEPT: The likelihood function L(π|y) tells us how "likely" different
#          parameter values are, given our observed data.
#
# For geometric distribution: first success on trial y
#   P(Y = y | π) = (1-π)^(y-1) * π
#
# When y=3: L(π) = (1-π)^2 * π
#
# WHY THIS MATTERS: Likelihood is THE foundation of statistical inference.
#                   MLE, Bayesian inference, neural network training - all use it.

def problem_4_2():
    """Plot likelihood function for geometric distribution with y=3."""

    print("\n" + "=" * 60)
    print("PROBLEM 4.2: Likelihood Function for Geometric Distribution")
    print("=" * 60)

    # -------------------------------------------
    # STEP 1: Define the observation
    # -------------------------------------------
    y = 3  # First success on observation 3

    # -------------------------------------------
    # STEP 2: Create array of possible π values
    # -------------------------------------------
    # π must be between 0 and 1 (it's a probability)
    # Use many points for smooth curve
    pi_values = np.linspace(0.001, 0.999, 500)

    # -------------------------------------------
    # STEP 3: Compute likelihood for each π
    # -------------------------------------------
    # L(π | y=3) = (1-π)^(y-1) * π = (1-π)^2 * π
    likelihood = ((1 - pi_values) ** (y - 1)) * pi_values

    # -------------------------------------------
    # STEP 4: Find the MLE (maximum likelihood estimate)
    # -------------------------------------------
    # For geometric: MLE of π = 1/y
    pi_mle = 1 / y
    likelihood_at_mle = ((1 - pi_mle) ** (y - 1)) * pi_mle

    # -------------------------------------------
    # STEP 5: Plot the likelihood function
    # -------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(pi_values, likelihood, 'b-', linewidth=2, label='Likelihood L(π|y=3)')

    # Mark the MLE
    plt.axvline(pi_mle, color='red', linestyle='--', linewidth=1.5,
                label=f'MLE: π = 1/{y} = {pi_mle:.4f}')
    plt.scatter([pi_mle], [likelihood_at_mle], color='red', s=100, zorder=5)

    plt.xlabel('π (probability of success)', fontsize=12)
    plt.ylabel('L(π | y=3)', fontsize=12)
    plt.title('Likelihood Function for Geometric Distribution\nObservation: First Success on Trial y=3',
              fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Add annotation
    plt.annotate(f'Maximum at π = {pi_mle:.4f}\nL = {likelihood_at_mle:.4f}',
                 xy=(pi_mle, likelihood_at_mle),
                 xytext=(pi_mle + 0.15, likelihood_at_mle + 0.02),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    save_plot('problem_4_2_plot.png')
    plt.close()

    # -------------------------------------------
    # STEP 6: Report
    # -------------------------------------------
    print(f"\nObservation: y = {y} (first success on trial 3)")
    print(f"Likelihood function: L(π) = (1-π)^{y-1} × π = (1-π)² × π")
    print(f"\nMaximum Likelihood Estimate (MLE): π = 1/y = {pi_mle:.4f}")
    print(f"Likelihood at MLE: {likelihood_at_mle:.4f}")

    print("\n--- INTERPRETATION ---")
    print("The likelihood function shows how 'compatible' each π value is with")
    print("observing y=3. The MLE (1/3 ≈ 0.333) is our best single guess for π.")
    print("Note how the curve is asymmetric - this reflects uncertainty in our estimate.")

    return pi_mle


# =============================================================================
# PROBLEM 4.11: Confidence Intervals and Outlier Effects
# =============================================================================
# CONCEPT: A 95% CI gives a range where we're "95% confident" the true
#          population mean lies. But it's sensitive to outliers!
#
# Formula: CI = x̄ ± t_(α/2, n-1) × (s / √n)
#
# WHY THIS MATTERS: Understanding CI robustness is crucial in data science.
#                   One bad data point can completely mislead your analysis.

def problem_4_11():
    """Construct 95% CI and examine outlier effects."""

    print("\n" + "=" * 60)
    print("PROBLEM 4.11: Confidence Intervals and Outlier Effects")
    print("=" * 60)

    # -------------------------------------------
    # STEP 1: Enter the data
    # -------------------------------------------
    # TV watching hours for 10 Islamic subjects in 2018 GSS
    data_original = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 4])

    # Create contaminated data (4 incorrectly recorded as 24)
    data_outlier = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 24])

    # -------------------------------------------
    # STEP 2: Write a function to compute CI
    # -------------------------------------------
    # This is a REUSABLE pattern - you'll use this again!
    def compute_95_ci(data, label=""):
        """Compute 95% confidence interval for the mean."""
        n = len(data)
        x_bar = np.mean(data)
        s = np.std(data, ddof=1)  # ddof=1 for sample std dev
        se = s / np.sqrt(n)

        # t critical value for 95% CI with n-1 degrees of freedom
        t_crit = stats.t.ppf(0.975, df=n-1)

        # Margin of error
        margin = t_crit * se

        # CI bounds
        lower = x_bar - margin
        upper = x_bar + margin

        print(f"\n{label}")
        print(f"  n = {n}")
        print(f"  Mean (x̄) = {x_bar:.2f}")
        print(f"  Std Dev (s) = {s:.2f}")
        print(f"  Std Error (SE) = {se:.2f}")
        print(f"  t-critical (α=0.05, df={n-1}) = {t_crit:.3f}")
        print(f"  95% CI: ({lower:.2f}, {upper:.2f})")

        return lower, upper, x_bar, se

    # -------------------------------------------
    # STEP 3: Compute CIs for both datasets
    # -------------------------------------------
    print("\n--- PART (a): Original Data ---")
    ci_orig = compute_95_ci(data_original, "Original Data: [0,0,1,1,1,2,2,3,3,4]")

    print("\n--- PART (b): Data with Outlier ---")
    ci_outlier = compute_95_ci(data_outlier, "Outlier Data: [0,0,1,1,1,2,2,3,3,24]")

    # -------------------------------------------
    # STEP 4: Compare and interpret
    # -------------------------------------------
    print("\n--- COMPARISON ---")
    ci_width_orig = ci_orig[1] - ci_orig[0]
    ci_width_outlier = ci_outlier[1] - ci_outlier[0]

    print(f"Original CI width:  {ci_width_orig:.2f} hours")
    print(f"Outlier CI width:   {ci_width_outlier:.2f} hours")
    print(f"Width increase:     {ci_width_outlier/ci_width_orig:.1f}x")

    print("\n--- INTERPRETATION ---")
    print("The outlier (24 instead of 4) dramatically affects the CI:")
    print(f"  - Mean shifted from {ci_orig[2]:.2f} to {ci_outlier[2]:.2f}")
    print(f"  - CI width increased by {ci_width_outlier/ci_width_orig:.1f}x")
    print("This demonstrates that CIs for means are NOT robust to outliers.")
    print("A single transcription error can completely invalidate inference.")

    # -------------------------------------------
    # STEP 5: Visualize
    # -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot data points
    axes[0].scatter(range(len(data_original)), data_original, s=100, alpha=0.7, label='Original')
    axes[0].scatter(range(len(data_outlier)), data_outlier, s=100, alpha=0.7, marker='x', label='With outlier')
    axes[0].set_xlabel('Observation Index')
    axes[0].set_ylabel('Hours of TV')
    axes[0].set_title('Data Comparison')
    axes[0].legend()

    # Plot CIs
    y_positions = [1, 0]
    labels = ['Original', 'With Outlier']
    colors = ['steelblue', 'coral']
    cis = [ci_orig, ci_outlier]

    for i, (ci, label, color) in enumerate(zip(cis, labels, colors)):
        lower, upper, mean, _ = ci
        axes[1].plot([lower, upper], [y_positions[i], y_positions[i]],
                     color=color, linewidth=3, label=label)
        axes[1].scatter([mean], [y_positions[i]], color=color, s=100, zorder=5)
        axes[1].scatter([lower, upper], [y_positions[i], y_positions[i]],
                        color=color, s=50, marker='|')

    axes[1].set_xlabel('Hours of TV Watching')
    axes[1].set_ylabel('')
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(labels)
    axes[1].set_title('95% Confidence Intervals Comparison')
    axes[1].legend()

    plt.tight_layout()
    save_plot('problem_4_11_plot.png')
    plt.close()
    return ci_orig, ci_outlier


# =============================================================================
# PROBLEM 5.6: Hypothesis Testing for Proportions (Frequentist + Bayesian)
# =============================================================================
# CONCEPT: Test if Republican candidate has majority support (π > 0.50)
#          Compare frequentist p-values with Bayesian posterior probabilities
#
# State A: 59/100 prefer Republican
# State B: 525/1000 prefer Republican
#
# WHY THIS MATTERS: This directly compares frequentist vs Bayesian thinking.
#                   As an AI student, you MUST understand both paradigms.

def problem_5_6():
    """Compare polling evidence using frequentist and Bayesian approaches."""

    print("\n" + "=" * 60)
    print("PROBLEM 5.6: Frequentist vs Bayesian Hypothesis Testing")
    print("=" * 60)

    # -------------------------------------------
    # STEP 1: Define the data
    # -------------------------------------------
    # State A: 59 out of 100
    n_A, x_A = 100, 59
    p_hat_A = x_A / n_A

    # State B: 525 out of 1000
    n_B, x_B = 1000, 525
    p_hat_B = x_B / n_B

    print(f"\nState A: {x_A}/{n_A} = {p_hat_A:.1%} prefer Republican")
    print(f"State B: {x_B}/{n_B} = {p_hat_B:.1%} prefer Republican")

    # -------------------------------------------
    # PART (a): Frequentist Approach
    # -------------------------------------------
    print("\n--- PART (a): Frequentist Hypothesis Test ---")
    print("H₀: π = 0.50 (no majority)")
    print("Hₐ: π > 0.50 (Republican majority)")

    # One-proportion z-test
    # z = (p_hat - p_0) / sqrt(p_0 * (1-p_0) / n)
    p_0 = 0.50

    # State A
    z_A = (p_hat_A - p_0) / np.sqrt(p_0 * (1 - p_0) / n_A)
    pvalue_A = 1 - stats.norm.cdf(z_A)  # One-tailed (>)

    # State B
    z_B = (p_hat_B - p_0) / np.sqrt(p_0 * (1 - p_0) / n_B)
    pvalue_B = 1 - stats.norm.cdf(z_B)

    print(f"\nState A: z = {z_A:.2f}, p-value = {pvalue_A:.4f}")
    print(f"State B: z = {z_B:.2f}, p-value = {pvalue_B:.4f}")

    print("\n  INTERPRETATION:")
    if pvalue_A < pvalue_B:
        print(f"  State A has STRONGER evidence (smaller p-value: {pvalue_A:.4f} < {pvalue_B:.4f})")
    else:
        print(f"  State B has STRONGER evidence (smaller p-value: {pvalue_B:.4f} < {pvalue_A:.4f})")

    print("\n  WHY? State A has a larger sample proportion (59% vs 52.5%),")
    print("  which matters more than State B's larger sample size for THIS question.")

    # -------------------------------------------
    # PART (b): Bayesian Approach
    # -------------------------------------------
    print("\n--- PART (b): Bayesian Analysis ---")
    print("Prior: Beta(50, 50) - strong prior belief π is near 0.50")

    # Prior parameters
    alpha_prior, beta_prior = 50, 50

    # Posterior after observing data:
    # Beta(alpha_prior + x, beta_prior + n - x)

    # State A posterior
    alpha_A = alpha_prior + x_A
    beta_A = beta_prior + (n_A - x_A)

    # State B posterior
    alpha_B = alpha_prior + x_B
    beta_B = beta_prior + (n_B - x_B)

    # P(π < 0.50 | data) - probability of being WRONG about Republican victory
    prob_wrong_A = stats.beta.cdf(0.50, alpha_A, beta_A)
    prob_wrong_B = stats.beta.cdf(0.50, alpha_B, beta_B)

    print(f"\nState A: Posterior Beta({alpha_A}, {beta_A})")
    print(f"  P(π < 0.50 | data) = {prob_wrong_A:.6f}")

    print(f"\nState B: Posterior Beta({alpha_B}, {beta_B})")
    print(f"  P(π < 0.50 | data) = {prob_wrong_B:.6f}")

    print("\n  INTERPRETATION:")
    if prob_wrong_A < prob_wrong_B:
        print(f"  State A has STRONGER evidence for Republican victory")
        print(f"  (lower posterior probability of being wrong: {prob_wrong_A:.6f} < {prob_wrong_B:.6f})")
    else:
        print(f"  State B has STRONGER evidence for Republican victory")
        print(f"  (lower posterior probability of being wrong: {prob_wrong_B:.6f} < {prob_wrong_A:.6f})")

    print("\n  NOTE: The prior Beta(50,50) is quite strong and 'pulls' estimates")
    print("  toward 0.50. State B's larger sample overcomes this pull more.")

    # -------------------------------------------
    # STEP 3: Visualize
    # -------------------------------------------
    pi_range = np.linspace(0.35, 0.75, 500)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prior
    prior = stats.beta.pdf(pi_range, alpha_prior, beta_prior)

    # State A
    posterior_A = stats.beta.pdf(pi_range, alpha_A, beta_A)
    axes[0].plot(pi_range, prior, 'k--', linewidth=2, label='Prior Beta(50,50)')
    axes[0].plot(pi_range, posterior_A, 'b-', linewidth=2, label=f'Posterior Beta({alpha_A},{beta_A})')
    axes[0].axvline(0.50, color='red', linestyle=':', label='π = 0.50')
    axes[0].fill_between(pi_range[pi_range < 0.50], 0,
                         posterior_A[pi_range < 0.50], alpha=0.3, color='red')
    axes[0].set_xlabel('π (Republican proportion)')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'State A: 59/100\nP(π < 0.50) = {prob_wrong_A:.4f}')
    axes[0].legend()

    # State B
    posterior_B = stats.beta.pdf(pi_range, alpha_B, beta_B)
    axes[1].plot(pi_range, prior, 'k--', linewidth=2, label='Prior Beta(50,50)')
    axes[1].plot(pi_range, posterior_B, 'b-', linewidth=2, label=f'Posterior Beta({alpha_B},{beta_B})')
    axes[1].axvline(0.50, color='red', linestyle=':', label='π = 0.50')
    axes[1].fill_between(pi_range[pi_range < 0.50], 0,
                         posterior_B[pi_range < 0.50], alpha=0.3, color='red')
    axes[1].set_xlabel('π (Republican proportion)')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'State B: 525/1000\nP(π < 0.50) = {prob_wrong_B:.4f}')
    axes[1].legend()

    plt.tight_layout()
    save_plot('problem_5_6_plot.png')
    plt.close()
    return (pvalue_A, pvalue_B), (prob_wrong_A, prob_wrong_B)


# =============================================================================
# PROBLEM 5.23: Chi-Squared Test of Independence
# =============================================================================
# CONCEPT: Test whether two categorical variables are independent
#          (sentiment toward AI vs gender)
#
# Steps: 1) Create contingency table
#        2) Compute expected frequencies under independence
#        3) Calculate χ² statistic
#        4) Get p-value and interpret
#
# WHY THIS MATTERS: Chi-squared tests are EVERYWHERE in data science
#                   A/B testing, feature selection, categorical analysis

def problem_5_23():
    """Chi-squared test for AI sentiment independence from gender."""

    print("\n" + "=" * 60)
    print("PROBLEM 5.23: Chi-Squared Test of Independence")
    print("=" * 60)

    # -------------------------------------------
    # STEP 1: Generate the dataset (as specified)
    # -------------------------------------------
    import pandas as pd

    np.random.seed(2024)  # MUST use this seed for reproducibility!
    n = 300

    sentiments = ["Positive", "Neutral", "Negative"]
    genders = ["Male", "Female", "Other"]

    df_ai = pd.DataFrame({
        "sentiment": np.random.choice(sentiments, size=n, p=[0.44, 0.33, 0.23]),
        "gender": np.random.choice(genders, size=n, p=[0.49, 0.48, 0.03]),
        "age": np.random.randint(18, 75, size=n),
        "ai_usage_frequency": np.random.choice(["Daily", "Weekly", "Rarely", "Never"], size=n),
        "trust_in_ai": np.random.randint(1, 6, size=n),
    })

    print(f"\nDataset created with {len(df_ai)} observations")
    print("\nFirst few rows:")
    print(df_ai.head())

    # -------------------------------------------
    # PART (a): Create contingency table
    # -------------------------------------------
    print("\n--- PART (a): Contingency Table ---")

    # pd.crosstab is your friend for contingency tables!
    contingency_table = pd.crosstab(df_ai['sentiment'], df_ai['gender'], margins=True)
    print("\nContingency Table (Sentiment × Gender):")
    print(contingency_table)

    # -------------------------------------------
    # PART (b): Chi-squared test
    # -------------------------------------------
    print("\n--- PART (b): Chi-Squared Test ---")
    print("H₀: Sentiment toward AI is INDEPENDENT of gender")
    print("Hₐ: Sentiment toward AI is NOT independent of gender")

    # Get table without margins for the test
    observed = pd.crosstab(df_ai['sentiment'], df_ai['gender'])

    # Perform chi-squared test
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)

    print(f"\nObserved frequencies:")
    print(observed)

    print(f"\nExpected frequencies (under independence):")
    expected_df = pd.DataFrame(expected,
                               index=observed.index,
                               columns=observed.columns)
    print(expected_df.round(2))

    print(f"\nTest Results:")
    print(f"  Chi-squared statistic: {chi2:.2f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  P-value: {p_value:.4f}")

    # -------------------------------------------
    # PART (c): Interpretation
    # -------------------------------------------
    print("\n--- PART (c): Interpretation ---")

    alpha = 0.05
    if p_value < alpha:
        print(f"Since p-value ({p_value:.4f}) < α ({alpha}), we REJECT H₀.")
        print("There IS a statistically significant association between")
        print("sentiment toward AI and gender.")
    else:
        print(f"Since p-value ({p_value:.4f}) ≥ α ({alpha}), we FAIL TO REJECT H₀.")
        print("There is NOT sufficient evidence of an association between")
        print("sentiment toward AI and gender.")

    print("\nContext: In this simulated dataset, sentiment and gender were")
    print("generated INDEPENDENTLY (using np.random.choice separately).")
    print(f"So finding p-value = {p_value:.4f} is consistent with the data generation.")

    # -------------------------------------------
    # STEP 5: Visualize
    # -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stacked bar chart
    sentiment_by_gender = pd.crosstab(df_ai['gender'], df_ai['sentiment'], normalize='index')
    sentiment_by_gender.plot(kind='bar', stacked=True, ax=axes[0],
                             color=['#66c2a5', '#fc8d62', '#8da0cb'])
    axes[0].set_xlabel('Gender')
    axes[0].set_ylabel('Proportion')
    axes[0].set_title('Sentiment Distribution by Gender')
    axes[0].legend(title='Sentiment')
    axes[0].tick_params(axis='x', rotation=0)

    # Heatmap of residuals
    residuals = (observed.values - expected) / np.sqrt(expected)
    im = axes[1].imshow(residuals, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
    axes[1].set_xticks(range(len(observed.columns)))
    axes[1].set_xticklabels(observed.columns)
    axes[1].set_yticks(range(len(observed.index)))
    axes[1].set_yticklabels(observed.index)
    axes[1].set_xlabel('Gender')
    axes[1].set_ylabel('Sentiment')
    axes[1].set_title('Standardized Residuals\n(Red = more than expected, Blue = less)')
    plt.colorbar(im, ax=axes[1], label='Standardized Residual')

    # Add text annotations
    for i in range(len(observed.index)):
        for j in range(len(observed.columns)):
            axes[1].text(j, i, f'{residuals[i,j]:.2f}',
                        ha='center', va='center', fontsize=10)

    plt.tight_layout()
    save_plot('problem_5_23_plot.png')
    plt.close()
    return chi2, p_value, contingency_table


# =============================================================================
# MAIN: Run all examples
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ASSIGNMENT 4.1 - EXAMPLE SOLUTIONS")
    print("  These 5 problems demonstrate patterns for the remaining problems")
    print("=" * 70)

    # Run each problem
    problem_4_1()
    problem_4_2()
    problem_4_11()
    problem_5_6()
    problem_5_23()

    print("\n" + "=" * 70)
    print("  ALL EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nKEY PATTERNS TO APPLY TO REMAINING PROBLEMS:")
    print("  • Problem 4.14, 4.31: Use the CI pattern from 4.11")
    print("  • Problem 5.8: Use t-test pattern (similar structure to 5.6)")
    print("  • Problem 5.10: Two-sample t-test (compare two groups like 5.6)")
    print("\nRemember to cite AI assistance per your syllabus!")
