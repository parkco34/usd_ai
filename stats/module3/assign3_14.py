#!/usr/bin/env python
"""
On each bet in a sequence of bets, you win 1 dollar with probability 0.50 and lose 1 dollar (i.e., win negative 1 dollar) with probability 0.50. Let $Y$ denote the total of your winnings and losings after 100 bets. Giving your reasoning, state the approximate distribution of $Y.$
"""
import numpy as np

n_bets = 100
p_win = 1/2

# Possible outcomes
results = np.array([1, -1])
probs = np.array([p_win, 1 - p_win])

# Theoretical
mew = n_bets * (p_win * 1 + (1 - p_win) * (-1))
var = n_bets * 1
std = np.sqrt(var)

print("THEORETICAL RESULTS")
print("=" * 30)
print(f"Mean of Y: {mew}")
print(f"Variance of Y: {var}")
print(f"Standard Deviation of Y: {std}")
print("Approximate distribution: Y ~ Normal(0, 100)")

# Simulation
rng = np.random.default_rng(73)
n_sim = int(1e5)

# Generate random sample
X = rng.choice(results, size=(n_sim, n_bets), p=probs)

# Total wins
Y = X.sum(axis=1)

print("SIMULATION RESULTS")
print("=" * 30)
print(f"Simulated mean of Y: {Y.mean():.2f}")
print(f"Simulated SD of Y: {Y.std(ddof=1):.2f}")

# Interpretation
print("\nINTERPRETATION")
print("=" * 30)
print(
    "After 100 fair bets, total winnings are centered near $0.\n"
    "The variability is about $10, so outcomes within roughly\n"
    "[-20, 20] dollars are common. The simulated distribution\n"
    "closely matches a Normal(0, 100), confirming the CLT approximation."
)

