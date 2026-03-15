#!/usr/bin/env python
"""
Maximizing the objective function
Objective function: Z = 4x + 3y
SUBJECT TO:
    x ≥ 0
    y ≥2
    2y ≤ 25 - x
    4y ≥ 2x - 8
    y ≤ 2x -5
"""
import pulp as pl

# Instanitating problem class
example = pl.LpProblem("Example_Problem", pl.LpMaximize)

# Decision variables w/ constraints
x = pl.LpVariable("x", lowBound=0, cat="Continuous")
y = pl.LpVariable("y", lowBound=2, cat="Continuous")

# Objective function and constraints are added using '+='
# First the objective function, 
example += 4 * x + 3 * y, "Z"
# Then, Constraints
example += 2 * y <= 25 - x
example += 4 * y >= 2 * x - 8
example += y <= 2 * x - 5

print(f"Problem: {example}")
print(f"\nSolution to problem:\n {example.solve()}")
print(pl.LpStatus[example.status])

# Retrieve the values of the decision variables
for var in example.variables():
    print(f"{var.name} = {var.varValue}")

# Output the optimal solution
print(pl.value(example.objective))









