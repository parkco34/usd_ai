#!/usr/bin/env python
"""
You run a small factory that produces two items: A ($40 profit per unit) and B ($60 profit per unit). Each product requires time on two different machines: Machine 1 and Machine 2. The factory has a limited number of hours available on each machine per week.

Your goal is to maximize total profit, while staying within available machine hours and meeting a minimum quota for product B.

It takes 2 hours on Machine 1 and 3 hours on Machine 2 to make a single unit of item A.

It takes 4 hours on Machine 1 and 2 hours on Machine 2 to make a single unit of item B.

You have 100 hours of time on Machine 1 available per week. You have 80 hours of time on Machine 2 per week. You can also choose to run Machine 2 on Saturdays for overtime: this will get you 20 extra hours of time, but will cost you $500. You are also required to produce at least 10 units of item B per week.

LET:
x = number of units of Product A to produce (continuous)
y = number of units of Product B to produce (continuous)
z = 1 if we operate Machine 2 on Saturday for overtime (binary)
1) Write the objective function for total profit.

2) Write the constraints for this problem.

3) Solve this problem using the PuLP library. What is the optimal combination of products A and B to produce? What is the maximum profit you can make in a week? Should you run Machine B on overtime?
================================================================
1) Decision Variables:
    x = units of Product A (per week)
    y = Units of Product B (per week)
    z = 1 if Machine 2 goes into overtime (Saturday), else 0

2) Parameters:
    ?

3) Constraints:
    Machine 1:
        2x + 4y <= 100

    Machine 2:
        3x + 2y <= 80 + 20z

    B_min: y >= 10
    Nonegativity/Binary:
        x >= 0
        y >= 0
        z ∈ (0, 1)
"""
import pulp as pl
from textwrap import dedent

def build_model(model_name):
    """
    Maximization model for weekly profit, subject to resources and time constraints.
    ----------------------------------------------------------v-
    INPUT:
        model_name: (str) Name of model

    OUTPUT:
        model: (pulp.LpProblem) 
    """
    # Input validation
    if not isinstance(model_name, str):
        raise TypeError("\nInput type invalid")

    return pl.LpProblem(model_name, sense=pl.LpMaximize)

# Build optimization model
model = build_model("two_products")

#  Decision variables
x = pl.LpVariable(name="Product_A_units", lowBound=0, cat="Continuous")
y = pl.LpVariable(name="Product_B_units", lowBound=0, cat="Continuous")
z = pl.LpVariable(name="overtime", cat=pl.LpBinary)

# Objective function (weekly proft)
# Penalty of $500 only when z=1 (overtime)
model += 40 * x + 60 * y - 500 * z, "Weekly_profit"

# Adding constraints
# Machine 1
model += 2 * x + 4 * y <= 100, "machine_1_capacity"

# Machine 2
model += 3 * x + 2 * y <= 80 + 20 * z, "machine_2_capacity"

# Min production for B
model += y >= 10, "minimum_B_quota"

# Solving model
solver = pl.PULP_CBC_CMD(msg=False)
model.solve(solver)

# Solution
stat = pl.LpStatus[model.status]
opt_x = pl.value(x)
opt_y = pl.value(y)
opt_z = pl.value(z)
optimal = pl.value(model.objective)

# Output
print(f"\nSolver status: {stat}")
print(f"x = units of Product A = {opt_x:.3f}")
print(f"y = units of Product B = {opt_y:.3f}")
print(f"Overtime decision = {int(round(opt_z))}")

print("\nObjective Value: (maximum weekly profit)")
print(f"${optimal:.2f}")

# Interpretation
print(dedent(f"""
INTERPRETATION
----------------
The optimal weekly profit is achieved via {opt_x:.2f} units of
Product A and {opt_y:.2f} units of Product B.

Thus, we get a maximum weekly profit of ${optimal:.2f}.

Since z = 0, the model suggests we don't run machine 2 on a Saturday (now overtime).
      """))







