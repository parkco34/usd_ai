"""
Assignment 3.2: Real-World Optimization with PuLP

This script formulates and solves the mixed-integer linear programming problem
from Assignment 3.2. The assignment asks us to maximize weekly profit for a
small factory producing two products, A and B, subject to machine-capacity
constraints, a minimum production quota for product B, and an optional binary
choice to purchase overtime on Machine 2.

Mathematical formulation
------------------------
Decision variables
    x = units of Product A produced per week
    y = units of Product B produced per week
    z = 1 if Machine 2 overtime is used on Saturday, 0 otherwise

Objective function
    Maximize weekly profit:

        P(x, y, z) = 40x + 60y - 500z

Interpretation:
    - Each unit of A contributes $40 profit.
    - Each unit of B contributes $60 profit.
    - If overtime is used, there is a fixed cost of $500.

Constraints
    Machine 1 capacity:
        2x + 4y <= 100

    Machine 2 capacity:
        3x + 2y <= 80 + 20z

    Minimum quota for product B:
        y >= 10

    Nonnegativity and binary structure:
        x >= 0
        y >= 0
        z in {0, 1}

Why this is a mixed-integer linear program
------------------------------------------
The objective and constraints are linear in the decision variables. However,
this is not a pure linear program because z must be binary. Therefore, the
problem is a mixed-integer linear program (MILP).
"""

from pulp import (
    LpBinary,
    LpMaximize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    value,
)


# -----------------------------------------------------------------------------
# Step 1: Build the optimization model
# -----------------------------------------------------------------------------
"""
We construct a maximization model because the assignment explicitly asks for
maximum weekly profit.

Symbolically, we want:

    max_{x, y, z}  40x + 60y - 500z

subject to the resource and quota constraints.
"""
model = LpProblem(name="assignment_3_2_factory_profit_maximization", sense=LpMaximize)


# -----------------------------------------------------------------------------
# Step 2: Define decision variables
# -----------------------------------------------------------------------------
"""
Variable definitions:

x:
    Continuous number of units of Product A.
    Because the assignment itself defines x as continuous, we model x >= 0.

y:
    Continuous number of units of Product B.
    Again, the assignment defines y as continuous, so we model y >= 0.

z:
    Binary overtime decision for Machine 2.
    z = 0 means no overtime.
    z = 1 means purchase 20 additional Machine 2 hours at a fixed cost of $500.
"""
x = LpVariable(name="x_product_A_units", lowBound=0, cat="Continuous")
y = LpVariable(name="y_product_B_units", lowBound=0, cat="Continuous")
z = LpVariable(name="z_machine_2_overtime", cat=LpBinary)


# -----------------------------------------------------------------------------
# Step 3: Add the objective function
# -----------------------------------------------------------------------------
"""
Objective:

    max P = 40x + 60y - 500z

This encodes the economic tradeoff directly.
The term -500z is a fixed-charge penalty: it only activates when z = 1.
"""
model += 40 * x + 60 * y - 500 * z, "weekly_profit"


# -----------------------------------------------------------------------------
# Step 4: Add constraints
# -----------------------------------------------------------------------------
"""
Machine 1 constraint:
    Product A uses 2 hours on Machine 1.
    Product B uses 4 hours on Machine 1.
    Total available weekly hours on Machine 1 = 100.

Therefore:
    2x + 4y <= 100
"""
model += 2 * x + 4 * y <= 100, "machine_1_capacity"

"""
Machine 2 constraint:
    Product A uses 3 hours on Machine 2.
    Product B uses 2 hours on Machine 2.
    Base weekly capacity on Machine 2 = 80 hours.
    Optional overtime adds 20 hours if z = 1.

Therefore:
    3x + 2y <= 80 + 20z

This is the key logical linkage between the binary variable z and the usable
Machine 2 capacity.
"""
model += 3 * x + 2 * y <= 80 + 20 * z, "machine_2_capacity_with_overtime"

"""
Minimum production quota for Product B:

    y >= 10

This means the feasible region is restricted so that any valid production plan
must include at least 10 units of Product B.
"""
model += y >= 10, "minimum_B_quota"


# -----------------------------------------------------------------------------
# Step 5: Solve the model
# -----------------------------------------------------------------------------
"""
We use CBC, the default open-source solver commonly packaged with PuLP.
"""
solver = PULP_CBC_CMD(msg=False)
model.solve(solver)


# -----------------------------------------------------------------------------
# Step 6: Extract and report the solution
# -----------------------------------------------------------------------------
status = LpStatus[model.status]
optimal_x = value(x)
optimal_y = value(y)
optimal_z = value(z)
optimal_profit = value(model.objective)

print("=" * 72)
print("ASSIGNMENT 3.2 - REAL-WORLD OPTIMIZATION WITH PULP")
print("=" * 72)
print(f"Solver status: {status}")
print()
print("Optimal decision variables")
print("-" * 72)
print(f"x = units of Product A = {optimal_x:.4f}")
print(f"y = units of Product B = {optimal_y:.4f}")
print(f"z = overtime decision    = {int(round(optimal_z))}")
print()
print("Objective value")
print("-" * 72)
print(f"Maximum weekly profit = ${optimal_profit:,.2f}")
print()


# -----------------------------------------------------------------------------
# Step 7: Verify the constraints numerically
# -----------------------------------------------------------------------------
"""
It is often useful, especially in an academic setting, to verify the solution by
substituting the optimizer back into the constraints.
"""
machine_1_used = 2 * optimal_x + 4 * optimal_y
machine_2_used = 3 * optimal_x + 2 * optimal_y
machine_1_available = 100
machine_2_available = 80 + 20 * optimal_z

print("Constraint check")
print("-" * 72)
print(f"Machine 1 used: {machine_1_used:.4f} <= {machine_1_available:.4f}")
print(f"Machine 2 used: {machine_2_used:.4f} <= {machine_2_available:.4f}")
print(f"Product B quota: {optimal_y:.4f} >= 10.0000")
print()


# -----------------------------------------------------------------------------
# Step 8: Provide an interpretation in context
# -----------------------------------------------------------------------------
"""
Interpretation of the optimizer:

If z = 0 in the optimal solution, then the model is telling us that the marginal
benefit of the extra 20 Machine 2 hours is not large enough to justify the fixed
$500 overtime charge.

If z = 1, then overtime is economically worthwhile despite the fixed cost.
"""
overtime_used = int(round(optimal_z))

print("Interpretation")
print("-" * 72)
print(
    f"The optimal weekly production plan is to produce {optimal_x:.2f} units of "
    f"Product A and {optimal_y:.2f} units of Product B."
)
print(
    f"This yields a maximum weekly profit of ${optimal_profit:,.2f}."
)

if overtime_used == 1:
    print(
        "Because z = 1, the model recommends running Machine 2 on Saturday "
        "for overtime."
    )
else:
    print(
        "Because z = 0, the model recommends not running Machine 2 on Saturday "
        "for overtime."
    )


# -----------------------------------------------------------------------------
# Step 9: Optional academic-integrity note
# -----------------------------------------------------------------------------
"""
The assignment instructions state that AI-assisted contributions should be
explicitly disclosed and explained if used in the submitted work. Add a short
comment or note in your final submission if appropriate.
"""
