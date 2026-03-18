#!/usr/bin/env python
import pulp as pl

lp_501 = pulp.LpProblem("LP_501", pulp.LpMaximize)

x = pulp.LpVariable('x', lowBound=0, cat='Continuous')
y = pulp.LpVariable('y', lowBound=2, cat='Continuous')

#Maximize Z = 4x + 3Y
lp_501 += 4 * x + 3 * y, "Z"

#Constraints

#2y <= 25-x
lp_501 += 2*y <= 25-x

#4y >= 2x-8
lp_501 += 4*y >= 2*x-8

#y<=2x-5
lp_501 += y <= 2*x-5

lp_501.variables()

lp_501.solve()
pulp.LpStatus[lp_501.status]

#Possible status codes
#Not Solved: Status prior to solving the problem.
#Optimal: An optimal solution has been found.
#Infeasible: There are no feasible solutions (e.g. if you set the constraints x <= 1 and x >=2).
#Unbounded: The constraints are not bounded, maximising the solution will tend towards infinity (e.g. if the only constraint was x >= 3).
#Undefined: The optimal solution may exist but may not have been found.

print(x.varValue)
print(y.varValue)
