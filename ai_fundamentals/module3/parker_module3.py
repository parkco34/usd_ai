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
"""
import pulp as pl


