#!/usr/bin/env python
"""
?
**Problem 4.2**

Two fair dice are rolled. Let (X) equal the product of the two dice. Compute (P(X = i)) for (i = 1, 2, \ldots, 36).
"""
from fractions import Fraction
from collections import Counter

D = range(1, 7)
products = [d1 * d2 for d1 in D for d2 in D]

counts = Counter(products)

pmf = {k: Fraction(i, 36) for k, i in counts.items()}

E = sum(k*p for k, p in pmf.items())

breakpoint()

