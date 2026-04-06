#!/usr/bin/env python
"""
============================
SUPPORT VECTOR MACHINE (SVM)
============================
    (Explain Support Vector Machines)

OBJECTIVE FUNCTION:
    T(w, b) = (lambda/2) * ||w||^2 + (1/n) * sum(max(0, 1 - yi[w * Xi + b]))

    1st term: MAXIMUM MARGIN OF HYPERPLANE
    2nd term: HINGE LOSS FUNCTION:
        (Describe the Hinge loss function)

"""
from numpy.linalg import norm
import numpy as np
import pandas as pd

def max_margin(lamb, weight):
    """
    Maximizes the Margin of a hyperplane for Support Vector Machine (SVM).
    It's the product of the square of the weight vector and half the
    REGULARIZATION parameter (the 1/2 is for convenience for deriving the
    gradients)
    ------------------------------------------------------------
    INPUT:
        lamb: (float) Regularization parameter
        weight: (float) Weight vector

    OUTPUT:
        spread_them_cheeks: (float) The maximum margin        
    """ 
    # lolz
    spread_them_cheeks = 0.5 * (lamb * norm(weight))
    return spread_them_cheeks




weights = np.array()
spread = max_margin()
breakpoint()

