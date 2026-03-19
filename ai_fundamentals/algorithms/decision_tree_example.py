#!/usr/bin/env python
"""
Predicts whether a student will pass class based on whether
the student works, is taking other online courses,
or has a background in computer science, math, or other.
------------------------------------------------------------
Source: 
    https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c
------------------------------------------------------------
BINARY CLASSIFICATION -> Adjust for Multiclassification ?
"""
import pandas as pd
import numpy as np

def read_file(path):
    if not isinstance(path, str):
        raise ValueError("Path provided is in wrong format!\nMust be a string")

    try:
        if path.endswith("csv"):
            dframe = pd.read_csv(path)

        elif path.endswith("txt") or path.endswith("dat"):
            dframe = pd.read_csv(path, sep=r"\s+")

    except Exception as err:
        print(f"OOPZ!\n{err}")

    return dframe

def data_entropy(target):
    """
    Entropy is the measure of uncertainty in a dataset.
    H = - ∑pi*log_2(pi) = 0, where
    H = 0 (Pure dataset; all data points belong to same class)
    H = 1 (Totaly impure dataset; All data points are equally distributed among different classes)
    --------------------------------------------------------
    INPUT:
        target: (pd.Series) Target attribute

    OUTPUT:
        entropy: (float) 
    """
    # 










# Example usage
PATH = "exam_results.txt"
df = read_file(PATH)

breakpoint()
