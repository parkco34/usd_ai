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
    """
    reads file depending on file extension.
    --------------------------------------------
    INPUT:
        path: (str) Path (absolute/relative) to data file.

    OUTPUT:
        X, y: (tuple of np.arrays) Features and target
    """
    if not isinstance(path, str):
        raise ValueError("Path provided is in wrong format!\nMust be a string")

    try:
        if path.endswith("csv") or path.endswith("txt"):
            dframe = pd.read_csv(path)

        elif path.endswith("dat"):
            dframe = pd.read_csv(path, sep=r"\s+")

    except Exception as err:
        print(f"OOPZ!\n{err}")

    X, y = (np.array(dframe[dframe.columns[:-1]]), np.array(dframe[dframe.columns[-1]]))

    return X, y

def data_entropy(target, verbose=False):
    """
    Entropy is the measure of uncertainty in a dataset.
    H = - ∑pi*log_2(pi) = 0, where
    (Math note: Minus sign comes from the fact that the log2 of any number smaller than 1 is a negeative number)
    H = 0 (Pure dataset; all data points belong to same class)
    H = 1 (Totaly impure dataset; All data points are equally distributed among different classes)
    --------------------------------------------------------
    INPUT:
        target: (np.ndarray) Target attribute

    OUTPUT:
        entropy: (float) 
    """
    # Proportions for probability of each class
    classes, counts = np.unique(target, return_counts=True)

    # Probabilites
    p_fail, p_pass = (counts/target.size)[0], (counts/target.size)[1]

    # Entropy calculation
    entropy = -sum([prob * np.log2(prob) for prob in (p_fail, p_pass) if prob>0])
   
    # for debugging
    if verbose:
        print(f"Pfail = {p_fail}")
        print(f"Ppass = {p_pass}")
        print(f"Data entopry is: {entropy}")

    return entropy

def attribute_entropy(X, y, feat_idx):
    """
    Weighted entropy for given feature (Conditional entropy).
    H(S|A) = ∑(|Si| / |S|) * H(Si), i ∈ values of A.
    ------------------------------------------------------
    For each unique value of a feature, filter the TARGET to that subset,
    compute the entropy of that target subset, weigh it by proportion, then
    sum them.
    ---------------------------------------------------------
    INPUT:
        X: (np.ndarray) Feature matrix
        y: (np.ndarray) Target labels <-- Entropy is about this
        feat_idx: (int) Column of X to split on

    OUTPUT:
        weighted_entropy: (float) 
    """
    # initialize weighted entropy
    weighted_entropy = 0

    # Get unique values for the feature
    uniques, counts = np.unique(X[:, feat_idx], return_counts=True)

    # Iterate thru unique values, calculating weighted entropy
    for val in uniques:
        # Create mask for filtering
        mask = X[:, feat_idx] == val

        # Corresponding subset for target
        subset_y = y[X[:, feat_idx] == val]

        # Probabilities for subset
        weight = subset_y.size / X.size

        # calculate the entropy for the given feature
        subset_entropy = data_entropy(subset_y)

    return weighted_entropy

def information_gain(X, y, feat_idx):
    """
    INFORMATION GAIN represents how much information a feature provides for the target variable.
    ------------------------------------------------
    Calculate a potential split from each variable, then calculate the average entropy across both or all nodes, concluding with the change in entropy via parent node.
    -----------------------------------------------------
    =====
    STEPS
    =====
    1. Probability of pass/fail @ each node
    2. Parent Entropy calculation
    3. Average Entropy of child nodes, which is the weighted entropy of all subnodes
    """
    



# Example usage
PATH = "exam_results.txt"
df = read_file(PATH)
X, y = read_file(PATH)

attribute_entropy(X, y, 3)
