#!/usr/bin/env python
"""
+++++++++
ALGORITHM
+++++++++
====================================================
function LEARN-DECISION-TREE(examples, attributes, parent_examples) returns a tree

    if examples is empty then
        return PLURALITY-VALUE(parent_examples)

    else if all examples have the same classification then
        return the classification

    else if attributes is empty then
        return PLURALITY-VALUE(examples)

    else
        A ← argmax_{a ∈ attributes} IMPORTANCE(a, examples)
        tree ← a new decision tree with root test A

        for each value v of A do
            exs ← { e : e ∈ examples and e.A = v }
            subtree ← LEARN-DECISION-TREE(exs, attributes − A, examples)
            add a branch to tree with label (A = v) and subtree subtree

        return tree
=====================================================
DECISION TREE FROM SCRATCH
==========================
Reference:
    https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c
"""
import numpy as np
import pandas as pd

def read_file(path):
    """
    Reads data into DataFrame
    --------------------------------------------
    INPUT:
        path: (str) Path (absolute/relative) to data file.
    OUTPUT:
        dframe: (pd.DataFrame)
    """
    # input validation
    if not isinstance(path, str) and not (path.endswith("txt") or path.endswith("csv")):
        raise ValueError("\nInput is fucked!")

    # Depends on file extension
    if path.endswith("dat"):
        return pd.read_csv(path, sep=r"\s+")

    else:
        return pd.read_csv(path)

def encode_categoricals(data, verbose=False):
    """
    Encodes categorical data into binary 0/1's
    ----------------------------------------
    INPUT:
        data: (array-like) Data with values to convert to 0/1's

    OUTPUT:
        encoded_data: (array-like)
    """
    # Ensure only two values in array
    uniques, counts =  np.unique(data, return_counts=True)
    
    if len(uniques) > 2:
        raise TypeError("\nCannot encode to binary type given the data given!")

    # Mapping
    mapping = {uniques[0]: 0, uniques[1]: 1}

    # IMplement mapping
    encoded_data = np.array([mapping[val] for val in data])

    if verbose:
        print(f"Uniques: {uniques}")
        print(f"Mapping: {mapping}")

    return encoded_data

def split_convert(dataframe, encode=False):
    """
    Split data into X and y (features and target) and converts to numpy arrays.
    Optionally encodes categorical data to binary.
    -------------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame) Dataset
    OUTPUT:
        X: (np.ndarray) Features
        y: (np.ndarray) Target
    """
    X = np.array(dataframe[dataframe.columns[:-1]])    
    y = np.array(dataframe[dataframe.columns[-1]])
    
    if encode:
        # Encode target feature
        y = encode_categoricals(y)

    return X, y

def parent_entropy(target):
    """
    H(S) = - sum(pi * log2(pi))
    Measures uncertainty in dataset S.
    - H(S) = 0  → pure (all same class)
    - H(S) = 1  → max impurity (equally distributed)
    -------------------------------------------------
    INPUT:
        target: (np.ndarray) Target data
    OUTPUT:
        entropy: (float)
    """
    # Probabilities
    total = len(target)
    _, counts = np.unique(target, return_counts=True)
    probabilities = counts / total

    # Entropy calculation
    entropy = -sum([prob * np.log2(prob) for prob in probabilities if prob > 0])

    return entropy

def avg_child_entropy(X, y, feat_idx):
    """
    Weighted average entropy of child nodes after splitting on a feature.
    For each unique value of the feature:
        weight = |subset| / |total|
        sum += weight * H(subset)

        ? --> Why weight the child values (why AVERAGE?)
    --------------------------------------------------
    INPUT:
        X: (np.ndarray) Features
        y: (np.ndarray) Target
        feat_idx: (int) Column index of feature to split on
    OUTPUT:
        weighted_kid_entropy: (float)
    """
    # Initialize child weighted entropy
    weighted_kid_entropy = 0.0

    # Feature values
    feature_vals = X[:, feat_idx].astype(str)
    
    # Iterate thru values for weighted child entropy
    for val in np.unique(feature_vals):
        # Create mask for filtering target values
        mask = feature_vals == val

        # Create subset for entropy calculation
        y_subset = y[mask]
   
        # Weigh the little bastards
        weight = len(y_subset) / len(y)

        # Average child entropy
        weighted_kid_entropy += weight * parent_entropy(y_subset)

    print(f"Bastards entropy: {weighted_kid_entropy:.3f}") 

    return weighted_kid_entropy

def info_gain(X, y, feat_idx):
    """
    IG(S, feature) = H(parent) - H_avg(children)
    Measures how much splitting on this feature reduces uncertainty.
    --------------------------------------------------
    INPUT:
        X: (np.ndarray) Features
        y: (np.ndarray) Target
        feat_idx: (int) Feature index
    OUTPUT:
        information_gain: (float)
    """
    return parent_entropy(y) - avg_child_entropy(X, y, feat_idx)

def best_split_ig(X, y, feat_names):
    """
    Select the feature with the largest information gain.
    -------------------------------------------
    HOW: ?

    -------------------------------------------
    INPUT:
        X: (np.ndarray) Features
        y: (np.ndarray) Target
        feat_names: (list of str) Column names
    OUTPUT:
        best_ig: (float)
        best_feat_name: (str)
        best_feat_idx: (int)
    """

def majority_class(y):
    """
    y_hat = mode(y)
    Returns most common class label (used for leaf predictions).
    ----------------------------------------------------------------
    INPUT:
        y: (np.ndarray) Target
    OUTPUT:
        (str/float) Most common value
    """


# Class Node goes here ?
class Node(object):
    """
    Tree node. Internal nodes store feat_idx, feat_name, kids dict.
    Leaf nodes store prediction and is_leaf=True.
    """
    pass


def build_tree(X, y, feat_names):
    """
    Recursively builds the decision tree.
    BASE CASES (return leaf):
        1. Node is pure: |unique(y)| == 1
        2. No features remain: X.shape[1] == 0
        3. No gain: best IG <= 0

    RECURSIVE CASE:
        1. Find best feature via best_split_ig
        2. For each unique value of that feature:
           - Mask & filter X, y
           - Delete the split column from X
           - Recurse on subset
        3. Return internal Node with kids dict
    ---------------------------------------------------
    INPUT:
        X: (np.ndarray) Features
        y: (np.ndarray) Target
        feat_names: (list of str)
    OUTPUT:
        node: (Node)
    """
    # ====== Base cases =======
    pass


def print_tree(root_node, indent=""):
    """
    ?? REDO ON YOUR OWN ??
    Recursively prints the tree. Indent string grows with depth.
    ---------
    INPUT:
        root_node: (Node)
        indent: (str)
    """

def predict_one(node, x, feat_names):
    """
    Predicts class label for one example by walking the tree.
    At each internal node: find feature index via feat_names.index(),
    pull value from x, follow matching child. If unseen value, fall
    back to majority_class.
    -------------------------------------------------------
    INPUT:
        node: (Node) Root of tree
        x: (np.ndarray) Single row
        feat_names: (list of str) Original column names
    OUTPUT:
        prediction: (str)
    """

def predict_all(root_node, X, feat_names):
    """
    Predicts class labels for all rows via list comprehension
    over predict_one.
    --------------------------------------------------
    INPUT:
        root_node: (Node) Root of trained tree
        X: (np.ndarray) Feature space
        feat_names: (list of str)
    OUTPUT:
        predictions: (np.ndarray)
    """



def main():
    PATH = "data/niblings.txt"
    df = read_file(PATH)
    X, y = split_convert(df, encode=True)
    data_entropy = parent_entropy(y)

    kids = avg_child_entropy(X, y, 0)
    # Info gain
    information_gain = info_gain(X, y, 0)
    

    breakpoint()


if __name__ == "__main__":
    main()
