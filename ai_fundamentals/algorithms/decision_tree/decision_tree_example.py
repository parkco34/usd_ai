#!/usr/bin/env python
"""
DECISION TREE FROM SCRATCH
==========================
Reference:
    https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c
"""
import pandas as pd
import numpy as np

def read_file(path):
    """
    Reads data into DataFrame
    --------------------------------------------
    INPUT:
        path: (str) Path (absolute/relative) to data file.

    OUTPUT:
        X, y: (tuple of np.arrays) Features and target
    """
    if not isinstance(path, str):
        raise ValueError("Path provided is in wrong format\nMust be a string")

    if path.endswith("csv") or path.endswith("txt"):
        dframe = pd.read_csv(path)

    elif path.endswith("dat"):
        dframe = pd.read_csv(path, sep=r"\s+")

    return dframe

def split_convert(dataframe):
    """
    Split data into X and y (features and target) and converts the data into numpy arrays for efficient calculations.
    -------------------------------------------------
    INPUT:
        dataframe: (pd.DataFrame) Dataset 

    OUTPUT:
        (tuple)
        X: (np.ndarray) Features
        y: (np.ndarray) Target
    """
    X = np.asarray(dataframe[dataframe.columns[:-1]])
    y = np.asarray(dataframe[dataframe.columns[-1]])

    return X, y

def encode_categorical(data):
    """
    Encodes categorical values to 0s(-) and 1s(+) incase libraries used which would need numerical values for processing.
    ------------------------------------------------
    INPUT:
        data: (np.ndarray)

    OUTPUT:
        encoded_data: (np.ndarray) 
    """
    # ?? Have no idea yet ??
    pass

# Maybe try this function from Claude?
def encode_categorical(data):
    """
    Encodes categorical values to integers per column.
    Each unique string in a column gets a unique integer.
    ------------------------------------------------
    INPUT:
        data: (np.ndarray)
    OUTPUT:
        encoded_data: (np.ndarray)
    """
    encoded_data = np.empty_like(data, dtype=int)

    for col in range(data.shape[1]):
        unique_vals = np.unique(data[:, col])
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        encoded_data[:, col] = [mapping[val] for val in data[:, col]]

    return encoded_data

def parent_entropy(target, verbose=False):
    """
    ENTROPY:
        H(S) = - sum(pi * log2(pi))
    - Measures the amount of uncertainty in the data set S.
    Selects for root node, the smallest entropy (or larger information gain IG(S)).
    - H(S) = 0 (pure: All data points belong to same class)
    - H(S) = 1 (Totally impure: All data points are equally distributed among the different classes)

    pi = Probability of class i
    S = Total number of examples
    ------------------------------------------------
    INPUT:
        target: (np.ndarray) Target data

    OUTPUT:
        entropy: (float) Entropy of entire dataset from which the feature with
        smalles entropy is selected as the root node
    """
    # Get unique values and the count of each (binary)
    uniques, count = np.unique(target, return_counts=True)

    # probabilities (+/-)
    probs = count / target.size

    # Calculating entropy
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
    return entropy

def avg_child_entropy(X, y, feat_idx):
    """
    Average entropy of child nodes for this feature
    =============================================
    1) Initialize weighted entropy as 0 and get all the unique values for each attribute

    2) Iterate thru the attributes:
        - Create MASK for filtering for thee target values based on current attribute values
        - Calculate subset of data corresponding to target values
        - get weights (size of subset / total)
        - calculate subset entropy
        - Summing -> weighted_entropy each iteration
    --------------------------------------------------
    INPUT:
        X: (np.ndarray) Attributes
        y: (np.ndarray) Target
        feat_idx: (int) Index number for current feature

    OUTPUT:
        weighted_kid_entropy: (float) Weighted child entropy
    """
    # initialize weighted average
    weighted_kid_entropy = 0.0

    # Unique values for feature
    uniques = np.unique(X[:, feat_idx])

    # Iterate thru values to calculate entropy
    for val in uniques:
        # mask for filtering data
        mask = X[:, feat_idx] == val
        # Filter via mask
        subset_y = y[mask]

        # subset size divided by total of target
        weight = subset_y.size / y.size

        # entropy calculation
        child_entropy = parent_entropy(subset_y)

        # Weighted average
        weighted_kid_entropy += weight * child_entropy

    return weighted_kid_entropy

def info_gain(X, y, feat_idx):
    """
    IG(S) = parent_entropy - avg_child_entropy
    Measures how important an attribute of the feature vector is, using it to
    DECIDE the ordering of attributes in the nodes of the tree.
    --------------------------------------------------
    INPUT:
        X: (np.ndarray) Attributes
        y: (np.ndarray) Target
        feat_idx: (int) Feature index for filtering

    OUTPUT:
        information_gain: (float) Information gain lolz
    """
    # Subtract parent entropy from kids entropy
    return parent_entropy(y) - avg_child_entropy(X, y, feat_idx)

def best_split_ig(X, y, feat_names):
    """
    Select attribute with the largest information gain as decision node.
    =========================================
    1) Initialize return variables since the work done is in a loop
    2) Iterate thru features to find best info gain
    3) Compare against previous info gain an assign bests accordingly
    -------------------------------------------
    INPUT:
        x: (np.ndarray) attributes
        y: (np.ndarray) target
        feat_names: (list of str) Column names

    OUTPUT:
        (tuple)
        best_ig: (float) HIghest information gain
        best_feat_name: (str) Best attribute for split
        best_feat_idx: (int) Column index of best attribute
    """
    # Intialize some shit
    best_feat_idx = None
    best_feat_name = None
    best_ig = -np.inf

    # Iterate thru feature indices and compare info gains
    for idx in range(X.shape[1]):
        # Calculate info gain for given feature
        ig = info_gain(X, y, idx)

        # If best info gain hasn't been assigned, assign it the first value
        if best_ig < 0:
            best_ig = ig

        # Compare to determine best info gain (largest)
        if ig >= best_ig:
            best_ig = ig
            best_feat_idx = idx
            best_feat_name = feat_names[idx]

    return best_ig, best_feat_name, best_feat_idx

def majority_class(y):
    """
    y_hat = mode(y)
    Returns most common class label.
    Since the recursion has to stop at some point, the leaf should predict the
    class majority.
    ----------------------------------------------------------------
    INPUT:
        y: (np.ndarray) Target feature

    OUTPUT:
        (float) Mode common value
    """
    # Get values and their frequencies
    values, counts = np.unique(y, return_counts=True)
    
    # argmax returns the indices of max values along the axis
    return values[np.argmax(counts)]


class Node:
    """
    At each non-leaf node, look at each feature and branch by its values.
    This requires:
        - feature index
        - feature name
        - dictionary of children nodes for each value

    At the leaf node, return class label y_hat = mode(y)
    """
    def __init__(
        self,
        kids=None,
        feat_idx = None,
        feat_name = None,
        prediction=None,
        is_leaf=False,
        majority_class=None,
        info_gain=None
    ):
        self.feat_idx = feat_idx
        self.feat_name = feat_name
        self.kids = kids if kids is not None else {}
        self.prediction = prediction
        self.is_leaf = is_leaf

        # Useful fallback if prediction path missing
        self.majority_class = majority_class

        # For debugging
        self.info_gain = info_gain

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(prediction={self.prediction})"

        return f"Node(feature={self.feat_name}, Children={list(self.kids.keys())})"


def build_tree(X, y, feat_names):
    """
    Recursively builds the decision tree.
    A node is returned at every call of this function.
    Should stop when:
        1 - NODE IS PURE
            ~ All labels are the same |unique(y)| = 1

        2 - NO FEATURES REMAIN
            ~ no columns left to split on

        3 - NO GAIN FROM SPLITTING
            ~ If best information gain is zero, splitting won't reduce entropy

    Recursive case:
        1 - best_split_ig to find which feature to split on
        2 - If none of the base cases triggered, branch on every unique value
        of best feature
        3 - For each unique value:
            - Create boolean mask equalling best feature column == value
            - Filter both X and y into subsets
            - Delete best feature column from subset X
            - Recursively call build_tree on subset, where the return value is
            the childe node for the value
        4 - Store each kid in dictionary: {'value': child_node, ...}
        5 - Return a Node with that dictionary as kids, plus the feature index
        and name.
    --------------------------------------------------------------------------------
    INPUT:
        X: (np.ndarray) Attributes
        y: (np.ndarray) Target
        feat_names: (list of str) Names of features (duh)

    OUTPUT:
        tree: (?)
    """
    # Base cases (when to stop and return leaf node)
    if len(np.unique(y)) == 1:
        # y[0] is used since there's only one unique value (arbitrary)
        node = Node(prediction=y[0], is_leaf=True)

        return node

    # Check if there's any features to split on
    if X.shape[1] == 0 or len(feat_names) == 0:
        node = Node(
            # Get MODE
            prediction=majority_class(y),
            is_leaf=True
        )

        return node

    # Get best splits
    best_ig, best_feat_name, best_feat_idx = best_split_ig(X, y, feat_names)

    # Base case 3:  No gain from splitting
    if best_ig <= 0:
        node = Node(
            # Get MODE
            prediction=majority_class(y),
            is_leaf=True
                   )
        return node

    # Build new list of feature names, excluding the one I'm about to split on mothafuckazzz
    # Each recursive call gets a table with one fewer column via list comprehension
    remaining_names = [name for name in feat_names if name != best_feat_name]

    # ========== RECURSION ============
    # Initialize empty dictionary
    kids = {}

    # Iterate thru unique values for given feature
    for val in np.unique(X[:, best_feat_idx]):
        # Mask for filtering data based on best feature index
        mask = X[:, best_feat_idx] == val
        
        # Filter for corresponding rows
        X_subset = X[mask]
        y_subset = y[mask]

        # REmove feature column I split on
        X_subset = np.delete(X_subset, best_feat_idx, axis=1)

        # Recurse on smaller table
        kids[val] = build_tree(X_subset, y_subset, remaining_names)

    node = Node(
        kids=kids, 
        feat_idx=best_feat_idx,
        feat_name=best_feat_name,
        is_leaf=False,
        majority_class=majority_class(y),
        info_gain=best_ig
    )
    
    return node

def print_tree(root_node, indent=""):
    """
    +++++++++++++++++++++++++++++++++++++++ 
    Claude did this --> ? REDO ON MY OWN !!!
    +++++++++++++++++++++++++++++++++++++++

    PRINTS THE TREE
    ---------------
    INDENT TRICK:
        Indent is a string of spaces, so each recursive call adds more spaces, making deeper nodes appear further right, visually showing the tree strucuture.
    -----------------------------------
    INPUT:
        root_node: (Node class)

    OUTPUT:
        None
    """
    # Base case: leaf node
    if root_node.is_leaf:
        print(f"{indent} -> Predict: {root_node.prediction}")
        return None

    # Recursive case: Internal root_node
    # Printing feature root_node splits on
    print(f"{indent}[{root_node.feat_name}] (IG={root_node.info_gain:.3f})")

    # Iterate thru each branch value, getting keys/values
    for val, child in root_node.kids.items():
        # Branch label
        print(f"{indent} {root_node.feat_name} = {val}")

        # Recurse into da kid wiith more indentation
        print_tree(child, indent + "   ")

def predict_one(node, x, feat_names):
    """
    Predicts the class label for a single example by walking down the tree.
    ==================================================
    HOW:
    Starting at the root node.  At each internal node:
        1) split on feature F
        2) Find where F is in feat_names to get column index
        3) Pull value from x at the index
        4) Find child node that matches the value
        5) Movde to child node
    
    Repeating until you land on a leaf, then return prediction

    WHY feat_names is NEEDED:
        Tree stores feature names on each node, not indices.
        We can't use feat_idx on the node since the indices wer valid only at the recursion level where the node was created.
        Columns shift after deletion, but NAMES neve change.

    UNSEEN VALUES:
        If the test example has a value the tree hasn't seen during training,
        there's no child branch for it.  WE call back to majority_class -  the
            most common label among al training examples reached at this node.
    -------------------------------------------------------
    INPUT:
        node: (class Node)
        x: (np.ndarray) Single row
        feat_names: (list of str) All original column neames, in original order

    OUTPUT:
        prediction: (str) Predicted class label
    """
    # If node not a leaf node, start splitting...
    while not node.is_leaf:
        # get index of feature name
        feat_idx = feat_names.index(node.feat_name)

        # Get value from x (row) at the index position
        value = x[feat_idx]

        # Ensure value is in node for matching a child node value
        if value in node.kids.keys():
            # replace current node with child node
            node = node.kids[value]

        else:
            # return MODE
            return node.majority_class

    return node.prediction

def predict_all(node):
    """

    """
    pass

        

# ===== MAIN ======
PATH = "data/exam_results.txt"
df = read_file(PATH)

# remove useless column
df.drop("Resp srl no", axis=1, inplace=True)
X, y = split_convert(df)
feat_names = list(df.columns[:-1])
root_node = build_tree(X, y, feat_names)

# This will show the work in the order the computer does it (depth-first)
#print(f"Root Node: {root_node}")

# Proper output of decision tree
print_tree(root_node)


breakpoint()
