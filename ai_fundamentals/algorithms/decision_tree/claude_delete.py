#!/usr/bin/env python
def build_tree(X, y, feat_names):
    """
    Recursively builds a decision tree using the ID3 algorithm.

    WHAT IS RECURSION:
        A function that calls itself on a smaller version of the problem.
        Every recursive function needs:
            - BASE CASES: conditions where it stops and returns a result
            - RECURSIVE CASE: where it breaks the problem down and calls itself

        Think of it like peeling an onion. Each call peels one layer (one feature).
        The base cases are when you hit the center — nothing left to peel.

    WHAT THIS FUNCTION DOES:
        1. Checks if we should stop (base cases)
        2. If not, finds the best feature to split on
        3. Groups the data by each value of that feature
        4. For each group, calls itself on that smaller group
        5. Packages everything into a Node and returns it

    WHAT GETS RETURNED:
        Always a Node object. Either:
            - A leaf Node (is_leaf=True, has a prediction)
            - An internal Node (is_leaf=False, has children that are also Nodes)

    PARAMETERS:
        X:          (np.ndarray) Feature matrix — rows are examples, columns are features
        y:          (np.ndarray) Target labels — one per row of X
        feat_names: (list of str) Column names, same length as X.shape[1]

    RETURNS:
        Node: The root of the (sub)tree built from this data
    """

    # ===================== BASE CASE 1 =====================
    # CHECK: Are all the labels the same?
    #
    # np.unique(y) returns an array of distinct values in y.
    # If its length is 1, every example has the same class.
    # There is nothing to split — we already know the answer.
    #
    # WHAT WE RETURN:
    #   A leaf Node whose prediction is that single class label.
    #   y[0] works because they're ALL the same value.
    #   kids=None because leaves have no children.
    # ===========================================================
    if len(np.unique(y)) == 1:
        return Node(kids=None, prediction=y[0], is_leaf=True)

    # ===================== BASE CASE 2 =====================
    # CHECK: Are there any features left to split on?
    #
    # X.shape[1] gives the number of columns (features).
    # Each recursive call DELETES the column we split on,
    # so this number shrinks by 1 every level.
    # When it hits 0, we have no features left.
    #
    # WHAT WE RETURN:
    #   A leaf Node predicting the majority class.
    #   We can't split, so our best guess is the most common label.
    # ===========================================================
    if X.shape[1] == 0 or len(feat_names) == 0:
        return Node(kids=None, prediction=majority_class(y), is_leaf=True)

    # ===================== FIND BEST SPLIT =====================
    # This calls your best_split_ig function, which loops through
    # every remaining feature, computes info gain for each, and
    # returns the winner.
    #
    # best_ig:        (float) the highest information gain found
    # best_feat_name: (str)   the column name of that feature
    # best_feat_idx:  (int)   the column INDEX of that feature in X
    # ===========================================================
    best_ig, best_feat_name, best_feat_idx = best_split_ig(X, y, feat_names)

    # ===================== BASE CASE 3 =====================
    # CHECK: Did splitting actually help?
    #
    # If best_ig is 0 (or negative from floating point issues),
    # no feature reduces entropy. Splitting is pointless.
    #
    # WHAT WE RETURN:
    #   Same as base case 2 — a leaf with the majority vote.
    # ===========================================================
    if best_ig <= 0:
        return Node(kids=None, prediction=majority_class(y), is_leaf=True)

    # ===================== PREPARE FOR RECURSION =====================
    # Build a NEW list of feature names that excludes the one
    # we're about to split on.
    #
    # WHY: Each recursive call gets a table with one fewer column.
    #       The names list must match. If X has 3 columns and we
    #       remove column 1, the new X has 2 columns, so we need
    #       a list of 2 names.
    #
    # HOW: List comprehension — keep every name that ISN'T the
    #       best feature's name.
    # ===========================================================
    remaining_names = [name for name in feat_names if name != best_feat_name]

    # ===================== RECURSIVE CASE =====================
    # This is where the tree actually branches.
    #
    # children is a dictionary: {feature_value: child_Node}
    # For example, if splitting on "Weather" with values
    # ["Sunny", "Rainy", "Cloudy"], we get:
    #   {"Sunny": Node(...), "Rainy": Node(...), "Cloudy": Node(...)}
    #
    # STEP BY STEP inside the loop:
    #
    #   1. MASK — a boolean array, True where the best feature
    #      column equals the current value.
    #      Example: X[:, 2] == "Sunny" → [True, False, True, False, ...]
    #
    #   2. SUBSET ROWS — X[mask] keeps only the rows where mask
    #      is True. Same for y[mask]. This is the GROUP of examples
    #      that share this feature value.
    #
    #   3. DELETE COLUMN — np.delete(X_subset, best_feat_idx, axis=1)
    #      removes the column we just split on.
    #      axis=1 means "delete a column" (axis=0 would delete a row).
    #      We remove it because we already used it — it has no more
    #      information to contribute.
    #
    #   4. RECURSE — call build_tree on the smaller table.
    #      It returns a Node (could be a leaf or another internal node).
    #      Store it in the children dictionary under this value.
    # ===========================================================
    children = {}

    for val in np.unique(X[:, best_feat_idx]):
        # Step 1: Boolean mask — which rows match this value?
        mask = X[:, best_feat_idx] == val

        # Step 2: Keep only matching rows
        X_subset = X[mask]
        y_subset = y[mask]

        # Step 3: Remove the feature column we split on
        X_subset = np.delete(X_subset, best_feat_idx, axis=1)

        # Step 4: Recurse on the smaller table
        children[val] = build_tree(X_subset, y_subset, remaining_names)

    # ===================== RETURN INTERNAL NODE =====================
    # Package everything into a Node.
    #
    # kids:           the dictionary of children we just built
    # feat_idx:       which column this node splits on (valid for THIS level only)
    # feat_name:      the string name (stable across all levels)
    # is_leaf:        False — this is an internal decision node
    # majority_class: stored as a fallback for prediction on unseen values
    # info_gain:      stored for debugging / printing
    # ===========================================================
    return Node(
        kids=children,
        feat_idx=best_feat_idx,
        feat_name=best_feat_name,
        is_leaf=False,
        majority_class=majority_class(y),
        info_gain=best_ig
    )


def predict_one(node, x, feat_names):
    """
    Predicts the class label for a SINGLE example by walking down the tree.

    HOW IT WORKS:
        Start at the root node. At each internal node:
            1. The node says "I split on feature F"
            2. Look up where F is in feat_names to get the column index
            3. Pull the value from x at that index
            4. Find the child node that matches that value
            5. Move to that child
        Repeat until you land on a leaf, then return its prediction.

    WHY WE NEED feat_names:
        The tree stores feature NAMES (strings) on each node, not indices.
        But x is a flat array — x[0], x[1], x[2], etc.
        feat_names is the bridge:
            col_idx = feat_names.index("Weather")  → gives us 2
            value   = x[2]                          → gives us "Sunny"

        We can't use feat_idx stored on the node because those indices
        were valid only at the recursion level where the node was created.
        Columns shift after deletion, but NAMES never change.

    UNSEEN VALUES:
        If the test example has a value the tree never saw during training
        (e.g., "Foggy" when training only had "Sunny"/"Rainy"),
        there's no child branch for it. We fall back to majority_class —
        the most common label among all training examples that reached
        this node.

    PARAMETERS:
        node:       (Node) The root of the tree (or subtree)
        x:          (np.ndarray) A single row — one example's feature values
        feat_names: (list of str) ALL original column names, in original order

    RETURNS:
        The predicted class label
    """
    # Keep walking until we reach a leaf
    while not node.is_leaf:

        # What feature does this node split on?
        # .index() searches the list and returns the position
        col_idx = feat_names.index(node.feat_name)

        # What value does this example have for that feature?
        value = x[col_idx]

        # Follow the matching child branch
        if value in node.kids:
            node = node.kids[value]
        else:
            # Value wasn't seen during training — use fallback
            return node.majority_class

    # We've reached a leaf — return its stored prediction
    return node.prediction


def predict_all(tree, X, feat_names):
    """
    Predicts class labels for EVERY row in X.

    HOW IT WORKS:
        A list comprehension that calls predict_one on each row.
        np.array() wraps the result so it's an ndarray, not a Python list.
        This makes it easy to compare against y for accuracy.

    PARAMETERS:
        tree:       (Node) Root of the trained decision tree
        X:          (np.ndarray) Feature matrix — all test examples
        feat_names: (list of str) Original column names

    RETURNS:
        (np.ndarray) One prediction per row of X
    """
    # "for x in X" iterates over ROWS of a 2D array
    # Each x is a 1D array (one example's features)
    return np.array([predict_one(tree, x, feat_names) for x in X])


def print_tree(node, indent=""):
    """
    Recursively prints the tree in a readable format.

    HOW IT WORKS:
        Same recursive pattern as build_tree:
        - Base case: node is a leaf → print its prediction and stop
        - Recursive case: print the feature name, then loop through
          each (value, child) pair and recurse with deeper indentation

    THE INDENT TRICK:
        indent is a string of spaces. Each recursive call adds more spaces.
        This makes deeper nodes appear further right, visually showing
        the tree structure:
            [Weather]
              Weather = Sunny:
                [Humidity]
                  Humidity = High:
                    → Predict: No
                  Humidity = Normal:
                    → Predict: Yes
              Weather = Rainy:
                → Predict: No

    PARAMETERS:
        node:   (Node) Current node to print
        indent: (str)  Spaces for current depth level
    """
    # Base case: leaf node
    if node.is_leaf:
        print(f"{indent}→ Predict: {node.prediction}")
        return

    # Recursive case: internal node
    # Print which feature this node splits on
    print(f"{indent}[{node.feat_name}]  (IG={node.info_gain:.4f})")

    # Loop through each branch (value → child node)
    for value, child in node.kids.items():
        # Print the branch label
        print(f"{indent}  {node.feat_name} = {value}:")
        # Recurse into the child with more indentation
        print_tree(child, indent + "    ")
