#!/usr/bin/env python
# ============================================================
# BFS IMPLEMENTATION GUIDE
# Reference the full solution only after attempting each step
# ============================================================

# ---------------------------------------------------------------
# STEP 1: Define your raw data
# ---------------------------------------------------------------
edges = []  # Your edge list goes here as (source, dest, cost) tuples


# ---------------------------------------------------------------
# STEP 2: Build the Graph
# ---------------------------------------------------------------
def build_adjacency_list(edges):
    """
    PURPOSE:
        Convert a flat edge list into a dictionary for fast neighbor lookup.

    WHY THIS MATTERS:
        Your raw data is a list of tuples. Every time you ask "who are
        node A's neighbors?", searching the list costs O(E) — you scan
        every edge. A dictionary gives you O(1) lookup instead.

    WHAT TO BUILD:
        A dict where:
            key   = a node (string)
            value = list of (neighbor, cost) tuples

        Example result:
            { "A": [("B", 2), ("C", 3)], "B": [], ... }

    HOW TO DO IT:
        1. Create an empty dict called graph
        2. Loop over edges, unpacking each tuple into (source, dest, cost)
        3. If source isn't in graph yet, add it with an empty list
        4. If dest isn't in graph yet, add it with an empty list
           (important! leaf nodes have no outgoing edges but must exist)
        5. Append (dest, cost) to graph[source]
        6. Return graph
    """
    pass  # YOUR CODE HERE


# ---------------------------------------------------------------
# STEP 3: Find the Root Node
# ---------------------------------------------------------------
def find_root(graph):
    """
    PURPOSE:
        Identify the starting node — the one with no incoming edges.

    CONCEPT:
        Root node = appears as a SOURCE but never as a DESTINATION.

        Mathematically this is a set difference operation:
            root ∈ Sources − Destinations

        Sources      = all keys in your graph dict
        Destinations = every node that appears inside any neighbor list

    HOW TO DO IT:
        1. Build a set of ALL source nodes  → graph.keys()
        2. Build a set of ALL destination nodes → loop graph.values(),
           then loop each (neighbor, cost) pair, collect neighbor
        3. Subtract: sources - destinations
        4. Return the single node remaining (.pop())

    NOTE:
        In a well-formed directed tree there is exactly one root.
        If your graph had cycles or multiple roots this would need
        more sophisticated handling.
    """
    pass  # YOUR CODE HERE


# ---------------------------------------------------------------
# STEP 4: The BFS Algorithm Itself
# ---------------------------------------------------------------
def breadth_first_search(graph, start, goal):
    """
    PURPOSE:
        Find a path from start to goal by exploring level-by-level.

    CORE IDEA — THE QUEUE:
        BFS uses a FIFO queue (First In, First Out).
        Nodes discovered first get processed first.
        This is what guarantees level-by-level (shallow before deep) traversal.

        Compare:
            Queue (FIFO)  → BFS  (explores wide, finds shallowest path)
            Stack (LIFO)  → DFS  (explores deep, not guaranteed optimal)

    WHAT GOES IN THE QUEUE:
        Each entry in the queue tracks THREE things:
            (current_node, path_so_far, cumulative_cost)

        You need path_so_far so that when you reach the goal,
        you can return the full route — not just the final node.

    THE VISITED SET:
        Without it, you'll loop forever on cycles.
        Mark a node visited the moment you ADD it to the queue,
        not when you pop it — otherwise you enqueue duplicates.

    HOW TO DO IT:
        1. Initialize queue with a single entry: (start, [start], 0)
        2. Initialize visited as a set containing start
        3. While the queue is not empty:
            a. Pop from the FRONT (index 0) — this is what makes it FIFO
            b. Unpack into current_node, path, cost
            c. If current_node == goal → return (path, cost)
            d. Loop over graph[current_node] to get (neighbor, edge_cost)
            e. If neighbor not in visited:
                - Add neighbor to visited
                - Build new_path = path + [neighbor]
                - Build new_cost = cost + edge_cost
                - Append (neighbor, new_path, new_cost) to queue
        4. If the loop ends with no goal found → return (None, None)

    PARAMETERS:
        graph (dict) : Adjacency list built in Step 2
        start (str)  : Starting node label
        goal  (str)  : Target node label

    RETURNS:
        (path, total_cost) if goal is reachable, else (None, None)
    """
    pass  # YOUR CODE HERE


# ---------------------------------------------------------------
# STEP 5: Wire It All Together
# ---------------------------------------------------------------
"""
WHAT TO DO HERE:
    1. Call build_adjacency_list(edges) → store result as graph
    2. Call find_root(graph)            → store result as start
    3. Define your goal node as a string (e.g., "G")
    4. Call breadth_first_search(graph, start, goal) → unpack into path, cost
    5. Print the start, goal, path (joined with " -> "), and cost
"""

