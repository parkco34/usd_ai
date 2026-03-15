#!/usr/bin/env python
"""
Breadth First Search is a graph traversal algorithm that explores nodes by
layer, visiting all neighbors at current depth before moving to next level.
- Used when all actions have the same cost.

MEAURES:
    Completeness - Solution guaranteed to find solution
    Cost OPtimality - Lowest path cost of solution
    Time Complexity - Duh
    Space Complexity - Duh
"""
import numpy as np

# Dataset
edges = [
    ("A","B",2),
    ("A","C",3),
    ("A","D",4),
    ("B","E",4),
    ("B","F",5),
    ("C","J",7),
    ("D","H",4),
    ("F","D",2),
    ("H","G",3),
    ("J","G",4)
]

def adjacency_list(edges):
    """
    Converts edge list to  adjacency dictionary for O(1) neighbir lookup.
    Each origin node contains its own list of tuples with: (destination node, cost).
    """
    # Intiate empty dictionary
    graph = {}

    # Iterate over edges, unpacking each tuple into (source, destination, cost)
    for origin, destination, cost in edges:
        # Ensure every node is a key by adding an empty list when node isn't in graph
        if origin not in graph:
            graph[origin] = []

        # Leaf nodes have no outgoing edges but must exist
        if destination not in graph:
            graph[destination] =  []

        # Append (destination, cost) to graph[origin]
        graph[origin].append((destination, cost))

    return graph

# Find root node
def find_root(graph):
    """
    Identify root node (no incoming edges), appearing as ORIGIN but has no destination.
    --------------------------------------------------------
    Mathematically: A set-difference operation:
        Origins = All keys in graph
        Destinations = Every node that appears inside any neighbor list...
    """
    # Build set of ALL origin nodes
    origins = set(graph.keys())

    # Initialize destination node set
    destinations = set()

    # Build set of ALL destination nodes by iterating thru graph values
    for val in graph.values():
        # Iterate thru each (neighbor, cost) pair, collecting neighbor
        for dest, cost in val:
            destinations.add(dest)

    # Set difference operation
    roots = origins - destinations

    return roots.pop()

def breadth_first_search(graph, root, goal):
    """
    Find a path from start to goal by exploring level-by-level.
    BFS uses a FIFO queue.
    Nodes discovered first get processed first.
    --------------------------------------------------------
    Compare: 
        Queue (FIFO) -> BFS (explores wide, finds shallpwest path)
        Stack (LIFO) -> DFS (explores deep, not guaranteed optimal)
    -------------------------------------------------------
    Queue:
        Each entry in queue tracks: (current_node, path_sofar, cumulative cost)
        path_sofar for when you reach the goal, you return the full route.

    Visited set:
        Mark node as visited the moment you add it to queue.,
        not when you POP it, otherwise you enqueue duplicates.
    """
    # Initialize queue with single entry (root, [root], cost=0)
    queue = [(root, [root], 0)]

    # Initialize visited items as a set w/ root
    seen = set(root)

    # While queue isn't empty, POP from front
    while queue:
        # FIFO 
        current, path, cost = queue.pop(0)

        # Check if goal is reached, and if so return path, cost
        if current == goal:
            return path, cost

        # Iterate over graph[current] to get (neighbor, edge_cost)
        for neighbor, edge_cost in graph[current]:
            # If neighbor isn't in seen (visited), add neightbor visited, build new_path = path + [neighbor]
            # new_cost = cost + edge_cost
            # append(neighbor, new_path, new_cost) to queue
            if neighbor not in seen:
                seen.add(neighbor)
                new_path = path + [neighbor]
                new_cost = cost + edge_cost

                queue.append((neighbor, new_path, new_cost))

    # If  loop ends with no goal found, return (None, None)
    return None, None

def main():
    # ====== Build graph ===========
    graph = adjacency_list(edges)

    # ======== Find root node ========
    root = find_root(graph)

    # ======= BFS algorithm ===========
    # Goal node
    GOAL = "G"

    # BSF algorithm
    path, cost = breadth_first_search(graph, root, GOAL)

    # Output results
    print(f"Root node: {root}")
    print(f"Goal node: {GOAL}")
    print(f"Path: {path}")
    print(f"Cost: {cost}")

if __name__ == "__main__":
    main()


# ============================================================
# BFS IMPLEMENTATION GUIDE
# Reference the full solution only after attempting each step
# ============================================================

# ---------------------------------------------------------------
# STEP 1: Define your raw data
# ---------------------------------------------------------------
# edges = []  # Your edge list goes here as (source, dest, cost) tuples


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










