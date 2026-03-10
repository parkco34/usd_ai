#!/usr/bin/env python
"""
Breadth First Search is a graph traversal algorithm that explores nodes by
layer, visiting all neighbors at current depth before moving to next level.
- Used when all actions have the same cost.
MEASURES:
    Completeness  - Guaranteed to find solution if one exists
    Cost Optimal  - Yes, when all edge costs are equal
    Time          - O(b^d) where b=branching factor, d=depth of solution
    Space         - O(b^d) — keeps ALL nodes in memory (biggest weakness)
"""

# Dataset
edges = [
    ("A","B",2), ("A","C",3), ("A","D",4),
    ("B","E",4), ("B","F",5), ("C","J",7),
    ("D","H",4), ("F","D",2), ("H","G",3), ("J","G",4)
]


def build_adjacency_list(edges):
    """
    Converts edge list into adjacency dict for O(1) neighbor lookup.

    An adjacency list maps each node -> list of (neighbor, cost) tuples.
    This is the standard graph representation for traversal algorithms.

    Without this, finding neighbors of node V requires scanning ALL edges
    every time — O(E) per lookup vs O(1) with a dict.

    Example:
        edges = [("A","B",2), ("A","C",3)]
        result = {"A": [("B",2), ("C",3)], "B": [], "C": []}
    """
    graph = {}

    for source, dest, cost in edges:
        # Ensure every node exists as a key, even if it has no outgoing edges
        if source not in graph:
            graph[source] = []
        if dest not in graph:
            graph[dest] = []

        graph[source].append((dest, cost))

    return graph


def find_root(graph):
    """
    Root node = node that appears as a SOURCE but never as a DESTINATION.
    Mathematically: root ∈ Sources − Destinations  (set difference)

    In a well-formed tree there is exactly one such node.
    """
    all_sources = set(graph.keys())
    all_destinations = set()

    for neighbors in graph.values():
        for (node, _) in neighbors:
            all_destinations.add(node)

    roots = all_sources - all_destinations
    return roots.pop()  # Returns the single root node


def breadth_first_search(graph, start, goal):
    """
    BFS explores the graph LEVEL BY LEVEL using a FIFO queue.

    KEY INSIGHT — Why a queue (FIFO)?
        - Queue ensures we process nodes in the order they were discovered
        - This guarantees we always explore shallower nodes before deeper ones
        - A stack (LIFO) would give you DFS instead

    FRONTIER vs EXPLORED:
        - Frontier  : Nodes discovered but not yet processed (the queue)
        - Explored  : Nodes already processed (visited set)
        - Together they prevent infinite loops in cyclic graphs

    PATH TRACKING:
        - We store the full path alongside each node in the queue
        - Each queue entry is (current_node, path_so_far, cost_so_far)
        - When we reach the goal, we already have the complete path

    PARAMETERS:
        graph (dict) : Adjacency list from build_adjacency_list()
        start (str)  : Starting node label
        goal  (str)  : Target node label

    RETURNS:
        (path, total_cost) tuple if goal found, else (None, None)
    """
    # Queue entries: (current_node, path_taken, cumulative_cost)
    queue = [(start, [start], 0)]

    # Visited set prevents re-processing nodes in cycles
    visited = set()
    visited.add(start)

    while queue:
        # FIFO: pop from the FRONT of the list (index 0)
        current_node, path, cost = queue.pop(0)

        # Goal check — BFS guarantees this is the SHALLOWEST solution
        if current_node == goal:
            return path, cost

        # Expand: add all unvisited neighbors to the back of the queue
        for (neighbor, edge_cost) in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                new_cost = cost + edge_cost
                queue.append((neighbor, new_path, new_cost))

    # Exhausted all reachable nodes without finding goal
    return None, None


# ===== MAIN =====
graph = build_adjacency_list(edges)
start = find_root(graph)
goal  = "G"

path, cost = breadth_first_search(graph, start, goal)

print(f"Start : {start}")
print(f"Goal  : {goal}")
print(f"Path  : {' -> '.join(path)}")
print(f"Cost  : {cost}")
