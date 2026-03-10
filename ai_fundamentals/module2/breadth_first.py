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

def breadth_first_search(graph, visited):
    """
    PROBLEM : Statespace - SET of all possible states in the environment.
    BREADTH FIRST SEARCH where we choose a node N, with minimum value of some
    evaluation function f(n), and on each iteration we choose a node on the
    frontier with minimum f(n) value, returning the state if it's a goal state,
    otherwise expanding to generate child nodes.
    """
    pass

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

# Find root node
def find_root(graph):
    """
    Identify root node (no incoming edges), appearing as ORIGIN but has no destination.
    --------------------------------------------------------
    Mathematically: A set-difference operation:
        Origins = All keys in graph
        Destinations = Every node that appears inside any neighbor list...
    """
    pass

def breadth_first_search(graph, start, goal):
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
    pass

# ====== Build graph ===========
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


# ======== Find root node ========
# Build set of ALL origin nodes
origins = set(graph.keys())

# Initialize destination node set
destinations = set()

# Build set of ALL destination nodes by iterating thru graph values
for val in graph.values():
    # Iterate thru each (neighbor, cost) pair, collecting neighbor
    for dest, cost in val:
        destinations.add(dest)

roots = origins - destinations
root = roots.pop()

# ======= BFS algorithm ===========
# Initialize queue with single entry
queue = [root]

# Initialize visited items as a set w/ root


    











