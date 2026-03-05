#!/usr/bin/env python
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from math import hypot

# COnstants: coordinates, locations, and type of network
ORACLE_PARK = "Oracle Park, San Francisco, California, USA"
PALACE_FINE_ARTS = "Palace of Fine Arts, San Francisco, California, USA"
PLACE_NAME = "San Francisco, California, USA"
# Drivable roads only
NETWORK_TYPE="drive"
# Edge length in meters
WEIGHT="length"

# Output coordinates
print("Oracle Park:", ORACLE_PARK)
print("Palace of Fine Arts:", PALACE_FINE_ARTS)

# Download and create MultiDirect graph w/in boundaries
# 'drive' param for network type
G = ox.graph_from_place(
    query=PLACE_NAME,
    network_type=NETWORK_TYPE, 
    # Merges nodes along straight road segment
    simplify=True
)

def meters_to_miles(meters):
    """
    Converts meters to miles
    ---------------------------
    INPUT:
        meters: (float) 

    OUTPUT:
        miles: (float)
    """
    return meters / 1609.34

def route_length(graph_obj, route, weight):
    """
    Accumulates the edge weights along a route (list of nodes), choosing the
    minimum-weight parallel edge that's parallel between the consecutive nodes.
    ------------------------------------------
    INPUT:
        graph_obj: (Graph Object) G
        route: (list) Nodes
        weight: (str) 'length'

    OUTPUT:
        total: (float) Sum of edge weights along route.
    """
    # Initialize total
    total = 0.0

    # Iterate through edge pairs like: (A, B), (B, C), (C, D)
    for u, v in zip(route[:-1], route[1:]):
        # Get dictionary of edges (u, v)
        data = G.get_edge_data(u, v)

        # Choose 'best' edge
        best = list(data.values())[0].get(weight, 0)
        # Cumulative sum
        total += best

    return total

def astar(graph_obj, goal):
    """
    Admissable heuristic function for A* using Euclidean disance (meters) from node to goal node.
    INPUT:
        graph_obj: (nx Graph object) G
        goal: (int) ?

    OUTPUT:
        heur: (heuristic function)
    """
    # Latitude/longitude
    goal_y = G.nodes[goal]["y"]
    goal_x = G.nodes[goal]["x"]

    # First class heuristic Euclidean function
    def euclid(n1, n2):
        """
        Euclidean heuristic function
        """
        y1, y2 = G.nodes[n1]["y"], G.nodes[n2]["y"]
        x1, x2 = G.nodes[n1]["x"], G.nodes[n2]["x"]

        return hypot(x2 - x1, y2 - y1)

# ===== Main ======
# Coordinates
origin = ox.geocode(ORACLE_PARK)       # (lat, lon)
destination = ox.geocode(PALACE_FINE_ARTS)    # (lat, lon)

# Make Graph object
G = ox.graph_from_place(PLACE_NAME, network_type=NETWORK_TYPE)

# Nodes
origin_node = ox.distance.nearest_nodes(G, X=origin[1], Y=origin[0])
dest_node = ox.distance.nearest_nodes(G, X=destination[1], Y=destination[0])


dijkstra_route = nx.shortest_path(
    G, 
    source=origin_node, 
    target=dest_node, 
    weight=WEIGHT, 
    method="dijkstra")

dijkstra_dist= route_length(G, dijkstra_route, weight=WEIGHT)
