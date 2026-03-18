#!/usr/bin/env python
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from math import hypot

# Constants: coordinates, locations, and type of network
ORACLE_PARK = "Oracle Park, San Francisco, California, USA"
PALACE_FINE_ARTS = "Palace of Fine Arts, San Francisco, California, USA"
PLACE_NAME = "San Francisco, California, USA"
# Drivable roads only
NETWORK_TYPE="drive"
# Edge length in meters
WEIGHT="length"

# Output coordinates
print("Origin:", ORACLE_PARK)
print("Destination:", PALACE_FINE_ARTS)

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
        data = graph_obj.get_edge_data(u, v)

        # Use min() across all parallel edges
        best = min(attrs.get(weight, 0) for attrs in data.values())
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
    goal_y = graph_obj.nodes[goal]["y"]
    goal_x = graph_obj.nodes[goal]["x"]

    # First class heuristic Euclidean function
    def geo_path(n1, n2):
        """
        Heuristic function
        """
        y1 = graph_obj.nodes[n1]["y"]
        x1 = graph_obj.nodes[n1]["x"]
        
        # orthodrome
        return ox.distance.great_circle(y1, x1, goal_y, goal_x)

    return geo_path

# ===== Main ======
# Coordinates
origin = ox.geocode(ORACLE_PARK)       # (lat, lon)
destination = ox.geocode(PALACE_FINE_ARTS)    # (lat, lon)

# Nodes
origin_node = ox.distance.nearest_nodes(G, X=origin[1], Y=origin[0])
destination_node = ox.distance.nearest_nodes(G, X=destination[1], Y=destination[0])

# Dijkstra Algorithm
dijkstra_route = nx.shortest_path(
    G,
    source=origin_node,
    target=destination_node,
    weight=WEIGHT,
    method="dijkstra")

dijkstra_dist= route_length(G, dijkstra_route, weight=WEIGHT)

# A* Algorithm
heuristic = astar(G, goal=destination_node)
astar_route = nx.astar_path(
    G,
    source=origin_node,
    target=destination_node,
    heuristic=heuristic,
    weight=WEIGHT)

astar_distance = route_length(G, astar_route, weight=WEIGHT)

# OUtput for Dijkstra path
print(f"Origin node: {origin_node}")
print(f"Destination node: {destination_node}")
print(f"\nOrigin Coordinates (Oracle Park): {origin}")
print(f"Destination Coordinates (Palace of Fine Arts): {destination}")

print("\n====== Dijkstra (Shortest Path) ========")
print(f"\nDijlstra Route Length: {meters_to_miles(dijkstra_dist):.1f} miles")

# Output for A* w/ Heuristic
print("\n======== A* w/ Heuristic ======")
print(f"A* Route length: {meters_to_miles(astar_distance):.1f} miles")

# Visualization
fig1, ax1 = ox.plot_graph_route(G, dijkstra_route, route_linewidth=4, node_size=0, bgcolor="white", show=False, close=False)
ax1.set_title("Oracle Park → Palace of Fine Arts (Dijkstra)", fontsize=14)
plt.show()

fig2, ax2 = ox.plot_graph_route(G, astar_route, route_linewidth=4, node_size=0, bgcolor="white", show=False, close=False)
ax2.set_title("Oracle Park → Palace of Fine Arts (A*)", fontsize=14)
plt.show()



