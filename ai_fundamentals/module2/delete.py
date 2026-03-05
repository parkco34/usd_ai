#!/usr/bin/env python
"""
Assignment 2.2 — Advanced Search Problems (NetworkX + OSMnx)

Goal:
- Find the shortest driving-distance path between:
  (1) Oracle Park, San Francisco, CA
  (2) Palace of Fine Arts, San Francisco, CA

Algorithms used (via NetworkX):
- Dijkstra (lowest-cost-first) on edge attribute 'length' (meters)
- A* using an admissible heuristic = straight-line (great-circle) distance to goal (meters)

Style reference: lecture.py :contentReference[oaicite:0]{index=0}
"""

import math
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox


# =========================
# ====== CONFIG ===========
# =========================
PLACE = "San Francisco, California, USA"
NETWORK_TYPE = "drive"  # driving network

ORACLE_PARK = "Oracle Park, San Francisco, California, USA"
PALACE_FINE_ARTS = "Palace of Fine Arts, San Francisco, California, USA"

WEIGHT = "length"  # OSMnx edge lengths are in meters


# =========================
# ====== HELPERS ==========
# =========================
def meters_to_miles(m: float) -> float:
    return m / 1609.344


def route_length_meters(G: nx.MultiDiGraph, route: list[int], weight: str = "length") -> float:
    """
    Sum the edge weights along a route (node list). For MultiDiGraph, choose the
    minimum-weight parallel edge between consecutive nodes.
    """
    total = 0.0

    for u, v in zip(route[:-1], route[1:]):
        data = G.get_edge_data(u, v)
        breakpoint()
        if data is None:
            raise ValueError(f"No edge data between {u} -> {v}")

        # MultiDiGraph: data is dict keyed by edge keys
        # Choose the edge with minimum weight (common approach in OSMnx examples)
        best = min(data.values(), key=lambda d: d.get(weight, float("inf")))
        total += float(best.get(weight, 0.0))

    return total


def astar_heuristic_factory(G: nx.MultiDiGraph, goal: int):
    """
    Create an admissible heuristic function for A*:
    h(n) = great-circle distance (meters) from node n to goal node.

    Using straight-line distance is admissible for road networks when edge costs are >= 0
    and represent distances, because any drivable path cannot be shorter than the
    straight-line distance between endpoints.
    """
    goal_y = float(G.nodes[goal]["y"])  # latitude
    goal_x = float(G.nodes[goal]["x"])  # longitude

    def h(n1: int, n2_ignored: int) -> float:
        y = float(G.nodes[n1]["y"])
        x = float(G.nodes[n1]["x"])
        return ox.distance.great_circle(y, x, goal_y, goal_x)

    return h


# =========================
# ====== MAIN =============
# =========================
def main():
    # OSMnx settings (cache saves time on repeated runs)
    ox.settings.use_cache = True
    ox.settings.log_console = False

    print("# ====== Build street network (drive) ========")
    G = ox.graph_from_place(PLACE, network_type=NETWORK_TYPE, simplify=True)

    # Ensure edge lengths exist (usually they do, but this makes it explicit/robust)
    G = ox.distance.add_edge_lengths(G)

    print(f"Nodes: {len(G.nodes):,} | Edges: {len(G.edges):,}")

    print("\n# ====== Geocode origin/destination ========")
    origin_point = ox.geocode(ORACLE_PARK)       # (lat, lon)
    dest_point = ox.geocode(PALACE_FINE_ARTS)    # (lat, lon)

    print(f"Origin (Oracle Park): {origin_point}")
    print(f"Destination (Palace of Fine Arts): {dest_point}")

    # Nearest nodes on the drive graph
    # OSMnx uses X=longitude, Y=latitude
    origin_node = ox.distance.nearest_nodes(G, X=origin_point[1], Y=origin_point[0])
    dest_node = ox.distance.nearest_nodes(G, X=dest_point[1], Y=dest_point[0])

    print(f"\nNearest origin node: {origin_node}")
    print(f"Nearest dest node:   {dest_node}")

    # ------------------------------------------------------------
    # 1) Dijkstra (lowest-cost-first) — optimal for nonnegative weights
    # ------------------------------------------------------------
    print("\n# ====== Dijkstra (Shortest path by driving distance) ========")
    dijkstra_route = nx.shortest_path(G, source=origin_node, target=dest_node, weight=WEIGHT, method="dijkstra")
    dijkstra_dist_m = route_length_meters(G, dijkstra_route, weight=WEIGHT)

    print(f"Dijkstra route length: {dijkstra_dist_m:,.1f} m  ({meters_to_miles(dijkstra_dist_m):.2f} miles)")
    print(f"Dijkstra route nodes:  {len(dijkstra_route):,}")

    # ------------------------------------------------------------
    # 2) A* — optimal if heuristic is admissible
    # ------------------------------------------------------------
    print("\n# ====== A* (Shortest path by driving distance + heuristic) ========")
    h = astar_heuristic_factory(G, goal=dest_node)
    astar_route = nx.astar_path(G, source=origin_node, target=dest_node, heuristic=h, weight=WEIGHT)
    astar_dist_m = route_length_meters(G, astar_route, weight=WEIGHT)

    print(f"A* route length: {astar_dist_m:,.1f} m  ({meters_to_miles(astar_dist_m):.2f} miles)")
    print(f"A* route nodes:  {len(astar_route):,}")

    # ------------------------------------------------------------
    # Visualize routes
    # ------------------------------------------------------------
    print("\n# ====== Plot routes ========")
    fig1, ax1 = ox.plot_graph_route(G, dijkstra_route, route_linewidth=4, node_size=0, bgcolor="white", show=False, close=False)
    ax1.set_title("Oracle Park → Palace of Fine Arts (Dijkstra)", fontsize=14)
    plt.show()

    fig2, ax2 = ox.plot_graph_route(G, astar_route, route_linewidth=4, node_size=0, bgcolor="white", show=False, close=False)
    ax2.set_title("Oracle Park → Palace of Fine Arts (A*)", fontsize=14)
    plt.show()

    # Quick sanity: these should typically match (both optimal for distance)
    print("\n# ====== Compare results ========")
    delta_m = abs(dijkstra_dist_m - astar_dist_m)
    print(f"|Dijkstra - A*| distance difference: {delta_m:.6f} meters")
    if delta_m < 1e-6:
        print("Same optimal distance (as expected).")
    else:
        print("Slight difference (can happen due to parallel edges / tie-breaking).")


if __name__ == "__main__":
    main()

