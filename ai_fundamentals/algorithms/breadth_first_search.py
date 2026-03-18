#!/usr/bin/env python
import time


class Queue(object):
    def __init__(self):
        # Empty list
        self.items = []

    def __len__(self):
        """
        Determine number of items in container
        """
        return len(self.items)

    def __repr__(self):
        """
        Provides the string version of the obejct.
        """
        return f"Queue(front -> {self.items} <- back)"

    def enqueue(self, element, verbose=False):
        """
        Adds element to queue.
        Includes 'verbose' flag so when I use this class in Breadth First
        Search, it won't print a million things out to the console unless I
        want it to for debugging purposes.
        """
        self.items.append(element)

        if verbose:
            print(f"\nAdding {element} to the Queue")
            print(f"New Queue: {self.items}")

    def dequeue(self, verbose=False):
        """
        Removes element from queue
        """
        # Ensure list isn't empty
        if not self.items:
            raise IndexError("Cannot Dequeue an empty list!")

        current_node = self.items.pop(0)

        if verbose:
            print(f"\nRemoving {current_node} from the Queue")
            print(f"New Queue: {self.items}")

        return current_node

    def is_empty(self):
        """
        Ensures the Queue isn't empty when popping off the mothafucka, returning True/False.
        """
        return len(self) == 0

    def peek(self, verbose=False):
        """
        Take a look at the front item without removing it.
        """
        if self.is_empty():
            raise IndexError("Cannot peek at an empty list!")

        if verbose:
            print(f"Front item: {self.items[0]}")

        return self.items[0]


def build_graph(edges):
    """
    Converts edges into a Graph.
    Returns a dictionary where each key is a node and the values are that
    node's neighbors.
    ------------------------------
    Iterates thru edges and builds the dictionary representing my graph.
    """
    # Initiate empty dictionary
    graph = {}

    # Get nodes as keys (unique)
    for i in range(len(edges)):
        # Unique Node
        key = edges[i][0]

        # No duplicate keys!
        if key not in graph:
            # Create empty list for each node to place neighbors and costs in
            graph[key] = []
            
        # Append (destination, cost) to node w/in list
        graph[key].append((edges[i][1], edges[i][2]))

    return graph

def bfs(graph, start, goal):
    """
    Breadth-First Search on a directed graph.

    Explores nodes layer by layer, guaranteeing the SHORTEST PATH in terms of
    number of hops, but not COST since BFS ignores edge weights.
    ------------------------------------------------------------------------
    INPUT:
        graph: (dict) 
        start: (str) Root node
        goal: (str) End node

    OUTPUT:
        (list or None) Path from start to goal as a list of node names.
            Returns None if no path exists
    ------------------------------------------------------------------------
    ALGORITHM
    ==========
    1. Create Queue, enqueueing initial path w/ root node
    2. Create visited set, adding root node to it right away.
    3. While queue isn't empty:
        - Dequeue front path
        - Current node is last element of the path
        - If current_node == goal, return path... DONE.
        - Look up current node's neighbors in graph
        - For each (neighbor, cost) in neighbors:
            - If neighbor not visited:
                * Add neighbor to visited
                * Build new path: current_path + [neighbor]
                * Enqueue new path
    4. If loop end without returning, return None
    """
    # Create Queue
    que = Queue()

    # Enqueue the path containing root node (enqueuing a  list (path))
    que.enqueue([(start, 0)])

    # Visited set w/ root node
    visited = set()
    visited.add(start)
    
    while not que.is_empty():
        # Dequeue  PATH
        current_path = que.dequeue()
        current_node = current_path[-1][0]

        # If the current node is the goal node, return the PATH
        if current_node == goal:
            # Sum costs in path
            cost = sum([cst for tup in current_path for cst in tup if
                    isinstance(cst, int)])

            # Output Steps and cost
            print(f"\nSteps to goal: {len(current_path)}")
            print(f"Cost to get to solution: {cost}")

            print(f"\nThe path to goal node is {current_path}")
            print(f"Goal node is {goal}")
            print(f"Nodes visited: {visited}")
            print(f"Number of nodes visited before solution was found: {len(visited)}")
            return current_path

        # Find neighbors; .get() used to return an empty list incase of
        # KeyError
        for adjacent in graph.get(current_node, []):

            if adjacent[0] not in visited:
                # Add neighbor to visited set
                visited.add(adjacent[0])
                
                
                # Build new path
                new_path = current_path + [adjacent]
               
                # Enqueue current path + list of neighbors
                que.enqueue(new_path)

    return None

def main():
    # Example usage
    cities = ["Austin", "Boston", "Chicago", "Denver", "Eugene"]
    print(f"Cities to visit: {cities}")

    # ------- Time it for BIG O notation examination for Queue -------
    sizes = [1000, 5000, 10000, 50000, 100000]

    print(f"{'n':>10} {'time (sec)':>12}")

    for n in sizes:
        que = Queue()

        for i in range(n):
            que.enqueue(i)

        start = time.perf_counter()

        while not que.is_empty():
            que.dequeue()

        end = time.perf_counter()
        # O(n^2) since after each dequeue, items in list get shifted to the
        # left which causes this inefficiency
        print(f"{n:>10} {end - start:>12.6f}")

    # ++++++++++++ Build graph and implement BFS ++++++++++++++++++
    # Directed edges (dataset)
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

    graph = build_graph(edges)
    print(f"\nGraph: {graph}")

    # BFS
    ROOT = list(graph.keys())[0]
    GOAL = "E"

    bfs(graph, ROOT, GOAL)

if __name__ == "__main__":
    main()
