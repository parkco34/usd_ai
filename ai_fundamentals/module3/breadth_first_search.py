#!/usr/bin/env python
"""
1) What objects does the algporithm need to exist?
2) 
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


class Queue(object):
    def __init__(self):
        # Empty list
        self.items = []

    def __len__():
        """
        Determine number of items in container
        """
        return len(self.items)

    def __repr__(self):
        """
        Provides the string version of the obejct.
        """
        return f"Queue(front -> {self.items} <- back)"

    def enqueue(self, element):
        """
        Adds element to queue
        """
        print(f"\nAdding {element} to the Queue")
        self.items.append(element)
        print(f"New Queue: {self.items}")

    def dequeue(self):
        """
        Removes element from queue
        """
        # Ensure list isn't empty
        if not self.items:
            raise IndexError("Cannot Dequeue an empty list!")

        current_node = self.items.pop(0)

        print(f"\nRemoving {current_node} from the Queue")
        print(f"New Queue: {self.items}")

        return current_node

    def is_empty():
        """
        Ensures the Queue isn't empty when popping off the mothafucka, returning True/False.
        """
        if not self.items:
            return True

        else:
            return False

    def peek():
        """
        Take a look at the front item without removing it.
        """
        print(f"Front item: {self.items[0]}")


def main():
    # Example usage
    cities = ["Austin", "Boston", "Chicago", "Denver", "Eugene"]
    print(f"Cities to visit: {cities}")

    # Queue object
    que = Queue()
    print(f"Initial queue: {que.items}")

    for city in cities:
        # Enqueue
        que.enqueue(city)

    for item in cities:
        que.dequeue()

if __name__ == "__main__":
    main()
