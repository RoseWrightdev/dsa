from collections import deque

class Graph:
    """
    A graph represented using an adjacency matrix.
    """
    def __init__(self, num_vertices, directed=False):
        # Initialize the matrix with all zeros
        ...

    def add_edge(self, v1, v2, weight=1):
        # Add an edge between two vertices
        ...

    def remove_edge(self, v1, v2):
        # Remove an edge between two vertices
        ...

    def get_neighbors(self, vertex):
        # Get all neighbors of a given vertex
        ...

    def __str__(self):
        # A simple string representation of the matrix
        ...

# =========================================
# --- Graph Traversal Algorithms
# =========================================

def bfs(graph, start_vertex):...
def dfs(graph, start_vertex):...

# =========================================
# --- Common Graph Problems
# =========================================

def has_path():...
def shortest_path_unweighted():... # (Hint: Use BFS)