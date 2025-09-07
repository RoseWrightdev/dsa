import collections
import heapq
import turtle
import math
from typing import List, Set, Dict, Tuple, Any, Deque, Optional

# ======================================================================
# I. GRAPH REPRESENTATIONS
# ======================================================================

# --- Representation 1: Using Node Objects ---
# This is an object-oriented approach, similar to Trees or Linked Lists.
# Each node is an object that knows about its neighbors.

class GraphNode:
    """Represents a vertex in the graph using an object."""
    def __init__(self, val: Any):
        self.val = val
        self.neighbors: List['GraphNode'] = []

# --- Representation 2: Using an Adjacency List ---
# This is a common and efficient way to represent a graph with a single
# manager object. The algorithms below are written to work with this class.

class Graph:
    """Represents a graph using an adjacency list (a dictionary)."""
    def __init__(self, directed: bool = False):
        ...

    def add_edge(self, u: Any, v: Any, weight: int = 1) -> None:
        ...

    def get_nodes(self) -> List[Any]:
        ...

    def __str__(self) -> str:
        ...

# ======================================================================
# II. TRAVERSAL ALGORITHMS
# ======================================================================

def bfs(graph: Graph, start_node: Any) -> List[Any]:
    ...

def dfs(graph: Graph, start_node: Any) -> List[Any]:
    ...

# ======================================================================
# III. CYCLE DETECTION
# ======================================================================

def is_cycle_undirected(graph: Graph) -> bool:
    ...

def is_cycle_directed(graph: Graph) -> bool:
    ...

# ======================================================================
# IV. SHORTEST PATH
# ======================================================================

def dijkstra_shortest_path(graph: Graph, start_node: Any) -> Dict[Any, float]:
    ...

def shortest_path_unweighted(graph: Graph, start: Any, end: Any) -> List[Any]:
    ...

# ======================================================================
# V. MINIMUM SPANNING TREE (MST)
# ======================================================================

def prim_mst(graph: Graph) -> Tuple[int, List[Tuple[Any, Any, int]]]:
    ...

# ======================================================================
# VI. TOPOLOGICAL SORTING
# ======================================================================

def topological_sort(graph: Graph) -> List[Any]:
    ...

# ======================================================================
# VII. UNION-FIND (DISJOINT SET UNION)
# ======================================================================

class UnionFind:
    def __init__(self, nodes: List[Any]):
        ...

    def find(self, i: Any) -> Any:
        ...

    def union(self, i: Any, j: Any) -> None:
        ...

# ======================================================================
# VIII. GRID-BASED PROBLEMS (ISLANDS, ROTTEN TOMATOES, FLOOD FILL)
# ======================================================================

def number_of_islands(grid: List[List[str]]) -> int:
    ...

def rotten_tomatoes(grid: List[List[int]]) -> int:
    ...

def flood_fill(image: List[List[int]], sr: int, sc: int, new_color: int) -> List[List[int]]:
    ...

# ======================================================================
# IX. VISUALIZATION
# ======================================================================

def visualize_graph_turtle(graph: Graph) -> None:
    ...