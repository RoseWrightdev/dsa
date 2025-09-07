import collections
import heapq
import turtle
import math
from typing import List, Set, Dict, Tuple, Any, Deque, Optional

# ======================================================================
# I. GRAPH REPRESENTATION (ADJACENCY LIST)
# ======================================================================

class Graph:
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

