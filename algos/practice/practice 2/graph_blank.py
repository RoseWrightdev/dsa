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
        self.graph: Dict[Any, List[Tuple[Any, int]]] = collections.defaultdict(list)
        self.directed = directed

    def add_edge(self, u: Any, v: Any, weight: int = 1) -> None:
        self.graph[u].append((v, weight))

    def get_nodes(self) -> List[Any]:
        return list(self.graph.keys())

    def __str__(self) -> str:
        s = ""
        for i in self.graph:
            s += f"{i} -> {self.graph[i]}\n"
        return s

# ======================================================================
# II. TRAVERSAL ALGORITHMS
# ======================================================================

def bfs(graph: Graph, start_node: Any) -> List[Any]:
    # edgecases -> what if the starting node isn't in the graph? then return an empty list
    # init set, and queue
    # we need a set so we don't infly consume the same verticies
    # we need a queue so we can track throughout time what we have visited linearly, 1, 2, 3, 4
    # first in, first out
    # init the result
    if start_node not in graph.graph:
        return []
    visited = set() # use a set to prevent bfs from consuming the same nodes in an inf loop
    visited.add(start_node)
    # first in first out -> track throughout time what we have visited first
    queue = collections.deque([start_node])
    # the resulting order
    traversal_order = []
    # queue
    while queue:
        # the vertext of a graph is 
        vertex = queue.popleft()
        traversal_order.append(vertex)
        # add to set, sets must be uquie
        visited.add(vertex)
        for neighbor, _ in graph.graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return traversal_order


def dfs(graph: Graph, start_node: Any) -> List[Any]:

    # first check edgecases:
    # what if there the starting node isn't in the graph? 
    # then we return []

    # next we init a "visited" set and the "traversal_order" result
    
    # then we define the rec dfs helper function
    # this function first updates the visited set and travsal order
    # then it iterates through the neighbors of the vertex and
    # recursively calls rec_dfs(vertex) if the vertex isn't in the set
    if start_node not in graph.graph:
        return []
    visited = set()
    traversal_order = []

    def rec_dfs(vertex):
        visited.add(vertex)
        traversal_order.append(vertex)
        for neighbor, weight in graph.graph[vertex]:
            if neighbor not in visited:
                rec_dfs(neighbor)
    rec_dfs(start_node)
    return traversal_order

def dfs_iter(graph):...

# ======================================================================
# III. CYCLE DETECTION
# ======================================================================

def is_cycle_undirected(graph: Graph) -> bool:
    
    # Plan:
    # edgecase -> if there aren't any nodes in the graph, return false
    # init visited set to track if we have seen a vertex
    # if we have seen a vertex already while traversing
    # then there is a cycle in the graph
    # otherwise, there ins't a cycle
    # falsify pattern ->
    # prove the condition false through the graph, otherwise assume it is true
    if not graph.graph:
        return False
    visited = set()

    def dfs(vertex, parent):
        visited.add(vertex)
        for neighbor, _ in graph.graph[vertex]:
            if neighbor not in visited:
                if dfs(neighbor, vertex):
                    return True
            elif neighbor != parent:
                return True
        return False

    for vertex in graph.get_nodes():
        if vertex not in visited:
            if dfs(vertex, None):
                return True
    return False

def is_cycle_directed(graph: Graph) -> bool:
    # Cycle detection in a directed graph using DFS and recursion stack
    if not graph.graph:
        return False
    visited = set()
    rec_stack = set()

    def dfs(vertex):
        visited.add(vertex)
        rec_stack.add(vertex)
        for neighbor, _ in graph.graph[vertex]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(vertex)
        return False

    for vertex in graph.get_nodes():
        if vertex not in visited:
            if dfs(vertex):
                return True
    return False

# ======================================================================
# IV. SHORTEST PATH
# ======================================================================

def dijkstra_shortest_path(graph: Graph, start_node: Any) -> Dict[Any, float]:
    pq = [(0, start_node)]
    distances = {node: float('inf') for node in graph.get_nodes()}
    if start_node in distances:
        distances[start_node] = 0
    
    while pq:
        current_dist, current_vertex = heapq.heappop(pq)
        if current_dist > distances.get(current_vertex, float('inf')):
            continue
        for neighbor, weight in graph.graph[current_vertex]:
            distance = current_dist + weight
            

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