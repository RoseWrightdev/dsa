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
        self.directed: bool = directed

    def add_edge(self, u: Any, v: Any, weight: int = 1) -> None:
        self.graph[u].append((v, weight))
        if not self.directed:
            self.graph[v].append((u, weight))

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
    if start_node not in graph.graph:
        return []
    visited: Set[Any] = set()
    queue: Deque[Any] = collections.deque([start_node])
    traversal_order: List[Any] = []
    visited.add(start_node)
    while queue:
        vertex = queue.popleft()
        traversal_order.append(vertex)
        for neighbor, weight in graph.graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return traversal_order

def dfs(graph: Graph, start_node: Any) -> List[Any]:
    if start_node not in graph.graph:
        return []
    visited: Set[Any] = set()
    traversal_order: List[Any] = []
    def _dfs_recursive(vertex: Any) -> None:
        visited.add(vertex)
        traversal_order.append(vertex)
        for neighbor, weight in graph.graph[vertex]:
            if neighbor not in visited:
                _dfs_recursive(neighbor)
    _dfs_recursive(start_node)
    return traversal_order

# ======================================================================
# III. CYCLE DETECTION
# ======================================================================

def is_cycle_undirected(graph: Graph) -> bool:
    visited: Set[Any] = set()
    for node in graph.get_nodes():
        if node not in visited:
            if _is_cycle_undirected_util(graph, node, visited, -1):
                return True
    return False

def _is_cycle_undirected_util(graph: Graph, u: Any, visited: Set[Any], parent: Any) -> bool:
    visited.add(u)
    for v, weight in graph.graph[u]:
        if v not in visited:
            if _is_cycle_undirected_util(graph, v, visited, u):
                return True
        elif v != parent:
            return True
    return False

def is_cycle_directed(graph: Graph) -> bool:
    visited: Set[Any] = set()
    recursion_stack: Set[Any] = set()
    for node in graph.get_nodes():
        if node not in visited:
            if _is_cycle_directed_util(graph, node, visited, recursion_stack):
                return True
    return False

def _is_cycle_directed_util(graph: Graph, u: Any, visited: Set[Any], recursion_stack: Set[Any]) -> bool:
    visited.add(u)
    recursion_stack.add(u)
    for v, weight in graph.graph[u]:
        if v not in visited:
            if _is_cycle_directed_util(graph, v, visited, recursion_stack):
                return True
        elif v in recursion_stack:
            return True
    recursion_stack.remove(u)
    return False

# ======================================================================
# IV. SHORTEST PATH
# ======================================================================

def dijkstra_shortest_path(graph: Graph, start_node: Any) -> Dict[Any, float]:
    pq: List[Tuple[int, Any]] = [(0, start_node)]
    distances: Dict[Any, float] = {node: float('inf') for node in graph.get_nodes()}
    if start_node in distances:
        distances[start_node] = 0
    while pq:
        current_dist, current_vertex = heapq.heappop(pq)
        if current_dist > distances.get(current_vertex, float('inf')):
            continue
        for neighbor, weight in graph.graph[current_vertex]:
            distance = current_dist + weight
            if distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances

def shortest_path_unweighted(graph: Graph, start: Any, end: Any) -> List[Any]:
    if start == end: return [start]
    queue: Deque[Tuple[Any, List[Any]]] = collections.deque([(start, [start])])
    visited: Set[Any] = {start}
    while queue:
        vertex, path = queue.popleft()
        for neighbor, _ in graph.graph[vertex]:
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []

# ======================================================================
# V. MINIMUM SPANNING TREE (MST)
# ======================================================================

def prim_mst(graph: Graph) -> Tuple[int, List[Tuple[Any, Any, int]]]:
    if not graph.graph:
        return 0, []
    start_node = next(iter(graph.graph))
    pq: List[Tuple[int, Any, Any]] = [(weight, start_node, neighbor) for neighbor, weight in graph.graph[start_node]]
    heapq.heapify(pq)
    visited: Set[Any] = {start_node}
    mst_cost: int = 0
    mst_edges: List[Tuple[Any, Any, int]] = []
    while pq and len(visited) < len(graph.get_nodes()):
        weight, u, v = heapq.heappop(pq)
        if v not in visited:
            visited.add(v)
            mst_cost += weight
            mst_edges.append((u, v, weight))
            for neighbor, neighbor_weight in graph.graph[v]:
                if neighbor not in visited:
                    heapq.heappush(pq, (neighbor_weight, v, neighbor))
    return mst_cost, mst_edges

# ======================================================================
# VI. TOPOLOGICAL SORTING
# ======================================================================

def topological_sort(graph: Graph) -> List[Any]:
    if not graph.directed:
        raise TypeError("Topological sort is only for directed graphs.")
    in_degree: Dict[Any, int] = {node: 0 for node in graph.get_nodes()}
    for u in graph.graph:
        for v, _ in graph.graph[u]:
            in_degree[v] += 1
    queue: Deque[Any] = collections.deque([node for node in in_degree if in_degree[node] == 0])
    topo_order: List[Any] = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v, _ in graph.graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    if len(topo_order) == len(graph.get_nodes()):
        return topo_order
    else:
        raise ValueError("Graph has a cycle, topological sort not possible.")

# ======================================================================
# VII. UNION-FIND (DISJOINT SET UNION)
# ======================================================================

class UnionFind:
    def __init__(self, nodes: List[Any]):
        self.parent: Dict[Any, Any] = {node: node for node in nodes}
        self.rank: Dict[Any, int] = {node: 0 for node in nodes}

    def find(self, i: Any) -> Any:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: Any, j: Any) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j
                if self.rank[root_i] == self.rank[root_j]:
                    self.rank[root_j] += 1

# ======================================================================
# VIII. GRID-BASED PROBLEMS (ISLANDS, ROTTEN TOMATOES, FLOOD FILL)
# ======================================================================

def number_of_islands(grid: List[List[str]]) -> int:
    if not grid: return 0
    rows, cols = len(grid), len(grid[0])
    count = 0
    def _dfs_sink(r: int, c: int) -> None:
        if not (0 <= r < rows and 0 <= c < cols and grid[r][c] == '1'):
            return
        grid[r][c] = '0'
        _dfs_sink(r + 1, c)
        _dfs_sink(r - 1, c)
        _dfs_sink(r, c + 1)
        _dfs_sink(r, c - 1)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                _dfs_sink(r, c)
                count += 1
    return count

def rotten_tomatoes(grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    queue: Deque[Tuple[int, int, int]] = collections.deque()
    fresh_oranges = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))
            elif grid[r][c] == 1:
                fresh_oranges += 1
    time = 0
    while queue:
        r, c, time = queue.popleft()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh_oranges -= 1
                queue.append((nr, nc, time + 1))
    return time if fresh_oranges == 0 else -1

def flood_fill(image: List[List[int]], sr: int, sc: int, new_color: int) -> List[List[int]]:
    rows, cols = len(image), len(image[0])
    original_color = image[sr][sc]
    if original_color == new_color:
        return image
    def _dfs(r: int, c: int) -> None:
        if not (0 <= r < rows and 0 <= c < cols and image[r][c] == original_color):
            return
        image[r][c] = new_color
        _dfs(r + 1, c)
        _dfs(r - 1, c)
        _dfs(r, c + 1)
        _dfs(r, c - 1)
    _dfs(sr, sc)
    return image

# ======================================================================
# IX. VISUALIZATION
# ======================================================================

def visualize_graph_turtle(graph: Graph) -> None:
    wn = turtle.Screen()
    wn.title("Graph Visualization")
    wn.bgcolor("white")
    t = turtle.Turtle()
    t.speed(0)
    t.hideturtle()
    nodes = graph.get_nodes()
    num_nodes = len(nodes)
    node_positions: Dict[Any, Tuple[float, float]] = {}
    radius = 200
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / num_nodes
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        node_positions[node] = (x, y)
    t.pencolor("gray")
    for u in graph.graph:
        for v, weight in graph.graph[u]:
            if v in node_positions:
                t.penup()
                t.goto(node_positions[u])
                t.pendown()
                t.goto(node_positions[v])
    for node, pos in node_positions.items():
        t.penup()
        t.goto(pos)
        t.pendown()
        t.pencolor("black")
        t.fillcolor("skyblue")
        t.begin_fill()
        t.circle(20)
        t.end_fill()
        t.penup()
        t.goto(pos[0], pos[1] - 10)
        t.write(str(node), align="center", font=("Arial", 12, "bold"))
    wn.mainloop()
