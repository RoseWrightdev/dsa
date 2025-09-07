from collections import deque
import math

class Graph:
    """
    A graph represented using an adjacency matrix.
    For unweighted graphs, 1 indicates an edge, 0 indicates no edge.
    For weighted graphs, the value is the edge weight.
    """
    def __init__(self, num_vertices: int, directed: bool = False):
        """
        Initializes the graph with a given number of vertices.
        """
        self.num_vertices = num_vertices
        self.directed = directed
        # Initialize the matrix with 0s (no edges)
        self.adj_matrix = [[0] * num_vertices for _ in range(num_vertices)]

    def add_edge(self, v1: int, v2: int, weight: int = 1):
        """
        Adds an edge between v1 and v2.
        Time: O(1)
        """
        if 0 <= v1 < self.num_vertices and 0 <= v2 < self.num_vertices:
            self.adj_matrix[v1][v2] = weight
            if not self.directed:
                self.adj_matrix[v2][v1] = weight # For undirected graphs

    def remove_edge(self, v1: int, v2: int):
        """
        Removes an edge between v1 and v2.
        Time: O(1)
        """
        if 0 <= v1 < self.num_vertices and 0 <= v2 < self.num_vertices:
            self.adj_matrix[v1][v2] = 0
            if not self.directed:
                self.adj_matrix[v2][v1] = 0

    def get_neighbors(self, vertex: int) -> list[int]:
        """
        Gets all neighbors of a given vertex.
        Time: O(V) - Must scan the entire row.
        """
        neighbors = []
        if 0 <= vertex < self.num_vertices:
            for i in range(self.num_vertices):
                if self.adj_matrix[vertex][i] > 0:
                    neighbors.append(i)
        return neighbors

    def __str__(self) -> str:
        """A simple string representation of the adjacency matrix."""
        return '\n'.join(' '.join(map(str, row)) for row in self.adj_matrix)

# =========================================
# --- Graph Traversal Algorithms
# =========================================

def bfs(graph: Graph, start_vertex: int) -> list[int]:
    """
    Performs a Breadth-First Search on the graph.
    Returns the list of visited vertices in order.

    Time: O(V^2) - The get_neighbors call inside the loop is O(V),
                   and the loop runs V times in a connected graph.
    Space: O(V) - For the visited set and the queue.
    """
    visited = [False] * graph.num_vertices
    q = deque([start_vertex])
    result = []
    
    visited[start_vertex] = True
    
    while q:
        vertex = q.popleft()
        result.append(vertex)
        
        for neighbor in graph.get_neighbors(vertex):
            if not visited[neighbor]:
                visited[neighbor] = True
                q.append(neighbor)
    return result

def dfs(graph: Graph, start_vertex: int) -> list[int]:
    """
    Performs a Depth-First Search on the graph.
    Returns the list of visited vertices in order.

    Time: O(V^2) - Same reasoning as BFS.
    Space: O(V) - For the visited set and recursion stack.
    """
    visited = [False] * graph.num_vertices
    result = []

    def _dfs_recursive(vertex):
        visited[vertex] = True
        result.append(vertex)
        for neighbor in graph.get_neighbors(vertex):
            if not visited[neighbor]:
                _dfs_recursive(neighbor)
    
    _dfs_recursive(start_vertex)
    return result

# =========================================
# --- Common Graph Problems
# =========================================

def shortest_path_unweighted(graph: Graph, start: int, end: int) -> list[int]:
    """
    Finds the shortest path in an unweighted graph using BFS.
    Returns the path as a list of vertices, or an empty list if no path exists.

    Time: O(V^2)
    Space: O(V)
    """
    if start == end:
        return [start]
        
    q = deque([(start, [start])]) # (current_vertex, current_path)
    visited = {start}

    while q:
        vertex, path = q.popleft()
        for neighbor in graph.get_neighbors(vertex):
            if neighbor == end:
                return path + [end]
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                q.append((neighbor, new_path))
    return [] # No path found