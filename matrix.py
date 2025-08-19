from typing import List
from collections import deque

# =========================================
# --- Matrix Traversal
# =========================================

def dfs_traversal(matrix: List[List[int]]) -> List[int]:
    """
    Traverses a matrix using Depth-First Search (DFS) recursively.
    Returns a list of elements in the order they were visited.

    Time: O(M*N) - Visits each cell once.
    Space: O(M*N) - For the visited set and recursion stack.
    """
    if not matrix:
        return []
    rows, cols = len(matrix), len(matrix[0])
    visited = set()
    result = []

    def dfs(r, c):
        if not (0 <= r < rows and 0 <= c < cols) or (r, c) in visited:
            return
        
        visited.add((r, c))
        result.append(matrix[r][c])
        
        # Explore neighbors (down, right, up, left)
        dfs(r + 1, c)
        dfs(r, c + 1)
        dfs(r - 1, c)
        dfs(r, c - 1)

    dfs(0, 0)
    return result

def bfs_traversal(matrix: List[List[int]]) -> List[int]:
    """
    Traverses a matrix using Breadth-First Search (BFS) iteratively.
    Returns a list of elements in the order they were visited.

    Time: O(M*N) - Visits each cell once.
    Space: O(M*N) - For the visited set and the queue.
    """
    if not matrix:
        return []
    rows, cols = len(matrix), len(matrix[0])
    visited = set()
    q = deque([(0, 0)])
    visited.add((0, 0))
    result = []

    while q:
        r, c = q.popleft()
        result.append(matrix[r][c])
        
        # Explore neighbors
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return result

def spiral_traversal(matrix: List[List[int]]) -> List[int]:
    """
    Traverses a matrix in a spiral order and returns the elements.

    Time: O(M*N) - Visits each cell once.
    Space: O(M*N) - For the result list. O(1) if not storing the result.
    """
    if not matrix:
        return []
    
    result = []
    rows, cols = len(matrix), len(matrix[0])
    top, bottom, left, right = 0, rows - 1, 0, cols - 1

    while top <= bottom and left <= right:
        # Traverse Right
        for c in range(left, right + 1):
            result.append(matrix[top][c])
        top += 1
        
        # Traverse Down
        for r in range(top, bottom + 1):
            result.append(matrix[r][right])
        right -= 1

        if top <= bottom:
            # Traverse Left
            for c in range(right, left - 1, -1):
                result.append(matrix[bottom][c])
            bottom -= 1

        if left <= right:
            # Traverse Up
            for r in range(bottom, top - 1, -1):
                result.append(matrix[r][left])
            left += 1
            
    return result

# =========================================
# --- Matrix Manipulation
# =========================================

def rotate_90_degrees_clockwise(matrix: List[List[int]]) -> None:
    """
    Rotates a square (N x N) matrix 90 degrees clockwise in-place.
    The rotation is achieved by first transposing the matrix and then
    reversing each row.

    Time: O(M*N) - Visits each cell a constant number of times.
    Space: O(1) - The rotation is done in-place.
    """
    n = len(matrix)
    
    # 1. Transpose the matrix (rows become columns and vice-versa)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
            
    # 2. Reverse each row
    for i in range(n):
        matrix[i].reverse()
    
    # 

def set_matrix_zeroes(matrix: List[List[int]]) -> None:
    """
    If an element in an M x N matrix is 0, its entire row and
    column are set to 0. This is done in-place.

    Time: O(M*N)
    Space: O(1) - Uses the first row/col for storage instead of a new set.
    """
    rows, cols = len(matrix), len(matrix[0])
    first_row_has_zero = any(matrix[0][c] == 0 for c in range(cols))
    first_col_has_zero = any(matrix[r][0] == 0 for r in range(rows))

    # Use first row and col to mark zeroes
    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][c] == 0:
                matrix[0][c] = 0
                matrix[r][0] = 0

    # Zero out cells based on marks in the first row and col
    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][0] == 0 or matrix[0][c] == 0:
                matrix[r][c] = 0

    # Zero out the first row and col if needed
    if first_row_has_zero:
        for c in range(cols):
            matrix[0][c] = 0
    if first_col_has_zero:
        for r in range(rows):
            matrix[r][0] = 0

# =========================================
# --- Searching in a Matrix
# =========================================

def search_in_2d_matrix(matrix: List[List[int]], target: int) -> bool:
    """
    Searches for a value in an M x N matrix where:
    - Integers in each row are sorted from left to right.
    - The first integer of each row is greater than the last of the previous row.

    This structure allows for an efficient binary search.

    Time: O(log(M*N)) - Standard binary search on the conceptual 1D array.
    Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False
        
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1
    
    while left <= right:
        mid_idx = (left + right) // 2
        mid_val = matrix[mid_idx // cols][mid_idx % cols]
        
        if mid_val == target:
            return True
        elif mid_val < target:
            left = mid_idx + 1
        else:
            right = mid_idx - 1
            
    return False