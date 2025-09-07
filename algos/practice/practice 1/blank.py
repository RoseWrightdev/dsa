from typing import List, Any, Optional, Dict, Tuple, Deque, Set
import collections
import bisect
import math
import heapq
import unittest

# ======================================================================
# ------------------------- TREES --------------------------------------
# ======================================================================


class Node:
    """A node in a binary tree."""

    def __init__(
        self, val: Any, left: "Optional[Node]" = None, right: "Optional[Node]" = None
    ):
        self.val = val
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.val})"


def make_perfect_tree(height: int, value: int = 1) -> Optional[Node]:
    if height <= 0:
        return None
    root = Node(value)
    root.left = make_perfect_tree(height - 1, value * 2)
    root.right = make_perfect_tree(height - 1, value * 2 + 1)
    return root


def print_tree(root: Optional[Node], prefix: str = "", is_left: bool = True): ...


def serialize(root: Optional[Node]) -> List[int]:
    result = []

    def dfs(node: Optional[Node]):
        if node is None:
            result.append(-1)
            return
        result.append(node.val)
        dfs(node.left)
        dfs(node.right)

    dfs(root)
    return result


def deserialize(data: List[int]) -> Optional[Node]: ...


def bfs_level_order(root: Optional[Node]) -> Optional[List[List[Any]]]:
    if root is None:
        return None
    q = collections.deque()
    q.append([root])
    res = []
    while q:
        level_size = len(q)
        curr_level = []
        for _ in range(level_size):
            node = q.popleft()
            curr_level.append(node.val)
            if node.left is not None:
                q.append(node.left)
            if node.right is not None:
                q.append(node.right)
        res.append(curr_level)
    return res


def dfs_preorder(root: Optional[Node]) -> List[Any]:
    res = []

    def dfs(node: Optional[Node]):
        if not node:
            return
        res.append(node)
        dfs(node.left)
        dfs(node.right)

    dfs(root)
    return res


def dfs_inorder(root: Optional[Node]) -> List[Any]:
    res = []

    def dfs(node: Optional[Node]) -> None:
        nonlocal res
        if not node:
            return None
        dfs(node.left)
        res.append(node)
        dfs(node.right)

    dfs(root)
    return res


def dfs_postorder(root: Optional[Node]) -> List[Any]:
    res = []

    def dfs(node: Optional[Node]) -> None:
        nonlocal res
        if not node:
            return None
        dfs(node.left)
        dfs(node.right)
        res.append(node)

    dfs(root)
    return res


def right_view(root: Optional[Node]) -> Optional[List[Any]]:
    if root is None:
        return None
    q = collections.deque()
    q.append([root])
    res = []
    while q:
        level_len = len(q)
        curr = []
        for i in range(level_len):
            node = q.popleft()
            if i == level_len - 1:
                curr.append(node)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(curr)
    return res


def left_view(root: Optional[Node]) -> Optional[List[Any]]:
    if root is None:
        return None
    q = collections.deque()
    q.append(root)
    res = []
    while q:
        q_len = len(q)
        curr = []
        for i in range(q_len):
            node = q.popleft()
            if i == 0:
                curr.append(node)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(curr)
    return res


def top_view(root: Optional[Node]) -> Optional[List[Any]]:
    if root is None:
        return None
    hd_map = {}
    q = collections.deque((root, 0))
    min_hd, max_hd = 0, 0
    while q:
        node, hd = q.popleft()
        if hd not in hd_map:
            hd_map[hd] = node.val
        max_hd = max(hd, max_hd)
        min_hd = min(hd, min_hd)
        if node.left:
            q.append((node.left, hd - 1))
        if node.right:
            q.append((node.right, hd + 1))
    return [hd_map[i] for i in range(min_hd, max_hd + 1)]


def top_view_rec(
    root: Optional[Node], hd_map: Dict[int, Any], hd: int, min_max: tuple[int, int]
) -> None:
    if root is None:
        return
    if hd not in hd_map:
        hd_map[hd] = root.val
        min_max = min(min_max[0], hd), max(min_max[1], hd)
    top_view_rec(root.left, hd_map, hd - 1, min_max)
    top_view_rec(root.right, hd_map, hd + 1, min_max)


def top_view_recursive(root: Optional[Node]) -> Optional[List[Any]]:
    if root is None:
        return None
    hd_map = {}
    min_max = [0, 0]
    top_view_rec(root, hd_map, 0, min_max)
    return [hd_map[i] for i in range(min_max[0], min_max[1] + 1)]


def bottom_view(root: Optional[Node]) -> Optional[List[Any]]:
    if root is None:
        return None
    q = collections.deque((root, 0))
    hd_map = {}
    min_hd, max_hd = 0, 0
    while q:
        node, hd = q.popleft()
        hd_map[hd] = node.val
        max_hd = max(max_hd, hd)
        min_hd = min(min_hd, hd)
        if node.left:
            q.append((node.val, hd - 1))
        if node.right:
            q.append((node.val, hd + 1))
    return [hd_map[i] for i in range(min_hd, max_hd + 1)]


def height(root: Optional[Node]) -> int:
    if root is None:
        return 0
    return max(height(root.left), height(root.right)) + 1


def is_same_tree(p: Optional[Node], q: Optional[Node]) -> bool:
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)


def is_symmetrical(root: Optional[Node]) -> bool:
    """Checks if a tree is a mirror of itself."""
    if not root:
        return True

    def is_mirror(t1: Optional[Node], t2: Optional[Node]) -> bool:
        if not t1 and not t2:
            return True
        if not t1 or not t2 or t1.val != t2.val:
            return False
        return is_mirror(t1.right, t2.left) and is_mirror(t1.left, t2.right)

    return is_mirror(root.left, root.right)


def is_subtree(root: Optional[Node], subRoot: Optional[Node]) -> bool:
    """
    Checks if subRoot is a subtree of root.
    Time: O(M*N) in the worst case, where M and N are the number of nodes.
    Space: O(H) for recursion stack.
    """
    if not root:
        return False
    if is_same_tree(root, subRoot):
        return True
    return is_subtree(root.left, subRoot) or is_subtree(root.right, subRoot)


def invert_tree(root: Optional[Node]) -> Optional[Node]:
    if not root:
        return None
    invert_tree(root.left)
    invert_tree(root.right)
    root.left, root.right = root.right, root.left


def lowest_common_ancestor(
    root: Optional[Node], p: "Node", q: "Node"
) -> Optional[Node]: ...


def diameter_of_binary_tree(root: Optional[Node]) -> int:
    diameter = 0

    def depth(node):
        nonlocal diameter
        if not node:
            return 0
        left_depth = depth(node.left)
        right_depth = depth(node.right)
        diameter = max(diameter, left_depth + right_depth)

        return 1 + max(left_depth, right_depth)

    depth(root)
    return diameter


def has_path_sum(root: Optional[Node], targetSum: int) -> bool:
    if not root:
        return False
    if not root.left and not root.right:
        return targetSum == root.val
    remaining_sum = targetSum - root.val
    return has_path_sum(root.left, remaining_sum) or has_path_sum(
        root.right, remaining_sum
    )


# ======================================================================
# ------------------------- BINARY SEARCH TREES ------------------------
# ======================================================================


class TreeNode:
    """A node in a Binary Search Tree."""

    def __init__(self, key: Any):
        self.key = key
        self.left: Optional["TreeNode"]
        self.right: Optional["TreeNode"]


def insert(root: Optional[TreeNode], key: Any) -> TreeNode:
    if root is None:
        return TreeNode(key)
    if key < root.left:
        root.left = insert(root.left, key)
    elif key > root.right:
        root.right = insert(root.right, key)
    return root


def is_valid_bst(root: Optional[Node]) -> bool: ...


def search(root: Optional[TreeNode], key: Any) -> Optional[TreeNode]: ...


def min_value_node(node: TreeNode) -> TreeNode: ...


def delete(root: Optional[TreeNode], key: Any) -> Optional[TreeNode]: ...


def inorder_traversal(root: Optional[TreeNode]) -> List[Any]: ...


def max_bst(root: Optional[TreeNode]) -> Optional[Any]: ...


def floor(root: Optional[TreeNode], key: Any) -> Optional[TreeNode]: ...


def ceil(root: Optional[TreeNode], key: Any) -> Optional[TreeNode]: ...


def inorder_successor(root: Optional[TreeNode], p: TreeNode) -> Optional[TreeNode]: ...


def kth_smallest_element(root: Optional[TreeNode], k: int) -> Optional[Any]: ...


# ========================================================================
# --------------------------- LINKED LISTS -------------------------------
# ========================================================================


class ListNode:
    """A node in a singly linked list."""

    def __init__(self, val: Any = 0, next: "Optional[ListNode]" = None):
        self.val = val
        self.next = next


class DoublyListNode:
    """A node in a doubly linked list."""

    def __init__(self, val: Any = 0):
        self.val = val
        self.next: Optional["DoublyListNode"] = None
        self.prev: Optional["DoublyListNode"] = None


def make_singly_linked_list(values: List[Any]) -> Optional[ListNode]:
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for value in values[1:]:
        current.next = ListNode(value)
        current = current.next
    return head


def print_linked_list(head: Optional[ListNode]) -> str:
    while head:
        print(head.val)
        head = head.next


def get_length(head: Optional[ListNode]) -> int:
    count = 0
    while head:
        count += 1
        head = head.next
    return count


def search_list(head: Optional[ListNode], key: Any) -> bool:
    while head:
        if head.val == key:
            return True
        head = head.next
    return False


def delete_node_by_key(head: Optional[ListNode], key: Any) -> Optional[ListNode]:
    if not head:
        return None
    dummy = ListNode(0, head)
    prev, curr = dummy, head

    while curr:
        if curr.val == key:
            prev.next = curr.next
            break
        prev = curr
        curr = curr.next

    return dummy.next


def has_cycle(head: Optional[ListNode]) -> bool:
    fast = slow = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]: ...


def find_middle_node(head: Optional[ListNode]) -> Optional[ListNode]:
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow


def merge_two_sorted_lists(
    l1: Optional[ListNode], l2: Optional[ListNode]
) -> Optional[ListNode]: ...


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    dummy = ListNode(0, head)

    fast, slow = dummy, dummy

    for _ in range(n + 1):
        fast = fast.next

    while fast:
        slow = slow.next
        fast = fast.next

    slow.next = slow.next.next

    return dummy.next


def reorder_list(head: Optional[ListNode]) -> None: ...


class RandomListNode:
    """A node with an additional 'random' pointer."""

    def __init__(
        self, x: int, next: "RandomListNode" = None, random: "RandomListNode" = None
    ): ...


def copy_random_list(
    head: "Optional[RandomListNode]",
) -> "Optional[RandomListNode]": ...


# ======================================================================
# ------------------------- STACKS -------------------------------------
# ======================================================================


class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item: Any) -> None:
        self.stack.append(item)

    def pop(self) -> Any:
        return self.stack.pop()

    def peek(self) -> Any:
        return self.stack[0]

    def is_empty(self) -> bool:
        return len(self.stack) == 0

    def size(self) -> int:
        return len(self.stack)


def is_valid_parentheses(s: str) -> bool:
    stack = []
    for c in s:
        if c == "(":
            stack.append(c)
        elif c == ")":
            if not stack:
                return False
            else:
                stack.pop()
    return len(stack) == 0


def evaluate_reverse_polish_notation(tokens: List[str]) -> int:
    opps = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: int(x / y),
    }
    stk = []
    for token in tokens:
        if token not in opps:
            stk.append(int(token))
        else:
            y = stk.pop()
            x = stk.pop()
            result = opps[token](x, y)
            stk.append(result)
    return stk[0]


def generate_parentheses(n: int) -> List[str]: ...


class MinStack:
    def __init__(self):
        self.stk = []
        self.min_stk = []

    def push(self, val: int) -> None:
        self.stk.append(val)
        if val < self.min_stk[-1]:
            self.min_stk.append(val)

    def pop(self) -> None:
        self.stk.pop()
        self.min_stk.pop()

    def top(self) -> int:
        return self.stk[-1]

    def getMin(self) -> int:
        return self.min_stk[-1]


# ======================================================================
# ------------------------- QUEUES -------------------------------------
# ======================================================================


class QueueWithLinkedList:
    def __init__(self, max_len: int):
        self.head = None  # front of queue
        self.tail = None  # back of queue
        self._size = 0
        self._max_len = max_len

    def enqueue(self, item: Any) -> None:
        node = ListNode(item)
        if not self.tail:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        if self._size >= self._max_len:
            self.dequeue()
        else:
            self._size += 1

    def dequeue(self) -> Any:
        if not self.head:
            return None
        val = self.head.val
        self.head = self.head.next
        if not self.head:
            self.tail = None
        self._size -= 1
        return val

    def peek(self) -> Any:
        if not self.head:
            return None
        return self.head.val

    def is_empty(self) -> bool:
        return self.head is None

    def size(self) -> int:
        return self._size


class Queue:
    def __init__(self): ...
    def enqueue(self, item: Any) -> None: ...
    def dequeue(self) -> Any: ...
    def peek(self) -> Any: ...
    def is_empty(self) -> bool: ...
    def size(self) -> int: ...


class MovingAverage:
    def __init__(self, size: int):
        self.size = size
        self.q = collections.deque()
        self.sum = 0

    def next(self, val: int) -> float:
        self.q.appendleft(val)
        self.sum += val
        if len(self.q) > self.size:
            removed_val = self.q.pop()
            self.sum -= removed_val
        return self.sum / len(self.q)


class MyCircularQueue:
    def __init__(self, k: int): ...
    def enQueue(self, value: int) -> bool: ...
    def deQueue(self) -> bool: ...
    def Front(self) -> int: ...
    def Rear(self) -> int: ...
    def isEmpty(self) -> bool: ...
    def isFull(self) -> bool: ...


# ======================================================================
# ------------------------- HEAPS --------------------------------------
# ======================================================================


class MinHeap:
    def __init__(self):
        self.heap = []

    def _parent_index(self, index):
        return (index - 1) // 2

    def _left_child_index(self, index):
        return (index * 2) + 1

    def _right_child_index(self, index):
        return (index * 2) + 2

    def push(self, value: int) -> None: ...
    def pop(self) -> int: ...
    def peek(self) -> int: ...


class KthLargest:
    def __init__(self, k: int, nums: List[int]): ...
    def add(self, val: int) -> int: ...


class MedianFinder:
    def __init__(self): ...
    def addNum(self, num: int) -> None: ...
    def findMedian(self) -> float: ...


def merge_k_sorted_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]: ...


def top_k_frequent_elements(nums: List[int], k: int) -> List[int]: ...


def task_scheduler(tasks: List[str], n: int) -> int: ...


def k_smallest_pairs(nums1: List[int], nums2: List[int], k: int) -> List[List[int]]: ...


# ======================================================================
# ------------------------- GRAPHS -------------------------------------
# ======================================================================


class GraphNode:
    def __init__(self, val: Any): ...


class Graph:  # Adjacency List
    def __init__(self, directed: bool = False): ...
    def add_edge(self, u: Any, v: Any, weight: int = 1) -> None: ...
    def get_nodes(self) -> List[Any]: ...
    def __str__(self) -> str: ...


class GraphAdjMatrix:
    def __init__(self, num_vertices: int, directed: bool = False): ...
    def add_edge(self, v1: int, v2: int, weight: int = 1): ...
    def remove_edge(self, v1: int, v2: int): ...
    def get_neighbors(self, vertex: int) -> list[int]: ...
    def __str__(self) -> str: ...


def bfs(graph: Graph, start_node: Any) -> List[Any]: ...


def dfs(graph: Graph, start_node: Any) -> List[Any]: ...


def is_cycle_undirected(graph: Graph) -> bool: ...


def is_cycle_directed(graph: Graph) -> bool: ...


def dijkstra_shortest_path(graph: Graph, start_node: Any) -> Dict[Any, float]: ...


def shortest_path_unweighted(graph: Graph, start: Any, end: Any) -> List[Any]: ...


def prim_mst(graph: Graph) -> Tuple[int, List[Tuple[Any, Any, int]]]: ...


def topological_sort(graph: Graph) -> List[Any]: ...


class UnionFind:
    def __init__(self, nodes: List[Any]): ...
    def find(self, i: Any) -> Any: ...
    def union(self, i: Any, j: Any) -> None: ...


def number_of_islands(grid: List[List[str]]) -> int: ...


def rotten_tomatoes(grid: List[List[int]]) -> int: ...


def flood_fill(
    image: List[List[int]], sr: int, sc: int, new_color: int
) -> List[List[int]]: ...


def visualize_graph_turtle(graph: Graph) -> None: ...


# ======================================================================
# ------------------------- DYNAMIC PROGRAMMING ------------------------
# ======================================================================


def fibonacci_memoization(n: int) -> int: ...


def fibonacci_tabulation(n: int) -> int: ...


def climbing_stairs(n: int) -> int: ...


def longest_common_subsequence(text1: str, text2: str) -> int: ...


def coin_change(coins: List[int], amount: int) -> int: ...


def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int: ...


# ======================================================================
# ------------------------- SORTING ------------------------------------
# ======================================================================


def bubble_sort(arr: List[Any]) -> List[Any]:
    n = len(arr)
    for _ in range(n):
        for i in range(1, n):
            if arr[i - 1] > arr[i]:
                arr[i - 1], arr[i] = arr[i], arr[i - 1]
    return arr


def selection_sort(arr: List[Any]) -> List[Any]:
    n = len(arr)
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


def insertion_sort(arr: List[Any]) -> List[Any]:
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def merge_sort(arr: List[Any]) -> List[Any]:
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]

        merge_sort(left)
        merge_sort(right)

        i = j = k = 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1
    return arr

def quick_sort(arr: List[Any]) -> List[Any]: ...


def heap_sort(arr: List[Any]) -> List[Any]: ...


def counting_sort(arr: List[int]) -> List[int]: ...


# ======================================================================
# ------------------------- MATRIX -------------------------------------
# ======================================================================


def dfs_traversal(matrix: List[List[int]]) -> List[int]:
    if not matrix:
        return []
    rows, cols = len(matrix), len(matrix[0])
    visited = set()
    result = []

    def dfs(r: int, c: int):
        if not (0 <= r < rows and 0 <= c < cols) or (r, c) in visited:
            return
        visited.add((r, c))
        result.append(matrix[r][c])
        dfs(r + 1, c)
        dfs(r, c + 1)
        dfs(r - 1, c)
        dfs(r, c - 1)
    dfs(0, 0)

    return result


def bfs_traversal(matrix: List[List[int]]) -> List[int]:
    if not matrix:
        return []
    rows, cols = len(matrix), len(matrix[0])
    q = collections.deque([(0, 0)])
    visited = set()
    visited.add((0, 0))
    result = []
    while q:
        r, c = q.popleft()
        result.append(matrix[r][c])
        for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return result


def spiral_traversal(matrix: List[List[int]]) -> List[int]:
    if not matrix:
        return []
    
    result = []
    rows, cols = len(matrix), len(matrix[0])
    # bounds
    top, bottom, left, right = 0, rows - 1, 0, cols - 1

    while top <= bottom and left <= right:
        # traverse right
        for c in range(top, bottom + 1):
            result.append(matrix[top][c])
        top += 1

        # traverse down
        for r in range(left, right + 1):
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


def rotate_degs(matrix: List[List[int]], degs, dir: Optional[bool] = False) -> None:
    if not matrix or not matrix[0]:
        return
    
    rots = degs % 360
    rows, cols = len(matrix), len(matrix[0])

    def rotation(invert, reverse):
        if invert:
            for r in range(rows):
                for c in range(cols):
                    matrix[r][c] = matrix[r][c]
        if reverse:
            for i in range(rows):
                matrix[i].reverse()

    if rots == 1:
        if not dir:
            rotation(True, True)
        else: 
            rotation(True, False)
    elif rots == 2:
        rotation(False, True)
    elif rots == 3:
        if not dir:
            rotation(True, False)
        else: 
            rotation(True, False)

class TestRotateDegs(unittest.TestCase):
    matrix = [[k for k in range(3 * i)] for i in range(1, 3 + 1)]

    def ninity(self):



def set_matrix_zeroes(matrix: List[List[int]]) -> None: ...


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


def number_of_islands():...


def rotten_tomatoes():...


# ======================================================================
# ------------------------- SLIDING WINDOW -----------------------------
# ======================================================================


def sliding_window_template(): ...


def max_subarr_of_size_k(arr: List[int], k: int) -> int: ...


def smallest_subarray_with_given_sum(arr: List[int], target_sum: int) -> int: ...


def find_string_anagrams(s: str, pattern: str) -> List[int]: ...


# ======================================================================
# ------------------------- PREFIX SUM ---------------------------------
# ======================================================================


def prefix_sum_template(): ...


def subarray_sum_equals_k(nums: List[int], k: int) -> int: ...


class NumArray:
    def __init__(self, nums: List[int]): ...
    def sumRange(self, left: int, right: int) -> int: ...


def find_pivot_index(nums: List[int]) -> int: ...


# ======================================================================
# ------------------------- MONOTONIC STACK ----------------------------
# ======================================================================


def monotonic_stack_template(): ...


def next_greater_element_i(nums1: List[int], nums2: List[int]) -> List[int]: ...


def daily_temperatures(temperatures: List[int]) -> List[int]: ...


def remove_k_digits(num: str, k: int) -> str: ...


# ======================================================================
# ------------------------- ORDERED SET / FENWICK TREE -----------------
# ======================================================================


def simulate_ordered_set_with_bisect(): ...


class FenwickTree:
    def __init__(self, size: int): ...
    def update(self, index: int, delta: int) -> None: ...
    def query(self, index: int) -> int: ...


def count_smaller_numbers_after_self(nums: List[int]) -> List[int]: ...


def contains_nearby_almost_duplicate(nums: List[int], k: int, t: int) -> bool: ...


# ======================================================================
# ------------------------- GREEDY ALGORITHMS --------------------------
# ======================================================================


def greedy_algorithm_template(): ...


def max_subarray(nums: List[int]) -> int: ...


def can_jump(nums: List[int]) -> bool: ...


def merge_intervals(intervals: List[List[int]]) -> List[List[int]]: ...


def can_complete_circuit(gas: List[int], cost: List[int]) -> int: ...


# ======================================================================
# ------------------------- HASHMAPS -----------------------------------
# ======================================================================


class HashTable:
    def __init__(self, size: int = 100): ...
    def set(self, key: Any, value: Any) -> None: ...
    def get(self, key: Any) -> Optional[Any]: ...
    def delete(self, key: Any) -> None: ...
    def __str__(self) -> str: ...


def two_sum(nums: List[int], target: int) -> List[int]: ...


def is_valid_anagram(s: str, t: str) -> bool: ...


def group_anagrams(strs: List[str]) -> List[List[str]]: ...


def first_unique_char(s: str) -> int: ...


class LoggerRateLimiter:
    def __init__(self): ...
    def should_print_message(self, timestamp: int, message: str) -> bool: ...


class LRUCache:
    def __init__(self, capacity: int): ...
    def get(self, key: int) -> int: ...
    def put(self, key: int, value: int) -> None: ...


# ======================================================================
# ------------------------- BINARY SEARCH ------------------------------
# ======================================================================


def binary_search_template(): ...


def binary_search_exact(arr: List[Any], target: Any) -> int: ...


def find_first_occurrence(arr: List[Any], target: Any) -> int: ...


def find_last_occurrence(arr: List[Any], target: Any) -> int: ...


def search_in_rotated_sorted_array(nums: List[int], target: int) -> int: ...


def find_min_in_rotated_sorted_array(nums: List[int]) -> int: ...


def find_peak_element(nums: List[int]) -> int: ...


def search_a_2d_matrix_ii(matrix: List[List[int]], target: int) -> bool: ...


# ======================================================================
# ------------------------- BITMASKING ---------------------------------
# ======================================================================


def bitmasking_template(): ...


def get_bit(n: int, i: int) -> int: ...


def set_bit(n: int, i: int) -> int: ...


def clear_bit(n: int, i: int) -> int: ...


def generate_all_subsets(nums: List[Any]) -> List[List[Any]]: ...


def count_set_bits(n: int) -> int: ...


def single_number(nums: List[int]) -> int: ...


def is_power_of_two(n: int) -> bool: ...


# ======================================================================
# ------------------------- DIVIDE AND CONQUER -------------------------
# ======================================================================


def divide_and_conquer_template(): ...


def max_subarray_sum(arr: List[int]) -> int: ...


# ======================================================================
# ------------------------- BACKTRACKING -------------------------------
# ======================================================================


def backtracking_template(): ...


def permutations(nums: List[int]) -> List[List[int]]: ...


def subsets_backtracking(nums: List[int]) -> List[List[int]]: ...


def combination_sum(candidates: List[int], target: int) -> List[List[int]]: ...


# ======================================================================
# ------------------------- TRIES --------------------------------------
# ======================================================================


class TrieNode:
    def __init__(self): ...


class Trie:
    def __init__(self): ...
    def insert(self, word: str) -> None: ...
    def search(self, word: str) -> bool: ...
    def startsWith(self, prefix: str) -> bool: ...
