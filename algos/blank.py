from typing import List, Any, Optional, Dict, Tuple, Deque, Set
import collections
import bisect
import math
import heapq

# ======================================================================
# ------------------------- SORTING ------------------------------------
# ======================================================================


def sorting_template():
    """
    Use Cases:
    - Fundamental for many other algorithms (e.g., binary search, two pointers).
    - Organizing data for human readability or efficient processing.

    Pros & Cons:
    - O(n^2) sorts (Bubble, Selection, Insertion): Simple to implement, good for small
      or nearly sorted datasets. Inefficient for large datasets.
    - O(n log n) sorts (Merge, Quick, Heap): Efficient and scalable. Merge sort has
      stable sorting but requires O(n) space. Quick sort is often fastest in practice
      but has a worst-case O(n^2). Heap sort is in-place but not stable.
    - O(n+k) sorts (Counting): Extremely fast but only works for integers in a
      limited range.
    """
    ...


def bubble_sort(arr: List[int]) -> List[int]: ... 
    

def selection_sort(arr: List[Any]) -> List[Any]: ...


def insertion_sort(arr: List[Any]) -> List[Any]: ...


def merge_sort(arr: List[Any]) -> List[Any]: ...


def quick_sort(arr: List[Any]) -> List[Any]: ...


def heap_sort(arr: List[Any]) -> List[Any]: ...


def counting_sort(arr: List[int]) -> List[int]: ...


# ======================================================================
# ------------------------- HASHMAPS -----------------------------------
# ======================================================================


def hashmap_template():
    """
    Use Cases:
    - O(1) average time complexity for lookups, insertions, and deletions.
    - Counting frequencies of items.
    - Caching results (memoization).
    - Storing key-value pairs.

    Pros & Cons:
    - Pros: Extremely fast for lookups. Flexible key types.
    - Cons: Unordered (in older Python versions). High memory usage. Worst-case
      time complexity can be O(n) if many hash collisions occur.
    """
    ...


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
# ------------------------- STACKS -------------------------------------
# ======================================================================


def stack_template():
    """
    Principle: Last-In, First-Out (LIFO).
    Use Cases:
    - Parsing expressions (e.g., parentheses matching, RPN).
    - Simulating recursion iteratively (e.g., DFS).
    - Backtracking problems.
    - Monotonic stack problems.
    """
    ...


class Stack:
    def __init__(self): ...
    def push(self, item: Any) -> None: ...
    def pop(self) -> Any: ...
    def peek(self) -> Any: ...
    def is_empty(self) -> bool: ...
    def size(self) -> int: ...


def is_valid_parentheses(s: str) -> bool: ...


def evaluate_reverse_polish_notation(tokens: List[str]) -> int: ...


def generate_parentheses(n: int) -> List[str]: ...


class MinStack:
    def __init__(self): ...
    def push(self, val: int) -> None: ...
    def pop(self) -> None: ...
    def top(self) -> int: ...
    def getMin(self) -> int: ...


# ======================================================================
# ------------------------- QUEUES -------------------------------------
# ======================================================================


def queue_template():
    """
    Principle: First-In, First-Out (FIFO).
    Use Cases:
    - Breadth-First Search (BFS) in graphs and trees.
    - Managing tasks or requests in order (e.g., printer queue).
    - Rate limiting and sliding window problems.
    """
    ...


class QueueWithList:
    def __init__(self): ...
    def enqueue(self, item: Any) -> None: ...
    def dequeue(self) -> Any: ...
    def peek(self) -> Any: ...
    def is_empty(self) -> bool: ...
    def size(self) -> int: ...


class Queue:
    def __init__(self): ...
    def enqueue(self, item: Any) -> None: ...
    def dequeue(self) -> Any: ...
    def peek(self) -> Any: ...
    def is_empty(self) -> bool: ...
    def size(self) -> int: ...


class MovingAverage:
    def __init__(self, size: int): ...
    def next(self, val: int) -> float: ...


class MyCircularQueue:
    def __init__(self, k: int): ...
    def enQueue(self, value: int) -> bool: ...
    def deQueue(self) -> bool: ...
    def Front(self) -> int: ...
    def Rear(self) -> int: ...
    def isEmpty(self) -> bool: ...
    def isFull(self) -> bool: ...


# ======================================================================
# ------------------------- LINKED LISTS -------------------------------
# ======================================================================


def linked_list_template():
    """
    Principle: A sequence of nodes where each node points to the next.
    Use Cases:
    - Implementing other data structures like stacks and queues.
    - Situations requiring frequent insertions/deletions at unknown positions.

    Pros & Cons:
    - Pros: Dynamic size, efficient insertions/deletions.
    - Cons: No random access (O(n) to find an element). Higher memory overhead
      than arrays due to storing pointers.
    """
    ...


class ListNode:
    """A node in a singly linked list."""

    def __init__(self, val: Any = 0, next: "Optional[ListNode]" = None): ...


class DoublyListNode:
    """A node in a doubly linked list."""

    def __init__(self, val: Any = 0): ...


def make_singly_linked_list(values: List[Any]) -> Optional[ListNode]: ...


def print_linked_list(head: Optional[ListNode]) -> str: ...


def get_length(head: Optional[ListNode]) -> int: ...


def search_list(head: Optional[ListNode], key: Any) -> bool: ...


def delete_node_by_key(head: Optional[ListNode], key: Any) -> Optional[ListNode]: ...


def has_cycle(head: Optional[ListNode]) -> bool: ...


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]: ...


def find_middle_node(head: Optional[ListNode]) -> Optional[ListNode]: ...


def merge_two_sorted_lists(
    l1: Optional[ListNode], l2: Optional[ListNode]
) -> Optional[ListNode]: ...


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]: ...


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
# ------------------------- TWO POINTERS -------------------------------
# ======================================================================


def two_pointers_template():
    """
    Principle: Use two pointers to iterate through a data structure until
    they meet or satisfy a condition.
    Use Cases:
    - Finding pairs in a sorted array.
    - Reversing arrays or linked lists.
    - Palindrome checks.
    - Problems involving finding a subarray or subsequence.
    """
    ...


def two_sum_ii_sorted_input(numbers: List[int], target: int) -> List[int]: ...


def three_sum(nums: List[int]) -> List[List[int]]: ...


def container_with_most_water(height: List[int]) -> int: ...


def is_palindrome(s: str) -> bool: ...


def sort_colors(nums: List[int]) -> None: ...


# ======================================================================
# ------------------------- SLIDING WINDOW -----------------------------
# ======================================================================


def sliding_window_template():
    """
    Principle: Maintain a "window" (a subarray or substring) and slide it
    through the data to solve problems on contiguous parts of the data.
    Use Cases:
    - Finding min/max subarray of a fixed size.
    - Finding the smallest subarray that meets a certain condition.
    - String permutation/anagram problems.
    """
    ...


def max_subarr_of_size_k(arr: List[int], k: int) -> int: ...


def smallest_subarray_with_given_sum(arr: List[int], target_sum: int) -> int: ...


def find_string_anagrams(s: str, pattern: str) -> List[int]: ...


# ======================================================================
# ------------------------- BINARY SEARCH ------------------------------
# ======================================================================


def binary_search_template():
    """
    Principle: Efficiently search a sorted array by repeatedly dividing
    the search interval in half.
    Use Cases:
    - Finding an element in a sorted array.
    - Finding the first/last occurrence of an element.
    - Can be applied to the "answer space" for optimization problems
      (e.g., find the minimum speed to arrive on time).
    """
    ...


def binary_search_exact(arr: List[Any], target: Any) -> int: ...


def find_first_occurrence(arr: List[Any], target: Any) -> int: ...


def find_last_occurrence(arr: List[Any], target: Any) -> int: ...


def search_in_rotated_sorted_array(nums: List[int], target: int) -> int: ...


def find_min_in_rotated_sorted_array(nums: List[int]) -> int: ...


def find_peak_element(nums: List[int]) -> int: ...


def search_a_2d_matrix_ii(matrix: List[List[int]], target: int) -> bool: ...


# ======================================================================
# ------------------------- TREES --------------------------------------
# ======================================================================


def tree_template():
    """
    Principle: A hierarchical data structure with a root node and child nodes.
    Operations are often recursive.
    Use Cases:
    - Representing hierarchical data (file systems, organization charts).
    - Search trees for efficient searching (BSTs).
    - Heaps for priority queues.
    - Tries for string searching.
    """
    ...


class Node:
    """A node in a binary tree."""

    def __init__(
        self, val: Any, left: "Optional[Node]" = None, right: "Optional[Node]" = None
    ): ...

    def __str__(self) -> str: ...


def make_perfect_tree(height: int, value: int = 1) -> Optional[Node]: ...


def print_tree(root: Optional[Node], prefix: str = "", is_left: bool = True): ...


def serialize(root: Optional[Node]) -> List[int]: ...


def deserialize(data: List[int]) -> Optional[Node]: ...


def bfs_level_order(root: Optional[Node]) -> List[List[Any]]: ...


def dfs_preorder(root: Optional[Node]) -> List[Any]: ...


def dfs_inorder(root: Optional[Node]) -> List[Any]: ...


def dfs_postorder(root: Optional[Node]) -> List[Any]: ...


def right_view(root: Optional[Node]) -> List[Any]: ...


def left_view(root: Optional[Node]) -> List[Any]: ...


def top_view(root: Optional[Node]) -> List[Any]: ...


def bottom_view(root: Optional[Node]) -> List[Any]: ...


def height(root: Optional[Node]) -> int: ...


def is_same_tree(p: Optional[Node], q: Optional[Node]) -> bool: ...


def is_symmetrical(root: Optional[Node]) -> bool: ...


def is_valid_bst(root: Optional[Node]) -> bool: ...


def is_subtree(root: Optional[Node], subRoot: Optional[Node]) -> bool: ...


def invert_tree(root: Optional[Node]) -> Optional[Node]: ...


def lowest_common_ancestor(
    root: Optional[Node], p: "Node", q: "Node"
) -> Optional[Node]: ...


def diameter_of_binary_tree(root: Optional[Node]) -> int: ...


def has_path_sum(root: Optional[Node], targetSum: int) -> bool: ...


# ======================================================================
# ------------------------- BINARY SEARCH TREES ------------------------
# ======================================================================


def bst_template():
    """
    Principle: A binary tree where for each node, all keys in the left
    subtree are less than the node's key, and all keys in the right subtree
    are greater.
    Use Cases:
    - Efficiently storing and searching for items in sorted order.
    - Implementing ordered sets and maps.
    Pros & Cons:
    - Pros: O(log n) average time for search, insert, delete. Inorder traversal
      yields sorted data.
    - Cons: Can become unbalanced, leading to O(n) worst-case time.
    """
    ...


class TreeNode:
    """A node in a Binary Search Tree."""

    def __init__(self, key: Any): ...


def insert(root: Optional[TreeNode], key: Any) -> TreeNode: ...


def search(root: Optional[TreeNode], key: Any) -> Optional[TreeNode]: ...


def min_value_node(node: TreeNode) -> TreeNode: ...


def delete(root: Optional[TreeNode], key: Any) -> Optional[TreeNode]: ...


def inorder_traversal(root: Optional[TreeNode]) -> List[Any]: ...


def max_bst(root: Optional[TreeNode]) -> Optional[Any]: ...


def floor(root: Optional[TreeNode], key: Any) -> Optional[TreeNode]: ...


def ceil(root: Optional[TreeNode], key: Any) -> Optional[TreeNode]: ...


def inorder_successor(root: Optional[TreeNode], p: TreeNode) -> Optional[TreeNode]: ...


def kth_smallest_element(root: Optional[TreeNode], k: int) -> Optional[Any]: ...


# ======================================================================
# ------------------------- HEAPS --------------------------------------
# ======================================================================


def heap_template():
    """
    Principle: A specialized tree-based data structure that satisfies the
    heap property (e.g., in a min-heap, the parent is always smaller than
    its children).
    Use Cases:
    - Implementing priority queues.
    - Finding the k-th smallest/largest element.
    - Heap sort.
    - Graph algorithms like Dijkstra's and Prim's.
    """
    ...


class MinHeap:
    def __init__(self): ...
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


def graph_template():
    """
    Principle: A collection of nodes (vertices) and edges that connect them.
    Can be directed/undirected, weighted/unweighted.
    Representations:
    - Adjacency List: A map from each node to a list of its neighbors.
      Efficient for sparse graphs.
    - Adjacency Matrix: A V x V matrix where matrix[i][j] indicates an edge.
      Efficient for dense graphs and O(1) edge lookups.
    """
    ...


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
# ------------------------- MATRIX -------------------------------------
# ======================================================================


def matrix_template():
    """
    Principle: A 2D array or grid.
    Use Cases:
    - Representing game boards, images, or maps.
    - Problems often involve traversing the grid (like a graph) or
      performing transformations like rotation and setting values.
    """
    ...


def dfs_traversal(matrix: List[List[int]]) -> List[int]: ...


def bfs_traversal(matrix: List[List[int]]) -> List[int]: ...


def spiral_traversal(matrix: List[List[int]]) -> List[int]: ...


def rotate_degs_clockwise(matrix: List[List[int]]) -> None: ...


def set_matrix_zeroes(matrix: List[List[int]]) -> None: ...


def search_in_2d_matrix(matrix: List[List[int]], target: int) -> bool: ...


# ======================================================================
# ------------------------- GREEDY ALGORITHMS --------------------------
# ======================================================================


def greedy_algorithm_template():
    """
    Principle: Make the locally optimal choice at each step with the hope
    of finding a global optimum.
    Use Cases:
    - Optimization problems where a simple, local choice leads to the best result.
    - Interval scheduling, shortest path (Dijkstra's), minimum spanning tree (Prim's).
    Pros & Cons:
    - Pros: Often simpler and faster than DP.
    - Cons: Does not work for all optimization problems. Proving correctness
      can be difficult.
    """
    ...


def max_subarray(nums: List[int]) -> int: ...


def can_jump(nums: List[int]) -> bool: ...


def merge_intervals(intervals: List[List[int]]) -> List[List[int]]: ...


def can_complete_circuit(gas: List[int], cost: List[int]) -> int: ...


# ======================================================================
# ------------------------- DIVIDE AND CONQUER -------------------------
# ======================================================================


def divide_and_conquer_template():
    """
    Principle: Break a problem into smaller subproblems, solve them
    recursively, and combine the results.
    Use Cases:
    - Sorting (Merge Sort, Quick Sort).
    - Searching (Binary Search).
    - Problems on trees and arrays that can be naturally split.
    """
    ...


def max_subarray_sum(arr: List[int]) -> int: ...


# ======================================================================
# ------------------------- DYNAMIC PROGRAMMING ------------------------
# ======================================================================


def dynamic_programming_template():
    """
    Principle: Solve complex problems by breaking them down into simpler,
    overlapping subproblems and storing the results of subproblems to avoid
    re-computation (memoization or tabulation).
    Use Cases:
    - Optimization problems (e.g., find min/max/longest/shortest).
    - Counting problems (e.g., find the number of ways to do something).
    - Problems with optimal substructure and overlapping subproblems.
    """
    ...


def fibonacci_memoization(n: int) -> int: ...


def fibonacci_tabulation(n: int) -> int: ...


def climbing_stairs(n: int) -> int: ...


def longest_common_subsequence(text1: str, text2: str) -> int: ...


def coin_change(coins: List[int], amount: int) -> int: ...


def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int: ...


# ======================================================================
# ------------------------- BACKTRACKING -------------------------------
# ======================================================================


def backtracking_template():
    """
    Principle: An algorithmic technique for solving problems recursively by
    trying to build a solution incrementally, one piece at a time, removing
    those solutions that fail to satisfy the constraints of the problem at
    any point in time (this is the "backtracking").
    Use Cases:
    - Generating all possible solutions (permutations, combinations, subsets).
    - Solving constraint satisfaction problems (Sudoku, N-Queens).
    """
    ...


def permutations(nums: List[int]) -> List[List[int]]: ...


def subsets_backtracking(nums: List[int]) -> List[List[int]]: ...


def combination_sum(candidates: List[int], target: int) -> List[List[int]]: ...


# ======================================================================
# ------------------------- TRIES --------------------------------------
# ======================================================================


def trie_template():
    """
    Principle: A tree-like data structure that stores a dynamic set of strings.
    Each node represents a character, and paths from the root represent prefixes/words.
    Use Cases:
    - Autocomplete and search suggestions.
    - Spell checkers.
    - IP routing.
    Pros & Cons:
    - Pros: Very fast prefix-based searches (O(L) where L is length of prefix).
    - Cons: Can be memory-intensive if storing many long strings with few
      common prefixes.
    """
    ...


class TrieNode:
    def __init__(self): ...


class Trie:
    def __init__(self): ...
    def insert(self, word: str) -> None: ...
    def search(self, word: str) -> bool: ...
    def startsWith(self, prefix: str) -> bool: ...


# ======================================================================
# ------------------------- PREFIX SUM ---------------------------------
# ======================================================================


def prefix_sum_template():
    """
    Principle: Pre-calculate sums of prefixes of an array to answer
    range sum queries in O(1) time.
    Use Cases:
    - Range sum queries.
    - Finding subarrays with a specific sum.
    - Problems where the difference between two prefix sums is meaningful.
    """
    ...


def subarray_sum_equals_k(nums: List[int], k: int) -> int: ...


class NumArray:
    def __init__(self, nums: List[int]): ...
    def sumRange(self, left: int, right: int) -> int: ...


def find_pivot_index(nums: List[int]) -> int: ...


# ======================================================================
# ------------------------- MONOTONIC STACK ----------------------------
# ======================================================================


def monotonic_stack_template():
    """
    Principle: A stack where elements are always in a sorted order
    (increasing or decreasing).
    Use Cases:
    - Finding the next/previous greater/smaller element for all elements in an array.
    - Problems involving histograms or ranges defined by smaller/larger elements.
    """
    ...


def next_greater_element_i(nums1: List[int], nums2: List[int]) -> List[int]: ...


def daily_temperatures(temperatures: List[int]) -> List[int]: ...


def remove_k_digits(num: str, k: int) -> str: ...


# ======================================================================
# ------------------------- ORDERED SET / FENWICK TREE -----------------
# ======================================================================


def ordered_set_template():
    """
    Principle: A data structure that maintains a collection of elements in
    sorted order and supports efficient rank queries.
    Use Cases:
    - Counting smaller/larger elements.
    - Finding the k-th order statistic.
    - Problems requiring both updates and rank queries.
    Implementation:
    - `bisect` module is simple but has O(n) updates.
    - Fenwick Tree or Segment Tree provide O(log n) updates and queries.
    """
    ...


def simulate_ordered_set_with_bisect(): ...


class FenwickTree:
    def __init__(self, size: int): ...
    def update(self, index: int, delta: int) -> None: ...
    def query(self, index: int) -> int: ...


def count_smaller_numbers_after_self(nums: List[int]) -> List[int]: ...


def contains_nearby_almost_duplicate(nums: List[int], k: int, t: int) -> bool: ...


# ======================================================================
# ------------------------- BITMASKING ---------------------------------
# ======================================================================


def bitmasking_template():
    """
    Principle: Use the bits of an integer to represent a set or state.
    Use Cases:
    - Generating all subsets of a set.
    - Problems with a small number of boolean states that can be
      represented by bits.
    - Efficiently toggling and checking properties.
    """
    ...


def get_bit(n: int, i: int) -> int: ...


def set_bit(n: int, i: int) -> int: ...


def clear_bit(n: int, i: int) -> int: ...


def generate_all_subsets(nums: List[Any]) -> List[List[Any]]: ...


def count_set_bits(n: int) -> int: ...


def single_number(nums: List[int]) -> int: ...


def is_power_of_two(n: int) -> bool: ...
