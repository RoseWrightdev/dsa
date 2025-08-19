import collections
import bisect
from typing import List, Any, Dict

# ======================================================================
# I. ORDERED SET / SORTED SET PATTERN
# ======================================================================
# An "Ordered Set" is a data structure that stores a collection of unique
# elements in sorted order. It combines the properties of a balanced binary
# search tree and a dynamic array, allowing for efficient operations like:
#
# 1. Add/Remove element: O(log n)
# 2. Check for element: O(log n)
# 3. Find k-th smallest element: O(log n)
# 4. Find the rank of an element (number of items smaller than it): O(log n)
#
# Python does not have a built-in ordered set. This functionality can be
# simulated with the `bisect` module on a sorted list (with slow O(n)
# insertions/deletions) or implemented with more advanced structures like
# a Fenwick Tree (BIT) or a Segment Tree for specific use cases.

# ======================================================================
# II. SIMULATING WITH THE `bisect` MODULE
# ======================================================================
# The `bisect` module provides support for maintaining a list in sorted
# order without having to sort the list after each insertion.

def simulate_ordered_set_with_bisect():
    """
    Demonstrates using `bisect` to simulate ordered set operations.
    """
    ...

# ======================================================================
# III. FENWICK TREE (BINARY INDEXED TREE)
# ======================================================================
# A Fenwick Tree is a data structure that can efficiently update elements
# and calculate prefix sums in a table of numbers. It's a powerful tool
# for solving problems related to order and rank.

class FenwickTree:
    """
    A Fenwick Tree (or Binary Indexed Tree) implementation.
    It's 1-indexed internally for easier calculations.
    """
    def __init__(self, size: int):
        ...

    def update(self, index: int, delta: int) -> None:
        """
        Adds `delta` to the element at `index`.
        Time Complexity: O(log n)
        """
        ...

    def query(self, index: int) -> int:
        """
        Computes the prefix sum up to `index` (inclusive).
        Time Complexity: O(log n)
        """
        ...

# ======================================================================
# IV. ORDERED SET / FENWICK TREE PROBLEMS
# ======================================================================

def count_smaller_numbers_after_self(nums: List[int]) -> List[int]:
    """
    For each element in an array, count how many elements to its right are smaller.
    This is a classic problem solved efficiently with a Fenwick Tree.
    Time Complexity: O(n log n)
    Space Complexity: O(k) where k is the range of numbers.
    """
    ...

def contains_nearby_almost_duplicate(nums: List[int], k: int, t: int) -> bool:
    """
    Given an array, find out whether there are two distinct indices i and j
    such that `abs(nums[i] - nums[j]) <= t` and `abs(i - j) <= k`.
    This uses a "bucketing" approach, which is related to ordered sets.
    Time Complexity: O(n)
    Space Complexity: O(k)
    """
    ...
