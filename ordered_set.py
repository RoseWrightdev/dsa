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
    sorted_list = []

    # 1. Add element (O(n) due to list insertion)
    bisect.insort_left(sorted_list, 10)
    bisect.insort_left(sorted_list, 5)
    bisect.insort_left(sorted_list, 20)
    # sorted_list is now [5, 10, 20]

    # 2. Find rank of an element (O(log n))
    # `bisect_left` finds the insertion point, which is the count of smaller elements.
    rank_of_15 = bisect.bisect_left(sorted_list, 15) # returns 2

    # 3. Find k-th smallest element (O(1))
    kth_smallest = sorted_list[1] # returns 10 (for k=1)

    print(f"Sorted List: {sorted_list}")
    print(f"Rank of 15 (items smaller than 15): {rank_of_15}")
    print(f"1st smallest element: {kth_smallest}")

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
        self.tree = [0] * (size + 1)

    def update(self, index: int, delta: int) -> None:
        """
        Adds `delta` to the element at `index`.
        Time Complexity: O(log n)
        """
        index += 1 # 1-based index
        while index < len(self.tree):
            self.tree[index] += delta
            index += index & (-index) # Move to the next relevant index

    def query(self, index: int) -> int:
        """
        Computes the prefix sum up to `index` (inclusive).
        Time Complexity: O(log n)
        """
        index += 1 # 1-based index
        s = 0
        while index > 0:
            s += self.tree[index]
            index -= index & (-index) # Move to the parent index
        return s

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
    # Create a rank mapping for numbers to handle negatives and large values
    rank_map = {val: i for i, val in enumerate(sorted(set(nums)))}
    
    n = len(nums)
    result = [0] * n
    ft = FenwickTree(len(rank_map))
    
    # Iterate from right to left
    for i in range(n - 1, -1, -1):
        # Get rank of the current number
        rank = rank_map[nums[i]]
        
        # Query the Fenwick tree for the count of numbers with smaller rank
        # (which have already been processed and are to the right)
        result[i] = ft.query(rank - 1)
        
        # Update the Fenwick tree to mark the presence of the current number's rank
        ft.update(rank, 1)
        
    return result

def contains_nearby_almost_duplicate(nums: List[int], k: int, t: int) -> bool:
    """
    Given an array, find out whether there are two distinct indices i and j
    such that `abs(nums[i] - nums[j]) <= t` and `abs(i - j) <= k`.
    This uses a "bucketing" approach, which is related to ordered sets.
    Time Complexity: O(n)
    Space Complexity: O(k)
    """
    if t < 0 or k <= 0:
        return False

    buckets: Dict[int, int] = {}
    bucket_size = t + 1  # The size of each bucket

    for i, num in enumerate(nums):
        # Determine the bucket ID for the current number
        bucket_id = num // bucket_size

        # Check if the current bucket contains a close number
        if bucket_id in buckets:
            return True
        # Check the left bucket
        if (bucket_id - 1) in buckets and abs(num - buckets[bucket_id - 1]) <= t:
            return True
        # Check the right bucket
        if (bucket_id + 1) in buckets and abs(num - buckets[bucket_id + 1]) <= t:
            return True

        # Add current element to its bucket
        buckets[bucket_id] = num

        # Maintain a window of size k by removing the oldest element
        if i >= k:
            old_bucket_id = nums[i - k] // bucket_size
            # We need to be careful to only remove the bucket if the number
            # stored in it is the one we are removing.
            if old_bucket_id in buckets and buckets[old_bucket_id] == nums[i - k]:
                del buckets[old_bucket_id]

    return False
