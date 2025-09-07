import collections
from typing import List, Any, Dict

# ======================================================================
# I. PREFIX SUM PATTERN
# ======================================================================
# The Prefix Sum pattern is a powerful technique used to efficiently
# calculate the sum of elements in a given range of an array. The core
# idea is to pre-calculate a "prefix sum" array where each element
# `prefix[i]` stores the sum of all elements from the start of the
# original array up to index `i`.

# With this prefix sum array, the sum of any subarray `arr[i:j]` can be
# found in O(1) time by calculating `prefix[j] - prefix[i-1]`.

def prefix_sum_template():
    """
    A general template for solving problems with Prefix Sums.
    
    1. Create a prefix sum array (or use a hashmap for more complex problems).
       - Initialize it with a 0 at the beginning to handle edge cases
         where the subarray starts at index 0.
       - `prefix_sum[i+1] = prefix_sum[i] + arr[i]`
       
    2. Iterate through the array or the prefix sum array.
    
    3. Use the pre-calculated sums to solve the problem.
       - For a range sum `[i, j]`, the result is `prefix[j+1] - prefix[i]`.
       - For problems like "subarray sum equals k", a hashmap is often
         used to store the frequencies of prefix sums encountered so far.
         If `current_sum - k` exists in the hashmap, it means we have found
         a subarray that sums to k.
    """
    pass

# ======================================================================
# II. PREFIX SUM PROBLEMS
# ======================================================================

def subarray_sum_equals_k(nums: List[int], k: int) -> int:
    """
    Given an array of integers and an integer k, find the total number
    of continuous subarrays whose sum equals to k.
    Time Complexity: O(n)
    Space Complexity: O(n) for the hashmap.
    """
    ...

class NumArray:
    """
    A class to handle Range Sum Queries on an immutable array.
    It pre-computes the prefix sums to answer queries in O(1) time.
    """
    def __init__(self, nums: List[int]):
        """
        Initializes the prefix sum array.
        Time Complexity: O(n)
        """
        ...

    def sumRange(self, left: int, right: int) -> int:
        """
        Returns the sum of elements between indices left and right, inclusive.
        Time Complexity: O(1)
        """
        ...

def find_pivot_index(nums: List[int]) -> int:
    """
    Finds the "pivot index" of an array. The pivot index is the index where
    the sum of all the numbers strictly to the left of the index is equal to
    the sum of all the numbers strictly to the index's right.
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    ...
