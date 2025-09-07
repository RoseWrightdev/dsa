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
    count = 0
    current_sum = 0
    # A hashmap to store the frequency of prefix sums: {prefix_sum: frequency}
    prefix_sum_freq: Dict[int, int] = {0: 1} # Base case for subarrays starting at index 0

    for num in nums:
        current_sum += num
        # If (current_sum - k) is a known prefix sum, it means the subarray
        # between that prefix and the current position sums to k.
        if (current_sum - k) in prefix_sum_freq:
            count += prefix_sum_freq[current_sum - k]
        
        # Add the current prefix sum to the map
        prefix_sum_freq[current_sum] = prefix_sum_freq.get(current_sum, 0) + 1
        
    return count

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
        self.prefix_sums = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            self.prefix_sums[i+1] = self.prefix_sums[i] + nums[i]

    def sumRange(self, left: int, right: int) -> int:
        """
        Returns the sum of elements between indices left and right, inclusive.
        Time Complexity: O(1)
        """
        return self.prefix_sums[right + 1] - self.prefix_sums[left]

def find_pivot_index(nums: List[int]) -> int:
    """
    Finds the "pivot index" of an array. The pivot index is the index where
    the sum of all the numbers strictly to the left of the index is equal to
    the sum of all the numbers strictly to the index's right.
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    total_sum = sum(nums)
    left_sum = 0
    
    for i, num in enumerate(nums):
        # The right sum is the total sum minus the left sum and the current element
        right_sum = total_sum - left_sum - num
        if left_sum == right_sum:
            return i
        left_sum += num
        
    return -1
