from typing import List, Any, Tuple
import math

# ======================================================================
# I. DIVIDE AND CONQUER PATTERN
# ======================================================================
# "Divide and Conquer" is a powerful algorithmic paradigm that works by
# recursively breaking down a problem into two or more sub-problems of the
# same or related type, until these become simple enough to be solved
# directly. The solutions to the sub-problems are then combined to give a
# solution to the original problem.
#
# The process generally involves three steps:
# 1. **Divide**: Break the given problem into sub-problems of the same type.
# 2. **Conquer**: Recursively solve these sub-problems. If the sub-problems
#    are small enough, solve them directly (the base case).
# 3. **Combine**: Combine the solutions of the sub-problems to create a
#    solution to the original problem.

# ======================================================================
# II. DIVIDE AND CONQUER PROBLEMS
# ======================================================================

def merge_sort(arr: List[Any]) -> List[Any]:
    """
    Sorts an array using the Merge Sort algorithm. This is a canonical
    example of Divide and Conquer.
    - Divide: The array is divided into two halves.
    - Conquer: Each half is sorted recursively.
    - Combine: The two sorted halves are merged together.
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    # A new copy is created to avoid modifying the list in place during recursion
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])

    # Combine step: merge the sorted halves
    return _merge(left_half, right_half)

def _merge(left: List[Any], right: List[Any]) -> List[Any]:
    """Helper function to merge two sorted lists."""
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    # Append any remaining elements
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

def quick_sort(arr: List[Any]) -> List[Any]:
    """
    Sorts an array using the Quick Sort algorithm.
    - Divide: The array is partitioned into three parts based on a 'pivot':
      elements smaller than the pivot, elements equal to the pivot, and
      elements larger than the pivot.
    - Conquer: The sub-arrays of smaller and larger elements are sorted recursively.
    - Combine: The sorted smaller elements, the pivot elements, and the
      sorted larger elements are concatenated.
    Time Complexity: Average O(n log n), Worst O(n^2)
    Space Complexity: O(log n) for the recursion stack.
    """
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)

def binary_search(arr: List[Any], target: Any) -> int:
    """
    Finds the index of a target in a sorted array.
    - Divide: Check the middle element.
    - Conquer: Decide whether to search the left or right half.
    - Combine: The "combine" step is trivial; once the element is found, we're done.
    Time Complexity: O(log n)
    Space Complexity: O(1) (for iterative version).
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1 # Target not found

def max_subarray_sum(arr: List[int]) -> int:
    """
    Finds the maximum sum of a contiguous subarray using Divide and Conquer.
    - Divide: Split the array into two halves.
    - Conquer: Recursively find the max subarray sum in the left half and right half.
    - Combine: The max sum can be in the left half, right half, or it can be a
      subarray that crosses the midpoint. The combine step finds the max
      crossing subarray sum and returns the maximum of the three.
    Time Complexity: O(n log n)
    """
    def _find_max_sum(arr, low, high):
        if low == high:
            return arr[low] # Base case: only one element

        mid = (low + high) // 2

        # 1. Max subarray sum in left half
        max_left = _find_max_sum(arr, low, mid)
        # 2. Max subarray sum in right half
        max_right = _find_max_sum(arr, mid + 1, high)
        # 3. Max subarray sum crossing the midpoint
        max_crossing = _max_crossing_sum(arr, low, mid, high)

        return max(max_left, max_right, max_crossing)

    return _find_max_sum(arr, 0, len(arr) - 1)

def _max_crossing_sum(arr, low, mid, high):
    """Helper for finding the max sum of a subarray crossing the midpoint."""
    # Include elements on left of mid
    sm = 0
    left_sum = -math.inf
    for i in range(mid, low - 1, -1):
        sm += arr[i]
        if sm > left_sum:
            left_sum = sm

    # Include elements on right of mid
    sm = 0
    right_sum = -math.inf
    for i in range(mid + 1, high + 1):
        sm += arr[i]
        if sm > right_sum:
            right_sum = sm

    return left_sum + right_sum
