from typing import List, Any

# ======================================================================
# I. GREEDY ALGORITHMS PATTERN
# ======================================================================
# A "Greedy Algorithm" is an approach for solving optimization problems by
# making the locally optimal choice at each stage with the hope of finding
# a global optimum.
#
# This means that as the algorithm progresses, it makes a choice that
# seems best at that moment, without considering the future consequences
# of that choice. For many problems, this strategy works and produces a
# globally optimal solution. For others, it does not.

def greedy_algorithm_template():
    """
    A general thought process for designing a greedy algorithm.
    
    1. **Identify the Greedy Choice**: Determine what the "best" or "most
       optimal" choice is at any given step. This often involves picking
       the largest/smallest, earliest/latest, or most profitable option.
       
    2. **Sorting (Common Prerequisite)**: Many greedy problems require sorting
       the input first. This puts the data in an order that makes it easy
       to make the locally optimal choice repeatedly. For example, sorting
       by start time, end time, weight, etc.
       
    3. **Iterate and Build Solution**: Loop through the sorted data. At each
       step, make the greedy choice and update the state of your solution.
       
    4. **Proof of Correctness (Conceptual)**: The hardest part of a greedy
       algorithm is proving that the series of locally optimal choices
       actually leads to a globally optimal solution. For interviews, you
       often rely on recognizing classic problem types where the greedy
       approach is known to be correct.
    """
    pass

# ======================================================================
# II. GREEDY ALGORITHM PROBLEMS
# ======================================================================

def max_subarray(nums: List[int]) -> int:
    """
    Given an integer array, find the contiguous subarray (containing at least
    one number) which has the largest sum and return its sum.
    This is Kadane's Algorithm, a classic greedy approach.
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    ...

def can_jump(nums: List[int]) -> bool:
    """
    Given an array of non-negative integers where each element represents
    the maximum jump length at that position, determine if you can reach
    the last index.
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    ...

def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    """
    Given a collection of intervals, merge all overlapping intervals.
    Time Complexity: O(n log n) due to sorting.
    Space Complexity: O(n) for the output list.
    """
    ...

def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    """
    Given two integer arrays `gas` and `cost`, return the starting gas
    station's index if you can travel around the circuit once, otherwise return -1.
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    ...
