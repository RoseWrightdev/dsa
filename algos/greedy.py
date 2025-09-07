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
    # The greedy choice at each step is:
    # "Should I extend the current subarray, or start a new one here?"
    max_so_far = nums[0]
    current_max = nums[0]
    
    for i in range(1, len(nums)):
        # The locally optimal choice: either the current number itself, or
        # the current number added to the previous subarray sum.
        current_max = max(nums[i], current_max + nums[i])
        
        # Update the global maximum
        max_so_far = max(max_so_far, current_max)
        
    return max_so_far

def can_jump(nums: List[int]) -> bool:
    """
    Given an array of non-negative integers where each element represents
    the maximum jump length at that position, determine if you can reach
    the last index.
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    # The greedy choice is to always jump to the position that offers the
    # furthest possible reach.
    max_reach = 0
    for i, jump_length in enumerate(nums):
        # If the current index is beyond our max reach, we can't get here.
        if i > max_reach:
            return False
        # Update the maximum reach from the current position.
        max_reach = max(max_reach, i + jump_length)
        # If our reach is at or beyond the last index, we're done.
        if max_reach >= len(nums) - 1:
            return True
            
    return False

def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    """
    Given a collection of intervals, merge all overlapping intervals.
    Time Complexity: O(n log n) due to sorting.
    Space Complexity: O(n) for the output list.
    """
    if not intervals:
        return []
        
    # Prerequisite: Sort intervals based on their start time.
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current_start, current_end in intervals[1:]:
        last_merged_start, last_merged_end = merged[-1]
        
        # Greedy choice: If the current interval overlaps with the last
        # merged interval, merge them by extending the end boundary.
        if current_start <= last_merged_end:
            merged[-1] = [last_merged_start, max(last_merged_end, current_end)]
        else:
            # No overlap, so add the current interval as a new one.
            merged.append([current_start, current_end])
            
    return merged

def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    """
    Given two integer arrays `gas` and `cost`, return the starting gas
    station's index if you can travel around the circuit once, otherwise return -1.
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    # If total gas is less than total cost, it's impossible to complete the circuit.
    if sum(gas) < sum(cost):
        return -1
        
    # The greedy insight: If you start at station A and run out of gas before
    # reaching station B, then no station between A and B can be a valid start.
    # Therefore, the new potential start must be B.
    total_tank = 0
    current_tank = 0
    start_station = 0
    
    for i in range(len(gas)):
        total_tank += gas[i] - cost[i]
        current_tank += gas[i] - cost[i]
        
        # If we run out of gas, this can't be the start.
        if current_tank < 0:
            # The new potential start is the next station.
            start_station = i + 1
            # Reset the current tank for the new journey.
            current_tank = 0
            
    return start_station if total_tank >= 0 else -1
