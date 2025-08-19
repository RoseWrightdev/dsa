import collections
from typing import List, Any, Dict

# ======================================================================
# I. MONOTONIC STACK PATTERN
# ======================================================================
# A "Monotonic Stack" is a stack data structure where the elements are
# always in a specific sorted order (either monotonically increasing or
# monotonically decreasing).
#
# This pattern is highly effective for problems where we need to find the
# next or previous greater/smaller element for each element in an array.
# By maintaining the monotonic property, we can efficiently find these
# relationships in a single pass.

def monotonic_stack_template():
    """
    A general template for a "next greater element" type problem
    using a monotonically decreasing stack.
    
    1. Initialize a result array and an empty stack. The stack will store
       elements (or their indices) for which we are seeking a "next greater" element.
       
    2. Iterate through the input array. For each `current_element`:
    
    3. Use a `while` loop: `while stack and stack.top() < current_element`:
       - This condition means `current_element` is the "next greater element"
         for the element at the top of the stack.
       - Pop from the stack.
       - Process the popped element (e.g., update its result in a hashmap or result array).
       
    4. After the loop, push the `current_element` (or its index) onto the stack.
    
    5. After iterating through the entire input array, any elements remaining
       in the stack do not have a "next greater element".
    """
    pass

# ======================================================================
# II. MONOTONIC STACK PROBLEMS
# ======================================================================

def next_greater_element_i(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    For each element in `nums1`, find the first greater element to its
    right in `nums2`.
    Time Complexity: O(nums1.length + nums2.length)
    Space Complexity: O(nums2.length) for the hashmap and stack.
    """
    # Map from value to its next greater element
    next_greater_map: Dict[int, int] = {}
    stack: List[int] = []

    # Iterate through nums2 to populate the map
    for num in nums2:
        while stack and stack[-1] < num:
            next_greater_map[stack.pop()] = num
        stack.append(num)

    # For elements left in the stack, there is no greater element
    for num in stack:
        next_greater_map[num] = -1

    # Build the result for nums1
    return [next_greater_map[num] for num in nums1]

def daily_temperatures(temperatures: List[int]) -> List[int]:
    """
    Given a list of daily temperatures, return an array where each element
    is the number of days you have to wait until a warmer temperature.
    If there is no future day for which this is possible, keep 0 instead.
    Time Complexity: O(n)
    Space Complexity: O(n) for the stack in the worst case.
    """
    n = len(temperatures)
    answer = [0] * n
    # Stack will store indices of the temperatures
    stack: List[int] = []

    for i, temp in enumerate(temperatures):
        # While stack is not empty and current temp is warmer than temp at stack's top index
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            answer[prev_index] = i - prev_index
        stack.append(i)
        
    return answer

def remove_k_digits(num: str, k: int) -> str:
    """
    Given a non-negative integer `num` represented as a string, remove `k`
    digits from the number so that the new number is the smallest possible.
    This uses a monotonically increasing stack.
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    stack: List[str] = []

    for digit in num:
        # While we still have removals left (k>0) and the top of the stack
        # is greater than the current digit, pop from the stack.
        # This removes larger digits from the more significant places.
        while k > 0 and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)

    # If k > 0, it means the remaining digits are in increasing order.
    # Remove the largest digits from the end.
    stack = stack[:-k] if k > 0 else stack

    # Join the stack and remove leading zeros
    result = "".join(stack).lstrip('0')

    return result if result else "0"
