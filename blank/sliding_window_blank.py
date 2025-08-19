import collections
from typing import List, Any

# ======================================================================
# I. SLIDING WINDOW PATTERN
# ======================================================================
# The Sliding Window pattern is used to perform a required operation on a
# specific window size of a given array or linked list. It's an efficient
# way to solve problems that involve finding a subarray or substring that
# satisfies a given condition.

def sliding_window_template():
    """
    A general template for solving sliding window problems.
    
    1. Initialize window pointers (left=0, right=0), a result variable, and any
       data structures needed to track the window's state (e.g., current_sum,
       a hashmap for character counts).
       
    2. Iterate through the array with the 'right' pointer from 0 to n-1
       to expand the window.
       
    3. As you expand, add the new element at 'right' to the window's state.
    
    4. Use a 'while' loop to check if the current window is valid or meets
       the desired condition (e.g., `current_sum >= target`). This loop is key
       for dynamic-sized windows.
       
       - If the condition is met, update your result (e.g., find the minimum
         length, add to a list of results).
         
       - Now, shrink the window from the left by moving the 'left' pointer forward.
         Before moving 'left', remove its element's contribution from the
         window's state (e.g., subtract from `current_sum`, decrement a
         character count in a hashmap).
    """
    pass

# ======================================================================
# II. SLIDING WINDOW PROBLEMS
# ======================================================================

def max_subarr_of_size_k(arr: List[int], k: int) -> int:
    """
    Finds the maximum sum of any contiguous subarray of size 'k'.
    This is a fixed-size sliding window problem.
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    ...

def smallest_subarray_with_given_sum(arr: List[int], target_sum: int) -> int:
    """
    Finds the length of the smallest contiguous subarray whose sum is
    greater than or equal to 'target_sum'.
    This is a dynamic-size sliding window problem.
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    ...

def find_string_anagrams(s: str, pattern: str) -> List[int]:
    """
    Find all starting indices of a pattern's anagrams in a given string.
    This uses a sliding window with a hashmap.
    Time Complexity: O(S + P) where S and P are lengths of the strings.
    Space Complexity: O(P) or O(1) if the character set is fixed (e.g., 26 letters).
    """
    ...
