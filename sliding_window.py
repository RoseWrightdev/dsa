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
    max_sum = float('-inf')
    window_sum = 0
    window_start = 0
    
    for window_end in range(len(arr)):
        window_sum += arr[window_end]
        # Slide the window when we've hit the required size 'k'
        if window_end >= k - 1:
            max_sum = max(max_sum, window_sum)
            # Subtract the element going out of the window
            window_sum -= arr[window_start]
            # Slide the window ahead
            window_start += 1
            
    return max_sum if max_sum != float('-inf') else 0

def smallest_subarray_with_given_sum(arr: List[int], target_sum: int) -> int:
    """
    Finds the length of the smallest contiguous subarray whose sum is
    greater than or equal to 'target_sum'.
    This is a dynamic-size sliding window problem.
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    min_length = float('inf')
    window_sum = 0
    window_start = 0
    
    for window_end in range(len(arr)):
        window_sum += arr[window_end]
        
        # Shrink the window as small as possible while sum is still >= target
        while window_sum >= target_sum:
            min_length = min(min_length, window_end - window_start + 1)
            window_sum -= arr[window_start]
            window_start += 1
            
    return min_length if min_length != float('inf') else 0

def find_string_anagrams(s: str, pattern: str) -> List[int]:
    """
    Find all starting indices of a pattern's anagrams in a given string.
    This uses a sliding window with a hashmap.
    Time Complexity: O(S + P) where S and P are lengths of the strings.
    Space Complexity: O(P) or O(1) if the character set is fixed (e.g., 26 letters).
    """
    result_indices = []
    window_start = 0
    matched = 0
    char_frequency = collections.Counter(pattern)
    
    for window_end in range(len(s)):
        right_char = s[window_end]
        if right_char in char_frequency:
            char_frequency[right_char] -= 1
            if char_frequency[right_char] == 0:
                matched += 1
                
        # Check if all characters are matched
        if matched == len(char_frequency):
            result_indices.append(window_start)
            
        # Shrink the window if it has reached the pattern's length
        if window_end >= len(pattern) - 1:
            left_char = s[window_start]
            window_start += 1
            if left_char in char_frequency:
                if char_frequency[left_char] == 0:
                    matched -= 1
                char_frequency[left_char] += 1
                
    return result_indices
