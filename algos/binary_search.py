from typing import List, Any

# ======================================================================
# I. BINARY SEARCH PATTERN
# ======================================================================
# "Binary Search" is a highly efficient searching algorithm that works on
# sorted arrays. It operates by repeatedly dividing the search interval in
# half. If the value of the search key is less than the item in the middle
# of the interval, narrow the interval to the lower half. Otherwise, narrow
# it to the upper half. This process continues until the value is found or
# the interval is empty.
#
# Key Template:
# 1. Initialize two pointers, `left` and `right`, to the start and end of
#    the array, respectively.
# 2. Loop as long as `left <= right`.
# 3. Calculate the middle index: `mid = left + (right - left) // 2`. This
#    version avoids potential overflow in other languages.
# 4. Compare the middle element with the target:
#    - If they match, you've found the target.
#    - If the middle element is less than the target, the target must be
#      in the right half, so move the left pointer: `left = mid + 1`.
#    - If the middle element is greater than the target, the target must
#      be in the left half, so move the right pointer: `right = mid - 1`.
# 5. If the loop finishes, the target was not found.

# ======================================================================
# II. BINARY SEARCH VARIATIONS
# ======================================================================

def binary_search_exact(arr: List[Any], target: Any) -> int:
    """
    Finds the exact index of a target in a sorted array.
    Returns -1 if the target is not found.
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def find_first_occurrence(arr: List[Any], target: Any) -> int:
    """
    Finds the index of the first occurrence of a target in a sorted array.
    Returns -1 if the target is not found.
    """
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            right = mid - 1 # Continue searching in the left half
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result

def find_last_occurrence(arr: List[Any], target: Any) -> int:
    """
    Finds the index of the last occurrence of a target in a sorted array.
    Returns -1 if the target is not found.
    """
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            left = mid + 1 # Continue searching in the right half
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result

# ======================================================================
# III. CLASSIC BINARY SEARCH PROBLEMS
# ======================================================================

def search_in_rotated_sorted_array(nums: List[int], target: int) -> int:
    """
    Searches for a target in a sorted array that has been rotated at some
    unknown pivot.
    Time Complexity: O(log n)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        
        # Check if the left half is sorted
        if nums[left] <= nums[mid]:
            # Check if the target is in the sorted left half
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Otherwise, the right half must be sorted
        else:
            # Check if the target is in the sorted right half
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
                
    return -1

def find_min_in_rotated_sorted_array(nums: List[int]) -> int:
    """
    Finds the minimum element in a sorted array that has been rotated.
    The minimum element is the pivot point.
    Time Complexity: O(log n)
    """
    left, right = 0, len(nums) - 1
    
    # If the array is not rotated at all
    if nums[left] <= nums[right]:
        return nums[left]
        
    while left <= right:
        mid = left + (right - left) // 2
        
        # Check if mid is the minimum element
        # 1. Is it greater than its next element?
        if mid < len(nums) - 1 and nums[mid] > nums[mid + 1]:
            return nums[mid + 1]
        # 2. Is it smaller than its previous element?
        if mid > 0 and nums[mid] < nums[mid - 1]:
            return nums[mid]
            
        # Decide which way to go
        if nums[mid] > nums[left]: # Minimum is in the right half
            left = mid + 1
        else: # Minimum is in the left half (or mid is the minimum)
            right = mid - 1
            
    return -1 # Should not be reached with valid inputs

# ======================================================================
# IV. ADVANCED BINARY SEARCH APPLICATIONS
# ======================================================================

def find_peak_element(nums: List[int]) -> int:
    """
    A peak element is an element that is strictly greater than its neighbors.
    Find and return the index of any peak element.
    The key insight is to use binary search on the "slope" of the array.
    Time Complexity: O(log n)
    """
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        # If the slope is increasing at mid, a peak must be to the right.
        if nums[mid] < nums[mid + 1]:
            left = mid + 1
        # If the slope is decreasing, mid could be a peak, or one is to the left.
        else:
            right = mid
    # `left` and `right` converge to a peak.
    return left

def search_a_2d_matrix_ii(matrix: List[List[int]], target: int) -> bool:
    """
    Searches for a value in an m x n matrix where each row and each
    column is sorted in ascending order.
    This isn't a traditional binary search, but it uses the same principle
    of efficiently eliminating part of the search space.
    Time Complexity: O(m + n)
    """
    if not matrix or not matrix[0]:
        return False
        
    rows, cols = len(matrix), len(matrix[0])
    # Start from the top-right corner
    row, col = 0, cols - 1
    
    while row < rows and col >= 0:
        current_val = matrix[row][col]
        if current_val == target:
            return True
        # If the target is smaller, we can eliminate the entire column
        elif target < current_val:
            col -= 1
        # If the target is larger, we can eliminate the entire row
        else:
            row += 1
            
    return False
