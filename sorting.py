from typing import List, Any

# ======================================================================
# I. COMPARISON-BASED SORTS (O(n^2) Complexity)
# ======================================================================
# These are simple but generally inefficient for large datasets.
# They are good for understanding the fundamentals of sorting.

def bubble_sort(arr: List[Any]) -> List[Any]:
    """
    Bubble Sort repeatedly steps through the list, compares adjacent elements
    and swaps them if they are in the wrong order.
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    n = len(arr)
    # A copy is made to avoid modifying the original list
    arr_copy = arr[:]
    for i in range(n):
        # Flag to optimize if the list becomes sorted early
        swapped = False
        for j in range(0, n - i - 1):
            if arr_copy[j] > arr_copy[j + 1]:
                arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
                swapped = True
        if not swapped:
            break
    return arr_copy

def selection_sort(arr: List[Any]) -> List[Any]:
    """
    Selection Sort divides the list into a sorted and an unsorted sublist.
    It repeatedly finds the minimum element from the unsorted part and
    moves it to the sorted part.
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    n = len(arr)
    arr_copy = arr[:]
    for i in range(n):
        # Find the minimum element in the remaining unsorted array
        min_idx = i
        for j in range(i + 1, n):
            if arr_copy[j] < arr_copy[min_idx]:
                min_idx = j
        # Swap the found minimum element with the first element
        arr_copy[i], arr_copy[min_idx] = arr_copy[min_idx], arr_copy[i]
    return arr_copy

def insertion_sort(arr: List[Any]) -> List[Any]:
    """
    Insertion Sort builds the final sorted array one item at a time.
    It is much less efficient on large lists than more advanced algorithms
    such as quicksort, heapsort, or merge sort.
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    arr_copy = arr[:]
    for i in range(1, len(arr_copy)):
        key = arr_copy[i]
        # Move elements of arr[0..i-1], that are greater than key,
        # to one position ahead of their current position
        j = i - 1
        while j >= 0 and key < arr_copy[j]:
            arr_copy[j + 1] = arr_copy[j]
            j -= 1
        arr_copy[j + 1] = key
    return arr_copy

# ======================================================================
# II. EFFICIENT COMPARISON-BASED SORTS (O(n log n) Complexity)
# ======================================================================

def merge_sort(arr: List[Any]) -> List[Any]:
    """
    Merge Sort is a classic divide-and-conquer algorithm. It divides the
    input array into two halves, calls itself for the two halves, and then
    merges the two sorted halves.
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        # Recursive call on each half
        merge_sort(left_half)
        merge_sort(right_half)

        # Merge the two halves
        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
    return arr

def quick_sort(arr: List[Any]) -> List[Any]:
    """
    Quick Sort is another divide-and-conquer algorithm. It picks an
    element as a 'pivot' and partitions the given array around the
    picked pivot.
    Time Complexity: Average O(n log n), Worst O(n^2)
    Space Complexity: O(log n) for the recursion stack.
    """
    # This is a common, readable implementation.
    # For a true in-place sort, one would pass indices and modify the list directly.
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)

def heap_sort(arr: List[Any]) -> List[Any]:
    """
    Heap Sort uses a binary heap data structure. It first converts the
    list into a max-heap. Then, it repeatedly swaps the root (max element)
    with the last element, reduces the heap size by one, and heapifies the root.
    Time Complexity: O(n log n)
    Space Complexity: O(1) (in-place)
    """
    n = len(arr)
    arr_copy = arr[:]

    def heapify(arr, n, i):
        largest = i  # Initialize largest as root
        left = 2 * i + 1
        right = 2 * i + 2

        # See if left child of root exists and is greater than root
        if left < n and arr[i] < arr[left]:
            largest = left

        # See if right child of root exists and is greater than root
        if right < n and arr[largest] < arr[right]:
            largest = right

        # Change root, if needed
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]  # swap
            heapify(arr, n, largest)

    # Build a maxheap.
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr_copy, n, i)

    # One by one extract elements
    for i in range(n - 1, 0, -1):
        arr_copy[i], arr_copy[0] = arr_copy[0], arr_copy[i]  # swap
        heapify(arr_copy, i, 0)
        
    return arr_copy

# ======================================================================
# III. NON-COMPARISON SORTS (Linear Time Complexity)
# ======================================================================
# These sorts work for specific types of input (e.g., integers in a known range).

def counting_sort(arr: List[int]) -> List[int]:
    """
    Counting Sort works by counting the number of objects having distinct
    key values. It is only suitable for direct use in situations where the
    variation in keys is not significantly greater than the number of items.
    Time Complexity: O(n + k) where k is the range of the input.
    Space Complexity: O(k)
    """
    if not arr:
        return []
        
    max_val = max(arr)
    count_array = [0] * (max_val + 1)
    
    # Store count of each character
    for num in arr:
        count_array[num] += 1
        
    # Build the output character array
    sorted_arr = []
    for i, count in enumerate(count_array):
        sorted_arr.extend([i] * count)
        
    return sorted_arr
