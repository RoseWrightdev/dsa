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
    ...

def selection_sort(arr: List[Any]) -> List[Any]:
    """
    Selection Sort divides the list into a sorted and an unsorted sublist.
    It repeatedly finds the minimum element from the unsorted part and
    moves it to the sorted part.
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    ...

def insertion_sort(arr: List[Any]) -> List[Any]:
    """
    Insertion Sort builds the final sorted array one item at a time.
    It is much less efficient on large lists than more advanced algorithms
    such as quicksort, heapsort, or merge sort.
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    ...

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
    ...

def quick_sort(arr: List[Any]) -> List[Any]:
    """
    Quick Sort is another divide-and-conquer algorithm. It picks an
    element as a 'pivot' and partitions the given array around the
    picked pivot.
    Time Complexity: Average O(n log n), Worst O(n^2)
    Space Complexity: O(log n) for the recursion stack.
    """
    ...

def heap_sort(arr: List[Any]) -> List[Any]:
    """
    Heap Sort uses a binary heap data structure. It first converts the
    list into a max-heap. Then, it repeatedly swaps the root (max element)
    with the last element, reduces the heap size by one, and heapifies the root.
    Time Complexity: O(n log n)
    Space Complexity: O(1) (in-place)
    """
    ...

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
    ...
