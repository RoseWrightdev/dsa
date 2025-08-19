import heapq
import collections
from typing import List, Any, Tuple, Optional

# ======================================================================
# I. HELPER CLASSES (e.g., for Linked List problems)
# ======================================================================

class ListNode:
    """A simple ListNode class for context in heap problems."""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# ======================================================================
# II. MIN-HEAP IMPLEMENTATION FROM SCRATCH
# ======================================================================

class MinHeap:
    """A Min-Heap implementation using a list."""
    def __init__(self):
        ...

    def _parent_index(self, index: int) -> int:
        ...

    def _left_child_index(self, index: int) -> int:
        ...

    def _right_child_index(self, index: int) -> int:
        ...

    def _swap(self, i: int, j: int) -> None:
        ...

    def _sift_up(self, index: int) -> None:
        ...

    def _sift_down(self, index: int) -> None:
        ...

    def push(self, value: int) -> None:
        ...

    def pop(self) -> int:
        ...

    def peek(self) -> int:
        ...

# ======================================================================
# III. CLASSIC HEAP PROBLEMS (using Python's `heapq`)
# ======================================================================

class KthLargest:
    """
    Design a class to find the k-th largest element in a stream of numbers.
    """
    def __init__(self, k: int, nums: List[int]):
        ...

    def add(self, val: int) -> int:
        ...

class MedianFinder:
    """
    Design a class to find the median from a data stream.
    """
    def __init__(self):
        ...

    def addNum(self, num: int) -> None:
        ...

    def findMedian(self) -> float:
        ...

def merge_k_sorted_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    ...

# ======================================================================
# IV. MORE ADVANCED HEAP PROBLEMS
# ======================================================================

def top_k_frequent_elements(nums: List[int], k: int) -> List[int]:
    ...

def task_scheduler(tasks: List[str], n: int) -> int:
    ...

def k_smallest_pairs(nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
    ...
