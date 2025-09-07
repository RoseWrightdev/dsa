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

# This class demonstrates the underlying logic of a heap.
# A heap is a complete binary tree, which can be efficiently
# represented using a dynamic array (a list in Python).

class MinHeap:
    """
    A Min-Heap implementation using a list.
    The parent of an element at index i is at (i-1)//2.
    The left child is at 2*i + 1.
    The right child is at 2*i + 2.
    """
    def __init__(self):
        """Initializes an empty heap."""
        self.heap: List[int] = []

    def _parent_index(self, index: int) -> int:
        return (index - 1) // 2

    def _left_child_index(self, index: int) -> int:
        return 2 * index + 1

    def _right_child_index(self, index: int) -> int:
        return 2 * index + 2

    def _swap(self, i: int, j: int) -> None:
        """Swaps two elements in the heap list."""
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def _sift_up(self, index: int) -> None:
        """
        Moves an element up the heap to maintain the heap property.
        Used after inserting a new element.
        """
        parent_idx = self._parent_index(index)
        # While the node is not the root and is smaller than its parent
        while index > 0 and self.heap[index] < self.heap[parent_idx]:
            self._swap(index, parent_idx)
            index = parent_idx
            parent_idx = self._parent_index(index)

    def _sift_down(self, index: int) -> None:
        """
        Moves an element down the heap to maintain the heap property.
        Used after removing the root element.
        """
        max_index = len(self.heap) - 1
        while True:
            left_idx = self._left_child_index(index)
            right_idx = self._right_child_index(index)
            smallest = index # Assume the current node is the smallest

            if left_idx <= max_index and self.heap[left_idx] < self.heap[smallest]:
                smallest = left_idx
            if right_idx <= max_index and self.heap[right_idx] < self.heap[smallest]:
                smallest = right_idx

            if smallest != index:
                self._swap(index, smallest)
                index = smallest # Move down to the new position
            else:
                break # The heap property is satisfied

    def push(self, value: int) -> None:
        """
        Adds a new value to the heap.
        Time Complexity: O(log N)
        """
        self.heap.append(value)
        self._sift_up(len(self.heap) - 1)

    def pop(self) -> int:
        """
        Removes and returns the minimum element (the root) from the heap.
        Time Complexity: O(log N)
        """
        if not self.heap:
            raise IndexError("pop from an empty heap")
        
        # Swap the root with the last element
        self._swap(0, len(self.heap) - 1)
        min_value = self.heap.pop()
        
        # Restore the heap property from the root
        if self.heap:
            self._sift_down(0)
            
        return min_value

    def peek(self) -> int:
        """
        Returns the minimum element without removing it.
        Time Complexity: O(1)
        """
        if not self.heap:
            raise IndexError("peek from an empty heap")
        return self.heap[0]

# ======================================================================
# III. CLASSIC HEAP PROBLEMS (using Python's `heapq`)
# ======================================================================
# Note: Python's `heapq` module implements a min-heap. To simulate a
# max-heap, we can store negated values.

class KthLargest:
    """
    Design a class to find the k-th largest element in a stream of numbers.
    We maintain a min-heap of size k. The root of this heap is the k-th largest element.
    """
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.heap: List[int] = nums
        heapq.heapify(self.heap)
        # Keep popping from the heap until it's of size k
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        """
        Adds a new value to the stream and returns the current k-th largest element.
        Time Complexity: O(log K)
        """
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]

class MedianFinder:
    """
    Design a class to find the median from a data stream.
    We use two heaps:
    - A max-heap (`small_half`) to store the smaller half of the numbers.
    - A min-heap (`large_half`) to store the larger half of the numbers.
    This keeps the medians at the top of the heaps.
    """
    def __init__(self):
        self.small_half: List[int] = []  # Max-heap (negated values)
        self.large_half: List[int] = []  # Min-heap

    def addNum(self, num: int) -> None:
        """Adds a number to the data structure."""
        # Push to max-heap and then move the largest from it to the min-heap
        heapq.heappush(self.small_half, -num)
        if self.small_half and self.large_half and (-self.small_half[0] > self.large_half[0]):
            val = -heapq.heappop(self.small_half)
            heapq.heappush(self.large_half, val)

        # Balance the heaps to ensure their sizes differ by at most 1
        if len(self.small_half) > len(self.large_half) + 1:
            val = -heapq.heappop(self.small_half)
            heapq.heappush(self.large_half, val)
        if len(self.large_half) > len(self.small_half) + 1:
            val = heapq.heappop(self.large_half)
            heapq.heappush(self.small_half, -val)

    def findMedian(self) -> float:
        """Returns the median of all elements so far."""
        if len(self.small_half) > len(self.large_half):
            return -self.small_half[0]
        if len(self.large_half) > len(self.small_half):
            return self.large_half[0]
        
        # Even number of elements, return average of the two middle values
        return (-self.small_half[0] + self.large_half[0]) / 2.0

def merge_k_sorted_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Merges k sorted linked lists into one sorted linked list.
    We use a min-heap to efficiently find the smallest node among the heads
    of all lists at any time.
    Time Complexity: O(N log K) where N is the total number of nodes and K is the number of lists.
    Space Complexity: O(K) for the heap.
    """
    min_heap: List[Tuple[int, int, ListNode]] = []
    for i, l in enumerate(lists):
        if l:
            # The (i) is a tie-breaker to handle nodes with the same value
            heapq.heappush(min_heap, (l.val, i, l))

    dummy = ListNode()
    tail = dummy
    
    while min_heap:
        val, i, node = heapq.heappop(min_heap)
        tail.next = node
        tail = tail.next
        if node.next:
            heapq.heappush(min_heap, (node.next.val, i, node.next))
            
    return dummy.next

# ======================================================================
# IV. MORE ADVANCED HEAP PROBLEMS
# ======================================================================

def top_k_frequent_elements(nums: List[int], k: int) -> List[int]:
    """
    Given an integer array nums and an integer k, return the k most frequent elements.
    1. Count frequencies of all numbers using a hashmap.
    2. Use a min-heap to keep track of the top k frequent elements.
    Time Complexity: O(N log K) - N for counting, N for pushing to heap of size K.
    Space Complexity: O(N + K) - N for hashmap, K for heap.
    """
    if k == len(nums):
        return nums

    # 1. Count frequencies
    counts = collections.Counter(nums)
    
    # 2. Use a min-heap of size k
    # The heap will store tuples of (frequency, number)
    min_heap: List[Tuple[int, int]] = []
    for num, freq in counts.items():
        heapq.heappush(min_heap, (freq, num))
        if len(min_heap) > k:
            heapq.heappop(min_heap)
            
    # The heap now contains the top k elements
    return [num for freq, num in min_heap]

def task_scheduler(tasks: List[str], n: int) -> int:
    """
    Given a list of CPU tasks and a cooldown period 'n', find the minimum
    time to finish all tasks.
    Time Complexity: O(T * n) where T is the number of unique tasks.
    Space Complexity: O(T)
    """
    counts = collections.Counter(tasks)
    # We use a max-heap to always process the most frequent task
    max_heap = [-count for count in counts.values()]
    heapq.heapify(max_heap)
    
    time = 0
    queue = collections.deque() # Stores [-count, available_time]
    
    while max_heap or queue:
        time += 1
        
        if max_heap:
            count = heapq.heappop(max_heap) + 1 # Decrement task count
            if count != 0:
                # This task needs to cool down
                queue.append([count, time + n])
                
        if queue and queue[0][1] == time:
            # A cooled-down task is ready to be added back to the heap
            heapq.heappush(max_heap, queue.popleft()[0])
            
    return time

def k_smallest_pairs(nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
    """
    Given two sorted integer arrays, find the k pairs with the smallest sums.
    A pair (u, v) consists of one element from the first array and one from the second.
    Time Complexity: O(K log K)
    Space Complexity: O(K)
    """
    if not nums1 or not nums2:
        return []

    min_heap: List[Tuple[int, int, int]] = []
    result: List[List[int]] = []
    
    # Push the first element of nums2 paired with each element of nums1
    for i in range(min(k, len(nums1))):
        heapq.heappush(min_heap, (nums1[i] + nums2[0], i, 0))
        
    while min_heap and len(result) < k:
        sum_val, i, j = heapq.heappop(min_heap)
        result.append([nums1[i], nums2[j]])
        
        # If there's a next element in nums2 for the current element from nums1, push it
        if j + 1 < len(nums2):
            heapq.heappush(min_heap, (nums1[i] + nums2[j + 1], i, j + 1))
            
    return result