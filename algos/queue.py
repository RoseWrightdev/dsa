from collections import deque
from typing import Any, List

class QueueWithList:
    """
    A simple queue implementation using a standard Python list.
    NOTE: This is for educational purposes. It is INEFFICIENT because
    popping from the beginning of a list is an O(N) operation.
    """
    def __init__(self):
        """Initializes an empty queue."""
        self._items: List[Any] = []

    def enqueue(self, item: Any) -> None:
        """Adds an item to the back of the queue. Time: O(1)"""
        self._items.append(item)

    def dequeue(self) -> Any:
        """
        Removes and returns the front item of the queue.
        Raises an IndexError if the queue is empty.
        Time: O(N) - Inefficient due to shifting all elements.
        """
        if self.is_empty():
            raise IndexError("dequeue from an empty queue")
        return self._items.pop(0)

    def peek(self) -> Any:
        """
        Returns the front item without removing it.
        Time: O(1)
        """
        if self.is_empty():
            raise IndexError("peek from an empty queue")
        return self._items[0]

    def is_empty(self) -> bool:
        """Checks if the queue is empty. Time: O(1)"""
        return not self._items

    def size(self) -> int:
        """Returns the number of items in the queue. Time: O(1)"""
        return len(self._items)

class Queue:
    """
    An EFFICIENT queue implementation using Python's `collections.deque`.
    `deque` is a double-ended queue, optimized for fast appends and pops
    from both ends.
    """
    def __init__(self):
        """Initializes an empty queue."""
        self._items = deque()

    def enqueue(self, item: Any) -> None:
        """
        Adds an item to the back of the queue.
        Time: O(1)
        """
        self._items.append(item)

    def dequeue(self) -> Any:
        """
        Removes and returns the front item of the queue.
        Raises an IndexError if the queue is empty.
        Time: O(1)
        """
        if self.is_empty():
            raise IndexError("dequeue from an empty queue")
        return self._items.popleft()

    def peek(self) -> Any:
        """
        Returns the front item without removing it.
        Time: O(1)
        """
        if self.is_empty():
            raise IndexError("peek from an empty queue")
        return self._items[0]

    def is_empty(self) -> bool:
        """Checks if the queue is empty. Time: O(1)"""
        return not self._items

    def size(self) -> int:
        """Returns the number of items in the queue. Time: O(1)"""
        return len(self._items)

# =========================================
# --- Classic Queue Problems
# =========================================

def number_of_islands(grid: List[List[str]]) -> int:
    """
    Counts the number of islands in a 2D grid. An island is surrounded by
    water and is formed by connecting adjacent lands horizontally or vertically.
    This is a classic Breadth-First Search (BFS) problem.

    Time: O(M*N) - We visit each cell in the grid at most once.
    Space: O(min(M,N)) - The max size of the queue in the worst case.
    """
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    num_islands = 0
    q = deque()

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                num_islands += 1
                q.append((r, c))
                grid[r][c] = '0' # Mark as visited

                while q:
                    row, col = q.popleft()
                    directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
                    for dr, dc in directions:
                        nr, nc = row + dr, col + dc
                        if (0 <= nr < rows and 0 <= nc < cols and
                                grid[nr][nc] == '1'):
                            q.append((nr, nc))
                            grid[nr][nc] = '0' # Mark as visited
    return num_islands

class MovingAverage:
    """
    Calculates the moving average of a stream of numbers.
    """
    def __init__(self, size: int):
        """Initializes the window size and the queue."""
        self.queue = deque()
        self.size = size
        self.current_sum = 0.0

    def next(self, val: int) -> float:
        """
        Adds a new value and returns the moving average.

        Time: O(1) - All operations are constant time.
        Space: O(N) - Where N is the size of the window.
        """
        if len(self.queue) == self.size:
            # Remove the oldest element to make room
            self.current_sum -= self.queue.popleft()
        
        self.queue.append(val)
        self.current_sum += val
        return self.current_sum / len(self.queue)


class MyCircularQueue:
    """
    An implementation of a circular queue (or circular buffer).
    It has a fixed size and overwrites the oldest elements when full.
    """
    def __init__(self, k: int):
        self.queue = [0] * k
        self.capacity = k
        self.size = 0
        self.head = 0
        self.tail = -1

    def enQueue(self, value: int) -> bool:
        """Inserts an element into the circular queue."""
        if self.isFull():
            return False
        self.tail = (self.tail + 1) % self.capacity
        self.queue[self.tail] = value
        self.size += 1
        return True

    def deQueue(self) -> bool:
        """Deletes an element from the circular queue."""
        if self.isEmpty():
            return False
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return True

    def Front(self) -> int:
        """Gets the front item from the queue."""
        return self.queue[self.head] if not self.isEmpty() else -1

    def Rear(self) -> int:
        """Gets the last item from the queue."""
        return self.queue[self.tail] if not self.isEmpty() else -1

    def isEmpty(self) -> bool:
        """Checks whether the circular queue is empty or not."""
        return self.size == 0

    def isFull(self) -> bool:
        """Checks whether the circular queue is full or not."""
        return self.size == self.capacity # type: ignore