import collections
from typing import List, Dict, Tuple, Any, Optional

# ======================================================================
# I. HASHMAP/HASHTABLE IMPLEMENTATION
# ======================================================================

class HashTable:
    """A basic implementation of a hash table."""
    def __init__(self, size: int = 100):
        ...

    def _hash(self, key: Any) -> int:
        ...

    def set(self, key: Any, value: Any) -> None:
        ...

    def get(self, key: Any) -> Optional[Any]:
        ...

    def delete(self, key: Any) -> None:
        ...

    def __str__(self) -> str:
        ...

# ======================================================================
# II. CLASSIC HASHMAP PROBLEMS
# ======================================================================

def two_sum(nums: List[int], target: int) -> List[int]:
    ...

def is_valid_anagram(s: str, t: str) -> bool:
    ...

def group_anagrams(strs: List[str]) -> List[List[str]]:
    ...

def first_unique_char(s: str) -> int:
    ...

# ======================================================================
# III. ADVANCED HASHMAP PROBLEMS
# ======================================================================

class LoggerRateLimiter:
    """
    Design a logger that prevents the same message from being printed more
    than once every 10 seconds.
    """
    def __init__(self):
        ...

    def should_print_message(self, timestamp: int, message: str) -> bool:
        ...

class DLinkedNode:
    """A node for a doubly linked list, used in the LRU Cache."""
    def __init__(self, key: int = 0, val: int = 0):
        ...

class LRUCache:
    """
    Implements a Least Recently Used (LRU) Cache. This is a classic
    problem combining a hashmap (for O(1) lookups) and a doubly
    linked list (for O(1) additions/removals of nodes).
    """
    def __init__(self, capacity: int):
        ...

    def _remove_node(self, node: DLinkedNode) -> None:
        ...

    def _add_to_head(self, node: DLinkedNode) -> None:
        ...

    def get(self, key: int) -> int:
        ...

    def put(self, key: int, value: int) -> None:
        ...
