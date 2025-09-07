from typing import Optional, Any, List, Dict

# ======================================================================
# I. NODE AND LIST IMPLEMENTATIONS
# ======================================================================

class ListNode:
    """A node in a singly linked list."""
    def __init__(self, val: Any = 0, next: 'Optional[ListNode]' = None):
        ...

class DoublyListNode:
    """A node in a doubly linked list."""
    def __init__(self, val: Any = 0):
        ...

# ======================================================================
# II. LIST CREATION & UTILITIES
# ======================================================================

def make_singly_linked_list(values: List[Any]) -> Optional[ListNode]:
    ...

def print_linked_list(head: Optional[ListNode]) -> str:
    ...

# ======================================================================
# III. BASIC OPERATIONS
# ======================================================================

def get_length(head: Optional[ListNode]) -> int:
    ...

def search_list(head: Optional[ListNode], key: Any) -> bool:
    ...

def delete_node_by_key(head: Optional[ListNode], key: Any) -> Optional[ListNode]:
    ...

# ======================================================================
# IV. CLASSIC LINKED LIST PROBLEMS
# ======================================================================

def has_cycle(head: Optional[ListNode]) -> bool:
    ...

def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    ...

def find_middle_node(head: Optional[ListNode]) -> Optional[ListNode]:
    ...

def merge_two_sorted_lists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    ...

def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    ...

# ======================================================================
# V. ADVANCED LINKED LIST PROBLEMS
# ======================================================================

def reorder_list(head: Optional[ListNode]) -> None:
    ...

class RandomListNode:
    """A node with an additional 'random' pointer."""
    def __init__(self, x: int, next: 'RandomListNode' = None, random: 'RandomListNode' = None):
        ...

def copy_random_list(head: 'Optional[RandomListNode]') -> 'Optional[RandomListNode]':
    ...
