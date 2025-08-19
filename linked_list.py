from typing import Optional, Any, List, Dict

# ======================================================================
# I. NODE AND LIST IMPLEMENTATIONS
# ======================================================================

class ListNode:
    """A node in a singly linked list."""
    def __init__(self, val: Any = 0, next: 'Optional[ListNode]' = None):
        self.val = val
        self.next = next

class DoublyListNode:
    """A node in a doubly linked list."""
    def __init__(self, val: Any = 0):
        self.val = val
        self.next: Optional['DoublyListNode'] = None
        self.prev: Optional['DoublyListNode'] = None

# ======================================================================
# II. LIST CREATION & UTILITIES
# ======================================================================

def make_singly_linked_list(values: List[Any]) -> Optional[ListNode]:
    """Creates a singly linked list from a list of values."""
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    return head

def print_linked_list(head: Optional[ListNode]) -> str:
    """Returns a string representation of the linked list (e.g., "1 -> 2 -> 3")."""
    if not head:
        return "Empty List"
    parts = []
    current = head
    while current:
        parts.append(str(current.val))
        current = current.next
    return " -> ".join(parts)

# ======================================================================
# III. BASIC OPERATIONS
# ======================================================================

def get_length(head: Optional[ListNode]) -> int:
    """Returns the number of nodes in a linked list."""
    count = 0
    current = head
    while current:
        count += 1
        current = current.next
    return count

def search_list(head: Optional[ListNode], key: Any) -> bool:
    """Checks if a key exists in the linked list."""
    current = head
    while current:
        if current.val == key:
            return True
        current = current.next
    return False

def delete_node_by_key(head: Optional[ListNode], key: Any) -> Optional[ListNode]:
    """Deletes the first occurrence of a node with the given key."""
    dummy = ListNode(0, head)
    prev, current = dummy, head
    
    while current:
        if current.val == key:
            prev.next = current.next
            break
        prev = current
        current = current.next
        
    return dummy.next

# ======================================================================
# IV. CLASSIC LINKED LIST PROBLEMS
# ======================================================================

def has_cycle(head: Optional[ListNode]) -> bool:
    """
    Checks if a linked list has a cycle using Floyd's Tortoise and Hare algorithm.
    Time Complexity: O(N)
    Space Complexity: O(1)
    """
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Reverses a singly linked list iteratively.
    Time Complexity: O(N)
    Space Complexity: O(1)
    """
    prev, current = None, head
    while current:
        next_temp = current.next  # Store the next node
        current.next = prev       # Reverse the current node's pointer
        prev = current            # Move prev one step forward
        current = next_temp       # Move current one step forward
    return prev

def find_middle_node(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Finds the middle node of a linked list using the slow and fast pointer method.
    If the list has an even number of nodes, it returns the second of the two middle nodes.
    Time Complexity: O(N)
    Space Complexity: O(1)
    """
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

def merge_two_sorted_lists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Merges two sorted linked lists into a single sorted linked list.
    Time Complexity: O(N + M) where N and M are the lengths of the lists.
    Space Complexity: O(1)
    """
    dummy = ListNode()
    tail = dummy
    
    while l1 and l2:
        if l1.val < l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next
        
    # Attach the remaining part of the non-empty list
    tail.next = l1 or l2
    
    return dummy.next

def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    Removes the N-th node from the end of the list.
    Uses a two-pointer approach (fast and slow).
    Time Complexity: O(L) where L is the length of the list.
    Space Complexity: O(1)
    """
    dummy = ListNode(0, head)
    fast, slow = dummy, dummy
    
    # Move fast pointer n+1 steps ahead
    for _ in range(n + 1):
        fast = fast.next
        
    # Move both pointers until fast reaches the end
    while fast:
        slow = slow.next
        fast = fast.next
        
    # slow is now at the node before the target node
    slow.next = slow.next.next
    
    return dummy.next

# ======================================================================
# V. ADVANCED LINKED LIST PROBLEMS
# ======================================================================

def reorder_list(head: Optional[ListNode]) -> None:
    """
    Reorders the list from L0→L1→…→Ln-1→Ln to L0→Ln→L1→Ln-1→L2→Ln-2→…
    This is done in-place.
    1. Find the middle of the list.
    2. Split the list into two halves.
    3. Reverse the second half.
    4. Merge the two halves.
    Time Complexity: O(N)
    Space Complexity: O(1)
    """
    if not head or not head.next:
        return

    # 1. Find the middle
    slow, fast = head, head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # 2. Split and 3. Reverse the second half
    second_half = reverse_list(slow.next)
    slow.next = None # End the first half

    # 4. Merge the two lists
    first_half = head
    while second_half:
        temp1, temp2 = first_half.next, second_half.next
        first_half.next = second_half
        second_half.next = temp1
        first_half, second_half = temp1, temp2

class RandomListNode:
    """A node with an additional 'random' pointer."""
    def __init__(self, x: int, next: 'RandomListNode' = None, random: 'RandomListNode' = None):
        self.val = int(x)
        self.next = next
        self.random = random

def copy_random_list(head: 'Optional[RandomListNode]') -> 'Optional[RandomListNode]':
    """
    Creates a deep copy of a linked list where each node has a random pointer.
    Uses a hashmap to store the mapping from old nodes to new nodes.
    Time Complexity: O(N)
    Space Complexity: O(N) for the hashmap.
    """
    if not head:
        return None
        
    old_to_new_map: Dict[RandomListNode, RandomListNode] = {}
    
    # First pass: create all new nodes and map old to new
    current = head
    while current:
        old_to_new_map[current] = RandomListNode(current.val)
        current = current.next
        
    # Second pass: assign next and random pointers for the new nodes
    current = head
    while current:
        new_node = old_to_new_map[current]
        new_node.next = old_to_new_map.get(current.next)
        new_node.random = old_to_new_map.get(current.random)
        current = current.next
        
    return old_to_new_map[head]
