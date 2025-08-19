import collections
from typing import List, Dict, Tuple, Any, Optional

# ======================================================================
# I. HASHMAP/HASHTABLE IMPLEMENTATION
# ======================================================================

# This class demonstrates how a hashmap can be built from scratch.
# It uses "chaining" with a list to handle hash collisions.

class HashTable:
    """A basic implementation of a hash table."""
    def __init__(self, size: int = 100):
        """Initializes the hash table with a fixed-size list (the 'buckets')."""
        self.size: int = size
        self.table: List[List[Tuple[Any, Any]]] = [[] for _ in range(self.size)]

    def _hash(self, key: Any) -> int:
        """
        A simple hash function to convert a key into an index.
        Uses Python's built-in hash() and the modulo operator.
        """
        return hash(key) % self.size

    def set(self, key: Any, value: Any) -> None:
        """
        
         +++++++++ exists.
        Time Complexity: Average O(1), Worst O(N) (if many collisions occur).
        """
        key_hash = self._hash(key)
        key_value_pair = (key, value)
        
        bucket = self.table[key_hash]
        for i, pair in enumerate(bucket):
            k, v = pair
            if key == k:
                # Key already exists, update the value
                bucket[i] = key_value_pair
                return
        
        # Key does not exist, append the new pair
        bucket.append(key_value_pair)

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieves the value associated with a given key.
        Returns None if the key is not found.
        Time Complexity: Average O(1), Worst O(N).
        """
        key_hash = self._hash(key)
        bucket = self.table[key_hash]
        for pair in bucket:
            k, v = pair
            if key == k:
                return v
        return None # Key not found

    def delete(self, key: Any) -> None:
        """
        Deletes a key-value pair from the hash table.
        Time Complexity: Average O(1), Worst O(N).
        """
        key_hash = self._hash(key)
        bucket = self.table[key_hash]
        for i, pair in enumerate(bucket):
            k, v = pair
            if key == k:
                bucket.pop(i)
                return

    def __str__(self) -> str:
        """String representation for printing the hash table."""
        items = []
        for bucket in self.table:
            if bucket:
                items.extend(bucket)
        return str(items)

# ======================================================================
# II. CLASSIC HASHMAP PROBLEMS
# ======================================================================

def two_sum(nums: List[int], target: int) -> List[int]:
    """
    Given a list of integers, return indices of the two numbers
    such that they add up to a specific target.
    Time Complexity: O(N) - We iterate through the list once.
    Space Complexity: O(N) - For the hashmap.
    """
    num_map: Dict[int, int] = {} # {number: index}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []

def is_valid_anagram(s: str, t: str) -> bool:
    """
    Given two strings, check if t is an anagram of s.
    An anagram is a word formed by rearranging the letters of another.
    Time Complexity: O(S + T) where S and T are lengths of the strings.
    Space Complexity: O(S) for the character count map.
    """
    if len(s) != len(t):
        return False
        
    char_counts = collections.Counter(s)
    
    for char in t:
        if char_counts[char] <= 0:
            return False
        char_counts[char] -= 1
        
    return True

def group_anagrams(strs: List[str]) -> List[List[str]]:
    """
    Given a list of strings, group the anagrams together.
    The key idea is to use a sorted version of a string as the hashmap key.
    Time Complexity: O(N * K log K) where N is the number of strings and K is the max length.
    Space Complexity: O(N * K) for storing the result.
    """
    anagram_map: Dict[str, List[str]] = collections.defaultdict(list)
    for s in strs:
        sorted_s = "".join(sorted(s))
        anagram_map[sorted_s].append(s)
    return list(anagram_map.values())

def first_unique_char(s: str) -> int:
    """
    Find the first non-repeating character in a string and return its index.
    If it does not exist, return -1.
    Time Complexity: O(N) - We iterate through the string twice in the worst case.
    Space Complexity: O(1) - The hashmap will store at most 26 characters.
    """
    counts = collections.Counter(s)
    for i, char in enumerate(s):
        if counts[char] == 1:
            return i
    return -1

# ======================================================================
# III. ADVANCED HASHMAP PROBLEMS
# ======================================================================

class LoggerRateLimiter:
    """
    Design a logger that prevents the same message from being printed more
    than once every 10 seconds.
    """
    def __init__(self):
        """Initializes a dictionary to store message timestamps."""
        self.message_timestamps: Dict[str, int] = {}

    def should_print_message(self, timestamp: int, message: str) -> bool:
        """
        Returns true if the message should be printed, false otherwise.
        Time Complexity: O(1)
        Space Complexity: O(M) where M is the number of unique messages.
        """
        if message not in self.message_timestamps:
            self.message_timestamps[message] = timestamp
            return True
            
        if timestamp - self.message_timestamps[message] >= 10:
            self.message_timestamps[message] = timestamp
            return True
        
        return False

class DLinkedNode:
    """A node for a doubly linked list, used in the LRU Cache."""
    def __init__(self, key: int = 0, val: int = 0):
        self.key = key
        self.val = val
        self.prev: Optional['DLinkedNode'] = None
        self.next: Optional['DLinkedNode'] = None

class LRUCache:
    """
    Implements a Least Recently Used (LRU) Cache. This is a classic
    problem combining a hashmap (for O(1) lookups) and a doubly
    linked list (for O(1) additions/removals of nodes).
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[int, DLinkedNode] = {}
        # Dummy head and tail nodes to handle edge cases smoothly
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: DLinkedNode) -> None:
        """Removes a node from the linked list."""
        prev, nxt = node.prev, node.next
        prev.next = nxt
        nxt.prev = prev

    def _add_to_head(self, node: DLinkedNode) -> None:
        """Adds a node right after the dummy head."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        """
        Retrieves an item from the cache. If found, it becomes the
        most recently used item.
        """
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            self._add_to_head(node)
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        """

        Adds or updates an item in the cache. If the cache is full,
        it removes the least recently used item.
        """
        if key in self.cache:
            # Update existing node
            node = self.cache[key]
            node.val = value
            self._remove_node(node)
            self._add_to_head(node)
        else:
            # Add new node
            if len(self.cache) == self.capacity:
                # Remove the least recently used item (the one before the tail)
                lru = self.tail.prev
                self._remove_node(lru)
                del self.cache[lru.key]
            
            new_node = DLinkedNode(key, value)
            self.cache[key] = new_node
            self._add_to_head(new_node)
