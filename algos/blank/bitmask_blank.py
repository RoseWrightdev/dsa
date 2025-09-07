from typing import List, Any, Tuple
import math

# ======================================================================
# I. BITMASKING PATTERN
# ======================================================================
# "Bitmasking" is a technique that uses the binary representation of numbers
# to solve problems. Each bit in an integer (the "mask") can represent a
# state, a property, or the presence/absence of an element in a set.
#
# This is extremely useful for problems involving subsets, permutations, or
# states that can be toggled on/off, as it allows for efficient storage
# and manipulation of this information.
#
# Common Bitwise Operators:
# - `&` (AND): `a & b` - Sets a bit to 1 only if both bits are 1.
# - `|` (OR): `a | b` - Sets a bit to 1 if at least one of the bits is 1.
# - `^` (XOR): `a ^ b` - Sets a bit to 1 only if the two bits are different.
# - `~` (NOT): `~a` - Inverts all the bits.
# - `<<` (Left Shift): `a << b` - Shifts bits of `a` to the left by `b` positions.
# - `>>` (Right Shift): `a >> b` - Shifts bits of `a` to the right by `b` positions.

# ======================================================================
# II. COMMON BITMASKING OPERATIONS
# ======================================================================

def get_bit(n: int, i: int) -> int:
    """Checks if the i-th bit of a number n is set (1) or not (0)."""
    ...

def set_bit(n: int, i: int) -> int:
    """Sets the i-th bit of a number n to 1."""
    ...

def clear_bit(n: int, i: int) -> int:
    """Clears the i-th bit of a number n to 0."""
    ...

# ======================================================================
# III. BITMASKING PROBLEMS
# ======================================================================

def generate_all_subsets(nums: List[Any]) -> List[List[Any]]:
    """
    Generates all possible subsets (the power set) of a given set of items.
    If a set has n elements, there are 2^n subsets. We can iterate from
    0 to 2^n - 1. Each number `i` in this range represents a unique subset.
    The j-th bit of `i` determines if the j-th element of `nums` is in the subset.
    Time Complexity: O(n * 2^n)
    Space Complexity: O(n * 2^n) to store the result.
    """
    ...

def count_set_bits(n: int) -> int:
    """
    Counts the number of set bits (1s) in the binary representation of an integer.
    The trick `n & (n - 1)` clears the least significant set bit.
    Time Complexity: O(k) where k is the number of set bits.
    """
    ...

def single_number(nums: List[int]) -> int:
    """
    Given a non-empty array of integers where every element appears twice
    except for one, find that single one.
    The XOR operator has the properties: x ^ x = 0 and x ^ 0 = x.
    So, XORing all numbers will cancel out the pairs, leaving the single number.
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    ...

def is_power_of_two(n: int) -> bool:
    """
    Checks if a given integer is a power of two.
    A power of two in binary has exactly one bit set to 1 (e.g., 8 is 1000).
    `n & (n - 1)` will be 0 if and only if `n` has only one set bit.
    We also need to handle the case where n is not positive.
    Time Complexity: O(1)
    """
    ...
