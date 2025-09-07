from typing import List, Any, Dict

# ======================================================================
# I. DYNAMIC PROGRAMMING (DP) PATTERN
# ======================================================================
# Dynamic Programming is an algorithmic technique for solving an optimization
# problem by breaking it down into simpler subproblems and utilizing the
# fact that the optimal solution to the overall problem depends upon the
# optimal solution to its subproblems.
#
# Key Characteristics of a DP Problem:
# 1. **Optimal Substructure**: An optimal solution to the problem can be
#    constructed from optimal solutions of its subproblems.
# 2. **Overlapping Subproblems**: The problem can be broken down into
#    subproblems which are reused several times.
#
# Main Approaches:
# - **Memoization (Top-Down)**: Solve the problem recursively, but store
#   the results of subproblems in a cache (e.g., a hashmap or array) to
#   avoid re-computation.
# - **Tabulation (Bottom-Up)**: Solve the problem iteratively by filling up
#   a table (DP array) from the base cases up to the final solution.

# ======================================================================
# II. DYNAMIC PROGRAMMING PROBLEMS
# ======================================================================

def fibonacci_memoization(n: int) -> int:
    """
    Calculates the n-th Fibonacci number using a top-down (memoization) approach.
    Time Complexity: O(n)
    Space Complexity: O(n) for the cache and recursion stack.
    """
    ...

def fibonacci_tabulation(n: int) -> int:
    """
    Calculates the n-th Fibonacci number using a bottom-up (tabulation) approach.
    Time Complexity: O(n)
    Space Complexity: O(n) for the DP table. (Can be optimized to O(1)).
    """
    ...

def climbing_stairs(n: int) -> int:
    """
    You are climbing a staircase. It takes n steps to reach the top.
    Each time you can either climb 1 or 2 steps. In how many distinct
    ways can you climb to the top?
    This is a classic DP problem that is equivalent to the Fibonacci sequence.
    Time Complexity: O(n)
    Space Complexity: O(1) (using optimized tabulation).
    """
    ...

def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Finds the length of the longest common subsequence between two strings.
    A subsequence is a sequence that can be derived from another sequence
    by deleting some or no elements without changing the order of the
    remaining elements.
    Time Complexity: O(m*n) where m and n are the lengths of the strings.
    Space Complexity: O(m*n) for the DP table.
    """
    ...

def coin_change(coins: List[int], amount: int) -> int:
    """
    Given coins of different denominations and a total amount of money,
    compute the fewest number of coins that you need to make up that amount.
    If that amount of money cannot be made up, return -1.
    Time Complexity: O(amount * len(coins))
    Space Complexity: O(amount)
    """
    ...

def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Given weights and values of n items, put these items in a knapsack of
    capacity W to get the maximum total value in the knapsack.
    You cannot break an item (0-1 property).
    Time Complexity: O(n * capacity)
    Space Complexity: O(n * capacity)
    """
    ...
