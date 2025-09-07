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
    cache: Dict[int, int] = {}
    
    def _fib(num: int) -> int:
        if num in cache:
            return cache[num]
        if num <= 1:
            return num
        
        cache[num] = _fib(num - 1) + _fib(num - 2)
        return cache[num]
        
    return _fib(n)

def fibonacci_tabulation(n: int) -> int:
    """
    Calculates the n-th Fibonacci number using a bottom-up (tabulation) approach.
    Time Complexity: O(n)
    Space Complexity: O(n) for the DP table. (Can be optimized to O(1)).
    """
    if n <= 1:
        return n
        
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
        
    return dp[n]

def climbing_stairs(n: int) -> int:
    """
    You are climbing a staircase. It takes n steps to reach the top.
    Each time you can either climb 1 or 2 steps. In how many distinct
    ways can you climb to the top?
    This is a classic DP problem that is equivalent to the Fibonacci sequence.
    Time Complexity: O(n)
    Space Complexity: O(1) (using optimized tabulation).
    """
    if n <= 2:
        return n
        
    # We only need to store the last two results.
    one_step_before = 2
    two_steps_before = 1
    
    for i in range(3, n + 1):
        current_ways = one_step_before + two_steps_before
        two_steps_before = one_step_before
        one_step_before = current_ways
        
    return one_step_before

def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Finds the length of the longest common subsequence between two strings.
    A subsequence is a sequence that can be derived from another sequence
    by deleting some or no elements without changing the order of the
    remaining elements.
    Time Complexity: O(m*n) where m and n are the lengths of the strings.
    Space Complexity: O(m*n) for the DP table.
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                # If characters match, add 1 to the result of the subproblem without these characters.
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                # If they don't match, take the maximum of the two possible subproblems.
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
    return dp[m][n]

def coin_change(coins: List[int], amount: int) -> int:
    """
    Given coins of different denominations and a total amount of money,
    compute the fewest number of coins that you need to make up that amount.
    If that amount of money cannot be made up, return -1.
    Time Complexity: O(amount * len(coins))
    Space Complexity: O(amount)
    """
    # dp[i] will be storing the minimum number of coins required for amount i
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0 # Base case: 0 coins are needed to make amount 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            # The minimum coins for amount `i` is the minimum of its current value
            # or taking one `coin` and adding it to the solution for `i - coin`.
            dp[i] = min(dp[i], dp[i - coin] + 1)
            
    return dp[amount] if dp[amount] != float('inf') else -1

def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Given weights and values of n items, put these items in a knapsack of
    capacity W to get the maximum total value in the knapsack.
    You cannot break an item (0-1 property).
    Time Complexity: O(n * capacity)
    Space Complexity: O(n * capacity)
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            item_weight = weights[i-1]
            item_value = values[i-1]
            
            # If the current item's weight is more than the current capacity,
            # we can't include it.
            if item_weight > w:
                dp[i][w] = dp[i-1][w]
            else:
                # The max value is either:
                # 1. Not including the item (dp[i-1][w])
                # 2. Including the item (value + value of remaining capacity)
                dp[i][w] = max(dp[i-1][w], item_value + dp[i-1][w - item_weight])
                
    return dp[n][capacity]
