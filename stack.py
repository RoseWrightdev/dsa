from typing import List, Any

class Stack:
    """
    A simple and efficient stack implementation using a Python list.
    The end of the list represents the top of the stack.
    """
    def __init__(self):
        """Initializes an empty stack."""
        self._items: List[Any] = []

    def push(self, item: Any) -> None:
        """
        Adds an item to the top of the stack.
        Time: O(1) - Amortized constant time for list append.
        """
        self._items.append(item)

    def pop(self) -> Any:
        """
        Removes and returns the top item of the stack.
        Raises an IndexError if the stack is empty.
        Time: O(1) - Constant time for list pop from the end.
        """
        if self.is_empty():
            raise IndexError("pop from an empty stack")
        return self._items.pop()

    def peek(self) -> Any:
        """
        Returns the top item of the stack without removing it.
        Raises an IndexError if the stack is empty.
        Time: O(1)
        """
        if self.is_empty():
            raise IndexError("peek from an empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        """
        Checks if the stack is empty.
        Time: O(1)
        """
        return not self._items

    def size(self) -> int:
        """
        Returns the number of items in the stack.
        Time: O(1)
        """
        return len(self._items)

# =========================================
# --- Classic Stack Problems
# =========================================

def is_valid_parentheses(s: str) -> bool:
    """
    Checks if a string of parentheses '()[]{}' is valid.
    An input string is valid if:
    1. Open brackets are closed by the same type of brackets.
    2. Open brackets are closed in the correct order.

    Time: O(N) - We iterate through the string once.
    Space: O(N) - In the worst case, we store all brackets on the stack.
    """
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for char in s:
        if char in mapping:  # It's a closing bracket
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:  # It's an opening bracket
            stack.append(char)
    return not stack # Must be empty at the end

def daily_temperatures(temperatures: List[int]) -> List[int]:
    """
    Given a list of daily temperatures, returns a list where each element
    is the number of days you have to wait until a warmer temperature.
    If there is no future day for which this is possible, keep 0 instead.

    Time: O(N) - Each element is pushed and popped at most once.
    Space: O(N) - The stack can store up to N elements.
    """
    n = len(temperatures)
    answer = [0] * n
    stack = []  # Stack will store indices: [index, temp]
    for i, temp in enumerate(temperatures):
        while stack and stack[-1][1] < temp:
            stack_i, stack_temp = stack.pop()
            answer[stack_i] = i - stack_i
        stack.append([i, temp])
    return answer

def evaluate_reverse_polish_notation(tokens: List[str]) -> int:
    """
    Evaluates an arithmetic expression in Reverse Polish Notation (RPN).
    Operators are +, -, *, /.

    Time: O(N) - We process each token once.
    Space: O(N) - The stack can grow up to the number of operands.
    """
    stack = []
    operators = {"+": lambda a, b: a + b,
                 "-": lambda a, b: a - b,
                 "*": lambda a, b: a * b,
                 "/": lambda a, b: int(a / b)} # Note integer division

    for token in tokens:
        if token in operators:
            operand2 = stack.pop()
            operand1 = stack.pop()
            result = operators[token](operand1, operand2)
            stack.append(result)
        else:
            stack.append(int(token))
    return stack.pop()

def generate_parentheses(n: int) -> List[str]:
    """
    Given n pairs of parentheses, generates all combinations of
    well-formed parentheses. Solved here using a stack-based backtracking approach.

    Time: O(4^n / sqrt(n)) - This is related to Catalan numbers.
    Space: O(N) - For the recursion depth and the stack used to build the string.
    """
    stack = []
    res = []

    def backtrack(open_count, closed_count):
        # Base case: the combination is complete
        if open_count == closed_count == n:
            res.append("".join(stack))
            return

        # Condition to add an opening parenthesis
        if open_count < n:
            stack.append("(")
            backtrack(open_count + 1, closed_count)
            stack.pop() # Backtrack

        # Condition to add a closing parenthesis
        if closed_count < open_count:
            stack.append(")")
            backtrack(open_count, closed_count + 1)
            stack.pop() # Backtrack

    backtrack(0, 0)
    return res


class MinStack:
    """
    A stack that supports push, pop, top, and retrieving the
    minimum element in constant time.

    Time: O(1) for all operations.
    Space: O(N) as it uses an auxiliary stack to track minimums.
    """
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        # Update the min_stack
        val = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(val)

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]