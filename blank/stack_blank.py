from typing import List, Any

class Stack:
    """A simple stack implementation using a list."""
    def __init__(self):
        # Initialize an empty list to store stack items
        ...

    def push(self, item: Any) -> None:
        # Add an item to the top of the stack
        ...

    def pop(self) -> Any:
        # Remove and return the top item of the stack
        # Should raise an error if the stack is empty
        ...

    def peek(self) -> Any:
        # Return the top item without removing it
        # Should raise an error if the stack is empty
        ...

    def is_empty(self) -> bool:
        # Check if the stack is empty
        ...

    def size(self) -> int:
        # Return the number of items in the stack
        ...

# =========================================
# --- Classic Stack Problems
# =========================================

def is_valid_parentheses():...
def daily_temperatures():...
def evaluate_reverse_polish_notation():...
def generate_parentheses():...
def min_stack():...