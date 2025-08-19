from typing import Optional, Any, List

# ======================================================================
# I. NODE IMPLEMENTATION
# ======================================================================

class TreeNode:
    """A node in a Binary Search Tree."""
    def __init__(self, key: Any):
        ...

# ======================================================================
# II. CORE BST OPERATIONS
# ======================================================================

def insert(root: Optional[TreeNode], key: Any) -> TreeNode:
    ...

def search(root: Optional[TreeNode], key: Any) -> Optional[TreeNode]:
    ...

def min_value_node(node: TreeNode) -> TreeNode:
    ...

def delete(root: Optional[TreeNode], key: Any) -> Optional[TreeNode]:
    ...

# ======================================================================
# III. UTILITY AND TRAVERSAL
# ======================================================================

def inorder_traversal(root: Optional[TreeNode]) -> List[Any]:
    ...

def max_bst(root: Optional[TreeNode]) -> Optional[Any]:
    ...

# ======================================================================
# IV. ADVANCED BST OPERATIONS
# ======================================================================

def floor(root: Optional[TreeNode], key: Any) -> Optional[TreeNode]:
    ...

def ceil(root: Optional[TreeNode], key: Any) -> Optional[TreeNode]:
    ...

def inorder_successor(root: Optional[TreeNode], p: TreeNode) -> Optional[TreeNode]:
    ...

# ======================================================================
# V. VALIDATION AND OTHER PROBLEMS
# ======================================================================

def is_valid_bst(root: Optional[TreeNode]) -> bool:
    ...

def kth_smallest_element(root: Optional[TreeNode], k: int) -> Optional[Any]:
    ...
