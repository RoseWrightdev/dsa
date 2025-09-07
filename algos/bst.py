from typing import Optional, Any, List

# ======================================================================
# I. NODE IMPLEMENTATION
# ======================================================================

class TreeNode:
    """A node in a Binary Search Tree."""
    def __init__(self, key: Any):
        self.key = key
        self.left: Optional['TreeNode'] = None
        self.right: Optional['TreeNode'] = None

# ======================================================================
# II. CORE BST OPERATIONS
# ======================================================================

def insert(root: Optional[TreeNode], key: Any) -> TreeNode:
    """
    Inserts a new key into the BST and returns the new root of the tree.
    If the tree is empty, it returns a new node.
    Time Complexity: Average O(log N), Worst O(N) (for a skewed tree).
    Space Complexity: Average O(log N), Worst O(N) (for recursion stack).
    """
    if root is None:
        return TreeNode(key)
    
    if key < root.key:
        root.left = insert(root.left, key)
    elif key > root.key:
        root.right = insert(root.right, key)
        
    return root

def search(root: Optional[TreeNode], key: Any) -> Optional[TreeNode]:
    """
    Searches for a key in a BST.
    Returns the node containing the key, or None if not found.
    Time Complexity: Average O(log N), Worst O(N).
    """
    if root is None or root.key == key:
        return root
    
    if key < root.key:
        return search(root.left, key)
    
    return search(root.right, key)

def min_value_node(node: TreeNode) -> TreeNode:
    """
    Given a non-empty binary search tree, return the node
    with the minimum key value found in that tree.
    """
    current = node
    while current and current.left is not None:
        current = current.left
    return current

def delete(root: Optional[TreeNode], key: Any) -> Optional[TreeNode]:
    """
    Deletes a key from the BST and returns the new root.
    Time Complexity: Average O(log N), Worst O(N).
    """
    if root is None:
        return root

    # Find the node to be deleted
    if key < root.key:
        root.left = delete(root.left, key)
    elif key > root.key:
        root.right = delete(root.right, key)
    else:
        # Node with only one child or no child
        if root.left is None:
            return root.right
        elif root.right is None:
            return root.left

        # Node with two children: Get the inorder successor (smallest in the right subtree)
        temp = min_value_node(root.right)
        
        # Copy the inorder successor's content to this node
        root.key = temp.key
        
        # Delete the inorder successor
        root.right = delete(root.right, temp.key)
        
    return root

# ======================================================================
# III. UTILITY AND TRAVERSAL
# ======================================================================

def inorder_traversal(root: Optional[TreeNode]) -> List[Any]:
    """
    Performs an inorder traversal, which visits nodes in sorted order for a BST.
    """
    res = []
    def _traverse(node):
        if node:
            _traverse(node.left)
            res.append(node.key)
            _traverse(node.right)
    _traverse(root)
    return res

def max_bst(root: Optional[TreeNode]) -> Optional[Any]:
    """Finds the maximum value in a BST."""
    if not root:
        return None
    current = root
    while current.right is not None:
        current = current.right
    return current.key

# ======================================================================
# IV. ADVANCED BST OPERATIONS
# ======================================================================

def floor(root: Optional[TreeNode], key: Any) -> Optional[TreeNode]:
    """
    Finds the floor of a key, which is the largest key in the BST
    less than or equal to the given key.
    """
    res = None
    while root:
        if root.key == key:
            return root
        if key < root.key:
            root = root.left
        else: # key > root.key
            res = root
            root = root.right
    return res

def ceil(root: Optional[TreeNode], key: Any) -> Optional[TreeNode]:
    """
    Finds the ceiling of a key, which is the smallest key in the BST
    greater than or equal to the given key.
    """
    res = None
    while root:
        if root.key == key:
            return root
        if key > root.key:
            root = root.right
        else: # key < root.key
            res = root
            root = root.left
    return res

def inorder_successor(root: Optional[TreeNode], p: TreeNode) -> Optional[TreeNode]:
    """
    Finds the inorder successor of a given node in a BST.
    The successor is the node with the smallest key greater than p.key.
    """
    successor = None
    while root:
        if p.key < root.key:
            successor = root
            root = root.left
        else:
            root = root.right
    return successor

# ======================================================================
# V. VALIDATION AND OTHER PROBLEMS
# ======================================================================

def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """
    Checks if a given binary tree is a valid Binary Search Tree.
    """
    def _validate(node, low=-float('inf'), high=float('inf')):
        if not node:
            return True
        if not (low < node.key < high):
            return False
        
        return (_validate(node.left, low, node.key) and
                _validate(node.right, node.key, high))
                
    return _validate(root)

def kth_smallest_element(root: Optional[TreeNode], k: int) -> Optional[Any]:
    """
    Finds the k-th smallest element in a BST.
    Leverages the property of inorder traversal.
    """
    stack = []
    current = root
    
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
            
        current = stack.pop()
        k -= 1
        if k == 0:
            return current.key
            
        current = current.right
        
    return None
