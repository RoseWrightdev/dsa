from collections import deque
from typing import Optional, Any, List

class Node:
    """A node in a binary tree."""
    def __init__(self, val: Any, left: 'Optional[Node]' = None, right: 'Optional[Node]' = None):
        self.val = val
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return str(self.val)

# =========================================
# --- Tree Creation & Utilities
# =========================================

def make_perfect_tree(height: int, value: int = 1) -> Optional[Node]:
    """
    Creates a perfect binary tree of a given height.
    In a perfect tree, all interior nodes have two children and all leaves are at the same level.
    """
    if height <= 0:
        return None
    root = Node(value)
    root.left = make_perfect_tree(height - 1, value * 2)
    root.right = make_perfect_tree(height - 1, value * 2 + 1)
    return root

def print_tree(root: Optional[Node], prefix: str = "", is_left: bool = True):
    """Prints the binary tree in a human-readable format."""
    if root is not None:
        print_tree(root.right, prefix + ("│   " if is_left else "    "), False)
        print(prefix + ("└── " if is_left else "┌── ") + str(root.val))
        print_tree(root.left, prefix + ("    " if is_left else "│   "), True)

# =========================================
# --- Serialization & Deserialization
# =========================================

def serialize(root: Optional[Node]) -> List[int]:
    """
    Serializes a tree into a list using pre-order traversal. Null children are marked with -1.
    Time: O(N) - visits each node once.
    Space: O(N) - for the result list and recursion stack.
    """
    result = []
    def dfs(node: Optional[Node]):
        if node is None:
            result.append(-1)
            return
        result.append(node.val)
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return result

def deserialize(data: List[int]) -> Optional[Node]:
    """
    Deserializes a list (created with the above serialize function) back into a tree.
    Time: O(N) - processes each value in the list once.
    Space: O(N) - for the recursion stack.
    """
    def dfs():
        val = next(data_iter)
        if val == -1:
            return None
        node = Node(val)
        node.left = dfs()
        node.right = dfs()
        return node
    
    data_iter = iter(data)
    return dfs()

# =========================================
# --- Tree Traversals & Views
# =========================================

def bfs_level_order(root: Optional[Node]) -> List[List[Any]]:
    """
    Performs a Level-Order Traversal (BFS), returning a list of levels.
    Time: O(N) - visits each node once.
    Space: O(W) - where W is the maximum width of the tree.
    """
    if not root:
        return []
    
    q = deque([root])
    result = []
    while q:
        level_size = len(q)
        current_level = []
        for _ in range(level_size):
            node = q.popleft()
            current_level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        result.append(current_level)
    return result

def dfs_preorder(root: Optional[Node]) -> List[Any]:
    """Pre-order traversal (Root -> Left -> Right)."""
    res = []
    def traverse(node):
        if not node:
            return
        res.append(node.val)
        traverse(node.left)
        traverse(node.right)
    traverse(root)
    return res

def dfs_inorder(root: Optional[Node]) -> List[Any]:
    """In-order traversal (Left -> Root -> Right)."""
    res = []
    def traverse(node):
        if not node:
            return
        traverse(node.left)
        res.append(node.val)
        traverse(node.right)
    traverse(root)
    return res

def dfs_postorder(root: Optional[Node]) -> List[Any]:
    """Post-order traversal (Left -> Right -> Root)."""
    res = []
    def traverse(node):
        if not node:
            return
        traverse(node.left)
        traverse(node.right)
        res.append(node.val)
    traverse(root)
    return res

def right_view(root: Optional[Node]) -> List[Any]:
    """
    Returns a list of nodes visible from the right side of the tree.
    Time: O(N) | Space: O(W)
    """
    if not root:
        return []

    result = []
    q = deque([root])
    while q:
        level_size = len(q)
        for i in range(level_size):
            node = q.popleft()
            # If it's the last node of the level, add it to the result
            if i == level_size - 1:
                result.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
    return result

def left_view(root: Optional[Node]) -> List[Any]:
    """
    Returns a list of nodes visible from the left side of the tree.
    Time: O(N) | Space: O(W)
    """
    if not root:
        return []

    result = []
    q = deque([root])
    while q:
        level_size = len(q)
        for i in range(level_size):
            node = q.popleft()
            # If it's the first node of the level, add it to the result
            if i == 0:
                result.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
    return result

def top_view(root: Optional[Node]) -> List[Any]:
    """
    Returns the top view of a binary tree.
    Time: O(N) with min/max tracking.
    Space: O(N) for the map and queue.
    """
    if not root:
        return []

    hd_map = {}  # {horizontal_distance: node.val}
    q = deque([(root, 0)])  # (node, horizontal_distance)
    min_hd, max_hd = 0, 0

    while q:
        node, hd = q.popleft()
        if hd not in hd_map:
            hd_map[hd] = node.val
        
        min_hd = min(min_hd, hd)
        max_hd = max(max_hd, hd)

        if node.left:
            q.append((node.left, hd - 1))
        if node.right:
            q.append((node.right, hd + 1))
            
    return [hd_map[i] for i in range(min_hd, max_hd + 1)]

def bottom_view(root: Optional[Node]) -> List[Any]:
    """
    Returns the bottom view of a binary tree.
    Time: O(N). Space: O(N).
    """
    if not root:
        return []

    hd_map = {}
    q = deque([(root, 0)])
    min_hd, max_hd = 0, 0

    while q:
        node, hd = q.popleft()
        # Always update the map with the latest node at this hd
        hd_map[hd] = node.val

        min_hd = min(min_hd, hd)
        max_hd = max(max_hd, hd)

        if node.left:
            q.append((node.left, hd - 1))
        if node.right:
            q.append((node.right, hd + 1))
            
    return [hd_map[i] for i in range(min_hd, max_hd + 1)]

# =========================================
# --- Tree Properties & Checks
# =========================================

def height(root: Optional[Node]) -> int:
    """
    Calculates the height (or max depth) of a tree.
    Time: O(N) | Space: O(H) for recursion stack, where H is tree height.
    """
    if not root:
        return 0
    return 1 + max(height(root.left), height(root.right))

def is_same_tree(p: Optional[Node], q: Optional[Node]) -> bool:
    """Checks if two trees are structurally identical and have the same node values."""
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)

def is_symmetrical(root: Optional[Node]) -> bool:
    """Checks if a tree is a mirror of itself."""
    if not root:
        return True

    def is_mirror(t1: Optional[Node], t2: Optional[Node]) -> bool:
        if not t1 and not t2:
            return True
        if not t1 or not t2 or t1.val != t2.val:
            return False
        return is_mirror(t1.right, t2.left) and is_mirror(t1.left, t2.right)

    return is_mirror(root.left, root.right)

def is_valid_bst(root: Optional[Node]) -> bool:
    """
    Checks if a binary tree is a valid Binary Search Tree (BST).
    Time: O(N) | Space: O(H)
    """
    def validate(node: Optional[Node], low: float = float('-inf'), high: float = float('inf')) -> bool:
        if not node:
            return True
        if not (low < node.val < high):
            return False
        return validate(node.left, low, node.val) and validate(node.right, node.val, high)
    
    return validate(root)

def is_subtree(root: Optional[Node], subRoot: Optional[Node]) -> bool:
    """
    Checks if subRoot is a subtree of root.
    Time: O(M*N) in the worst case, where M and N are the number of nodes.
    Space: O(H) for recursion stack.
    """
    if not root:
        return False
    if is_same_tree(root, subRoot):
        return True
    return is_subtree(root.left, subRoot) or is_subtree(root.right, subRoot)

# =========================================
# --- Classic Tree Problems
# =========================================

def invert_tree(root: Optional[Node]) -> Optional[Node]:
    """
    Inverts a binary tree by swapping left and right children.
    Time: O(N) | Space: O(H)
    """
    if not root:
        return None
    
    # Swap the children
    root.left, root.right = root.right, root.left
    
    # Recurse for the new left and right children
    invert_tree(root.left)
    invert_tree(root.right)
    
    return root

def lowest_common_ancestor(root: Optional[Node], p: 'Node', q: 'Node') -> Optional[Node]:
    """
    Finds the lowest common ancestor (LCA) of two nodes in a binary tree.
    Time: O(N) | Space: O(H)
    """
    if not root or root == p or root == q:
        return root

    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    if left and right:
        return root  # p and q are in different subtrees, so root is the LCA
    return left or right # p and q are in the same subtree

def diameter_of_binary_tree(root: Optional[Node]) -> int:
    """
    Calculates the diameter (longest path between any two nodes) of a binary tree.
    Time: O(N) | Space: O(H)
    """
    diameter = 0
    def depth(node):
        nonlocal diameter
        if not node:
            return 0
        left_depth = depth(node.left)
        right_depth = depth(node.right)
        # The diameter might be the path that goes through the current node
        diameter = max(diameter, left_depth + right_depth)
        # Return the height of the subtree rooted at this node
        return 1 + max(left_depth, right_depth)

    depth(root)
    return diameter

def has_path_sum(root: Optional[Node], targetSum: int) -> bool:
    """
    Checks if there is a root-to-leaf path that sums up to targetSum.
    Time: O(N) | Space: O(H)
    """
    if not root:
        return False
    # Check if it's a leaf node and the value matches the remaining sum
    if not root.left and not root.right:
        return targetSum == root.val
    
    # Recurse down, subtracting the current node's value from the target sum
    remaining_sum = targetSum - root.val
    return has_path_sum(root.left, remaining_sum) or has_path_sum(root.right, remaining_sum)