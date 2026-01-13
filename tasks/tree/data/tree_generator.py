import random
from typing import Dict, List, Tuple, Optional
from utils.tree_utils import TreeNode, TreeRenderer

class TreeGenerator:
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.renderer = TreeRenderer()
    
    def generate_random_tree(self, min_nodes: int = 3, max_nodes: int = 15) -> TreeNode:
        """Generate a random binary tree"""
        num_nodes = random.randint(min_nodes, max_nodes)
        
        # Create root
        root = TreeNode(1)
        nodes = [root]
        node_count = 1
        
        # BFS-style generation with random choices
        i = 0
        while i < len(nodes) and node_count < num_nodes:
            current = nodes[i]
            
            # Randomly decide to add left child
            if random.random() > 0.3 and node_count < num_nodes:
                node_count += 1
                current.left = TreeNode(node_count)
                nodes.append(current.left)
            
            # Randomly decide to add right child
            if random.random() > 0.3 and node_count < num_nodes:
                node_count += 1
                current.right = TreeNode(node_count)
                nodes.append(current.right)
            
            i += 1
        
        return root
    
    def generate_complete_tree(self, depth: int) -> TreeNode:
        """Generate a complete binary tree"""
        if depth == 0:
            return None
        
        root = TreeNode(1)
        queue = [root]
        node_count = 1
        
        while queue:
            current = queue.pop(0)
            current_depth = self._get_node_depth(root, current.val)
            
            if current_depth < depth:
                # Add left child
                node_count += 1
                current.left = TreeNode(node_count)
                queue.append(current.left)
                
                # Add right child if not the last level or if it's a complete tree
                if current_depth < depth - 1 or len(queue) == 0:
                    node_count += 1
                    current.right = TreeNode(node_count)
                    queue.append(current.right)
        
        return root
    
    def generate_perfect_tree(self, depth: int) -> TreeNode:
        """Generate a perfect binary tree"""
        if depth == 0:
            return None
        
        def build_perfect(current_depth: int, node_val: int) -> Tuple[TreeNode, int]:
            node = TreeNode(node_val)
            if current_depth == depth:
                return node, node_val
            
            node.left, next_val = build_perfect(current_depth + 1, node_val + 1)
            node.right, final_val = build_perfect(current_depth + 1, next_val + 1)
            return node, final_val
        
        tree, _ = build_perfect(1, 1)
        return tree
    
    def _get_node_depth(self, root: TreeNode, target: int) -> int:
        """Get the depth of a node with given value"""
        if not root:
            return -1
        
        if root.val == target:
            return 1
        
        left_depth = self._get_node_depth(root.left, target)
        if left_depth != -1:
            return left_depth + 1
        
        right_depth = self._get_node_depth(root.right, target)
        if right_depth != -1:
            return right_depth + 1
        
        return -1