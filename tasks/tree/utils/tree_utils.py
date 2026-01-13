from typing import Optional, List, Dict
import math

class TreeNode:
    def __init__(self, val: int):
        self.val = val
        # self.left = Optional['TreeNode']
        # self.right = Optional['TreeNode']
        self.left = None
        self.right = None

class TreeRenderer:
    def render_tree(self, root: Optional[TreeNode]) -> str:
        """Render tree according to the specified format"""
        if not root:
            return ""
        
        # Get tree structure
        levels = self._get_levels(root)
        max_depth = len(levels)
        
        if max_depth == 0:
            return ""
        
        # Calculate positions for each node
        node_positions = self._calculate_positions(levels, max_depth)
        
        # Render the tree
        result_lines = []
        
        for level_idx in range(max_depth):
            # Node line
            node_line = self._render_node_line(levels[level_idx], node_positions[level_idx])
            result_lines.append(node_line)
            
            # Connector line (if not last level)
            if level_idx < max_depth - 1:
                connector_line = self._render_connector_line(
                    levels[level_idx], levels[level_idx + 1], 
                    node_positions[level_idx], node_positions[level_idx + 1],
                    max_depth, level_idx + 2  # level_idx + 2 is the depth of children
                )
                result_lines.append(connector_line)
        
        return '\n'.join(result_lines)
    
    def render_tree_with_stars(self, complete_tree: TreeNode, original_tree: TreeNode) -> str:
        """Render tree with stars for new nodes"""
        if not complete_tree:
            return ""
        
        # Get original node values
        original_nodes = set()
        if original_tree:
            original_nodes = self._get_all_node_values(original_tree)
        
        # Create a copy of complete tree with stars for new nodes
        starred_tree = self._create_starred_tree(complete_tree, original_nodes)
        
        return self.render_tree(starred_tree)
    
    def _get_levels(self, root: TreeNode) -> List[List[TreeNode]]:
        """Get nodes at each level"""
        if not root:
            return []
        
        levels = []
        queue = [root]
        
        while queue:
            level_size = len(queue)
            current_level = []
            
            for _ in range(level_size):
                node = queue.pop(0)
                current_level.append(node)
                
                if node:
                    if node.left:
                        queue.append(node.left)
                    if node.right:
                        queue.append(node.right)
            
            levels.append(current_level)
            
            # Check if there are any actual nodes in the queue
            if not any(queue):
                break
        
        return levels
    
    def _calculate_positions(self, levels: List[List[TreeNode]], max_depth: int) -> List[List[int]]:
        """Calculate horizontal positions for each node"""
        positions = []
        
        # Calculate spacing for bottom level
        bottom_level_spacing = 3
        
        for level_idx in range(max_depth):
            level_positions = []
            level_depth_from_bottom = max_depth - level_idx
            
            # Calculate spacing for this level
            spacing = bottom_level_spacing + sum(4 * (2 ** i) for i in range(level_depth_from_bottom - 1))
            
            # Calculate positions for nodes in this level
            if level_idx == 0:
                # Root node position
                level_positions.append(0)
            else:
                # Calculate based on parent positions
                parent_positions = positions[level_idx - 1]
                parent_nodes = levels[level_idx - 1]
                
                for i, parent_node in enumerate(parent_nodes):
                    if parent_node and parent_node.left:
                        # Left child position
                        offset = 2 ** (max_depth - level_idx)
                        left_pos = parent_positions[i] - offset
                        level_positions.append(left_pos)
                    
                    if parent_node and parent_node.right:
                        # Right child position
                        offset = 2 ** (max_depth - level_idx)
                        right_pos = parent_positions[i] + offset
                        level_positions.append(right_pos)
            
            positions.append(level_positions)
        
        # Normalize positions to start from 0
        if positions:
            min_pos = min(min(level_pos) for level_pos in positions if level_pos)
            positions = [[pos - min_pos for pos in level_pos] for level_pos in positions]
        
        return positions
    
    def _render_node_line(self, nodes: List[TreeNode], positions: List[int]) -> str:
        """Render a line with nodes"""
        if not nodes or not positions:
            return ""
        
        max_pos = max(positions) if positions else 0
        line = [' '] * (max_pos + 10)  # Extra space for multi-digit numbers
        
        for node, pos in zip(nodes, positions):
            if node:
                node_str = str(node.val)
                for i, char in enumerate(node_str):
                    if pos + i < len(line):
                        line[pos + i] = char
        
        return ''.join(line).rstrip()
    
    def _render_connector_line(self, parent_nodes: List[TreeNode], child_nodes: List[TreeNode],
                              parent_positions: List[int], child_positions: List[int],
                              max_depth: int, child_level: int) -> str:
        """Render connector line between parent and child levels"""
        if not parent_nodes or not child_nodes:
            return ""
        
        max_pos = max(max(parent_positions), max(child_positions)) if parent_positions and child_positions else 0
        line = [' '] * (max_pos + 10)
        
        # Calculate offset for connectors
        offset = 2 ** (max_depth - child_level)
        
        child_idx = 0
        for parent_idx, parent_node in enumerate(parent_nodes):
            if not parent_node:
                continue
            
            parent_pos = parent_positions[parent_idx]
            
            # Left child connector
            if parent_node.left and child_idx < len(child_positions):
                connector_pos = parent_pos - offset
                if 0 <= connector_pos < len(line):
                    line[connector_pos] = '/'
                child_idx += 1
            
            # Right child connector
            if parent_node.right and child_idx < len(child_positions):
                connector_pos = parent_pos + offset
                if 0 <= connector_pos < len(line):
                    line[connector_pos] = '\\'
                child_idx += 1
        
        return ''.join(line).rstrip()
    
    def _get_all_node_values(self, root: TreeNode) -> set:
        """Get all node values in the tree"""
        if not root:
            return set()
        
        values = {root.val}
        if root.left:
            values.update(self._get_all_node_values(root.left))
        if root.right:
            values.update(self._get_all_node_values(root.right))
        
        return values
    
    def _create_starred_tree(self, tree: TreeNode, original_nodes: set) -> TreeNode:
        """Create a copy of tree with stars for nodes not in original"""
        if not tree:
            return None
        
        # Create new node
        if tree.val in original_nodes:
            new_node = TreeNode(tree.val)
        else:
            new_node = TreeNode('*')
        
        # Recursively copy children
        if tree.left:
            new_node.left = self._create_starred_tree(tree.left, original_nodes)
        if tree.right:
            new_node.right = self._create_starred_tree(tree.right, original_nodes)
        
        return new_node

class TreeAnalyzer:
    def get_all_nodes(self, root: TreeNode) -> List[int]:
        """Get all node values in the tree"""
        if not root:
            return []
        
        nodes = [root.val]
        if root.left:
            nodes.extend(self.get_all_nodes(root.left))
        if root.right:
            nodes.extend(self.get_all_nodes(root.right))
        
        return nodes
    
    def get_parent(self, root: TreeNode, target: int) -> Optional[int]:
        """Get parent of target node"""
        if not root or root.val == target:
            return None
        
        if (root.left and root.left.val == target) or (root.right and root.right.val == target):
            return root.val
        
        left_result = self.get_parent(root.left, target)
        if left_result is not None:
            return left_result
        
        return self.get_parent(root.right, target)
    
    def get_left_child(self, root: TreeNode, target: int) -> Optional[int]:
        """Get left child of target node"""
        node = self._find_node(root, target)
        if node and node.left:
            return node.left.val
        return None
    
    def get_right_child(self, root: TreeNode, target: int) -> Optional[int]:
        """Get right child of target node"""
        node = self._find_node(root, target)
        if node and node.right:
            return node.right.val
        return None
    
    def get_depth(self, root: TreeNode) -> int:
        """Get depth (number of levels) of the tree"""
        if not root:
            return 0
        
        left_depth = self.get_depth(root.left)
        right_depth = self.get_depth(root.right)
        
        return max(left_depth, right_depth) + 1
    
    def get_node_count(self, root: TreeNode) -> int:
        """Get total number of nodes in the tree"""
        if not root:
            return 0
        
        return 1 + self.get_node_count(root.left) + self.get_node_count(root.right)
    
    def _find_node(self, root: TreeNode, target: int) -> Optional[TreeNode]:
        """Find node with target value"""
        if not root:
            return None
        
        if root.val == target:
            return root
        
        left_result = self._find_node(root.left, target)
        if left_result:
            return left_result
        
        return self._find_node(root.right, target)

    def find_path_between_nodes(self, root: TreeNode, source, target) -> List:
        """Find the path between two nodes in the binary tree"""
        if not root:
            return []
        
        # If source and target are the same
        if source == target:
            if self._find_node(root, source):
                return [source]
            return []
        
        # Find path from root to source
        path_to_source = self._find_path_from_root(root, source)
        
        # Find path from root to target
        path_to_target = self._find_path_from_root(root, target)
        
        if not path_to_source or not path_to_target:
            return []  # One or both nodes not found
        
        # Find the lowest common ancestor (LCA) by finding the last common node
        lca_index = -1
        min_len = min(len(path_to_source), len(path_to_target))
        
        for i in range(min_len):
            if path_to_source[i] == path_to_target[i]:
                lca_index = i
            else:
                break
        
        if lca_index == -1:
            return []  # No common ancestor found (shouldn't happen in a tree)
        
        # Build the complete path: source -> ... -> LCA -> ... -> target
        
        # 1. Path from source to LCA (reverse the source path up to LCA, excluding LCA)
        path_source_to_lca = path_to_source[lca_index:]  # From LCA to source
        path_source_to_lca.reverse()  # Now from source to LCA
        
        # 2. Path from LCA to target (excluding LCA to avoid duplication)
        path_lca_to_target = path_to_target[lca_index + 1:]  # From LCA+1 to target
        
        # 3. Combine: source -> ... -> LCA + LCA -> ... -> target
        complete_path = path_source_to_lca + path_lca_to_target
        
        return complete_path

    def _find_path_from_root(self, root: TreeNode, target) -> List:
        """Find path from root to target node"""
        if not root:
            return []
        
        if root.val == target:
            return [target]
        
        # Try left subtree
        if root.left:
            left_path = self._find_path_from_root(root.left, target)
            if left_path:
                return [root.val] + left_path
        
        # Try right subtree
        if root.right:
            right_path = self._find_path_from_root(root.right, target)
            if right_path:
                return [root.val] + right_path
        
        return []

    def find_lowest_common_ancestor(self, root: TreeNode, node1, node2) -> Optional[TreeNode]:
        """Find the lowest common ancestor of two nodes"""
        if not root:
            return None
        
        # If current node is one of the target nodes
        if root.val == node1 or root.val == node2:
            return root
        
        # Search in left and right subtrees
        left_lca = self.find_lowest_common_ancestor(root.left, node1, node2)
        right_lca = self.find_lowest_common_ancestor(root.right, node1, node2)
        
        # If both nodes are found in different subtrees, current node is LCA
        if left_lca and right_lca:
            return root
        
        # Return non-null result
        return left_lca if left_lca else right_lca