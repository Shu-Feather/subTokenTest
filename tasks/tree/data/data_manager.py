import json
import random
from typing import List, Dict, Any
from data.tree_generator import TreeGenerator
from utils.tree_utils import TreeRenderer, TreeAnalyzer, TreeNode

class DataManager:
    def __init__(self, config):
        self.config = config
        self.tree_generator = TreeGenerator(config.max_depth)
        self.tree_renderer = TreeRenderer()
        self.tree_analyzer = TreeAnalyzer()
    
    def generate_test_data(self) -> List[Dict[str, Any]]:
        """Generate test data for specified tasks"""
        test_data = []
        
        if self.config.task_type == "task1":
            # Generate only Task 1 samples
            for _ in range(self.config.num_samples):
                sample = self._generate_task1_sample()
                test_data.append(sample)
        elif self.config.task_type == "task2":
            # Generate only Task 2 samples
            for _ in range(self.config.num_samples):
                sample = self._generate_task2_sample()
                test_data.append(sample)
        else:  # both
            # Generate both types
            num_task1 = int(self.config.num_samples * self.config.task1_ratio)
            num_task2 = self.config.num_samples - num_task1
            
            # Generate Task 1 samples
            for _ in range(num_task1):
                sample = self._generate_task1_sample()
                test_data.append(sample)
            
            # Generate Task 2 samples
            for _ in range(num_task2):
                sample = self._generate_task2_sample()
                test_data.append(sample)
            
            # Shuffle the data
            random.shuffle(test_data)
        
        return test_data
    
    def _generate_task1_sample(self) -> Dict[str, Any]:
        """Generate a Task 1 sample (structure question)"""
        # Generate random tree
        tree_type = random.choice(['random', 'complete', 'perfect'])
        
        if tree_type == 'random':
            tree = self.tree_generator.generate_random_tree()
        elif tree_type == 'complete':
            depth = random.randint(2, self.config.max_depth)
            tree = self.tree_generator.generate_complete_tree(depth)
        else:  # perfect
            depth = random.randint(2, min(4, self.config.max_depth))  # Perfect trees grow quickly
            tree = self.tree_generator.generate_perfect_tree(depth)
        
        tree_str = self.tree_renderer.render_tree(tree)
        
        # Generate question
        question_type = random.choice(['parent', 'left_child', 'right_child', 'depth', 'num_nodes'])
        
        if question_type in ['parent', 'left_child', 'right_child']:
            # Choose a random node
            all_nodes = self.tree_analyzer.get_all_nodes(tree)
            target_node = random.choice(all_nodes)
            
            if question_type == 'parent':
                question = f"What is the parent node of node {target_node}?"
                answer = self.tree_analyzer.get_parent(tree, target_node)
            elif question_type == 'left_child':
                question = f"What is the left child of node {target_node}?"
                answer = self.tree_analyzer.get_left_child(tree, target_node)
            else:  # right_child
                question = f"What is the right child of node {target_node}?"
                answer = self.tree_analyzer.get_right_child(tree, target_node)
            
            if answer is None:
                answer = "None"
            else:
                answer = str(answer)
        
        elif question_type == 'depth':
            question = "How many levels does this binary tree have?"
            answer = str(self.tree_analyzer.get_depth(tree))
        
        else:  # num_nodes
            question = "How many nodes does this binary tree have?"
            answer = str(self.tree_analyzer.get_node_count(tree))
        
        prompt = self._create_task1_prompt(tree_str, question)
        
        return {
            "task_type": "task1",
            "tree_structure": tree_str,
            "question": question,
            "question_type": question_type,
            "prompt": prompt,
            "expected_answer": answer
        }
    
    def _generate_task2_sample(self) -> Dict[str, Any]:
        """Generate a Task 2 sample (binary tree path analysis)"""
        # Generate a tree suitable for path analysis (using existing tree types)
        tree_type = random.choice(['random', 'complete', 'perfect'])
        
        if tree_type == 'random':
            tree = self.tree_generator.generate_random_tree()
        elif tree_type == 'complete':
            depth = random.randint(3, self.config.max_depth)
            tree = self.tree_generator.generate_complete_tree(depth)
        else:  # perfect
            depth = random.randint(3, min(4, self.config.max_depth))
            tree = self.tree_generator.generate_perfect_tree(depth)
        
        tree_str = self.tree_renderer.render_tree(tree)
        
        # Get all nodes for path selection
        all_nodes = self.tree_analyzer.get_all_nodes(tree)
        
        if len(all_nodes) < 2:
            # Fallback: generate a larger tree
            tree = self.tree_generator.generate_complete_tree(4)
            tree_str = self.tree_renderer.render_tree(tree)
            all_nodes = self.tree_analyzer.get_all_nodes(tree)
        
        # Select two different nodes for path analysis
        source_node, target_node = random.sample(all_nodes, 2)
        
        # Find the path between the two nodes
        path = self.tree_analyzer.find_path_between_nodes(tree, source_node, target_node)
        
        # Create path string representation
        if path:
            path_str = " -> ".join(map(str, path))
        else:
            # This shouldn't happen in a connected tree, but fallback
            path_str = f"{source_node} -> {target_node}"
        
        # Create question
        question = f"Find the path from node {source_node} to node {target_node} in the binary tree. Provide the path as a sequence of nodes separated by ' -> '."
        
        # Create prompt
        prompt = self._create_task2_prompt(tree_str, question)
        
        return {
            "task_type": "task2",
            "tree_structure": tree_str,
            "source_node": source_node,
            "target_node": target_node,
            "question": question,
            "prompt": prompt,
            "expected_answer": path_str,
            "path_analysis": {
                "source": source_node,
                "target": target_node,
                "path": path,
                "path_length": len(path) - 1 if path else -1
            }
        }
        
    def _generate_tree_for_path_analysis(self) -> TreeNode:
        """Generate a tree suitable for path analysis (reasonably dense with good connectivity)"""
        tree_types = ['complete', 'near_complete', 'balanced_random']
        tree_type = random.choice(tree_types)
        
        if tree_type == 'complete':
            depth = random.randint(3, min(self.config.max_depth, 5))
            return self.tree_generator.generate_complete_tree(depth)
        elif tree_type == 'near_complete':
            depth = random.randint(3, min(self.config.max_depth, 5))
            complete_tree = self.tree_generator.generate_complete_tree(depth)
            # Remove a few nodes to make it less perfect but still well-connected
            return self._remove_few_leaf_nodes(complete_tree)
        else:  # balanced_random
            min_nodes = 6
            max_nodes = min(20, 2 ** self.config.max_depth - 1)
            return self.tree_generator.generate_random_tree(min_nodes, max_nodes)
    
    def _remove_few_leaf_nodes(self, tree: TreeNode) -> TreeNode:
        """Remove a few leaf nodes to make the tree less perfect"""
        if not tree:
            return tree
        
        # Find all leaf nodes
        leaf_nodes = self._find_leaf_nodes(tree)
        
        # Remove at most 20% of leaf nodes
        if len(leaf_nodes) > 3:
            num_to_remove = random.randint(1, max(1, len(leaf_nodes) // 5))
            nodes_to_remove = random.sample(leaf_nodes, num_to_remove)
            
            for node_val in nodes_to_remove:
                self._remove_leaf_node(tree, node_val)
        
        return tree
    
    def _find_leaf_nodes(self, root: TreeNode) -> List[int]:
        """Find all leaf nodes in the tree"""
        if not root:
            return []
        
        # If it's a leaf node
        if not root.left and not root.right:
            return [root.val]
        
        leaf_nodes = []
        if root.left:
            leaf_nodes.extend(self._find_leaf_nodes(root.left))
        if root.right:
            leaf_nodes.extend(self._find_leaf_nodes(root.right))
        
        return leaf_nodes
    
    def _remove_leaf_node(self, root: TreeNode, target_val: int) -> bool:
        """Remove a leaf node from the tree"""
        if not root:
            return False
        
        # Check if left child is the target leaf
        if root.left and root.left.val == target_val and not root.left.left and not root.left.right:
            root.left = None
            return True
        
        # Check if right child is the target leaf
        if root.right and root.right.val == target_val and not root.right.left and not root.right.right:
            root.right = None
            return True
        
        # Recursively search
        return self._remove_leaf_node(root.left, target_val) or self._remove_leaf_node(root.right, target_val)
    
    def _create_task1_prompt(self, tree_str: str, question: str) -> str:
        """Create prompt for Task 1"""
        return f"""You are given a binary tree structure. Please analyze it carefully and answer the question.

Binary Tree Structure:
{tree_str}

Question: {question}

Please provide your answer following this format. Note that if there is no such node, answer 'None'. The final answer must appear **only** inside <answer> and </answer>.

---OUTPUT FORMAT---
[Your analysis here if any]
<answer>[just the number or 'None']</answer>"""
    
    def _create_task2_prompt(self, tree_str: str, question: str) -> str:
        """Create prompt for Task 2"""
        return f"""You are given a binary tree structure. Please analyze it carefully and find the requested path between two nodes.

    **Path Finding Concepts:**

    1. **Path Between Nodes**
    - A path between two nodes in a binary tree is the sequence of nodes you must traverse to get from the source node to the target node
    - In a binary tree, there is exactly one path between any two nodes
    - The path may go up from source to a common ancestor, then down to the target

    2. **How to Find Path**
    - Find the Lowest Common Ancestor (LCA) of the two nodes
    - The path goes: source → ... → LCA → ... → target
    - If one node is an ancestor of another, the LCA is the ancestor node

    3. **Path Format**
    - Format: source → intermediate_node1 → intermediate_node2 → ... → target
    - If source and target are the same node, the path is just that single node
    - Use " -> " (space-arrow-space) to separate nodes

    **Example:**
    For the tree:
    ```
        1
    / \\
    2   3
    /   / \\
    4   5   6
    ```

    - Path from 4 to 6: 4 -> 2 -> 1 -> 3 -> 6
    - Path from 2 to 5: 2 -> 1 -> 3 -> 5  
    - Path from 4 to 2: 4 -> 2

    Binary Tree Structure:
    {tree_str}

    Task: {question}

    Please provide your answer following this format. The final answer must appear **only** inside <answer> and </answer>.

    ---OUTPUT FORMAT---
    [Your analysis here if any]
    <answer>[The path in the format: node1 -> node2 -> ... -> nodeN]</answer>"""
    
    def save_test_data(self, test_data: List[Dict[str, Any]], file_path: str):
        """Save test data to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    def load_test_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load test data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
