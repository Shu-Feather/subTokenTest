import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any
from data.tree_generator import TreeGenerator
from utils.tree_utils import TreeRenderer, TreeAnalyzer, TreeNode
import random

class Task1Generator:
    def __init__(self, difficulty: str = "medium"):
        self.difficulty = difficulty
        self.depth_ranges = {
            "easy": (3, 4),
            "medium": (5, 5), 
            "hard": (6, 6)
        }
        self.max_depth = self.depth_ranges[difficulty][1]
        self.tree_generator = TreeGenerator(self.max_depth)
        self.tree_renderer = TreeRenderer()
        self.tree_analyzer = TreeAnalyzer()
    
    def generate_task1_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate Task 1 samples with specified difficulty"""
        samples = []
        count = 0
        none_answer_num = 0
        none_ratio = 0.1
        while count < num_samples:
            sample = self._generate_single_task1_sample()

            # Control the none ratio to be small
            if sample["expected_answer"] == "None":
                none_answer_num += 1
                if none_answer_num / num_samples > none_ratio:
                    continue

            sample["sample_id"] = count
            sample["difficulty"] = self.difficulty
            samples.append(sample)
            count += 1
        return samples
    
    def _generate_single_task1_sample(self) -> Dict[str, Any]:
        """Generate a single Task 1 sample with difficulty constraints"""
        # Get depth range for current difficulty
        min_depth, max_depth = self.depth_ranges[self.difficulty]
        target_depth = random.randint(min_depth, max_depth)
        
        # Choose tree generation strategy with bias towards fuller trees
        strategy = random.choices(
            ['complete', 'near_complete', 'high_fill'],
            weights=[0.4, 0.5, 0.1]  # 80% will be very full trees
        )[0]
        
        if strategy == 'complete':
            tree = self._generate_complete_tree_exact_depth(target_depth)
            tree_type = "complete"
        elif strategy == 'near_complete':
            tree = self._generate_near_complete_tree(target_depth)
            tree_type = "near_complete"
        else:  # high_fill
            fill_rate = random.uniform(0.75, 0.9)
            tree = self._generate_tree_with_exact_depth(target_depth, fill_rate)
            tree_type = "high_fill"
        
        # Verify the depth is correct
        actual_depth = self.tree_analyzer.get_depth(tree)
        if actual_depth != target_depth:
            # Retry if depth doesn't match
            max_retries = 10
            for _ in range(max_retries):
                tree = self._generate_complete_tree_exact_depth(target_depth)
                actual_depth = self.tree_analyzer.get_depth(tree)
                if actual_depth == target_depth:
                    tree_type = "complete"
                    break
        
        # Randomize the tree number
        tree = self._randomize_node_values(tree)

        tree_str = self.tree_renderer.render_tree(tree)
        
        # Generate question
        # question_type = random.choice(['parent', 'left_child', 'right_child', 'depth'])
        question_type = random.choice(['parent', 'left_child', 'right_child', 'num_nodes'])
        
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
        
        # elif question_type == 'depth':
        #     question = "How many levels does this binary tree have?"
        #     answer = str(self.tree_analyzer.get_depth(tree))
        
        else:  # num_nodes
            question = "How many nodes does this binary tree have?"
            answer = str(self.tree_analyzer.get_node_count(tree))
        
        prompt = self._create_task1_prompt(tree_str, question)
        
        return {
            "task_type": "task1",
            "tree_type": tree_type,
            "tree_depth": self.tree_analyzer.get_depth(tree),
            "tree_structure": tree_str,
            "question": question,
            "question_type": question_type,
            "prompt": prompt,
            "expected_answer": answer
        }
    
    def _generate_complete_tree_exact_depth(self, target_depth: int) -> TreeNode:
        """
        Generate a complete binary tree with exact depth.
        A complete tree has all levels fully filled except possibly the last level.
        """
        if target_depth <= 0:
            return None
        
        # Calculate number of nodes
        min_nodes = 2 ** (target_depth - 1)  # Minimum to reach target depth
        max_nodes = 2 ** target_depth - 1     # Perfect tree
        
        # Choose number of nodes, biased towards fuller trees (90-100% full)
        num_nodes = random.randint(
            min_nodes + (max_nodes - min_nodes) * 9 // 10,
            max_nodes
        )
        
        return self._build_complete_tree(num_nodes)
    
    def _build_complete_tree(self, num_nodes: int) -> TreeNode:
        """Build a complete binary tree with exactly num_nodes nodes using level-order"""
        if num_nodes == 0:
            return None
        
        root = TreeNode(1)
        queue = [root]
        node_counter = 1
        
        while node_counter < num_nodes:
            current = queue.pop(0)
            
            # Add left child
            if node_counter < num_nodes:
                node_counter += 1
                current.left = TreeNode(node_counter)
                queue.append(current.left)
            
            # Add right child
            if node_counter < num_nodes:
                node_counter += 1
                current.right = TreeNode(node_counter)
                queue.append(current.right)
        
        return root
    
    def _generate_near_complete_tree(self, target_depth: int) -> TreeNode:
        """Generate a nearly complete tree with a few nodes removed from last level"""
        # First generate a fuller complete tree
        tree = self._generate_complete_tree_exact_depth(target_depth)
        
        # Find nodes at the last level
        last_level_nodes = self._get_nodes_at_depth(tree, target_depth)
        
        # Remove 10-25% of nodes from last level randomly
        if len(last_level_nodes) > 3:
            num_to_remove = random.randint(1, max(1, len(last_level_nodes) // 6))
            nodes_to_remove = random.sample(last_level_nodes, num_to_remove)
            
            for node_val in nodes_to_remove:
                self._remove_node(tree, node_val)
        
        return tree
    
    def _generate_tree_with_exact_depth(self, target_depth: int, fill_rate: float = 0.75) -> TreeNode:
        """
        Generate a tree with EXACT target depth and controlled fullness
        
        Args:
            target_depth: The exact depth the tree should have
            fill_rate: How full the tree should be (0.0 to 1.0)
        """
        if target_depth <= 0:
            return None
        
        # Start with root
        root = TreeNode(1)
        node_counter = 1
        
        # First, create at least one path to guarantee the exact depth
        current = root
        for level in range(1, target_depth):
            node_counter += 1
            # Randomly choose left or right for the guaranteed path
            if random.random() < 0.5:
                current.left = TreeNode(node_counter)
                current = current.left
            else:
                current.right = TreeNode(node_counter)
                current = current.right
        
        # Now fill in additional nodes level by level with high probability
        # Use BFS to process nodes level by level
        queue = [root]
        
        while queue:
            current_node = queue.pop(0)
            current_depth = self._get_node_depth(root, current_node.val)
            
            # Don't add children if we're at max depth
            if current_depth >= target_depth:
                continue
            
            # Add left child with probability based on fill_rate
            if current_node.left is None:
                if random.random() < fill_rate:
                    node_counter += 1
                    current_node.left = TreeNode(node_counter)
                    queue.append(current_node.left)
            else:
                queue.append(current_node.left)
            
            # Add right child with probability based on fill_rate
            if current_node.right is None:
                if random.random() < fill_rate:
                    node_counter += 1
                    current_node.right = TreeNode(node_counter)
                    queue.append(current_node.right)
            else:
                queue.append(current_node.right)
        
        return root
    
    def _get_node_depth(self, root: TreeNode, target_val: int, current_depth: int = 1) -> int:
        """Helper function to get the depth of a specific node"""
        if root is None:
            return 0
        if root.val == target_val:
            return current_depth
        
        left_depth = self._get_node_depth(root.left, target_val, current_depth + 1)
        if left_depth > 0:
            return left_depth
        
        return self._get_node_depth(root.right, target_val, current_depth + 1)
    
    def _get_nodes_at_depth(self, root: TreeNode, target_depth: int, current_depth: int = 1) -> List[int]:
        """Get all node values at a specific depth"""
        if root is None:
            return []
        
        if current_depth == target_depth:
            return [root.val]
        
        result = []
        result.extend(self._get_nodes_at_depth(root.left, target_depth, current_depth + 1))
        result.extend(self._get_nodes_at_depth(root.right, target_depth, current_depth + 1))
        return result
    
    def _remove_node(self, root: TreeNode, target_val: int) -> bool:
        """Remove a leaf node (make it None in its parent)"""
        if root is None:
            return False
        
        # Check if left child is the target
        if root.left and root.left.val == target_val:
            # Only remove if it's a leaf
            if root.left.left is None and root.left.right is None:
                root.left = None
                return True
        
        # Check if right child is the target
        if root.right and root.right.val == target_val:
            # Only remove if it's a leaf
            if root.right.left is None and root.right.right is None:
                root.right = None
                return True
        
        # Recursively search in subtrees
        return self._remove_node(root.left, target_val) or self._remove_node(root.right, target_val)
    
    def _randomize_node_values(self, root: TreeNode) -> TreeNode:
        """
        Randomly reassign node values while maintaining tree structure
        
        Args:
            root: The root of the tree to randomize
        
        Returns:
            The same tree with randomized node values
        """
        if root is None:
            return None
        
        # Get all nodes in the tree
        all_nodes = self._collect_all_nodes(root)
        num_nodes = len(all_nodes)
        
        # Generate random unique values
        # Option 1: Random permutation of 1 to num_nodes
        # new_values = list(range(1, num_nodes + 1))
        # random.shuffle(new_values)
        
        # Option 2: Random values from a larger range (more random feel)
        new_values = random.sample(range(1, num_nodes * 5), num_nodes)
        
        # Assign new values to nodes
        for i, node in enumerate(all_nodes):
            node.val = new_values[i]
        
        return root

    def _collect_all_nodes(self, root: TreeNode) -> List[TreeNode]:
        """
        Collect all nodes in the tree using level-order traversal
        
        Args:
            root: The root of the tree
        
        Returns:
            List of all TreeNode objects in the tree
        """
        if root is None:
            return []
        
        nodes = []
        queue = [root]
        
        while queue:
            node = queue.pop(0)
            nodes.append(node)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return nodes
    
    def _create_task1_prompt(self, tree_str: str, question: str) -> str:
        """Create prompt for Task 1"""
        return f"""You are given a binary tree structure. Please analyze it carefully and answer the question.

First you need to understand the following fundamental concepts:

1. **Parent Node**
   - A parent node is a node that has one or more child nodes connected below it
   - Every node except the root has exactly one parent
   - The root node has no parent

2. **Left Child Node**
   - The left child is the node connected to the left side of a parent node
   - A node can have at most one left child
   - If a node has no left child, the answer is "None"
   - In tree diagrams, left children are typically shown connected with "/" from the parent

3. **Right Child Node**
   - The right child is the node connected to the right side of a parent node
   - A node can have at most one right child
   - If a node has no right child, the answer is "None"
   - In tree diagrams, right children are typically shown connected with "\\" from the parent

4. **Node Count (Total Number of Nodes)**
   - This is the total count of all nodes in the entire tree
   - Count every node including the root node and all internal and leaf nodes

**Tree Diagram Format:**
- Nodes are represented by numbers (1, 2, 3, etc.)
- "/" represents a connection to a left child
- "\\\\" represents a connection to a right child
- Spacing is used to show the hierarchical structure

**Example Tree:**
```
    1
   / \\
  2   3
 /   / \\
4   5   6
```

In this example:
- Parent of node 4: 2
- Left child of node 1: 2
- Right child of node 2: None
- Tree depth: 3 levels
- Total nodes: 6

Binary Tree Structure:
{tree_str}

Question: {question}

Please provide your answer following this format, that is, put only your anwser after the tag `### My answer is:`.
Note that if there is no such node, answer 'None'.

Following is the **example format**:

---OUTPUT FORMAT---
[Your analysis here if any]
### My answer is:
[Your answer here - just the number or 'None']"""

def main():
    parser = argparse.ArgumentParser(description="Generate Task 1 Binary Tree Questions")
    parser.add_argument("--difficulty", type=str, default="medium",
                       choices=["easy", "medium", "hard"],
                       help="Difficulty level: easy (3-4 levels), medium (5 levels), hard (6 levels)")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of Task 1 samples to generate")
    parser.add_argument("--output_dir", type=str, default="./generated_tasks",
                       help="Output directory for generated tasks")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output filename (if not specified, auto-generated based on difficulty and timestamp)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed generation information")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize generator
    generator = Task1Generator(difficulty=args.difficulty)
    
    if args.verbose:
        print(f"Generating {args.num_samples} Task 1 samples with {args.difficulty} difficulty")
        depth_range = generator.depth_ranges[args.difficulty]
        print(f"Tree depth range: {depth_range[0]}-{depth_range[1]} levels")
        print(f"Tree generation strategy: Biased towards fuller trees (complete and near-complete)")
    
    # Generate samples
    samples = generator.generate_task1_samples(args.num_samples)
    
    # Verify all samples meet depth requirements
    min_depth, max_depth = generator.depth_ranges[args.difficulty]
    invalid_samples = []
    for i, sample in enumerate(samples):
        depth = sample["tree_depth"]
        if depth < min_depth or depth > max_depth:
            invalid_samples.append((i, depth))
    
    if invalid_samples:
        print(f"\nWARNING: Found {len(invalid_samples)} samples with invalid depth:")
        for idx, depth in invalid_samples[:5]:  # Show first 5
            print(f"  Sample {idx}: depth {depth} (expected {min_depth}-{max_depth})")
        print("Regenerating invalid samples...")
        
        # Regenerate invalid samples
        for idx, _ in invalid_samples:
            max_retries = 20
            for retry in range(max_retries):
                new_sample = generator._generate_single_task1_sample()
                if min_depth <= new_sample["tree_depth"] <= max_depth:
                    new_sample["sample_id"] = idx
                    new_sample["difficulty"] = args.difficulty
                    samples[idx] = new_sample
                    break
    
    # Create metadata
    metadata = {
        "generation_info": {
            "difficulty": args.difficulty,
            "depth_range": generator.depth_ranges[args.difficulty],
            "total_samples": len(samples),
            "generation_timestamp": datetime.now().isoformat(),
            "task_type": "task1",
            "tree_generation_strategy": "Biased towards fuller trees (complete, near-complete)"
        },
        "difficulty_distribution": {},
        "question_type_distribution": {},
        "tree_type_distribution": {},
        "samples": samples
    }
    
    # Calculate distributions
    question_types = {}
    tree_types = {}
    depth_counts = {}
    
    for sample in samples:
        # Question type distribution
        qt = sample["question_type"]
        question_types[qt] = question_types.get(qt, 0) + 1
        
        # Tree type distribution
        tt = sample["tree_type"]
        tree_types[tt] = tree_types.get(tt, 0) + 1
        
        # Depth distribution
        depth = sample["tree_depth"]
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
    
    metadata["question_type_distribution"] = question_types
    metadata["tree_type_distribution"] = tree_types
    metadata["depth_distribution"] = depth_counts
    
    # Generate output filename
    if args.output_file:
        output_filename = args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"task1_{args.difficulty}_{len(samples)}samples_{timestamp}.json"
    
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Task 1 Generation Completed!")
    print(f"{'='*60}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Total samples: {len(samples)}")
    print(f"Depth range requirement: {min_depth}-{max_depth}")
    print(f"Actual depth distribution: {depth_counts}")
    print(f"Question type distribution: {question_types}")
    print(f"Tree type distribution: {tree_types}")
    print(f"Output saved to: {output_path}")
    
    # Verify all depths are within range
    all_valid = all(min_depth <= s["tree_depth"] <= max_depth for s in samples)
    if all_valid:
        print(f"\n✓ All samples meet the depth requirement ({min_depth}-{max_depth})")
    else:
        print(f"\n✗ WARNING: Some samples do not meet the depth requirement!")
    
    if args.verbose:
        print(f"\n{'='*60}")
        print(f"First 3 Samples Preview:")
        print(f"{'='*60}")
        for i, sample in enumerate(samples[:3]):
            print(f"\nSample {i+1}:")
            print(f"  Tree type: {sample['tree_type']}")
            print(f"  Depth: {sample['tree_depth']}")
            print(f"  Question: {sample['question']}")
            print(f"  Answer: {sample['expected_answer']}")
            if 'tree_structure' in sample:
                print(f"  Tree structure preview:")
                tree_lines = sample['tree_structure'].split('\n')
                for line in tree_lines:
                    print(f"    {line}")
                # if len(sample['tree_structure'].split('\n')) > 7:
                #     print(f"    ...")

if __name__ == "__main__":
    main()