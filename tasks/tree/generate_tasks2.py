import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any
from data.tree_generator import TreeGenerator
from utils.tree_utils import TreeRenderer, TreeAnalyzer, TreeNode
import random

class Task2Generator:
    def __init__(self, difficulty: str = "medium"):
        self.difficulty = difficulty
        self.depth_ranges = {
            "easy": (3, 3),
            "medium": (4, 4), 
            "hard": (5, 6)
        }
        self.max_depth = self.depth_ranges[difficulty][1]
        self.tree_generator = TreeGenerator(self.max_depth)
        self.tree_renderer = TreeRenderer()
        self.tree_analyzer = TreeAnalyzer()
    
    def generate_task2_samples(self, num_samples: int, threshold: int) -> List[Dict[str, Any]]:
        """Generate Task 2 samples with specified difficulty"""
        samples = []
        count = 0

        while count < num_samples:
            sample = self._generate_single_task2_sample()
            # filter small path length
            if sample["path_analysis"]["path_length"] < threshold:
                continue
            sample["sample_id"] = count
            sample["difficulty"] = self.difficulty
            samples.append(sample)

            count += 1
        
        return samples
    
    def _generate_single_task2_sample(self) -> Dict[str, Any]:
        """Generate a single Task 2 sample (path analysis)"""
        # Get depth range for current difficulty
        min_depth, max_depth = self.depth_ranges[self.difficulty]
        target_depth = random.randint(min_depth, max_depth)
        
        # Generate tree using similar strategy as Task1
        tree_type = random.choices(
            ['complete', 'near_complete', 'balanced'],
            weights=[0.6, 0.3, 0.1]  # Prefer more structured trees for path analysis
        )[0]
        
        if tree_type == 'complete':
            # Generate complete tree with exact depth
            min_nodes = 2 ** (target_depth - 1)
            max_nodes = 2 ** target_depth - 1
            num_nodes = random.randint(
                min_nodes + (max_nodes - min_nodes) * 3 // 4,  # 75-100% full
                max_nodes
            )
            tree = self._build_complete_tree(num_nodes)
        elif tree_type == 'near_complete':
            # Generate near complete tree
            tree = self._generate_near_complete_tree(target_depth)
        else:  # balanced
            # Generate balanced random tree
            min_nodes = target_depth + 2
            max_nodes = min(20, 2 ** target_depth - 1)
            tree = self.tree_generator.generate_random_tree(min_nodes, max_nodes)
        
        # Verify depth is correct
        actual_depth = self.tree_analyzer.get_depth(tree)
        if actual_depth != target_depth:
            # Fallback to complete tree
            complete_nodes = (2 ** target_depth) - 1
            tree = self._build_complete_tree(complete_nodes)
        
        # Randomize node values
        tree = self._randomize_node_values(tree)
        
        tree_str = self.tree_renderer.render_tree(tree)
        
        # Get all nodes for path selection
        all_nodes = self.tree_analyzer.get_all_nodes(tree)
        
        if len(all_nodes) < 2:
            # Fallback: generate a larger tree
            tree = self._build_complete_tree(7)  # 3-level complete tree
            tree = self._randomize_node_values(tree)
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
            # This shouldn't happen in a connected tree, regenerate
            path_str = f"{source_node} -> {target_node}"  # Fallback
        
        # Create question
        question = f"Find the path from node {source_node} to node {target_node} in the binary tree. Provide the path as a sequence of nodes separated by ' -> '."
        
        # Create prompt
        prompt = self._create_task2_prompt(tree_str, question)
        
        return {
            "task_type": "task2",
            "tree_structure": tree_str,
            "tree_depth": self.tree_analyzer.get_depth(tree),
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
    
    def _build_complete_tree(self, num_nodes: int) -> TreeNode:
        """Build a complete binary tree with exactly num_nodes nodes"""
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
        """Generate a nearly complete tree"""
        # Start with a complete tree
        min_nodes = 2 ** (target_depth - 1)
        max_nodes = 2 ** target_depth - 1
        num_nodes = random.randint(min_nodes, max_nodes - random.randint(1, 3))
        
        return self._build_complete_tree(num_nodes)
    
    def _randomize_node_values(self, root: TreeNode) -> TreeNode:
        """Randomly reassign node values while maintaining tree structure"""
        if not root:
            return None
        
        # Get all nodes in the tree
        all_nodes = self._collect_all_nodes(root)
        num_nodes = len(all_nodes)
        
        # Generate random unique values from a larger range
        new_values = random.sample(range(1, num_nodes * 5), num_nodes)
        
        # Assign new values to nodes
        for i, node in enumerate(all_nodes):
            node.val = new_values[i]
        
        return root
    
    def _collect_all_nodes(self, root: TreeNode) -> List[TreeNode]:
        """Collect all nodes in the tree using level-order traversal"""
        if not root:
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
    
    def _create_task2_prompt(self, tree_str: str, question: str) -> str:
        """Create prompt for Task 2"""
        return f"""You are given a binary tree structure. Please analyze it carefully and find the requested path between two nodes.

**Path Finding Concepts:**

1. **Path Between Nodes**
   - A path between two nodes in a binary tree is the shortest sequence of nodes you must traverse to get from the source node to the target node

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

Please provide your answer following this format:

---OUTPUT FORMAT---
[Your analysis here if any]
### My answer is:
[The path in the format: node1 -> node2 -> ... -> nodeN]"""

def main():
    parser = argparse.ArgumentParser(description="Generate Task 2 Binary Tree Path Analysis Questions")
    parser.add_argument("--difficulty", type=str, default="medium",
                       choices=["easy", "medium", "hard"],
                       help="Difficulty level: easy (3 levels), medium (4 levels), hard (5-6 levels)")
    parser.add_argument("--threshold", type=int, default=3,
                       help="the minimum length of generated path tasks")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of Task 2 samples to generate")
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
    generator = Task2Generator(difficulty=args.difficulty)
    
    if args.verbose:
        print(f"Generating {args.num_samples} Task 2 samples with {args.difficulty} difficulty")
        depth_range = generator.depth_ranges[args.difficulty]
        print(f"Tree depth range: {depth_range[0]}-{depth_range[1]} levels")
        print(f"Task type: Binary tree path analysis")
    
    # Generate samples
    samples = generator.generate_task2_samples(args.num_samples, args.threshold)
    
    # Verify samples
    min_depth, max_depth = generator.depth_ranges[args.difficulty]
    invalid_samples = []
    
    for i, sample in enumerate(samples):
        depth = sample["tree_depth"]
        if depth < min_depth or depth > max_depth:
            invalid_samples.append((i, f"depth_{depth}"))
        
        # Check if path is valid
        if not sample["path_analysis"]["path"]:
            invalid_samples.append((i, "empty_path"))
    
    if invalid_samples and args.verbose:
        print(f"\nWARNING: Found {len(invalid_samples)} problematic samples:")
        for idx, issue in invalid_samples[:3]:
            print(f"  Sample {idx}: {issue}")
    
    # Create metadata
    metadata = {
        "generation_info": {
            "difficulty": args.difficulty,
            "depth_range": generator.depth_ranges[args.difficulty],
            "total_samples": len(samples),
            "generation_timestamp": datetime.now().isoformat(),
            "task_type": "task2",
            "task_description": "Binary tree path analysis - finding paths between two nodes"
        },
        "difficulty_distribution": {},
        "path_length_distribution": {},
        "depth_distribution": {},
        "samples": samples
    }
    
    # Calculate distributions
    depth_counts = {}
    path_lengths = {}
    
    for sample in samples:
        # Depth distribution
        depth = sample["tree_depth"]
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        # Path length distribution
        path_len = sample["path_analysis"]["path_length"]
        path_lengths[path_len] = path_lengths.get(path_len, 0) + 1
    
    metadata["depth_distribution"] = depth_counts
    metadata["path_length_distribution"] = path_lengths
    
    # Generate output filename
    if args.output_file:
        output_filename = args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"task2_{args.difficulty}_{len(samples)}samples_{timestamp}.json"
    
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Task 2 Generation Completed!")
    print(f"{'='*60}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Total samples: {len(samples)}")
    print(f"Depth range requirement: {min_depth}-{max_depth}")
    print(f"Actual depth distribution: {depth_counts}")
    print(f"Path length distribution: {path_lengths}")
    print(f"Output saved to: {output_path}")
    
    # Verify all depths are within range
    all_valid = all(min_depth <= s["tree_depth"] <= max_depth for s in samples)
    if all_valid:
        print(f"\n✓ All samples meet the depth requirement ({min_depth}-{max_depth})")
    else:
        print(f"\n✗ WARNING: Some samples do not meet the depth requirement!")
    
    # Verify paths
    valid_paths = sum(1 for s in samples if s["path_analysis"]["path"])
    print(f"✓ {valid_paths}/{len(samples)} samples have valid paths")
    
    if args.verbose:
        print(f"\n{'='*60}")
        print(f"First 3 Samples Preview:")
        print(f"{'='*60}")
        for i, sample in enumerate(samples[:3]):
            print(f"\nSample {i+1}:")
            print(f"  Tree depth: {sample['tree_depth']}")
            print(f"  Source → Target: {sample['source_node']} → {sample['target_node']}")
            print(f"  Expected path: {sample['expected_answer']}")
            print(f"  Path length: {sample['path_analysis']['path_length']}")
            if 'tree_structure' in sample:
                print(f"  Tree structure preview:")
                tree_lines = sample['tree_structure'].split('\n')
                for line in tree_lines:
                    print(f"    {line}")
                # if len(sample['tree_structure'].split('\n')) > 7:
                #     print(f"    ...")
                
if __name__ == "__main__":
    main()
