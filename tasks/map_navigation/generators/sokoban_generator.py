# generators/sokoban_generator.py

import json
import random
import argparse
from typing import List, Dict, Tuple


class SokobanGenerator:
    """Generator for Sokoban environment test cases."""
    
    def __init__(self, size: int = 8):
        """
        Initialize Sokoban generator.
        
        Args:
            size: Size of the N x N grid
        """
        self.size = size
        self.grid = None
        
    def generate_map(self) -> List[List[str]]:
        """
        Generate a random Sokoban map with walls, one P, one X, and one O.
        P cannot be on the edges.
        
        Returns:
            2D grid representing the map
        """
        # Initialize grid with empty spaces
        grid = [['_' for _ in range(self.size)] for _ in range(self.size)]
        
        # Add walls on the border
        for i in range(self.size):
            grid[0][i] = '#'
            grid[self.size-1][i] = '#'
            grid[i][0] = '#'
            grid[i][self.size-1] = '#'
        
        # Add some random internal walls (20% of internal cells)
        internal_positions = [(i, j) for i in range(2, self.size-2) 
                            for j in range(2, self.size-2)]
        num_walls = int(len(internal_positions) * 0.2)
        wall_positions = random.sample(internal_positions, num_walls)
        for i, j in wall_positions:
            grid[i][j] = '#'
        
        # Get available positions (not on edges, not walls)
        available = [(i, j) for i in range(2, self.size-2) 
                    for j in range(2, self.size-2) if grid[i][j] == '_']
        
        # Place P, X, O
        if len(available) < 3:
            # If not enough space, reduce walls
            grid = [['_' for _ in range(self.size)] for _ in range(self.size)]
            for i in range(self.size):
                grid[0][i] = '#'
                grid[self.size-1][i] = '#'
                grid[i][0] = '#'
                grid[i][self.size-1] = '#'
            available = [(i, j) for i in range(2, self.size-2) 
                        for j in range(2, self.size-2)]
        
        special_positions = random.sample(available, 3)
        p_pos, x_pos, o_pos = special_positions
        
        grid[p_pos[0]][p_pos[1]] = 'P'
        grid[x_pos[0]][x_pos[1]] = 'X'
        grid[o_pos[0]][o_pos[1]] = 'O'
        
        self.grid = grid
        return grid
    
    def get_element_at(self, x: int, y: int) -> str:
        """Get element at coordinates (x, y)."""
        return self.grid[y][x]
    
    def find_element(self, element: str) -> Tuple[int, int]:
        """Find coordinates of an element. Returns (x, y)."""
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == element:
                    return (j, i)  # Return (x, y)
        return None
    
    def get_surrounding_elements(self, element: str) -> Dict[str, str]:
        """
        Get 8 surrounding elements of the given element.
        
        Returns:
            Dictionary with directions as keys and elements as values
        """
        pos = self.find_element(element)
        if not pos:
            return {}
        
        x, y = pos
        directions = {
            'up': (x, y-1),
            'down': (x, y+1),
            'left': (x-1, y),
            'right': (x+1, y),
            'up-left': (x-1, y-1),
            'up-right': (x+1, y-1),
            'down-left': (x-1, y+1),
            'down-right': (x+1, y+1),
        }
        
        result = {}
        for direction, (nx, ny) in directions.items():
            if 0 <= nx < self.size and 0 <= ny < self.size:
                result[direction] = self.grid[ny][nx]
            else:
                result[direction] = 'out-of-bounds'
        
        return result
    
    def get_relative_position(self, elem1: str, elem2: str) -> Tuple[int, int]:
        """
        Get relative position from elem1 to elem2.
        
        Returns:
            (dx, dy) where elem2 is at elem1 + (dx, dy)
        """
        pos1 = self.find_element(elem1)
        pos2 = self.find_element(elem2)
        
        if not pos1 or not pos2:
            return None
        
        return (pos2[0] - pos1[0], pos2[1] - pos1[1])
    
    def grid_to_string(self) -> str:
        """Convert grid to string representation."""
        return '\n'.join([' '.join(row) for row in self.grid])
    
    def generate_task_type_1(self, num_samples: int) -> List[Dict]:
        """Generate Task Type 1: Identify element at given coordinates."""
        tasks = []
        for _ in range(num_samples):
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            element = self.get_element_at(x, y)
            
            tasks.append({
                'task_type': 1,
                'question': f'What element is at coordinates ({x}, {y})?',
                'answer': element,
                'coordinates': {'x': x, 'y': y}
            })
        return tasks
    
    def generate_task_type_2(self, num_samples: int) -> List[Dict]:
        """Generate Task Type 2: Find coordinates of special elements."""
        tasks = []
        elements = ['P', 'X', 'O']
        
        for _ in range(num_samples):
            element = random.choice(elements)
            pos = self.find_element(element)
            
            tasks.append({
                'task_type': 2,
                'question': f'What are the coordinates of element {element}?',
                'answer': f'({pos[0]}, {pos[1]})',
                'element': element
            })
        return tasks
    
    def generate_task_type_3(self, num_samples: int) -> List[Dict]:
        """Generate Task Type 3: Identify 8 surrounding elements of P."""
        tasks = []
        surrounding = self.get_surrounding_elements('P')
        
        for _ in range(num_samples):
            tasks.append({
                'task_type': 3,
                'question': 'What are the 8 elements surrounding P (up, down, left, right, up-left, up-right, down-left, down-right)?',
                'answer': json.dumps(surrounding, sort_keys=True),
                'surrounding': surrounding
            })
        return tasks
    
    def generate_task_type_4(self, num_samples: int) -> List[Dict]:
        """Generate Task Type 4: Relative position between X and O."""
        tasks = []
        rel_pos = self.get_relative_position('X', 'O')
        
        for _ in range(num_samples):
            tasks.append({
                'task_type': 4,
                'question': 'What is the relative position from X to O? (Answer in format: (dx, dy))',
                'answer': f'({rel_pos[0]}, {rel_pos[1]})',
                'relative_position': {'dx': rel_pos[0], 'dy': rel_pos[1]}
            })
        return tasks
    
    def generate_dataset(self, 
                        num_maps: int,
                        tasks_per_type: Dict[int, int]) -> List[Dict]:
        """
        Generate complete dataset.
        
        Args:
            num_maps: Number of different maps to generate
            tasks_per_type: Dictionary mapping task type to number of tasks
            
        Returns:
            List of test cases
        """
        dataset = []
        
        for map_id in range(num_maps):
            # Generate new map
            grid = self.generate_map()
            map_str = self.grid_to_string()
            
            # Generate tasks for this map
            map_tasks = []
            
            if 1 in tasks_per_type:
                map_tasks.extend(self.generate_task_type_1(tasks_per_type[1]))
            if 2 in tasks_per_type:
                map_tasks.extend(self.generate_task_type_2(tasks_per_type[2]))
            if 3 in tasks_per_type:
                map_tasks.extend(self.generate_task_type_3(tasks_per_type[3]))
            if 4 in tasks_per_type:
                map_tasks.extend(self.generate_task_type_4(tasks_per_type[4]))
            
            # Add map info to each task
            for task in map_tasks:
                dataset.append({
                    'env_type': 'sokoban',
                    'map_id': map_id,
                    'map_size': self.size,
                    'map': map_str,
                    **task
                })
        
        return dataset


def main():
    parser = argparse.ArgumentParser(description='Generate Sokoban test data')
    parser.add_argument('--size', type=int, default=8, help='Size of the grid (N x N)')
    parser.add_argument('--num-maps', type=int, default=50, help='Number of different maps')
    parser.add_argument('--tasks-type-1', type=int, default=5, help='Number of type 1 tasks per map')
    parser.add_argument('--tasks-type-2', type=int, default=3, help='Number of type 2 tasks per map')
    parser.add_argument('--tasks-type-3', type=int, default=2, help='Number of type 3 tasks per map')
    parser.add_argument('--tasks-type-4', type=int, default=2, help='Number of type 4 tasks per map')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Generate dataset
    generator = SokobanGenerator(size=args.size)
    tasks_per_type = {
        1: args.tasks_type_1,
        2: args.tasks_type_2,
        3: args.tasks_type_3,
        4: args.tasks_type_4,
    }
    
    dataset = generator.generate_dataset(args.num_maps, tasks_per_type)
    
    # Save to file
    output_data = {
        'metadata': {
            'env_type': 'sokoban',
            'map_size': args.size,
            'num_maps': args.num_maps,
            'tasks_per_type': tasks_per_type,
            'total_tasks': len(dataset),
            'seed': args.seed,
        },
        'data': dataset
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Generated {len(dataset)} tasks and saved to {args.output}")


if __name__ == '__main__':
    main()
