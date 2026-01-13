import unittest
import json
from generators.sokoban_generator import SokobanGenerator
from generators.frozenlake_generator import FrozenLakeGenerator


class TestSokobanGenerator(unittest.TestCase):
    """Test cases for Sokoban generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = SokobanGenerator(size=8)
    
    def test_map_generation(self):
        """Test basic map generation."""
        grid = self.generator.generate_map()
        
        # Check grid size
        self.assertEqual(len(grid), 8)
        self.assertEqual(len(grid[0]), 8)
        
        # Check borders are walls
        for i in range(8):
            self.assertEqual(grid[0][i], '#')
            self.assertEqual(grid[7][i], '#')
            self.assertEqual(grid[i][0], '#')
            self.assertEqual(grid[i][7], '#')
    
    def test_special_elements_present(self):
        """Test that P, X, O are present in the map."""
        grid = self.generator.generate_map()
        
        p_count = sum(row.count('P') for row in grid)
        x_count = sum(row.count('X') for row in grid)
        o_count = sum(row.count('O') for row in grid)
        
        self.assertEqual(p_count, 1, "Should have exactly 1 P")
        self.assertEqual(x_count, 1, "Should have exactly 1 X")
        self.assertEqual(o_count, 1, "Should have exactly 1 O")
    
    def test_p_not_on_edges(self):
        """Test that P is not on the edges."""
        for _ in range(10):  # Test multiple generations
            grid = self.generator.generate_map()
            p_pos = self.generator.find_element('P')
            
            self.assertIsNotNone(p_pos)
            x, y = p_pos
            
            # P should not be on edges
            self.assertGreater(x, 0)
            self.assertLess(x, 7)
            self.assertGreater(y, 0)
            self.assertLess(y, 7)
    
    def test_find_element(self):
        """Test element finding."""
        grid = self.generator.generate_map()
        
        p_pos = self.generator.find_element('P')
        self.assertIsNotNone(p_pos)
        
        x, y = p_pos
        self.assertEqual(self.generator.get_element_at(x, y), 'P')
    
    def test_get_element_at(self):
        """Test getting element at coordinates."""
        grid = self.generator.generate_map()
        
        # Check a wall position
        elem = self.generator.get_element_at(0, 0)
        self.assertEqual(elem, '#')
    
    def test_get_surrounding_elements(self):
        """Test getting surrounding elements."""
        grid = self.generator.generate_map()
        surrounding = self.generator.get_surrounding_elements('P')
        
        # Should have 8 directions
        expected_directions = ['up', 'down', 'left', 'right', 
                             'up-left', 'up-right', 'down-left', 'down-right']
        for direction in expected_directions:
            self.assertIn(direction, surrounding)
    
    def test_get_relative_position(self):
        """Test relative position calculation."""
        grid = self.generator.generate_map()
        
        x_pos = self.generator.find_element('X')
        o_pos = self.generator.find_element('O')
        rel_pos = self.generator.get_relative_position('X', 'O')
        
        # Verify relative position
        expected_dx = o_pos[0] - x_pos[0]
        expected_dy = o_pos[1] - x_pos[1]
        
        self.assertEqual(rel_pos[0], expected_dx)
        self.assertEqual(rel_pos[1], expected_dy)
    
    def test_task_type_1_generation(self):
        """Test task type 1 generation."""
        grid = self.generator.generate_map()
        tasks = self.generator.generate_task_type_1(5)
        
        self.assertEqual(len(tasks), 5)
        for task in tasks:
            self.assertEqual(task['task_type'], 1)
            self.assertIn('question', task)
            self.assertIn('answer', task)
            self.assertIn('coordinates', task)
    
    def test_task_type_2_generation(self):
        """Test task type 2 generation."""
        grid = self.generator.generate_map()
        tasks = self.generator.generate_task_type_2(3)
        
        self.assertEqual(len(tasks), 3)
        for task in tasks:
            self.assertEqual(task['task_type'], 2)
            self.assertIn('element', task)
            self.assertIn(task['element'], ['P', 'X', 'O'])
    
    def test_dataset_generation(self):
        """Test complete dataset generation."""
        tasks_per_type = {1: 2, 2: 1, 3: 1, 4: 1}
        dataset = self.generator.generate_dataset(3, tasks_per_type)
        
        # Should have 3 maps * 5 tasks = 15 tasks
        self.assertEqual(len(dataset), 15)
        
        # Check task types distribution
        type_counts = {}
        for task in dataset:
            task_type = task['task_type']
            type_counts[task_type] = type_counts.get(task_type, 0) + 1
        
        self.assertEqual(type_counts[1], 6)  # 3 maps * 2 tasks
        self.assertEqual(type_counts[2], 3)  # 3 maps * 1 task


class TestFrozenLakeGenerator(unittest.TestCase):
    """Test cases for FrozenLake generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = FrozenLakeGenerator(size=8, num_holes=5)
    
    def test_map_generation(self):
        """Test basic map generation."""
        grid = self.generator.generate_map()
        
        # Check grid size
        self.assertEqual(len(grid), 8)
        self.assertEqual(len(grid[0]), 8)
    
    def test_special_elements_present(self):
        """Test that P and G are present."""
        grid = self.generator.generate_map()
        
        p_count = sum(row.count('P') for row in grid)
        g_count = sum(row.count('G') for row in grid)
        
        self.assertEqual(p_count, 1, "Should have exactly 1 P")
        self.assertEqual(g_count, 1, "Should have exactly 1 G")
    
    def test_holes_present(self):
        """Test that holes are present."""
        grid = self.generator.generate_map()
        o_count = sum(row.count('O') for row in grid)
        
        # Should have requested number of holes (or close to it)
        self.assertGreaterEqual(o_count, 1)
        self.assertLessEqual(o_count, self.generator.num_holes)
    
    def test_p_not_on_edges(self):
        """Test that P is not on the edges."""
        for _ in range(10):
            grid = self.generator.generate_map()
            p_pos = self.generator.find_element('P')
            
            self.assertIsNotNone(p_pos)
            x, y = p_pos
            
            # P should not be on edges
            self.assertGreater(x, 0)
            self.assertLess(x, 7)
            self.assertGreater(y, 0)
            self.assertLess(y, 7)
    
    def test_count_holes(self):
        """Test hole counting."""
        grid = self.generator.generate_map()
        count = self.generator.count_holes()
        
        # Manually count
        expected = sum(row.count('O') for row in grid)
        self.assertEqual(count, expected)
    
    def test_task_type_5_generation(self):
        """Test task type 5 (count holes) generation."""
        grid = self.generator.generate_map()
        tasks = self.generator.generate_task_type_5(2)
        
        self.assertEqual(len(tasks), 2)
        for task in tasks:
            self.assertEqual(task['task_type'], 5)
            self.assertIn('num_holes', task)
            
            # Answer should be string representation of count
            expected_count = self.generator.count_holes()
            self.assertEqual(task['answer'], str(expected_count))
    
    def test_dataset_generation(self):
        """Test complete dataset generation."""
        tasks_per_type = {1: 2, 2: 1, 3: 1, 4: 1, 5: 1}
        dataset = self.generator.generate_dataset(3, tasks_per_type)
        
        # Should have 3 maps * 6 tasks = 18 tasks
        self.assertEqual(len(dataset), 18)
        
        # All should be frozenlake env
        for task in dataset:
            self.assertEqual(task['env_type'], 'frozenlake')


if __name__ == '__main__':
    unittest.main()