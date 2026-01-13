"""
Board generator for creating valid Gomoku game states.
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Set, Any
from enum import Enum

class GameState(Enum):
    WHITE_WINS = "WHITE_WINS"
    BLACK_WINS = "BLACK_WINS"
    NO_WINNER = "NO_WINNER"
    INVALID_BOARD = "INVALID_BOARD"  

class WinDirection(Enum):
    HORIZONTAL = "HORIZONTAL"
    VERTICAL = "VERTICAL"
    DIAGONAL_DOWN = "DIAGONAL_DOWN"  # \
    DIAGONAL_UP = "DIAGONAL_UP"      # /

class BoardGenerator:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.board = np.full((board_size, board_size), 'E', dtype=str)
        
    def reset_board(self):
        """Reset the board to empty state"""
        self.board = np.full((self.board_size, self.board_size), 'E', dtype=str)
    
    def place_stone(self, row: int, col: int, stone: str) -> bool:
        """Place a stone on the board if position is empty"""
        if self.board[row, col] == 'E':
            self.board[row, col] = stone
            return True
        return False
    
    def get_board_density(self) -> float:
        """
        Calculate current board density (ratio of occupied positions to total positions)
        
        Returns:
            Float between 0 and 1 representing the density
        """
        total_positions = self.board_size * self.board_size
        occupied_positions = np.sum(self.board != 'E')
        return occupied_positions / total_positions
    
    def check_winner(self) -> GameState:
        """
        Check if there's a winner on the current board
        Ensures there is AT MOST ONE winner
        """
        state, _ = self.check_winner_with_direction()
        return state
    
    def check_winner_with_direction(self) -> Tuple[GameState, Optional[WinDirection]]:
        """
        Check if there's a winner and return the winning direction
        
        **IMPORTANT**: Ensures there is EXACTLY ONE winner (or no winner)
        - If both players have 5-in-a-row: returns INVALID_BOARD
        - If only one player has 5-in-a-row: returns that player's win
        - If one player has multiple winning lines: returns the first direction found
        
        Returns:
            Tuple of (GameState, WinDirection or None)
        """
        direction_map = {
            (0, 1): WinDirection.HORIZONTAL,
            (1, 0): WinDirection.VERTICAL,
            (1, 1): WinDirection.DIAGONAL_DOWN,
            (1, -1): WinDirection.DIAGONAL_UP
        }
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # Track all winners found
        white_wins = False
        black_wins = False
        first_win_direction = None
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] != 'E':
                    stone = self.board[row, col]
                    
                    for dr, dc in directions:
                        count = 1
                        # Check forward direction
                        r, c = row + dr, col + dc
                        while (0 <= r < self.board_size and 0 <= c < self.board_size and 
                               self.board[r, c] == stone):
                            count += 1
                            r, c = r + dr, c + dc
                            
                        # Check backward direction
                        r, c = row - dr, col - dc
                        while (0 <= r < self.board_size and 0 <= c < self.board_size and 
                               self.board[r, c] == stone):
                            count += 1
                            r, c = r - dr, c - dc
                        
                        if count >= 5:
                            # Found a winning line
                            if stone == 'W':
                                if not white_wins:
                                    white_wins = True
                                    if first_win_direction is None:
                                        first_win_direction = direction_map[(dr, dc)]
                            else:  # stone == 'B'
                                if not black_wins:
                                    black_wins = True
                                    if first_win_direction is None:
                                        first_win_direction = direction_map[(dr, dc)]
                            
                            # Early detection of invalid board
                            if white_wins and black_wins:
                                return GameState.INVALID_BOARD, None
        
        # Determine final state
        if white_wins and black_wins:
            return GameState.INVALID_BOARD, None
        elif white_wins:
            return GameState.WHITE_WINS, first_win_direction
        elif black_wins:
            return GameState.BLACK_WINS, first_win_direction
        else:
            return GameState.NO_WINNER, None
    
    def get_all_winning_lines(self) -> Dict[str, List[Tuple[WinDirection, List[Tuple[int, int]]]]]:
        """
        Find ALL winning lines on the board (for debugging/analysis)
        
        Returns:
            Dictionary mapping player ('W' or 'B') to list of winning lines
            Each winning line is (direction, positions)
        """
        direction_map = {
            (0, 1): WinDirection.HORIZONTAL,
            (1, 0): WinDirection.VERTICAL,
            (1, 1): WinDirection.DIAGONAL_DOWN,
            (1, -1): WinDirection.DIAGONAL_UP
        }
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        winning_lines = {'W': [], 'B': []}
        checked_lines = set()  # To avoid counting the same line multiple times
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] != 'E':
                    stone = self.board[row, col]
                    
                    for dr, dc in directions:
                        # Create a unique identifier for this potential line
                        # Use the "canonical" starting position (leftmost/topmost point)
                        canonical_start = self._get_canonical_line_start(row, col, dr, dc, stone)
                        line_id = (stone, canonical_start, (dr, dc))
                        
                        if line_id in checked_lines:
                            continue
                        
                        # Count and collect positions
                        positions = [(row, col)]
                        count = 1
                        
                        # Check forward direction
                        r, c = row + dr, col + dc
                        while (0 <= r < self.board_size and 0 <= c < self.board_size and 
                               self.board[r, c] == stone):
                            positions.append((r, c))
                            count += 1
                            r, c = r + dr, c + dc
                            
                        # Check backward direction
                        r, c = row - dr, col - dc
                        while (0 <= r < self.board_size and 0 <= c < self.board_size and 
                               self.board[r, c] == stone):
                            positions.insert(0, (r, c))
                            count += 1
                            r, c = r - dr, c - dc
                        
                        if count >= 5:
                            winning_lines[stone].append((direction_map[(dr, dc)], positions))
                            checked_lines.add(line_id)
        
        return winning_lines
    
    def _get_canonical_line_start(self, row: int, col: int, dr: int, dc: int, stone: str) -> Tuple[int, int]:
        """Get the canonical (leftmost/topmost) starting position of a line"""
        # Go backwards as far as possible
        while True:
            new_r, new_c = row - dr, col - dc
            if (0 <= new_r < self.board_size and 
                0 <= new_c < self.board_size and 
                self.board[new_r, new_c] == stone):
                row, col = new_r, new_c
            else:
                break
        return (row, col)
    
    def has_only_one_winning_direction(
        self, 
        winner: str, 
        expected_direction: WinDirection
    ) -> bool:
        """
        Check if the winner has EXACTLY ONE winning direction and it matches expected
        
        Args:
            winner: 'W' or 'B'
            expected_direction: The expected winning direction
        
        Returns:
            True if winner has only one winning line in the expected direction
        """
        all_wins = self.get_all_winning_lines()
        winner_lines = all_wins[winner]
        
        if len(winner_lines) == 0:
            return False
        
        # Check all winning lines are in the same direction
        directions_found = set(direction for direction, _ in winner_lines)
        
        if len(directions_found) != 1:
            return False
        
        # Check the direction matches expected
        return directions_found.pop() == expected_direction
    
    def verify_board_state(
        self, 
        expected_winner: Optional[str] = None,
        expected_direction: Optional[WinDirection] = None,
        allow_no_winner: bool = False,
        strict_direction: bool = True
    ) -> bool:
        """
        Verify that the board state matches expectations STRICTLY
        
        Args:
            expected_winner: 'W', 'B', or None (for no winner)
            expected_direction: Expected winning direction (if applicable)
            allow_no_winner: Whether NO_WINNER state is acceptable
            strict_direction: If True, ensures ONLY the expected direction wins
        
        Returns:
            True if board state matches expectations, False otherwise
        """
        state, direction = self.check_winner_with_direction()
        
        # Reject invalid boards
        if state == GameState.INVALID_BOARD:
            return False
        
        # Check for no winner case
        if expected_winner is None:
            if state != GameState.NO_WINNER:
                return False
            # Ensure NO player has any winning lines
            all_wins = self.get_all_winning_lines()
            if len(all_wins['W']) > 0 or len(all_wins['B']) > 0:
                return False
            return True
        
        # Check winner matches
        expected_state = GameState.WHITE_WINS if expected_winner == 'W' else GameState.BLACK_WINS
        if state != expected_state:
            return False
        
        # Check opponent does NOT win
        opponent = 'B' if expected_winner == 'W' else 'W'
        all_wins = self.get_all_winning_lines()
        if len(all_wins[opponent]) > 0:
            return False
        
        # Check direction matches (if specified)
        if expected_direction is not None:
            if direction != expected_direction:
                return False
            
            # If strict, ensure ONLY this direction wins
            if strict_direction:
                if not self.has_only_one_winning_direction(expected_winner, expected_direction):
                    return False
        
        return True
    
    def will_create_five_in_row(
        self, 
        row: int, 
        col: int, 
        stone: str
    ) -> Tuple[bool, Optional[WinDirection]]:
        """
        Check if placing a stone at (row, col) would create a 5-in-a-row
        
        Args:
            row: Row position
            col: Column position
            stone: 'W' or 'B'
        
        Returns:
            Tuple of (will_create, direction)
        """
        if self.board[row, col] != 'E':
            return False, None
        
        direction_map = {
            (0, 1): WinDirection.HORIZONTAL,
            (1, 0): WinDirection.VERTICAL,
            (1, 1): WinDirection.DIAGONAL_DOWN,
            (1, -1): WinDirection.DIAGONAL_UP
        }
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            
            # Check forward direction
            r, c = row + dr, col + dc
            while (0 <= r < self.board_size and 0 <= c < self.board_size and 
                   self.board[r, c] == stone):
                count += 1
                r, c = r + dr, c + dc
            
            # Check backward direction
            r, c = row - dr, col - dc
            while (0 <= r < self.board_size and 0 <= c < self.board_size and 
                   self.board[r, c] == stone):
                count += 1
                r, c = r - dr, c - dc
            
            if count >= 5:
                return True, direction_map[(dr, dc)]
        
        return False, None
    
    def generate_winning_board(
        self, 
        winner: str, 
        direction: Optional[WinDirection] = None,
        density: float = 0.3,
        max_attempts: int = 2000
    ) -> np.ndarray:
        """
        Generate a board where ONLY the specified player wins in EXACTLY the specified direction
        
        **STRICT GUARANTEES**:
        1. Only the specified player wins
        2. The win is ONLY in the specified direction (no other directions)
        3. The opponent has NO winning lines
        4. Density is close to target
        
        Args:
            winner: 'W' or 'B'
            direction: Specific winning direction (required)
            density: Target density of the board (0.0 to 1.0)
            max_attempts: Maximum number of generation attempts
        
        Returns:
            Generated board as numpy array
        """
        self.reset_board()
        
        # Clamp density to valid range
        density = max(0.0, min(1.0, density))
        
        # Direction is required for strict generation
        if direction is None:
            direction = random.choice(list(WinDirection))
        
        for attempt in range(max_attempts):
            self.reset_board()
            
            # Step 1: Place the winning line
            if not self._place_winning_line(winner, direction):
                continue
            
            # Step 2: Verify initial winning line
            state, actual_direction = self.check_winner_with_direction()
            expected_state = GameState.WHITE_WINS if winner == 'W' else GameState.BLACK_WINS
            
            if state != expected_state or actual_direction != direction:
                continue
            
            # Step 3: Add random stones VERY CAREFULLY
            success = self._add_random_stones_strict(winner, direction, density)
            if not success:
                continue
            
            # Step 4: COMPREHENSIVE VERIFICATION
            if self.verify_board_state(winner, direction, strict_direction=True):
                current_density = self.get_board_density()
                
                # Check density is acceptable (within 10% tolerance)
                if abs(current_density - density) <= 0.1:
                    return self.board.copy()
        
        # Fallback: create a minimal winning board
        self.reset_board()
        self._place_winning_line(winner, direction)
        
        # Try to add stones to fallback board
        self._add_random_stones_strict(winner, direction, density)
        
        # Verify fallback
        if not self.verify_board_state(winner, direction, strict_direction=True):
            # If even fallback fails, return minimal board
            self.reset_board()
            self._place_winning_line(winner, direction)
        
        return self.board.copy()
    
    def generate_no_winner_board(
        self, 
        density: float = 0.3,
        max_attempts: int = 2000
    ) -> np.ndarray:
        """
        Generate a board with NO winner (neither player has 5-in-a-row)
        
        **STRICT GUARANTEES**:
        1. No player has any 5-in-a-row
        2. Density is close to target
        
        Args:
            density: Target density of the board (0.0 to 1.0)
            max_attempts: Maximum number of generation attempts
        
        Returns:
            Generated board as numpy array
        """
        self.reset_board()
        
        # Clamp density to valid range
        density = max(0.0, min(1.0, density))
        
        for attempt in range(max_attempts):
            self.reset_board()
            self._add_random_stones_no_winner_strict(density)
            
            # STRICT VERIFICATION
            state, _ = self.check_winner_with_direction()
            current_density = self.get_board_density()
            
            # Check 1: No winner (and not invalid)
            if state != GameState.NO_WINNER:
                continue
            
            # Check 2: Verify no winning lines exist
            all_wins = self.get_all_winning_lines()
            if len(all_wins['W']) > 0 or len(all_wins['B']) > 0:
                continue
            
            # Check 3: Density is acceptable
            if abs(current_density - density) <= 0.1:
                return self.board.copy()
        
        # Fallback: return board with approximate density
        self.reset_board()
        self._add_random_stones_no_winner_strict(density)
        
        return self.board.copy()
    
    def _place_winning_line(
        self, 
        winner: str, 
        direction: WinDirection
    ) -> bool:
        """
        Place a winning line of EXACTLY 5 stones in the specified direction
        
        **Important**: Places exactly 5 stones, not more, to avoid multiple winning lines
        
        Args:
            winner: 'W' or 'B'
            direction: Winning direction
        
        Returns:
            True if successfully placed, False otherwise
        """
        # Map direction to delta values
        direction_map = {
            WinDirection.HORIZONTAL: (0, 1),
            WinDirection.VERTICAL: (1, 0),
            WinDirection.DIAGONAL_DOWN: (1, 1),
            WinDirection.DIAGONAL_UP: (1, -1)
        }
        
        dr, dc = direction_map[direction]
        
        # Try multiple times to find a valid position
        for _ in range(200):
            # Find a valid starting position based on direction
            if direction == WinDirection.HORIZONTAL:
                start_row = random.randint(0, self.board_size - 1)
                start_col = random.randint(0, self.board_size - 5)
            elif direction == WinDirection.VERTICAL:
                start_row = random.randint(0, self.board_size - 5)
                start_col = random.randint(0, self.board_size - 1)
            elif direction == WinDirection.DIAGONAL_DOWN:
                start_row = random.randint(0, self.board_size - 5)
                start_col = random.randint(0, self.board_size - 5)
            elif direction == WinDirection.DIAGONAL_UP:
                start_row = random.randint(0, self.board_size - 5)
                start_col = random.randint(4, self.board_size - 1)
            else:
                continue
            
            # Calculate positions for exactly 5 stones
            positions = []
            for i in range(5):
                row = start_row + i * dr
                col = start_col + i * dc
                positions.append((row, col))
            
            # Check all positions are valid and empty
            if not all(0 <= r < self.board_size and 0 <= c < self.board_size and 
                      self.board[r, c] == 'E' for r, c in positions):
                continue
            
            # Check that placing these stones won't extend beyond 5
            # Check before first stone
            before_r = start_row - dr
            before_c = start_col - dc
            if (0 <= before_r < self.board_size and 0 <= before_c < self.board_size and 
                self.board[before_r, before_c] == winner):
                continue
            
            # Check after last stone
            after_r = start_row + 5 * dr
            after_c = start_col + 5 * dc
            if (0 <= after_r < self.board_size and 0 <= after_c < self.board_size and 
                self.board[after_r, after_c] == winner):
                continue
            
            # Place the stones
            for row, col in positions:
                self.board[row, col] = winner
            
            return True
        
        return False
    
    def _add_random_stones_strict(
        self, 
        winner: str, 
        winning_direction: WinDirection,
        target_density: float = 0.3
    ) -> bool:
        """
        Add random stones with STRICT validation to prevent:
        1. Opponent from getting any 5-in-a-row
        2. Winner from getting 5-in-a-row in other directions
        3. Creating invalid board states
        
        Args:
            winner: The winning player ('W' or 'B')
            winning_direction: The direction in which winner should win
            target_density: Target board density (0.0 to 1.0)
        
        Returns:
            True if successfully added stones, False otherwise
        """
        opponent = 'B' if winner == 'W' else 'W'
        
        # Calculate how many stones we need to add
        total_positions = self.board_size * self.board_size
        target_stones = int(total_positions * target_density)
        current_stones = np.sum(self.board != 'E')
        stones_to_add = max(0, target_stones - current_stones)
        
        # Get empty positions
        empty_positions = [(r, c) for r in range(self.board_size) 
                          for c in range(self.board_size) if self.board[r, c] == 'E']
        
        if not empty_positions:
            return True
        
        # Limit stones to add by available positions
        stones_to_add = min(stones_to_add, len(empty_positions))
        
        # Shuffle positions for randomness
        random.shuffle(empty_positions)
        
        added_count = 0
        
        for row, col in empty_positions:
            if added_count >= stones_to_add:
                break
            
            # Decide which stone to place (slightly favor opponent for realism)
            stone = random.choices(
                [winner, opponent],
                weights=[0.4, 0.6]
            )[0]
            
            # Save current state
            original_board = self.board.copy()
            self.board[row, col] = stone
            
            # Check if this placement is safe
            is_safe = self._is_placement_safe(row, col, stone, winner, winning_direction)
            
            if is_safe:
                added_count += 1
            else:
                # Revert the placement
                self.board = original_board
        
        return True
    
    def _is_placement_safe(
        self,
        row: int,
        col: int,
        stone: str,
        expected_winner: str,
        expected_direction: WinDirection
    ) -> bool:
        """
        Check if placing a stone at (row, col) maintains board validity
        
        Args:
            row: Row position
            col: Column position
            stone: Stone that was just placed ('W' or 'B')
            expected_winner: The player who should win
            expected_direction: The direction in which they should win
        
        Returns:
            True if placement is safe, False otherwise
        """
        # Check 1: Verify board state is still correct
        state, actual_direction = self.check_winner_with_direction()
        
        # Board must not be invalid
        if state == GameState.INVALID_BOARD:
            return False
        
        # Winner must be correct
        expected_state = GameState.WHITE_WINS if expected_winner == 'W' else GameState.BLACK_WINS
        if state != expected_state:
            return False
        
        # Direction must be correct
        if actual_direction != expected_direction:
            return False
        
        # Check 2: Ensure ONLY expected direction wins
        all_wins = self.get_all_winning_lines()
        
        # Opponent must have NO wins
        opponent = 'B' if expected_winner == 'W' else 'W'
        if len(all_wins[opponent]) > 0:
            return False
        
        # Winner must have wins ONLY in expected direction
        winner_lines = all_wins[expected_winner]
        if len(winner_lines) == 0:
            return False
        
        for direction, _ in winner_lines:
            if direction != expected_direction:
                return False
        
        return True
    
    def _add_random_stones_no_winner_strict(self, target_density: float = 0.3):
        """
        Add random stones ensuring NO winner emerges (STRICT version)
        
        Strategy: Place stones carefully, avoiding any 5-in-a-row formations
        
        Args:
            target_density: Target board density (0.0 to 1.0)
        """
        # Calculate how many stones we need
        total_positions = self.board_size * self.board_size
        target_stones = int(total_positions * target_density)
        
        # Get all empty positions
        empty_positions = [(r, c) for r in range(self.board_size) 
                          for c in range(self.board_size)]
        
        # Shuffle for randomness
        random.shuffle(empty_positions)
        
        # Limit to available positions
        target_stones = min(target_stones, len(empty_positions))
        
        added_count = 0
        
        for row, col in empty_positions:
            if added_count >= target_stones:
                break
            
            stone = random.choice(['W', 'B'])
            
            # Check if placing this stone would create a 5-in-a-row
            original_board = self.board.copy()
            self.board[row, col] = stone
            
            # Verify no winner
            state, _ = self.check_winner_with_direction()
            all_wins = self.get_all_winning_lines()
            
            if state == GameState.NO_WINNER and len(all_wins['W']) == 0 and len(all_wins['B']) == 0:
                added_count += 1
            else:
                # Revert - this placement creates a winner
                self.board = original_board
        
        return True
    
    def board_to_string(self) -> str:
        """Convert board to string representation"""
        return '\n'.join(''.join(row) for row in self.board)


def generate_test_cases(
    board_size: int,
    num_cases: int,
    seed: int = None,
    outcome_distribution: Optional[Dict[str, float]] = None,
    direction_distribution: Optional[Dict[str, float]] = None,
    density_range: Tuple[float, float] = (0.2, 0.5),
    verbose: bool = False
) -> List[Tuple[str, str, Optional[str], float]]:
    """
    Generate test cases for the benchmark with STRICT verification
    
    **STRICT GUARANTEES**:
    - Each winning board has EXACTLY ONE winner
    - Each winning board wins in EXACTLY the SPECIFIED direction (no other directions)
    - Opponent has NO winning lines
    - No-winner boards have NO winners (neither player)
    - No INVALID boards (both players winning simultaneously)
    
    Args:
        board_size: Size of the board
        num_cases: Number of test cases to generate
        seed: Random seed for reproducibility
        outcome_distribution: Distribution of outcomes
            e.g., {"WHITE_WINS": 0.33, "BLACK_WINS": 0.33, "NO_WINNER": 0.34}
        direction_distribution: Distribution of winning directions
            e.g., {"HORIZONTAL": 0.25, "VERTICAL": 0.25, "DIAGONAL_DOWN": 0.25, "DIAGONAL_UP": 0.25}
        density_range: Tuple of (min_density, max_density) for board density
            e.g., (0.2, 0.5) means 20% to 50% of board filled
            Default: (0.2, 0.5)
        verbose: Print verification information
    
    Returns:
        List of (board_string, expected_result, direction, density) tuples
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Validate density range
    min_density, max_density = density_range
    min_density = max(0.0, min(1.0, min_density))
    max_density = max(0.0, min(1.0, max_density))
    if min_density > max_density:
        min_density, max_density = max_density, min_density
    
    # Default distributions
    if outcome_distribution is None:
        outcome_distribution = {
            "WHITE_WINS": 1/3,
            "BLACK_WINS": 1/3,
            "NO_WINNER": 1/3
        }
    
    if direction_distribution is None:
        direction_distribution = {
            "HORIZONTAL": 0.25,
            "VERTICAL": 0.25,
            "DIAGONAL_DOWN": 0.25,
            "DIAGONAL_UP": 0.25
        }
    
    # Normalize distributions
    outcome_total = sum(outcome_distribution.values())
    outcome_distribution = {k: v/outcome_total for k, v in outcome_distribution.items()}
    
    direction_total = sum(direction_distribution.values())
    direction_distribution = {k: v/direction_total for k, v in direction_distribution.items()}
    
    # Calculate number of cases for each outcome
    outcomes = list(outcome_distribution.keys())
    outcome_probs = list(outcome_distribution.values())
    
    # Generate random outcomes based on distribution
    outcome_counts = {}
    remaining = num_cases
    
    for i, outcome in enumerate(outcomes[:-1]):
        count = int(num_cases * outcome_probs[i])
        outcome_counts[outcome] = count
        remaining -= count
    outcome_counts[outcomes[-1]] = remaining  # Assign remaining to last category
    
    generator = BoardGenerator(board_size)
    test_cases = []
    
    # Prepare direction weights
    directions = [
        WinDirection.HORIZONTAL,
        WinDirection.VERTICAL,
        WinDirection.DIAGONAL_DOWN,
        WinDirection.DIAGONAL_UP
    ]
    direction_weights = [
        direction_distribution.get("HORIZONTAL", 0.25),
        direction_distribution.get("VERTICAL", 0.25),
        direction_distribution.get("DIAGONAL_DOWN", 0.25),
        direction_distribution.get("DIAGONAL_UP", 0.25)
    ]
    
    # Counters for verification statistics
    verification_stats = {
        "WHITE_WINS": {
            "generated": 0, 
            "verified": 0, 
            "direction_match": 0,
            "unique_direction": 0,
            "invalid": 0,
            "other_player_wins": 0,
            "wrong_direction": 0,
            "multiple_directions": 0
        },
        "BLACK_WINS": {
            "generated": 0, 
            "verified": 0, 
            "direction_match": 0,
            "unique_direction": 0,
            "invalid": 0,
            "other_player_wins": 0,
            "wrong_direction": 0,
            "multiple_directions": 0
        },
        "NO_WINNER": {
            "generated": 0, 
            "verified": 0,
            "invalid": 0,
            "has_winner": 0
        }
    }
    
    # Generate WHITE_WINS cases
    for _ in range(outcome_counts.get("WHITE_WINS", 0)):
        direction = random.choices(directions, weights=direction_weights)[0]
        density = random.uniform(min_density, max_density)
        board = generator.generate_winning_board('W', direction, density)
        board_str = '\n'.join(''.join(row) for row in board)
        actual_density = generator.get_board_density()
        
        # **STRICT VERIFICATION**
        verification_stats["WHITE_WINS"]["generated"] += 1
        state, actual_direction = generator.check_winner_with_direction()
        all_wins = generator.get_all_winning_lines()
        
        # Check for invalid board
        if state == GameState.INVALID_BOARD:
            verification_stats["WHITE_WINS"]["invalid"] += 1
            if verbose:
                print(f"[Fail] ERROR: WHITE_WINS board is INVALID (both players win)!")
                print(f"   White winning lines: {len(all_wins['W'])}")
                print(f"   Black winning lines: {len(all_wins['B'])}")
        
        # Check for wrong winner
        elif state == GameState.BLACK_WINS:
            verification_stats["WHITE_WINS"]["other_player_wins"] += 1
            if verbose:
                print(f"[Fail] ERROR: Expected WHITE_WINS but got BLACK_WINS!")
        
        # Check for correct winner
        elif state == GameState.WHITE_WINS:
            verification_stats["WHITE_WINS"]["verified"] += 1
            
            # Check opponent has NO winning lines
            if len(all_wins['B']) > 0:
                if verbose:
                    print(f"[Fail] ERROR: WHITE_WINS board has {len(all_wins['B'])} BLACK winning lines!")
            
            # Check direction matches
            if actual_direction == direction:
                verification_stats["WHITE_WINS"]["direction_match"] += 1
            else:
                verification_stats["WHITE_WINS"]["wrong_direction"] += 1
                if verbose:
                    print(f"[Fail] ERROR: WHITE_WINS direction mismatch. Expected: {direction.value}, Got: {actual_direction.value}")
            
            # Check for UNIQUE direction (no other directions)
            white_directions = set(d for d, _ in all_wins['W'])
            if len(white_directions) == 1 and direction in white_directions:
                verification_stats["WHITE_WINS"]["unique_direction"] += 1
            else:
                verification_stats["WHITE_WINS"]["multiple_directions"] += 1
                if verbose:
                    print(f"[Fail] ERROR: WHITE_WINS has multiple directions: {[d.value for d in white_directions]}")
        
        else:
            if verbose:
                print(f"[Fail] ERROR: WHITE_WINS verification failed. Got state: {state.value}")
        
        test_cases.append((board_str, "WHITE_WINS", direction.value, actual_density))
    
    # Generate BLACK_WINS cases
    for _ in range(outcome_counts.get("BLACK_WINS", 0)):
        direction = random.choices(directions, weights=direction_weights)[0]
        density = random.uniform(min_density, max_density)
        board = generator.generate_winning_board('B', direction, density)
        board_str = '\n'.join(''.join(row) for row in board)
        actual_density = generator.get_board_density()
        
        # **STRICT VERIFICATION**
        verification_stats["BLACK_WINS"]["generated"] += 1
        state, actual_direction = generator.check_winner_with_direction()
        all_wins = generator.get_all_winning_lines()
        
        # Check for invalid board
        if state == GameState.INVALID_BOARD:
            verification_stats["BLACK_WINS"]["invalid"] += 1
            if verbose:
                print(f"[Fail] ERROR: BLACK_WINS board is INVALID (both players win)!")
                print(f"   White winning lines: {len(all_wins['W'])}")
                print(f"   Black winning lines: {len(all_wins['B'])}")
        
        # Check for wrong winner
        elif state == GameState.WHITE_WINS:
            verification_stats["BLACK_WINS"]["other_player_wins"] += 1
            if verbose:
                print(f"[Fail] ERROR: Expected BLACK_WINS but got WHITE_WINS!")
        
        # Check for correct winner
        elif state == GameState.BLACK_WINS:
            verification_stats["BLACK_WINS"]["verified"] += 1
            
            # Check opponent has NO winning lines
            if len(all_wins['W']) > 0:
                if verbose:
                    print(f"[Fail] ERROR: BLACK_WINS board has {len(all_wins['W'])} WHITE winning lines!")
            
            # Check direction matches
            if actual_direction == direction:
                verification_stats["BLACK_WINS"]["direction_match"] += 1
            else:
                verification_stats["BLACK_WINS"]["wrong_direction"] += 1
                if verbose:
                    print(f"[Fail] ERROR: BLACK_WINS direction mismatch. Expected: {direction.value}, Got: {actual_direction.value}")
            
            # Check for UNIQUE direction (no other directions)
            black_directions = set(d for d, _ in all_wins['B'])
            if len(black_directions) == 1 and direction in black_directions:
                verification_stats["BLACK_WINS"]["unique_direction"] += 1
            else:
                verification_stats["BLACK_WINS"]["multiple_directions"] += 1
                if verbose:
                    print(f"[Fail] ERROR: BLACK_WINS has multiple directions: {[d.value for d in black_directions]}")
        
        else:
            if verbose:
                print(f"[Fail] ERROR: BLACK_WINS verification failed. Got state: {state.value}")
        
        test_cases.append((board_str, "BLACK_WINS", direction.value, actual_density))
    
    # Generate NO_WINNER cases
    for _ in range(outcome_counts.get("NO_WINNER", 0)):
        density = random.uniform(min_density, max_density)
        board = generator.generate_no_winner_board(density)
        board_str = '\n'.join(''.join(row) for row in board)
        actual_density = generator.get_board_density()
        
        # **STRICT VERIFICATION**
        verification_stats["NO_WINNER"]["generated"] += 1
        state, actual_direction = generator.check_winner_with_direction()
        all_wins = generator.get_all_winning_lines()
        
        if state == GameState.INVALID_BOARD:
            verification_stats["NO_WINNER"]["invalid"] += 1
            if verbose:
                print(f"[Fail] ERROR: NO_WINNER board is INVALID (both players win)!")
                print(f"   White winning lines: {len(all_wins['W'])}")
                print(f"   Black winning lines: {len(all_wins['B'])}")
        
        elif state != GameState.NO_WINNER:
            verification_stats["NO_WINNER"]["has_winner"] += 1
            if verbose:
                print(f"[Fail] ERROR: NO_WINNER board has winner: {state.value}, direction: {actual_direction}")
                if state == GameState.WHITE_WINS:
                    print(f"   White winning lines: {all_wins['W']}")
                elif state == GameState.BLACK_WINS:
                    print(f"   Black winning lines: {all_wins['B']}")
        
        else:
            verification_stats["NO_WINNER"]["verified"] += 1
            
            # Double check no winning lines exist
            if len(all_wins['W']) > 0 or len(all_wins['B']) > 0:
                if verbose:
                    print(f"[Fail] ERROR: NO_WINNER board has winning lines!")
                    print(f"   White: {len(all_wins['W'])}, Black: {len(all_wins['B'])}")
        
        test_cases.append((board_str, "NO_WINNER", None, actual_density))
    
    # Print STRICT verification statistics
    print("\n" + "="*80)
    print("STRICT Test Case Generation Verification Summary")
    print("="*80)
    
    total_invalid = 0
    total_wrong_winner = 0
    total_wrong_direction = 0
    total_multiple_directions = 0
    
    for outcome, stats in verification_stats.items():
        generated = stats["generated"]
        verified = stats["verified"]
        
        if generated > 0:
            verify_rate = verified / generated * 100
            print(f"\n{outcome}:")
            print(f"  Generated: {generated}")
            print(f"  Verified Correct: {verified} ({verify_rate:.1f}%)")
            
            if "direction_match" in stats:
                dir_match = stats["direction_match"]
                dir_rate = dir_match / generated * 100 if generated > 0 else 0
                print(f"  Direction Match: {dir_match} ({dir_rate:.1f}%)")
                
                unique_dir = stats["unique_direction"]
                unique_rate = unique_dir / generated * 100 if generated > 0 else 0
                print(f"  Unique Direction Only: {unique_dir} ({unique_rate:.1f}%)")
            
            # Error statistics
            if stats.get("invalid", 0) > 0:
                print(f"  [Fail] INVALID (both win): {stats['invalid']}")
                total_invalid += stats["invalid"]
            
            if stats.get("other_player_wins", 0) > 0:
                print(f"  [Fail] Wrong Winner: {stats['other_player_wins']}")
                total_wrong_winner += stats["other_player_wins"]
            
            if stats.get("wrong_direction", 0) > 0:
                print(f"  [Fail] Wrong Direction: {stats['wrong_direction']}")
                total_wrong_direction += stats["wrong_direction"]
            
            if stats.get("multiple_directions", 0) > 0:
                print(f"  [Fail] Multiple Directions: {stats['multiple_directions']}")
                total_multiple_directions += stats["multiple_directions"]
            
            if stats.get("has_winner", 0) > 0:
                print(f"  [Fail] Has Winner: {stats['has_winner']}")
                total_wrong_winner += stats["has_winner"]
    
    # Overall summary
    total_generated = sum(s["generated"] for s in verification_stats.values())
    total_verified = sum(s["verified"] for s in verification_stats.values())
    
    print(f"\n" + "-"*80)
    print(f"OVERALL:")
    print(f"  Total Generated: {total_generated}")
    print(f"  Total Verified: {total_verified} ({total_verified/total_generated*100:.1f}%)")
    
    error_found = False
    if total_invalid > 0:
        print(f"  [Fail] Total INVALID Boards: {total_invalid}")
        error_found = True
    if total_wrong_winner > 0:
        print(f"  [Fail] Total Wrong Winners: {total_wrong_winner}")
        error_found = True
    if total_wrong_direction > 0:
        print(f"  [Fail] Total Wrong Directions: {total_wrong_direction}")
        error_found = True
    if total_multiple_directions > 0:
        print(f"  [Fail] Total Multiple Winning Directions: {total_multiple_directions}")
        error_found = True
    
    if not error_found and total_verified == total_generated:
        print(f"  [Success] ALL BOARDS VERIFIED SUCCESSFULLY!")
        print(f"  [Success] Each winning board has EXACTLY ONE winning direction")
        print(f"  [Success] No opponent has any winning lines")
        print(f"  [Success] No invalid boards generated")
    else:
        print(f"  [Warning]  SOME BOARDS FAILED STRICT VERIFICATION!")
    
    print("="*80 + "\n")
    
    # Shuffle the test cases
    random.shuffle(test_cases)
    
    return test_cases


# Verification functions remain the same but with stricter checks
def verify_test_case(
    board_str: str, 
    expected_outcome: str, 
    expected_direction: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Verify a single test case with STRICT checks
    
    Args:
        board_str: String representation of the board
        expected_outcome: Expected outcome (WHITE_WINS, BLACK_WINS, NO_WINNER)
        expected_direction: Expected winning direction (if applicable)
        verbose: Print detailed information
    
    Returns:
        Dictionary with comprehensive verification results
    """
    # Parse board
    lines = board_str.strip().split('\n')
    board_size = len(lines)
    board = np.array([list(line.replace(' ', '')) for line in lines])
    
    # Create generator and set board
    generator = BoardGenerator(board_size)
    generator.board = board
    
    # Check state and direction
    state, actual_direction = generator.check_winner_with_direction()
    
    # Get all winning lines
    all_wins = generator.get_all_winning_lines()
    
    # Map string to enum
    state_map = {
        "WHITE_WINS": GameState.WHITE_WINS,
        "BLACK_WINS": GameState.BLACK_WINS,
        "NO_WINNER": GameState.NO_WINNER
    }
    expected_state = state_map.get(expected_outcome)
    
    # Verify outcome
    outcome_correct = (state == expected_state)
    
    # Verify direction (if applicable)
    direction_correct = True
    if expected_direction is not None:
        try:
            expected_dir_enum = WinDirection[expected_direction]
            direction_correct = (actual_direction == expected_dir_enum)
        except KeyError:
            direction_correct = False
    
    # Check for uniqueness of winner and direction
    has_unique_winner = True
    has_unique_direction = True
    
    if state == GameState.INVALID_BOARD:
        has_unique_winner = False
        has_unique_direction = False
    elif expected_outcome in ["WHITE_WINS", "BLACK_WINS"]:
        # Expected winner should have lines, opponent should not
        expected_winner = 'W' if expected_outcome == "WHITE_WINS" else 'B'
        opponent = 'B' if expected_winner == 'W' else 'W'
        
        # Check opponent has NO winning
        # Check opponent has NO winning lines
        if len(all_wins[opponent]) > 0:
            has_unique_winner = False
        
        # Check winner has ONLY the expected direction
        if expected_direction is not None:
            try:
                expected_dir_enum = WinDirection[expected_direction]
                winner_directions = set(d for d, _ in all_wins[expected_winner])
                
                # Must have exactly one direction and it must be the expected one
                if len(winner_directions) != 1 or expected_dir_enum not in winner_directions:
                    has_unique_direction = False
            except (KeyError, ValueError):
                has_unique_direction = False
    elif expected_outcome == "NO_WINNER":
        # No player should have any winning lines
        if len(all_wins['W']) > 0 or len(all_wins['B']) > 0:
            has_unique_winner = False
    
    result = {
        "outcome_correct": outcome_correct,
        "direction_correct": direction_correct,
        "has_unique_winner": has_unique_winner,
        "has_unique_direction": has_unique_direction,
        "actual_outcome": state.value,
        "actual_direction": actual_direction.value if actual_direction else None,
        "expected_outcome": expected_outcome,
        "expected_direction": expected_direction,
        "board_density": generator.get_board_density(),
        "white_winning_lines": len(all_wins['W']),
        "black_winning_lines": len(all_wins['B']),
        "white_winning_directions": list(set(d.value for d, _ in all_wins['W'])),
        "black_winning_directions": list(set(d.value for d, _ in all_wins['B'])),
        "all_winning_lines": {
            'W': [(d.value, pos) for d, pos in all_wins['W']],
            'B': [(d.value, pos) for d, pos in all_wins['B']]
        }
    }
    
    if verbose:
        print("\n" + "="*60)
        print("STRICT Test Case Verification")
        print("="*60)
        print(f"Expected: {expected_outcome}" + 
              (f" ({expected_direction})" if expected_direction else ""))
        print(f"Actual: {state.value}" + 
              (f" ({actual_direction.value})" if actual_direction else ""))
        print(f"\nWhite winning lines: {len(all_wins['W'])}")
        if all_wins['W']:
            print(f"  Directions: {result['white_winning_directions']}")
            for direction, positions in all_wins['W'][:2]:  # Show first 2
                print(f"    {direction.value}: {positions[:5]}...")
        print(f"Black winning lines: {len(all_wins['B'])}")
        if all_wins['B']:
            print(f"  Directions: {result['black_winning_directions']}")
            for direction, positions in all_wins['B'][:2]:  # Show first 2
                print(f"    {direction.value}: {positions[:5]}...")
        
        print(f"\nVerification Results:")
        print(f"  Outcome Correct: {'[Success]' if outcome_correct else '[Fail]'}")
        print(f"  Direction Correct: {'[Success]' if direction_correct else '[Fail]'}")
        print(f"  Unique Winner: {'[Success]' if has_unique_winner else '[Fail]'}")
        print(f"  Unique Direction: {'[Success]' if has_unique_direction else '[Fail]'}")
        print(f"  Board Density: {result['board_density']:.2%}")
        print("="*60 + "\n")
    
    return result


def batch_verify_test_cases(test_cases: List[Tuple[str, str, Optional[str], float]]) -> Dict[str, Any]:
    """
    Batch verify multiple test cases with STRICT validation
    
    Args:
        test_cases: List of (board_str, expected_outcome, direction, density) tuples
    
    Returns:
        Dictionary with batch verification statistics
    """
    stats = {
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "invalid_boards": 0,
        "wrong_outcome": 0,
        "wrong_direction": 0,
        "multiple_winners": 0,
        "multiple_directions": 0,
        "by_outcome": {}
    }
    
    for i, (board_str, expected_outcome, expected_direction, _) in enumerate(test_cases):
        result = verify_test_case(board_str, expected_outcome, expected_direction, verbose=False)
        
        # Track by outcome
        if expected_outcome not in stats["by_outcome"]:
            stats["by_outcome"][expected_outcome] = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "unique_direction_pass": 0
            }
        
        stats["by_outcome"][expected_outcome]["total"] += 1
        
        # Check if all verifications passed (STRICT)
        all_passed = (
            result["outcome_correct"] and 
            result["direction_correct"] and 
            result["has_unique_winner"] and
            result["has_unique_direction"]
        )
        
        if all_passed:
            stats["passed"] += 1
            stats["by_outcome"][expected_outcome]["passed"] += 1
            if result["has_unique_direction"]:
                stats["by_outcome"][expected_outcome]["unique_direction_pass"] += 1
        else:
            stats["failed"] += 1
            stats["by_outcome"][expected_outcome]["failed"] += 1
            
            # Track specific failures
            if result["actual_outcome"] == "INVALID_BOARD":
                stats["invalid_boards"] += 1
            if not result["outcome_correct"]:
                stats["wrong_outcome"] += 1
            if not result["direction_correct"]:
                stats["wrong_direction"] += 1
            if not result["has_unique_winner"]:
                stats["multiple_winners"] += 1
            if not result["has_unique_direction"]:
                stats["multiple_directions"] += 1
            
            # Print details of failed case
            print(f"\n[Fail] Test case {i+1} FAILED:")
            print(f"  Expected: {expected_outcome} ({expected_direction})")
            print(f"  Actual: {result['actual_outcome']} ({result['actual_direction']})")
            print(f"  White wins: {result['white_winning_lines']} (directions: {result['white_winning_directions']})")
            print(f"  Black wins: {result['black_winning_lines']} (directions: {result['black_winning_directions']})")
            if not result["has_unique_winner"]:
                print(f"  [Warning]  Multiple winners detected!")
            if not result["has_unique_direction"]:
                print(f"  [Warning]  Multiple winning directions detected!")
    
    # Print summary
    print("\n" + "="*80)
    print("STRICT Batch Verification Summary")
    print("="*80)
    print(f"Total Cases: {stats['total']}")
    print(f"Passed: {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)")
    print(f"Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    
    if stats['failed'] > 0:
        print(f"\nFailure Breakdown:")
        if stats['invalid_boards'] > 0:
            print(f"  [Fail] Invalid Boards (both win): {stats['invalid_boards']}")
        if stats['wrong_outcome'] > 0:
            print(f"  [Fail] Wrong Outcome: {stats['wrong_outcome']}")
        if stats['wrong_direction'] > 0:
            print(f"  [Fail] Wrong Direction: {stats['wrong_direction']}")
        if stats['multiple_winners'] > 0:
            print(f"  [Fail] Multiple Winners: {stats['multiple_winners']}")
        if stats['multiple_directions'] > 0:
            print(f"  [Fail] Multiple Winning Directions: {stats['multiple_directions']}")
    
    print(f"\nBy Outcome:")
    for outcome, outcome_stats in stats['by_outcome'].items():
        total = outcome_stats['total']
        passed = outcome_stats['passed']
        unique_dir = outcome_stats.get('unique_direction_pass', 0)
        print(f"  {outcome}: {passed}/{total} ({passed/total*100:.1f}%)")
        if outcome in ["WHITE_WINS", "BLACK_WINS"]:
            print(f"    Unique Direction: {unique_dir}/{total} ({unique_dir/total*100:.1f}%)")
    
    print("="*80 + "\n")
    
    return stats


def verify_dataset_file(filepath: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Verify all test cases in a dataset file with STRICT validation
    
    Args:
        filepath: Path to the dataset JSON file
        verbose: Print detailed information
    
    Returns:
        Dictionary with verification statistics
    """
    import json
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check format
    if isinstance(data, dict) and 'test_cases' in data:
        test_cases_data = data['test_cases']
    else:
        test_cases_data = data
    
    # Convert to format expected by batch_verify_test_cases
    test_cases = []
    for case in test_cases_data:
        board_str = case['board']
        expected = case['expected']
        direction = case.get('win_direction')
        density = case.get('density', 0.0)
        test_cases.append((board_str, expected, direction, density))
    
    print(f"\n{'='*80}")
    print(f"Verifying dataset: {filepath}")
    print(f"Total cases: {len(test_cases)}")
    print(f"{'='*80}\n")
    
    return batch_verify_test_cases(test_cases)