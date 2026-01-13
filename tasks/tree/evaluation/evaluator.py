import re
from typing import Optional, Tuple
from utils.tree_utils import TreeRenderer, TreeAnalyzer
import difflib

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.tree_renderer = TreeRenderer()
        self.tree_analyzer = TreeAnalyzer()
    
    def evaluate_response(self, task_type: str, response: str, expected_answer: str, 
                     tree_structure: Optional[str] = None) -> bool:
        """Evaluate model response based on task type"""
        if task_type == "task1":
            return self._evaluate_task1(response, expected_answer)
        elif task_type == "task2":
            return self._evaluate_task2_path_analysis(response, expected_answer)
        else:
            return False
    
    def evaluate_response_with_similarity(self, task_type: str, response: str, expected_answer: str, 
                                    tree_structure: Optional[str] = None) -> Tuple[bool, float]:
        """Evaluate model response and return accuracy and similarity score"""
        if task_type == "task1":
            is_correct = self._evaluate_task1(response, expected_answer)
            similarity = 1.0 if is_correct else 0.0
            return is_correct, similarity
        elif task_type == "task2":
            is_correct, similarity = self._evaluate_task2_path_with_similarity(response, expected_answer)
            return is_correct, similarity
        else:
            return False, 0.0
    
    def _extract_answer_for_verbose(self, response: str, task_type: str) -> Optional[str]:
        """Extract answer for verbose output (public method for main.py)"""
        if task_type == "task1":
            return self._extract_answer(response)
        elif task_type == "task2":
            return self._extract_path_answer(response)
        return None
    
    def _evaluate_task1(self, response: str, expected_answer: str) -> bool:
        """Evaluate Task 1 response (structure questions)"""
        # Extract answer wrapped in <answer> tags
        extracted_answer = self._extract_answer(response)
        
        # Normalize answers
        extracted_answer = extracted_answer.strip().lower()
        expected_answer = expected_answer.strip().lower()
        
        # Handle None case
        if expected_answer == "none":
            return extracted_answer in ["none", "null", "nothing", "no", "n/a"]
        
        # Handle numeric answers
        try:
            expected_num = int(expected_answer)
            extracted_num = int(extracted_answer)
            return expected_num == extracted_num
        except ValueError:
            return extracted_answer == expected_answer
    
    def _evaluate_task2_path_analysis(self, response: str, expected_answer: str) -> bool:
        """Evaluate Task 2 response (path analysis)"""
        extracted_path = self._extract_path_answer(response)
        
        if not extracted_path:
            return False
        
        # Normalize both paths
        extracted_normalized = self._normalize_path(extracted_path)
        expected_normalized = self._normalize_path(expected_answer)
        
        return extracted_normalized == expected_normalized
    
    def _evaluate_task2_path_with_similarity(self, response: str, expected_answer: str) -> Tuple[bool, float]:
        """Evaluate Task 2 response with similarity score"""
        extracted_path = self._extract_path_answer(response)
        
        if not extracted_path:
            return False, 0.0
        
        # Normalize both paths
        extracted_normalized = self._normalize_path(extracted_path)
        expected_normalized = self._normalize_path(expected_answer)
        
        # Check exact match
        is_correct = extracted_normalized == expected_normalized
        
        # Calculate similarity
        similarity = self._calculate_path_similarity(extracted_normalized, expected_normalized)
        
        return is_correct, similarity
    
    def _extract_answer(self, response: str) -> str:
        """Extract answer from <answer> tags; missing tags -> empty string."""
        if not isinstance(response, str):
            return ""
        match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return match.group(1).strip()

    def _extract_path_answer(self, response: str) -> str:
        """Extract path answer from <answer> tags; missing tags -> empty string."""
        return self._extract_answer(response)
        
    def _normalize_path(self, path_str: str) -> str:
        """Normalize a path string for comparison"""
        if not path_str:
            return ""
        
        # Remove extra whitespace and normalize arrows
        normalized = re.sub(r'\s*-+>\s*', ' -> ', path_str.strip())
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Extract just the numbers
        numbers = re.findall(r'\d+', normalized)
        
        return ' -> '.join(numbers) if numbers else ""
    
    def _calculate_path_similarity(self, extracted: str, expected: str) -> float:
        """Calculate similarity between two normalized paths"""
        if not extracted and not expected:
            return 1.0
        
        if not extracted or not expected:
            return 0.0
        
        # Split paths into node sequences
        extracted_nodes = extracted.split(' -> ')
        expected_nodes = expected.split(' -> ')
        
        # Calculate sequence similarity using difflib
        sequence_sim = difflib.SequenceMatcher(None, extracted_nodes, expected_nodes).ratio()
        
        # Calculate node overlap similarity
        extracted_set = set(extracted_nodes)
        expected_set = set(expected_nodes)
        
        if expected_set:
            overlap_sim = len(extracted_set & expected_set) / len(expected_set)
        else:
            overlap_sim = 1.0 if not extracted_set else 0.0
        
        # Check correct start and end points
        endpoint_sim = 0.0
        if extracted_nodes and expected_nodes:
            if extracted_nodes[0] == expected_nodes[0]:  # Same start
                endpoint_sim += 0.5
            if extracted_nodes[-1] == expected_nodes[-1]:  # Same end
                endpoint_sim += 0.5
        
        # Weighted combination
        final_similarity = (
            0.5 * sequence_sim +
            0.3 * overlap_sim +
            0.2 * endpoint_sim
        )
        
        return round(final_similarity, 3)
