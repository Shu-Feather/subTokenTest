"""
Evaluation system for cipher and decipher benchmark.
"""

import re
import string
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum

from ..ciphers.morse_code import MorseCode
from ..ciphers.caesar_cipher import CaesarCipher

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enumeration of task types."""
    MORSE_ENCODE = "morse_encode"
    MORSE_DECODE = "morse_decode"
    CAESAR_ENCODE = "caesar_encode"
    CAESAR_DECODE = "caesar_decode"


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    task_type: TaskType
    input_text: str
    expected_output: str
    model_output: str
    is_correct: bool
    similarity_score: float
    difficulty: Optional[str] = None  # 'easy', 'medium', 'hard'
    error_type: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class OutputProcessor:
    """Processes and cleans model outputs for evaluation."""
    
    @staticmethod
    def extract_answer(output: str, task_type: TaskType) -> str:
        """
        Extract the actual answer from model output using structured format.
        
        Args:
            output (str): Raw model output
            task_type (TaskType): Type of task
            
        Returns:
            str: Extracted answer
        """
        if not output:
            return ""
        
        # First, try to extract from structured format <answer> ... </answer>
        import re
        
        # Look for <answer> content </answer> pattern
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        answer_match = re.search(answer_pattern, output, re.DOTALL | re.IGNORECASE)
        
        if answer_match:
            cleaned = answer_match.group(1).strip()
        else:
            # Fallback to original extraction logic
            cleaned = output.strip()
            
            # Remove common prefixes and suffixes
            patterns_to_remove = [
                r'^(Answer|Result|Output|Final answer|Final result):\s*',
                r'^(Morse code|Encrypted text|Decrypted text|Text):\s*',
                r'^(Step \d+:.*?)*',
                r'(Here is|Here\'s|The answer is|The result is)\s*',
            ]
            
            for pattern in patterns_to_remove:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
            
            # For step-by-step responses, try to extract the final answer
            if 'final' in cleaned.lower() or 'step' in cleaned.lower():
                lines = cleaned.split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    if line and not line.startswith('Step'):
                        # Try to extract actual content after colons
                        if ':' in line:
                            line = line.split(':', 1)[-1].strip()
                        if line:
                            cleaned = line
                            break
        
        # Remove quotes and extra whitespace
        cleaned = cleaned.strip('\'"').strip()
        
        return cleaned
    
    @staticmethod
    def normalize_morse_code(morse_code: str) -> str:
        """
        Normalize Morse code for comparison.
        
        Args:
            morse_code (str): Raw Morse code
            
        Returns:
            str: Normalized Morse code
        """
        if not morse_code:
            return ""
        
        # Clean and normalize spacing
        normalized = re.sub(r'\s+', ' ', morse_code.strip())
        
        # Ensure word separators are properly formatted
        normalized = re.sub(r'\s*/\s*', ' / ', normalized)
        
        # Remove any non-Morse characters except dots, dashes, spaces, and forward slashes
        normalized = re.sub(r'[^.\-\s/]', '', normalized)
        
        return normalized.strip()
    
    @staticmethod
    def normalize_text(text: str, ignore_case: bool = True, ignore_punctuation: bool = True) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text (str): Input text
            ignore_case (bool): Whether to ignore case differences
            ignore_punctuation (bool): Whether to ignore punctuation
            
        Returns:
            str: Normalized text
        """
        if not text:
            return ""
        
        normalized = text.strip()
        
        if ignore_case:
            normalized = normalized.lower()
        
        if ignore_punctuation:
            # Remove punctuation but keep spaces
            normalized = ''.join(char if char not in string.punctuation else ' ' 
                               for char in normalized)
            normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized


class SimilarityCalculator:
    """Calculates similarity scores between expected and actual outputs."""
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        Args:
            s1 (str): First string
            s2 (str): Second string
            
        Returns:
            int: Levenshtein distance
        """
        if len(s1) < len(s2):
            return SimilarityCalculator.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def similarity_score(expected: str, actual: str) -> float:
        """
        Calculate similarity score between expected and actual output.
        
        Args:
            expected (str): Expected output
            actual (str): Actual output
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not expected and not actual:
            return 1.0
        
        if not expected or not actual:
            return 0.0
        
        distance = SimilarityCalculator.levenshtein_distance(expected, actual)
        max_length = max(len(expected), len(actual))
        
        return 1.0 - (distance / max_length) if max_length > 0 else 1.0


class CipherEvaluator:
    """Main evaluator for cipher and decipher tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator with configuration.
        
        Args:
            config (Dict[str, Any]): Evaluation configuration
        """
        self.config = config
        self.strict_match = config.get('strict_match', True)
        self.case_sensitive = config.get('case_sensitive', False)
        self.ignore_punctuation = config.get('ignore_punctuation', True)
        
        self.processor = OutputProcessor()
        self.similarity_calc = SimilarityCalculator()
    
    def evaluate_morse_encode(self, input_text: str, model_output: str) -> EvaluationResult:
        """
        Evaluate Morse code encoding task.
        
        Args:
            input_text (str): Original text to encode
            model_output (str): Model's Morse code output
            
        Returns:
            EvaluationResult: Evaluation result
        """
        # Generate expected output
        expected_morse = MorseCode.encode(input_text)
        
        # Process model output
        actual_morse = self.processor.extract_answer(model_output, TaskType.MORSE_ENCODE)
        actual_morse = self.processor.normalize_morse_code(actual_morse)
        
        # Evaluate
        is_correct = (expected_morse == actual_morse) if self.strict_match else False
        if not is_correct and not self.strict_match:
            # Calculate similarity score
            similarity = self.similarity_calc.similarity_score(expected_morse, actual_morse)
            is_correct = similarity > 0.9  # 90% similarity threshold
        else:
            similarity = 1.0 if is_correct else self.similarity_calc.similarity_score(expected_morse, actual_morse)
        
        return EvaluationResult(
            task_type=TaskType.MORSE_ENCODE,
            input_text=input_text,
            expected_output=expected_morse,
            model_output=actual_morse,
            is_correct=is_correct,
            similarity_score=similarity
        )
    
    def evaluate_morse_decode(self, morse_input: str, model_output: str) -> EvaluationResult:
        """
        Evaluate Morse code decoding task.
        
        Args:
            morse_input (str): Morse code to decode
            model_output (str): Model's decoded text output
            
        Returns:
            EvaluationResult: Evaluation result
        """
        # Generate expected output
        expected_text = MorseCode.decode(morse_input)
        
        # Process model output
        actual_text = self.processor.extract_answer(model_output, TaskType.MORSE_DECODE)
        
        # Normalize for comparison
        expected_normalized = self.processor.normalize_text(
            expected_text, self.case_sensitive, self.ignore_punctuation
        )
        actual_normalized = self.processor.normalize_text(
            actual_text, self.case_sensitive, self.ignore_punctuation
        )
        
        # Evaluate
        is_correct = (expected_normalized == actual_normalized) if self.strict_match else False
        if not is_correct and not self.strict_match:
            similarity = self.similarity_calc.similarity_score(expected_normalized, actual_normalized)
            is_correct = similarity > 0.9
        else:
            similarity = 1.0 if is_correct else self.similarity_calc.similarity_score(expected_normalized, actual_normalized)
        
        return EvaluationResult(
            task_type=TaskType.MORSE_DECODE,
            input_text=morse_input,
            expected_output=expected_text,
            model_output=actual_text,
            is_correct=is_correct,
            similarity_score=similarity
        )
    
    def evaluate_caesar_encode(self, input_text: str, shift: int, model_output: str) -> EvaluationResult:
        """
        Evaluate Caesar cipher encoding task.
        
        Args:
            input_text (str): Original text to encrypt
            shift (int): Caesar cipher shift value
            model_output (str): Model's encrypted output
            
        Returns:
            EvaluationResult: Evaluation result
        """
        # Generate expected output
        expected_encrypted = CaesarCipher.encode(input_text, shift)
        
        # Process model output
        actual_encrypted = self.processor.extract_answer(model_output, TaskType.CAESAR_ENCODE)
        
        # Evaluate
        is_correct = (expected_encrypted == actual_encrypted) if self.strict_match else False
        if not is_correct and not self.strict_match:
            similarity = self.similarity_calc.similarity_score(expected_encrypted, actual_encrypted)
            is_correct = similarity > 0.9
        else:
            similarity = 1.0 if is_correct else self.similarity_calc.similarity_score(expected_encrypted, actual_encrypted)
        
        return EvaluationResult(
            task_type=TaskType.CAESAR_ENCODE,
            input_text=input_text,
            expected_output=expected_encrypted,
            model_output=actual_encrypted,
            is_correct=is_correct,
            similarity_score=similarity,
            additional_info={'shift': shift}
        )
    
    def evaluate_caesar_decode(self, encrypted_input: str, shift: int, model_output: str) -> EvaluationResult:
        """
        Evaluate Caesar cipher decoding task.
        
        Args:
            encrypted_input (str): Encrypted text to decrypt
            shift (int): Caesar cipher shift value used for encryption
            model_output (str): Model's decrypted output
            
        Returns:
            EvaluationResult: Evaluation result
        """
        # Generate expected output
        expected_decrypted = CaesarCipher.decode(encrypted_input, shift)
        
        # Process model output
        actual_decrypted = self.processor.extract_answer(model_output, TaskType.CAESAR_DECODE)
        
        # Normalize for comparison if not case sensitive
        if not self.case_sensitive:
            expected_normalized = expected_decrypted.lower()
            actual_normalized = actual_decrypted.lower()
        else:
            expected_normalized = expected_decrypted
            actual_normalized = actual_decrypted
        
        # Evaluate
        is_correct = (expected_normalized == actual_normalized) if self.strict_match else False
        if not is_correct and not self.strict_match:
            similarity = self.similarity_calc.similarity_score(expected_normalized, actual_normalized)
            is_correct = similarity > 0.9
        else:
            similarity = 1.0 if is_correct else self.similarity_calc.similarity_score(expected_normalized, actual_normalized)
        
        return EvaluationResult(
            task_type=TaskType.CAESAR_DECODE,
            input_text=encrypted_input,
            expected_output=expected_decrypted,
            model_output=actual_decrypted,
            is_correct=is_correct,
            similarity_score=similarity,
            additional_info={'shift': shift}
        )
    
    def evaluate_single(self, task_type: TaskType, input_data: str, 
                       model_output: str, additional_info: Dict[str, Any] = None) -> EvaluationResult:
        """
        Evaluate a single task result.
        
        Args:
            task_type (TaskType): Type of task
            input_data (str): Input data for the task
            model_output (str): Model's output
            additional_info (Dict[str, Any]): Additional information (e.g., shift, difficulty)
            
        Returns:
            EvaluationResult: Evaluation result
        """
        try:
            # Extract difficulty if present
            difficulty = additional_info.get('difficulty', None) if additional_info else None
            
            if task_type == TaskType.MORSE_ENCODE:
                eval_result = self.evaluate_morse_encode(input_data, model_output)
            elif task_type == TaskType.MORSE_DECODE:
                eval_result = self.evaluate_morse_decode(input_data, model_output)
            elif task_type == TaskType.CAESAR_ENCODE:
                shift = additional_info.get('shift', 0) if additional_info else 0
                eval_result = self.evaluate_caesar_encode(input_data, shift, model_output)
            elif task_type == TaskType.CAESAR_DECODE:
                shift = additional_info.get('shift', 0) if additional_info else 0
                eval_result = self.evaluate_caesar_decode(input_data, shift, model_output)
            else:
                logger.error(f"Unknown task type: {task_type}")
                eval_result = EvaluationResult(
                    task_type=task_type,
                    input_text=input_data,
                    expected_output="",
                    model_output=model_output,
                    is_correct=False,
                    similarity_score=0.0,
                    difficulty=difficulty,
                    error_type="unknown_task_type"
                )
                return eval_result
            
            # Add difficulty to result
            eval_result.difficulty = difficulty
            return eval_result
            
        except Exception as e:
            logger.error(f"Error evaluating {task_type} task: {e}")
            # Create a failed evaluation result
            eval_result = EvaluationResult(
                task_type=task_type,
                input_text=input_data,
                expected_output="",
                model_output=model_output,
                is_correct=False,
                similarity_score=0.0,
                difficulty=additional_info.get('difficulty', None) if additional_info else None,
                error_type=str(e)
            )
            return eval_result
    
    def evaluate_batch(self, results: List[Tuple[TaskType, str, str, Any]]) -> List[EvaluationResult]:
        """
        Evaluate a batch of results.
        
        Args:
            results (List[Tuple]): List of (task_type, input, model_output, additional_info)
            
        Returns:
            List[EvaluationResult]: List of evaluation results
        """
        evaluations = []
        
        for task_type, input_data, model_output, additional_info in results:
            eval_result = self.evaluate_single(task_type, input_data, model_output, additional_info)
            evaluations.append(eval_result)
        
        return evaluations
    
    def get_summary_stats(self, evaluations: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Calculate summary statistics for evaluations.
        
        Args:
            evaluations (List[EvaluationResult]): List of evaluation results
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not evaluations:
            return {}
        
        # Overall stats
        total_count = len(evaluations)
        correct_count = sum(1 for eval_result in evaluations if eval_result.is_correct)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        avg_similarity = sum(eval_result.similarity_score for eval_result in evaluations) / total_count
        
        # Per-task stats
        task_stats = {}
        for task_type in TaskType:
            task_evaluations = [e for e in evaluations if e.task_type == task_type]
            if task_evaluations:
                task_correct = sum(1 for e in task_evaluations if e.is_correct)
                task_total = len(task_evaluations)
                task_accuracy = task_correct / task_total
                task_avg_similarity = sum(e.similarity_score for e in task_evaluations) / task_total
                
                task_stats[task_type.value] = {
                    'total': task_total,
                    'correct': task_correct,
                    'accuracy': task_accuracy,
                    'avg_similarity': task_avg_similarity
                }
        
        # Per-difficulty stats
        difficulty_stats = {}
        for difficulty in ['easy', 'medium', 'hard']:
            diff_evaluations = [e for e in evaluations if e.difficulty == difficulty]
            if diff_evaluations:
                diff_correct = sum(1 for e in diff_evaluations if e.is_correct)
                diff_total = len(diff_evaluations)
                diff_accuracy = diff_correct / diff_total
                diff_avg_similarity = sum(e.similarity_score for e in diff_evaluations) / diff_total
                
                difficulty_stats[difficulty] = {
                    'total': diff_total,
                    'correct': diff_correct,
                    'accuracy': diff_accuracy,
                    'avg_similarity': diff_avg_similarity
                }
        
        return {
            'overall': {
                'total': total_count,
                'correct': correct_count,
                'accuracy': accuracy,
                'avg_similarity': avg_similarity
            },
            'per_task': task_stats,
            'per_difficulty': difficulty_stats
        }