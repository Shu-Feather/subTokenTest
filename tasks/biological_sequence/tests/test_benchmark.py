"""
Unit tests for the Biological Sequence Manipulation Benchmark.
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_generator import BiologicalSequenceGenerator
from src.prompt_templates import PromptTemplates
from src.evaluator import SequenceEvaluator, EvaluationResult
from src.benchmark import BiologicalSequenceBenchmark


class TestBiologicalSequenceGenerator:
    """Test cases for the data generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = BiologicalSequenceGenerator()
    
    def test_dna_sequence_generation(self):
        """Test DNA sequence generation."""
        length = 10
        sequence = self.generator.generate_dna_sequence(length)
        
        assert len(sequence) == length
        assert all(base in 'ATCG' for base in sequence)
    
    def test_rna_sequence_generation(self):
        """Test RNA sequence generation."""
        length = 10
        sequence = self.generator.generate_rna_sequence(length)
        
        assert len(sequence) == length
        assert all(base in 'AUCG' for base in sequence)
    
    def test_protein_three_letter_generation(self):
        """Test three-letter protein sequence generation."""
        length = 5
        sequence = self.generator.generate_protein_three_letter(length)
        
        parts = sequence.split('-')
        assert len(parts) == length
        assert all(part in self.generator.aa_three_to_one for part in parts)
    
    def test_protein_one_letter_generation(self):
        """Test one-letter protein sequence generation."""
        length = 5
        sequence = self.generator.generate_protein_one_letter(length)
        
        # Should be no separators for one-letter format
        assert len(sequence) == length
        assert all(part in self.generator.aa_one_to_three for part in sequence)
    
    def test_dna_complement(self):
        """Test DNA complement generation."""
        test_cases = [
            ('A', 'T'),
            ('T', 'A'),
            ('G', 'C'),
            ('C', 'G'),
            ('ATCG', 'TAGC'),
            ('AAATTGCC', 'TTTAACGG')
        ]
        
        for input_seq, expected in test_cases:
            result = self.generator.get_dna_complement(input_seq)
            assert result == expected, f"Expected {expected}, got {result} for input {input_seq}"
    
    def test_rna_complement(self):
        """Test RNA complement generation."""
        test_cases = [
            ('A', 'U'),
            ('U', 'A'),
            ('G', 'C'),
            ('C', 'G'),
            ('AUCG', 'UAGC'),
            ('AAAUUGCC', 'UUUAACGG')
        ]
        
        for input_seq, expected in test_cases:
            result = self.generator.get_rna_complement(input_seq)
            assert result == expected, f"Expected {expected}, got {result} for input {input_seq}"
    
    def test_protein_conversion_three_to_one(self):
        """Test protein conversion from three-letter to one-letter."""
        test_cases = [
            ('GLY', 'G'),
            ('ARG', 'R'),
            ('PHE', 'F'),
            ('GLY-ARG-PHE', 'GRF'),  # No separators in output
            ('ALA-CYS-ASP-GLU', 'ACDE')  # No separators in output
        ]
        
        for input_seq, expected in test_cases:
            result = self.generator.convert_protein_three_to_one(input_seq)
            assert result == expected, f"Expected {expected}, got {result} for input {input_seq}"
    
    def test_protein_conversion_one_to_three(self):
        """Test protein conversion from one-letter to three-letter."""
        test_cases = [
            ('G', 'GLY'),
            ('R', 'ARG'),
            ('F', 'PHE'),
            ('GRF', 'GLY-ARG-PHE'),  # Input has no separators
            ('ACDE', 'ALA-CYS-ASP-GLU')  # Input has no separators
        ]
        
        for input_seq, expected in test_cases:
            result = self.generator.convert_protein_one_to_three(input_seq)
            assert result == expected, f"Expected {expected}, got {result} for input {input_seq}"
    
    def test_generate_test_cases(self):
        """Test test case generation."""
        num_cases = 10
        task_type = 'dna_complement'
        
        test_cases = self.generator.generate_test_cases(task_type, num_cases)
        
        assert len(test_cases) == num_cases
        for case in test_cases:
            assert 'task_type' in case
            assert 'input' in case
            assert 'expected_output' in case
            assert case['task_type'] == task_type
            
            # Verify correctness
            expected = self.generator.get_dna_complement(case['input'])
            assert case['expected_output'] == expected


class TestPromptTemplates:
    """Test cases for prompt templates."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.templates = PromptTemplates()
    
    def test_dna_complement_prompt(self):
        """Test DNA complement prompt generation."""
        input_seq = "ATCG"
        prompt = self.templates.get_dna_complement_prompt(input_seq)
        
        assert input_seq in prompt
        assert "complementary DNA sequence" in prompt
        assert "<ANSWER>" in prompt
        assert "</ANSWER>" in prompt
        assert "Adenine (A) always pairs with Thymine (T)" in prompt
    
    def test_rna_complement_prompt(self):
        """Test RNA complement prompt generation."""
        input_seq = "AUCG"
        prompt = self.templates.get_rna_complement_prompt(input_seq)
        
        assert input_seq in prompt
        assert "complementary RNA sequence" in prompt
        assert "<ANSWER>" in prompt
        assert "</ANSWER>" in prompt
        assert "Uracil (U)" in prompt
    
    def test_protein_three_to_one_prompt(self):
        """Test protein three-to-one conversion prompt."""
        input_seq = "GLY-ARG-PHE"
        prompt = self.templates.get_protein_three_to_one_prompt(input_seq)
        
        assert input_seq in prompt
        assert "three-letter" in prompt
        assert "one-letter" in prompt
        assert "GLY = G" in prompt
        assert "<ANSWER>" in prompt
        assert "WITHOUT any separators" in prompt  # New requirement
    
    def test_protein_one_to_three_prompt(self):
        """Test protein one-to-three conversion prompt."""
        input_seq = "GRF"  # Changed from "G-R-F"
        prompt = self.templates.get_protein_one_to_three_prompt(input_seq)
        
        assert input_seq in prompt
        assert "one-letter" in prompt
        assert "three-letter" in prompt
        assert "G = GLY" in prompt
        assert "<ANSWER>" in prompt
        assert "no separators" in prompt  # New requirement
    
    def test_get_prompt_for_task(self):
        """Test getting prompt for specific task type."""
        task_types = ['dna_complement', 'rna_complement', 'protein_three_to_one', 'protein_one_to_three']
        
        for task_type in task_types:
            input_seq = "TEST"
            prompt = self.templates.get_prompt_for_task(task_type, input_seq)
            assert isinstance(prompt, str)
            assert len(prompt) > 100  # Should be substantial prompt
    
    def test_extract_answer_from_response(self):
        """Test answer extraction from model responses."""
        test_cases = [
            ("<ANSWER>TAGC</ANSWER>", "TAGC"),
            ("The result is <ANSWER>GRF</ANSWER> based on the conversion.", "GRF"),  # Updated
            ("TAGC", "TAGC"),  # No tags, should return cleaned response
            ("Multiple lines\nGRF\nMore text", "GRF")  # Should find sequence-like line
        ]
        
        for response, expected in test_cases:
            result = self.templates.extract_answer_from_response(response)
            assert result == expected, f"Expected {expected}, got {result} for response {response}"


class TestSequenceEvaluator:
    """Test cases for the evaluator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = SequenceEvaluator()
    
    def test_normalize_sequence(self):
        """Test sequence normalization."""
        test_cases = [
            ("  ATCG  ", "dna_complement", "ATCG"),
            ("'TAGC'", "dna_complement", "TAGC"),
            ('"GRF"', "protein_three_to_one", "GRF"),  # Updated for new format
            ("GLY-ARG-PHE", "protein_one_to_three", "GLY-ARG-PHE"),  # Updated
            ("  GRF  ", "protein_three_to_one", "GRF")  # Updated
        ]
        
        for input_seq, task_type, expected in test_cases:
            result = self.evaluator._normalize_sequence(input_seq, task_type)
            assert result == expected, f"Expected {expected}, got {result} for input '{input_seq}' and task '{task_type}'"
    
    def test_evaluate_exact_match(self):
        """Test exact match evaluation."""
        # Correct matches
        is_correct, confidence, error = self.evaluator._evaluate_exact_match("ATCG", "ATCG", "dna_complement")
        assert is_correct == True
        assert confidence == 1.0
        assert error == ""
        
        # Incorrect matches
        is_correct, confidence, error = self.evaluator._evaluate_exact_match("ATCG", "ATCC", "dna_complement")
        assert is_correct == False
        assert confidence < 1.0
        assert "Expected: 'ATCG', Got: 'ATCC'" in error
        
        # Test protein sequences with new format
        is_correct, confidence, error = self.evaluator._evaluate_exact_match("GRF", "GRF", "protein_three_to_one")
        assert is_correct == True
        assert confidence == 1.0
    
    def test_calculate_similarity_score(self):
        """Test similarity score calculation."""
        # Identical sequences
        score = self.evaluator._calculate_similarity_score("ATCG", "ATCG")
        assert score == 1.0
        
        # Completely different
        score = self.evaluator._calculate_similarity_score("AAAA", "TTTT")
        assert score == 0.0
        
        # Partial match
        score = self.evaluator._calculate_similarity_score("ATCG", "ATCC")
        assert 0 < score < 1.0
    
    def test_evaluate_single_case(self):
        """Test single case evaluation."""
        test_case = {
            'task_type': 'dna_complement',
            'input': 'ATCG',
            'expected_output': 'TAGC'
        }
        
        # Correct answer
        result = self.evaluator.evaluate_single_case(test_case, 'TAGC', '<ANSWER>TAGC</ANSWER>')
        assert result.is_correct == True
        assert result.confidence_score == 1.0
        assert result.task_type == 'dna_complement'
        
        # Incorrect answer
        result = self.evaluator.evaluate_single_case(test_case, 'TACC', 'Wrong answer')
        assert result.is_correct == False
        assert result.confidence_score < 1.0
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        # Create mock results
        results = [
            EvaluationResult('dna_complement', 'ATCG', 'TAGC', 'TAGC', 'raw1', True, 1.0),
            EvaluationResult('dna_complement', 'GGCC', 'CCGG', 'CCGG', 'raw2', True, 1.0),
            EvaluationResult('rna_complement', 'AUCG', 'UAGC', 'UACC', 'raw3', False, 0.75),
        ]
        
        metrics = self.evaluator.calculate_metrics(results)
        
        assert 'overall' in metrics
        assert 'by_task' in metrics
        assert metrics['overall']['accuracy'] == 2/3
        assert metrics['overall']['total'] == 3
        assert metrics['overall']['correct'] == 2
        
        assert 'dna_complement' in metrics['by_task']
        assert 'rna_complement' in metrics['by_task']
        assert metrics['by_task']['dna_complement']['accuracy'] == 1.0
        assert metrics['by_task']['rna_complement']['accuracy'] == 0.0


class TestBiologicalSequenceBenchmark:
    """Test cases for the main benchmark class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.benchmark = BiologicalSequenceBenchmark()
    
    def test_initialization(self):
        """Test benchmark initialization."""
        assert isinstance(self.benchmark.generator, BiologicalSequenceGenerator)
        assert isinstance(self.benchmark.prompt_templates, PromptTemplates)
        assert isinstance(self.benchmark.evaluator, SequenceEvaluator)
        assert 'tasks' in self.benchmark.config
        assert 'models' in self.benchmark.config
    
    def test_get_default_config(self):
        """Test default configuration generation."""
        config = self.benchmark._get_default_config()
        
        assert 'benchmark_settings' in config
        assert 'tasks' in config
        assert 'models' in config
        
        # Check required tasks are present
        required_tasks = ['dna_complement', 'rna_complement', 'protein_three_to_one', 'protein_one_to_three']
        for task in required_tasks:
            assert task in config['tasks']
            assert 'enabled' in config['tasks'][task]
            assert 'num_cases' in config['tasks'][task]
    
    def test_generate_test_data(self):
        """Test test data generation."""
        # Configure for small test
        self.benchmark.config['tasks'] = {
            'dna_complement': {
                'enabled': True,
                'num_cases': 5,
                'sequence_length_range': [8, 10]
            },
            'rna_complement': {
                'enabled': False,
                'num_cases': 5,
                'sequence_length_range': [8, 10]
            }
        }
        
        test_data = self.benchmark.generate_test_data()
        
        assert 'dna_complement' in test_data
        assert 'rna_complement' not in test_data  # disabled
        assert len(test_data['dna_complement']) == 5
        
        # Verify data structure
        for case in test_data['dna_complement']:
            assert 'task_type' in case
            assert 'input' in case
            assert 'expected_output' in case
            assert case['task_type'] == 'dna_complement'
    
    def test_generate_test_data_with_filter(self):
        """Test test data generation with task filter."""
        test_data = self.benchmark.generate_test_data(['dna_complement'])
        
        assert 'dna_complement' in test_data
        # Other tasks should not be present even if enabled
        other_tasks = ['rna_complement', 'protein_three_to_one', 'protein_one_to_three']
        for task in other_tasks:
            assert task not in test_data


class MockModelInterface:
    """Mock model interface for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.call_count = 0
    
    def generate_response(self, prompt):
        self.call_count += 1
        
        # Return predefined responses based on prompt content
        if 'ATCG' in prompt:
            return '<ANSWER>TAGC</ANSWER>'
        elif 'AUCG' in prompt:
            return '<ANSWER>UAGC</ANSWER>'
        elif 'GLY-ARG-PHE' in prompt:
            return '<ANSWER>GRF</ANSWER>'  # Updated format
        elif 'GRF' in prompt:
            return '<ANSWER>GLY-ARG-PHE</ANSWER>'  # Updated format
        else:
            return '<ANSWER>UNKNOWN</ANSWER>'
    
    def __str__(self):
        return "MockModel"


class TestBenchmarkIntegration:
    """Integration tests for the complete benchmark workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.benchmark = BiologicalSequenceBenchmark()
        
        # Configure for minimal testing
        self.benchmark.config = {
            'benchmark_settings': {
                'results_dir': 'test_results',
                'save_raw_responses': True,
                'save_detailed_results': True
            },
            'tasks': {
                'dna_complement': {
                    'enabled': True,
                    'num_cases': 3,
                    'sequence_length_range': [4, 6]
                },
                'protein_three_to_one': {
                    'enabled': True,
                    'num_cases': 2,
                    'sequence_length_range': [3, 4]
                }
            },
            'models': {}
        }
    
    def test_run_single_model(self):
        """Test running benchmark on a single model."""
        # Generate test data
        test_data = self.benchmark.generate_test_data()
        
        # Create mock model
        mock_model = MockModelInterface()
        
        # Run benchmark
        results = self.benchmark.run_single_model(mock_model, test_data, verbose=False)
        
        assert 'dna_complement' in results
        assert 'protein_three_to_one' in results
        assert len(results['dna_complement']) == 3
        assert len(results['protein_three_to_one']) == 2
        
        # Check that model was called
        assert mock_model.call_count == 5  # 3 + 2 test cases
        
        # Verify result structure
        for task_results in results.values():
            for result in task_results:
                assert isinstance(result, EvaluationResult)
                assert hasattr(result, 'is_correct')
                assert hasattr(result, 'confidence_score')
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create model configuration
        model_configs = {
            'mock_model': {
                'provider': 'mock',  # This won't work with ModelFactory, but we'll mock it
                'model_name': 'mock',
                'parameters': {}
            }
        }
        
        # Generate test data
        test_data = self.benchmark.generate_test_data()
        
        # Verify we have test data
        assert len(test_data) > 0
        total_cases = sum(len(cases) for cases in test_data.values())
        assert total_cases == 5  # 3 DNA + 2 protein cases
        
        # Test data should have correct structure
        for task_type, cases in test_data.items():
            assert task_type in ['dna_complement', 'protein_three_to_one']
            for case in cases:
                assert 'input' in case
                assert 'expected_output' in case
                assert case['task_type'] == task_type
                
                # Verify expected outputs are correct
                if task_type == 'dna_complement':
                    expected = self.benchmark.generator.get_dna_complement(case['input'])
                    assert case['expected_output'] == expected
                elif task_type == 'protein_three_to_one':
                    expected = self.benchmark.generator.convert_protein_three_to_one(case['input'])
                    assert case['expected_output'] == expected


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.benchmark = BiologicalSequenceBenchmark()
    
    def test_invalid_task_type(self):
        """Test handling of invalid task types."""
        with pytest.raises(ValueError, match="Unknown task type"):
            self.benchmark.generator.generate_test_cases('invalid_task', 10)
    
    def test_empty_sequence_handling(self):
        """Test handling of empty sequences."""
        evaluator = SequenceEvaluator()
        
        # Empty sequences should be handled gracefully
        score = evaluator._calculate_similarity_score("", "")
        assert score == 1.0  # Empty sequences are identical
        
        score = evaluator._calculate_similarity_score("ATCG", "")
        assert score == 0.0  # No similarity with empty sequence
    
    def test_malformed_protein_sequences(self):
        """Test handling of malformed protein sequences."""
        evaluator = SequenceEvaluator()
        
        # Test sequences with different formatting for new protein format
        test_cases = [
            ("GRF", "GRF", "protein_three_to_one"),      # Exact match
            ("GRF", "  GRF  ", "protein_three_to_one"),  # Extra whitespace
            ("GLY-ARG-PHE", "GLY-ARG-PHE", "protein_one_to_three"),  # Exact match
            ("GLY-ARG-PHE", "  GLY - ARG - PHE  ", "protein_one_to_three")  # Extra whitespace
        ]
        
        for expected, actual, task_type in test_cases:
            is_correct, confidence, error = evaluator._evaluate_exact_match(expected, actual, task_type)
            # Should be marked as correct after normalization
            assert is_correct == True, f"Failed for expected='{expected}', actual='{actual}', task='{task_type}'"
    
    def test_case_insensitive_evaluation(self):
        """Test that evaluation is case-insensitive."""
        evaluator = SequenceEvaluator()
        
        test_cases = [
            ("ATCG", "atcg", "dna_complement"),
            ("GRF", "grf", "protein_three_to_one"),  # Updated format
            ("GLY-ARG-PHE", "gly-arg-phe", "protein_one_to_three")  # Updated format
        ]
        
        for expected, actual, task_type in test_cases:
            is_correct, confidence, error = evaluator._evaluate_exact_match(expected, actual, task_type)
            assert is_correct == True, f"Case insensitive comparison failed for {expected} vs {actual}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])