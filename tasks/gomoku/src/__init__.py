"""
Gomoku Benchmark Package

This package contains the core components for the Gomoku benchmark system.
"""

__version__ = "1.0.0"
__author__ = "Gomoku Benchmark Team"

# Import main classes for convenience
from .board_generator import BoardGenerator, generate_test_cases
from .model_interface import (
    ModelInterface, 
    OpenAIInterface, 
    DeepSeekInterface, 
    OllamaInterface, 
    TransformersInterface,
    create_model_interface
)
from .benchmark_runner import BenchmarkRunner
from .evaluator import (
    BenchmarkResults, 
    ResponseParser, 
    evaluate_responses, 
    print_evaluation_summary
)
from .vllm_interface import VLLMInterface, VLLMServerInterface

__all__ = [
    'BoardGenerator',
    'generate_test_cases',
    'ModelInterface',
    'OpenAIInterface',
    'DeepSeekInterface',
    'OllamaInterface',
    'VLLMInterface',
    'VLLMServerInterface',
    'TransformersInterface',
    'create_model_interface',
    'BenchmarkRunner',
    'BenchmarkResults',
    'ResponseParser',
    'evaluate_responses',
    'print_evaluation_summary'
]