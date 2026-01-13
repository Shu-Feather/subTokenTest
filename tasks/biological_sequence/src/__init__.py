"""
Biological Sequence Manipulation Benchmark

A comprehensive benchmark for evaluating Large Language Models (LLMs) 
on biological sequence manipulation tasks.
"""

__version__ = "1.0.0"
__author__ = "Biological Sequence Benchmark Team"

from .benchmark import BiologicalSequenceBenchmark
from .data_generator import BiologicalSequenceGenerator
from .evaluator import SequenceEvaluator
from .model_interface import ModelFactory
from .prompt_templates import PromptTemplates
from .vllm_interface import VLLMInterface, VLLMChatInterface

__all__ = [
    'BiologicalSequenceBenchmark',
    'BiologicalSequenceGenerator', 
    'SequenceEvaluator',
    'ModelFactory',
    'PromptTemplates',
    'VLLMInterface',
    'VLLMChatInterface'
]