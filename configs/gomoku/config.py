"""
Configuration file for the Chessboard Benchmark project.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    """Configuration for different model types"""
    model_name: str
    api_key: str = ""
    base_url: str = ""
    model_type: str = "openai"  # "openai", "deepseek", "ollama", "transformers"
    reasoning_effort: str = "low"  # "low", "medium", "high"
    max_tokens: int = 32768
    temperature: float = 0.6
    top_p: float = 0.95
    timeout: int = 1000
    results_dir: str = "./results" 
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 2
    enforce_eager: bool = True
    restricted_reasoning: bool = False
    batch_size: int = 4

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark parameters"""
    board_sizes: list = None
    test_counts: list = None
    output_dir: str = "results"
    data_dir: str = "data"
    verbose: bool = False  # Enable verbose logging
    test_file: str = None  # Load test cases from file
    save_response: Optional[str] = None
    
    def __post_init__(self):
        if self.board_sizes is None:
            self.board_sizes = [9, 15, 19]
        if self.test_counts is None:
            self.test_counts = [100, 200, 500]

# Default model configurations
DEFAULT_MODELS = {
    "gpt-5": ModelConfig(
        model_name="gpt-5",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model_type="openai",
        temperature=0.0,
        max_tokens=50000
    ),
    "o4-mini": ModelConfig(
        model_name="o4-mini",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model_type="openai",
        reasoning_effort="low",
        temperature=0.0,
        max_tokens=50000
    ),
    "gpt-4": ModelConfig(
        model_name="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model_type="openai",
        temperature=0.0,
        max_tokens=5000
    ),
    "gpt-3.5-turbo": ModelConfig(
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model_type="openai",
        temperature=0.0,
        max_tokens=5000
    ),
    "deepseek-v3": ModelConfig(
        model_name="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
        model_type="deepseek",
        temperature=0.0,
        max_tokens=5000
    ),
    "deepseek-r1": ModelConfig(
        model_name="deepseek-reasoner",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
        model_type="deepseek",
        temperature=0.0,
        max_tokens=50000
    ),
    # vLLM-based local models (recommended for local inference)
    "llama-3-8b": ModelConfig(
        model_name="/path/to/local/llama-3-8b-instruct",
        model_type="vllm",
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        batch_size=4
    ),
    "qwen2-7b": ModelConfig(
        model_name="/path/to/local/qwen-7b",
        model_type="vllm",
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        batch_size=4
    ),
    "qwen2.5-14b": ModelConfig(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        model_type="vllm",
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        batch_size=4
    ),
    "qwen2.5-32b": ModelConfig(
        model_name="/path/to/local/qwen-32b",
        model_type="vllm",
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
        enforce_eager=True,
        batch_size=4
    ),
    "deepseek-r1-distill-qwen-32b": ModelConfig(
        model_name="/path/to/local/r1-distill-qwen-32b",
        model_type="vllm",
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
        enforce_eager=True,
        batch_size=4
    ),
    "qwen2.5-7b": ModelConfig(
        model_name="/path/to/local/qwen-7b",
        model_type="vllm",
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
        enforce_eager=True,
        batch_size=4
    ),
    "deepseek-r1-distill-qwen-7b": ModelConfig(
        model_name="/path/to/local/r1-distill-qwen-7b",
        model_type="vllm",
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
        enforce_eager=True,
        batch_size=4
    ),
    # vLLM server (if running vllm serve)
    "vllm-server": ModelConfig(
        model_name="default",  # Will be overridden by server
        model_type="vllm-server",
        base_url="http://localhost:8000",
        api_key="EMPTY"
    ),
    # Legacy: Ollama models (deprecated, use vLLM instead)
    "qwen2.5": ModelConfig(
        model_name="qwen2.5",
        model_type="ollama",
        base_url="http://localhost:11434"
    ),
    "llama3.2": ModelConfig(
        model_name="llama3.2",
        model_type="ollama",
        base_url="http://localhost:11434"
    )
}

# Benchmark configuration
BENCHMARK_CONFIG = BenchmarkConfig()

# Prompt templates
SYSTEM_PROMPT = """You are an expert at analyzing Gomoku (Five-in-a-Row) game boards. Your task is to determine the current game state by analyzing the board configuration.

Rules:
- W represents a white stone
- B represents a black stone  
- E represents an empty position
- Victory condition: 5 consecutive stones of the same color in a row (horizontal, vertical, or diagonal)
- Only one player can win, or no one wins

You should analyze the board carefully and provide your reasoning. Then, wrap your final answer in <answer></answer> tags with exactly one of these three options:
- WHITE_WINS if white has 5 consecutive stones
- BLACK_WINS if black has 5 consecutive stones
- NO_WINNER if neither player has achieved victory

Example response format:
[Your analysis here...]
<answer>Your option here</answer>

Be careful to choose only one of the three options (WHITE_WINS, BLACK_WINS, NO_WINNER) and format your answer correctly."""

USER_PROMPT_TEMPLATE = """Please analyze this Gomoku board and determine the game state:

Board size: {board_size}x{board_size}
{board_representation}

What is the current game state? Remember to wrap your final answer in <answer></answer> tags."""
