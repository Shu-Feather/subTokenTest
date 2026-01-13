import logging
import sys
from typing import Dict, Any
import json


def setup_logger(verbose: bool = False) -> logging.Logger:
    """
    Setup logger for the benchmark.
    
    Args:
        verbose: Whether to enable verbose logging
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger('2d_nav_benchmark')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


def log_interaction(
    logger: logging.Logger,
    task_id: int,
    prompt: str,
    response: str,
    token_usage: Dict[str, int],
    verbose: bool = False
):
    """
    Log interaction with LLM.
    
    Args:
        logger: Logger instance
        task_id: ID of the current task
        prompt: Input prompt to the model
        response: Model's response
        token_usage: Dictionary with token usage info
        verbose: Whether to log detailed information
    """
    if verbose:
        logger.info(f"\n{'='*80}")
        logger.info(f"Task ID: {task_id}")
        logger.info(f"\n--- PROMPT ---\n{prompt}")
        logger.info(f"\n--- RESPONSE ---\n{response}")
        logger.info(f"\n--- TOKEN USAGE ---")
        logger.info(f"Prompt tokens: {token_usage.get('prompt_tokens', 'N/A')}")
        logger.info(f"Completion tokens: {token_usage.get('completion_tokens', 'N/A')}")
        logger.info(f"Total tokens: {token_usage.get('total_tokens', 'N/A')}")
        logger.info(f"{'='*80}\n")


def save_interaction(
    interactions: list,
    task_id: int,
    task_data: Dict[str, Any],
    prompt: str,
    response: str,
    parsed_answer: str,
    is_correct: bool,
    token_usage: Dict[str, int]
):
    """
    Save interaction data for later analysis.
    
    Args:
        interactions: List to append interaction data to
        task_id: ID of the current task
        task_data: Original task data
        prompt: Input prompt
        response: Model response
        parsed_answer: Extracted answer
        is_correct: Whether the answer is correct
        token_usage: Token usage information
    """
    interaction = {
        'task_id': task_id,
        'env_type': task_data.get('env_type'),
        'map_id': task_data.get('map_id'),
        'task_type': task_data.get('task_type'),
        'question': task_data.get('question'),
        'ground_truth': task_data.get('answer'),
        'prompt': prompt,
        'response': response,
        'parsed_answer': parsed_answer,
        'is_correct': is_correct,
        'token_usage': token_usage
    }
    interactions.append(interaction)


def log_interaction(
    logger: logging.Logger,
    task_id: int,
    prompt: str,
    response: str,
    token_usage: Dict[str, int],
    verbose: bool = False
):
    """
    Log interaction with LLM.
    
    Args:
        logger: Logger instance
        task_id: ID of the current task
        prompt: Input prompt to the model
        response: Model's response
        token_usage: Dictionary with token usage info
        verbose: Whether to log detailed information
    """
    if verbose:
        logger.info(f"\n{'='*80}")
        logger.info(f"Task ID: {task_id}")
        logger.info(f"\n--- PROMPT ---\n{prompt}")
        logger.info(f"\n--- RESPONSE ---\n{response}")
        logger.info(f"\n--- TOKEN USAGE ---")
        logger.info(f"Total tokens: {token_usage.get('total_tokens', 'N/A')}")
        logger.info(f"Prompt tokens: {token_usage.get('prompt_tokens', 'N/A')}")
        logger.info(f"Completion tokens: {token_usage.get('completion_tokens', 'N/A')}")
        logger.info(f"Reasoning tokens: {token_usage.get('reasoning_tokens', 0)}")
        logger.info(f"Visible output tokens: {token_usage.get('output_tokens', 'N/A')}")
        
        # Calculate and show thinking ratio if applicable
        if token_usage.get('reasoning_tokens', 0) > 0 and token_usage.get('total_tokens', 0) > 0:
            thinking_ratio = token_usage['reasoning_tokens'] / token_usage['total_tokens']
            logger.info(f"Thinking ratio: {thinking_ratio:.2%}")
        
        logger.info(f"{'='*80}\n")


def save_interaction(
    interactions: list,
    task_id: int,
    task_data: Dict[str, Any],
    prompt: str,
    response: str,
    parsed_answer: str,
    is_correct: bool,
    token_usage: Dict[str, int]
):
    """
    Save interaction data for later analysis.
    
    Args:
        interactions: List to append interaction data to
        task_id: ID of the current task
        task_data: Original task data
        prompt: Input prompt
        response: Model response
        parsed_answer: Extracted answer
        is_correct: Whether the answer is correct
        token_usage: Token usage information
    """
    interaction = {
        'task_id': task_id,
        'env_type': task_data.get('env_type'),
        'map_id': task_data.get('map_id'),
        'task_type': task_data.get('task_type'),
        'question': task_data.get('question'),
        'ground_truth': task_data.get('answer'),
        'prompt': prompt,
        'response': response,
        'parsed_answer': parsed_answer,
        'is_correct': is_correct,
        'token_usage': {
            'total_tokens': token_usage.get('total_tokens', 0),
            'prompt_tokens': token_usage.get('prompt_tokens', 0),
            'completion_tokens': token_usage.get('completion_tokens', 0),
            'reasoning_tokens': token_usage.get('reasoning_tokens', 0),
            'output_tokens': token_usage.get('output_tokens', 0),
        }
    }
    interactions.append(interaction)