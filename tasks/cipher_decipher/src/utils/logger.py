"""
Utility module for logging model interactions and responses for the Cipher & Decipher Benchmark.
"""

import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('benchmark.log')
    ]
)
logger = logging.getLogger(__name__)

class ResponseLogger:
    """Logger for saving model prompts and responses with token usage tracking."""
    
    def __init__(self, log_file: str):
        """
        Initialize response logger.
        
        Args:
            log_file (str): Path to log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.responses = []
        
        logger.info(f"Response logger initialized: {log_file}")
    
    def log_interaction(self, task_type: str, prompt: str, response: str, 
                    expected_output: str, extracted_answer: str,
                    is_correct: bool, difficulty: str = None,
                    usage_info: Dict[str, Any] = None,
                    additional_info: Dict[str, Any] = None,
                    similarity: float = None):  
        """
        Log a single model interaction with token usage information.
        
        Args:
            task_type (str): Type of task
            prompt (str): Input prompt
            response (str): Model response
            expected_output (str): Expected/golden answer
            extracted_answer (str): Answer extracted from response
            is_correct (bool): Whether the answer is correct
            difficulty (str): Difficulty level
            usage_info (Dict[str, Any]): Token usage information
            additional_info (Dict): Additional information
            similarity (float): Similarity score between extracted and golden answer
        """
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'task_type': task_type,
            'difficulty': difficulty,
            'prompt': prompt,
            'model_response': response,
            'golden_answer': expected_output,
            'extracted_answer': extracted_answer,
            'is_correct': is_correct,
            'similarity': similarity, 
            'token_usage': usage_info or {
                'total_tokens': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'reasoning_tokens': 0
            },
            'additional_info': additional_info or {}
        }
        self.responses.append(interaction)
    
    def save_to_file(self, format: str = 'json'):
        """
        Save logged interactions to file.
        
        Args:
            format (str): Output format ('json' or 'txt')
        """
        try:
            if format == 'json':
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    json.dump(self.responses, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(self.responses)} interactions to {self.log_file}")
            
            elif format == 'txt':
                txt_file = self.log_file.with_suffix('.txt')
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write("CIPHER & DECIPHER BENCHMARK - MODEL INTERACTIONS LOG\n")
                    f.write("="*80 + "\n\n")
                    
                    for i, interaction in enumerate(self.responses, 1):
                        f.write(f"\n{'='*80}\n")
                        f.write(f"INTERACTION #{i}\n")
                        f.write(f"{'='*80}\n")
                        f.write(f"Timestamp: {interaction['timestamp']}\n")
                        f.write(f"Task Type: {interaction['task_type']}\n")
                        if interaction.get('difficulty'):
                            f.write(f"Difficulty: {interaction['difficulty']}\n")
                        if interaction.get('additional_info'):
                            f.write(f"Additional Info: {interaction['additional_info']}\n")
                        
                        # Add token usage information
                        f.write(f"\n{'-'*80}\n")
                        f.write(f"TOKEN USAGE:\n{'-'*80}\n")
                        usage = interaction.get('token_usage', {})
                        f.write(f"Total Tokens: {usage.get('total_tokens', 0)}\n")
                        f.write(f"Input Tokens: {usage.get('input_tokens', 0)}\n")
                        f.write(f"Output Tokens: {usage.get('output_tokens', 0)}\n")
                        f.write(f"Reasoning Tokens: {usage.get('reasoning_tokens', 0)}\n")
                        
                        f.write(f"\n{'-'*80}\n")
                        f.write(f"PROMPT:\n{'-'*80}\n")
                        f.write(f"{interaction['prompt']}\n")
                        f.write(f"\n{'-'*80}\n")
                        f.write(f"MODEL RESPONSE:\n{'-'*80}\n")
                        f.write(f"{interaction['model_response']}\n")
                        f.write(f"\n{'-'*80}\n")
                        f.write(f"GOLDEN ANSWER:\n{'-'*80}\n")
                        f.write(f"{interaction['golden_answer']}\n")
                        f.write(f"\n{'-'*80}\n")
                        f.write(f"EXTRACTED ANSWER:\n{'-'*80}\n")
                        f.write(f"{interaction['extracted_answer']}\n")
                        f.write(f"\n{'-'*80}\n")
                        similarity = interaction.get('similarity')
                        similarity_str = f"{similarity:.4f}" if similarity is not None else "N/A"
                        f.write(f"SIMILARITY: {similarity_str}\n")
                        f.write(f"RESULT: {'✓ CORRECT' if interaction['is_correct'] else '✗ INCORRECT'}\n")
                        f.write(f"{'='*80}\n")
                
                logger.info(f"Saved {len(self.responses)} interactions to {txt_file}")
            
            # Save both formats
            if format == 'json':
                self.save_to_file('txt')
                
        except Exception as e:
            logger.error(f"Error saving interactions: {e}")
    
    def get_token_usage_summary(self) -> Dict[str, Any]:
        """
        Get summary of token usage across all logged interactions.
        
        Returns:
            Dict[str, Any]: Token usage summary statistics
        """
        if not self.responses:
            return {
                'total_interactions': 0,
                'total_tokens': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_reasoning_tokens': 0,
                'avg_tokens_per_interaction': 0,
                'by_task_type': {},
                'by_difficulty': {}
            }
        
        total_tokens = sum(r.get('token_usage', {}).get('total_tokens', 0) for r in self.responses)
        total_input = sum(r.get('token_usage', {}).get('input_tokens', 0) for r in self.responses)
        total_output = sum(r.get('token_usage', {}).get('output_tokens', 0) for r in self.responses)
        total_reasoning = sum(r.get('token_usage', {}).get('reasoning_tokens', 0) for r in self.responses)
        
        # Group by task type
        by_task = {}
        for r in self.responses:
            task_type = r.get('task_type', 'unknown')
            if task_type not in by_task:
                by_task[task_type] = {
                    'count': 0,
                    'total_tokens': 0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'reasoning_tokens': 0
                }
            usage = r.get('token_usage', {})
            by_task[task_type]['count'] += 1
            by_task[task_type]['total_tokens'] += usage.get('total_tokens', 0)
            by_task[task_type]['input_tokens'] += usage.get('input_tokens', 0)
            by_task[task_type]['output_tokens'] += usage.get('output_tokens', 0)
            by_task[task_type]['reasoning_tokens'] += usage.get('reasoning_tokens', 0)
        
        # Calculate averages for each task type
        for task_type in by_task:
            count = by_task[task_type]['count']
            if count > 0:
                by_task[task_type]['avg_total_tokens'] = by_task[task_type]['total_tokens'] / count
                by_task[task_type]['avg_input_tokens'] = by_task[task_type]['input_tokens'] / count
                by_task[task_type]['avg_output_tokens'] = by_task[task_type]['output_tokens'] / count
                by_task[task_type]['avg_reasoning_tokens'] = by_task[task_type]['reasoning_tokens'] / count
        
        # Group by difficulty
        by_difficulty = {}
        for r in self.responses:
            difficulty = r.get('difficulty', 'unknown')
            if difficulty not in by_difficulty:
                by_difficulty[difficulty] = {
                    'count': 0,
                    'total_tokens': 0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'reasoning_tokens': 0
                }
            usage = r.get('token_usage', {})
            by_difficulty[difficulty]['count'] += 1
            by_difficulty[difficulty]['total_tokens'] += usage.get('total_tokens', 0)
            by_difficulty[difficulty]['input_tokens'] += usage.get('input_tokens', 0)
            by_difficulty[difficulty]['output_tokens'] += usage.get('output_tokens', 0)
            by_difficulty[difficulty]['reasoning_tokens'] += usage.get('reasoning_tokens', 0)
        
        # Calculate averages for each difficulty
        for difficulty in by_difficulty:
            count = by_difficulty[difficulty]['count']
            if count > 0:
                by_difficulty[difficulty]['avg_total_tokens'] = by_difficulty[difficulty]['total_tokens'] / count
                by_difficulty[difficulty]['avg_input_tokens'] = by_difficulty[difficulty]['input_tokens'] / count
                by_difficulty[difficulty]['avg_output_tokens'] = by_difficulty[difficulty]['output_tokens'] / count
                by_difficulty[difficulty]['avg_reasoning_tokens'] = by_difficulty[difficulty]['reasoning_tokens'] / count
        
        return {
            'total_interactions': len(self.responses),
            'total_tokens': total_tokens,
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'total_reasoning_tokens': total_reasoning,
            'avg_tokens_per_interaction': total_tokens / len(self.responses) if self.responses else 0,
            'by_task_type': by_task,
            'by_difficulty': by_difficulty
        }