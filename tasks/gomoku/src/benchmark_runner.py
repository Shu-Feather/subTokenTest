"""
Main benchmark runner for the Gomoku evaluation system.
"""

import os
import re
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import asdict
import logging

from configs.gomoku.config import ModelConfig, BenchmarkConfig
from .board_generator import generate_test_cases
from .model_interface import create_model_interface
from .evaluator import evaluate_responses, BenchmarkResults
from src.response_logger import ResponseLogger

class BenchmarkRunner:
    """Main class for running the Gomoku benchmark"""
    
    def __init__(self, benchmark_config: BenchmarkConfig):
        self.config = benchmark_config
        self.setup_directories()
        self.setup_logging()
        self.test_file_data = None

        # Initialize response logger
        self.response_logger = ResponseLogger(benchmark_config.save_response) if benchmark_config.save_response else None
    
        # Load test file if specified
        if self.config.test_file:
            self.test_file_data = self.load_test_file(self.config.test_file)
    
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.data_dir, exist_ok=True)
        
        # Create logs directory if response logging is enabled
        if self.config.save_response:
            log_dir = os.path.dirname(self.config.save_response)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.output_dir, 'benchmark.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_test_file(self, test_file_path: str) -> Dict[str, List[Tuple]]:
        """
        Load test cases from a specified file
        
        Returns:
            Dictionary mapping board_size to list of test cases
        """
        self.logger.info(f"Loading test cases from {test_file_path}")
        
        try:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both formats: with metadata and without
            if isinstance(data, dict) and 'test_cases' in data:
                test_cases = data['test_cases']
                metadata = data.get('metadata', {})
                self.logger.info(f"Loaded test file with metadata: {metadata}")
            else:
                test_cases = data
            
            # Group test cases by board size
            grouped_cases = {}
            for case in test_cases:
                board_size = case.get('board_size')
                if board_size not in grouped_cases:
                    grouped_cases[board_size] = []
                grouped_cases[board_size].append((case['board'], case['expected']))
            
            self.logger.info(f"Loaded {len(test_cases)} test cases")
            for size, cases in grouped_cases.items():
                self.logger.info(f"  - {size}x{size}: {len(cases)} cases")
            
            return grouped_cases
            
        except Exception as e:
            self.logger.error(f"Failed to load test file {test_file_path}: {e}")
            raise
    
    def generate_or_load_test_data(
        self, 
        board_size: int, 
        num_cases: int
    ) -> List[Tuple[str, str]]:
        """Generate or load test data for given parameters"""
        
        # If test file is specified, use it
        if self.test_file_data is not None:
            if board_size in self.test_file_data:
                available_cases = self.test_file_data[board_size]
                
                # Use requested number of cases or all available
                if num_cases <= len(available_cases):
                    selected_cases = available_cases[:num_cases]
                else:
                    self.logger.warning(
                        f"Requested {num_cases} cases for {board_size}x{board_size}, "
                        f"but only {len(available_cases)} available. Using all available cases."
                    )
                    selected_cases = available_cases
                
                self.logger.info(
                    f"Using {len(selected_cases)} cases from test file "
                    f"for {board_size}x{board_size} board"
                )
                return selected_cases
            else:
                self.logger.warning(
                    f"Test file does not contain cases for {board_size}x{board_size} board. "
                    f"Falling back to generation/loading from data directory."
                )
        
        # Fall back to original behavior: load from data_dir or generate
        data_filename = f"test_data_{board_size}x{board_size}_{num_cases}.json"
        data_path = os.path.join(self.config.data_dir, data_filename)
        
        if os.path.exists(data_path):
            self.logger.info(f"Loading existing test data from {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle both formats
                if isinstance(data, dict) and 'test_cases' in data:
                    cases = data['test_cases']
                else:
                    cases = data
                
                return [(case['board'], case['expected']) for case in cases]
        
        self.logger.info(f"Generating new test data: {board_size}x{board_size}, {num_cases} cases")
        test_cases = generate_test_cases(board_size, num_cases)
        
        # Save test data
        data_to_save = [
            {
                'board': board,
                'expected': expected,
                'board_size': board_size
            }
            for board, expected in test_cases
        ]
        
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Test data saved to {data_path}")
        return test_cases
    
    def extract_answer_from_response(self, response: str) -> Any:
        """
        Parse model response to extract game state
        Returns: One of VALID_RESPONSES, "ERROR", or "PARSE_FAIL"
        """
        if not response or response == "ERROR":
            return "ERROR"
        
        # Clean the response
        cleaned_response = response.strip().upper()
        
        # Pattern matching for various response formats
        patterns = [
            # Exact matches with quotes or punctuation
            r'^["\']?(WHITE_WINS|BLACK_WINS|NO_WINNER)["\']?[.!]?$',
            
            # Match with explanatory text
            r'(?:ANSWER|RESULT|STATE|WINNER)?\s*:?\s*(WHITE_WINS|BLACK_WINS|NO_WINNER)',
            
            # Match in sentences
            r'\b(WHITE_WINS|BLACK_WINS|NO_WINNER)\b',
            
            # Alternative phrasings
            r'\b(WHITE\s+WINS?|BLACK\s+WINS?|NO\s+WINNER)\b',
            
            # Match "white wins", "black wins", "no winner"
            r'\b(WHITE|BLACK)\s+(WIN|WINS)\b',
            r'\bNO\s+(WINNER|WIN)\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cleaned_response)
            if match:
                matched_text = match.group(1).replace(' ', '_').upper()
                
                # Normalize variations
                if matched_text in ['WHITE_WIN', 'WHITE_WINS']:
                    return 'WHITE_WINS'
                elif matched_text in ['BLACK_WIN', 'BLACK_WINS']:
                    return 'BLACK_WINS'
                elif matched_text in ['NO_WINNER', 'NO_WIN']:
                    return 'NO_WINNER'
        
        # Check for semantic matches
        if any(phrase in cleaned_response for phrase in ['WHITE WIN', 'WHITE VICTORY', 'WHITE HAS WON']):
            return 'WHITE_WINS'
        elif any(phrase in cleaned_response for phrase in ['BLACK WIN', 'BLACK VICTORY', 'BLACK HAS WON']):
            return 'BLACK_WINS'
        elif any(phrase in cleaned_response for phrase in ['NO WIN', 'NO VICTOR', 'DRAW', 'TIE', 'NOBODY WINS']):
            return 'NO_WINNER'
        
        return "PARSE_FAIL"
    
    def run_single_model_test(
        self, 
        model_config: ModelConfig,
        board_size: int,
        num_cases: int
    ) -> Dict[str, Any]:
        """Run benchmark for a single model configuration"""
        
        self.logger.info(f"Running test for {model_config.model_name} "
                        f"(board_size={board_size}, cases={num_cases})")
        
        # Generate or load test data
        test_cases = self.generate_or_load_test_data(board_size, num_cases)
        
        # Update num_cases if fewer cases are available
        actual_num_cases = len(test_cases)
        if actual_num_cases != num_cases:
            self.logger.info(f"Using {actual_num_cases} cases instead of requested {num_cases}")
            num_cases = actual_num_cases
        
        # Create model interface
        try:
            model_interface = create_model_interface(model_config)
        except Exception as e:
            self.logger.error(f"Failed to create model interface: {e}")
            return self._create_error_result(model_config, board_size, num_cases, str(e))
        
        # Run evaluation
        start_time = time.time()
        predictions = []
        successful_cases = 0
        
        # Track token usage
        total_tokens = 0
        total_reasoning_tokens = 0
        total_output_tokens = 0
        
        for i, (board, expected) in enumerate(test_cases):
            try:
                # Get response with usage info (returns tuple: (response, usage_info))
                result = model_interface.generate_response(
                    board, 
                    board_size,
                    verbose=self.config.verbose
                )
                
                # Handle both old (str) and new (tuple) return formats
                if isinstance(result, tuple):
                    response, usage_info = result
                else:
                    # Backward compatibility
                    response = result
                    usage_info = None
                
                if response is not None:
                    # Extract answer from response
                    extracted_answer = self.extract_answer_from_response(response)
                    
                    # Check if answer is correct
                    is_correct = (extracted_answer == expected)
                    
                    # Add to predictions list for evaluator
                    predictions.append((expected, response))
                    successful_cases += 1
                    
                    # Track token usage
                    if usage_info:
                        total_tokens += usage_info.get('total_tokens', 0)
                        total_reasoning_tokens += usage_info.get('reasoning_tokens', 0)
                        total_output_tokens += usage_info.get('output_tokens', 0)
                    
                    # Log the response
                    if self.response_logger:
                        self.response_logger.log_response(
                            model_name=model_config.model_name,
                            board_size=board_size,
                            test_case_index=i,
                            board_representation=board,
                            model_response=response,
                            extracted_answer=extracted_answer,
                            expected_answer=expected,
                            is_correct=is_correct,
                            usage_info=usage_info,
                            metadata={
                                'model_type': model_config.model_type,
                                'test_config': f"{board_size}x{board_size}_{num_cases}"
                            }
                        )
                    
                    if self.config.verbose:
                        self.logger.info(f"Case {i+1}/{len(test_cases)}")
                        self.logger.info(f"  Expected: {expected}")
                        self.logger.info(f"  Extracted: {extracted_answer}")
                        self.logger.info(f"  Correct: {is_correct}")
                        if usage_info:
                            self.logger.info(f"  Tokens: {usage_info.get('total_tokens', 0)} "
                                           f"(reasoning: {usage_info.get('reasoning_tokens', 0)})")
                else:
                    predictions.append((expected, "ERROR"))
                    
                    # Log error case
                    if self.response_logger:
                        self.response_logger.log_response(
                            model_name=model_config.model_name,
                            board_size=board_size,
                            test_case_index=i,
                            board_representation=board,
                            model_response="ERROR: No response",
                            extracted_answer=None,
                            expected_answer=expected,
                            is_correct=False,
                            usage_info=None,
                            metadata={'error': 'No response from model'}
                        )
                    
                    if self.config.verbose:
                        self.logger.warning(f"Case {i+1}/{len(test_cases)} - ERROR: No response")
                    
            except Exception as e:
                self.logger.warning(f"Error processing case {i+1}: {e}")
                predictions.append((expected, "ERROR"))
                
                # Log error case
                if self.response_logger:
                    self.response_logger.log_response(
                        model_name=model_config.model_name,
                        board_size=board_size,
                        test_case_index=i,
                        board_representation=board,
                        model_response=f"ERROR: {str(e)}",
                        extracted_answer=None,
                        expected_answer=expected,
                        is_correct=False,
                        usage_info=None,
                        metadata={'error': str(e)}
                    )
            
            # Progress logging (only if not in verbose mode to avoid clutter)
            if not self.config.verbose and ((i + 1) % 10 == 0 or i == len(test_cases) - 1):
                self.logger.info(f"Processed {i+1}/{len(test_cases)} cases")
        
        end_time = time.time()
        
        # Evaluate results
        results = evaluate_responses(predictions)
        
        # Compile final results with token usage
        test_result = {
            'model_name': model_config.model_name,
            'model_type': model_config.model_type,
            'board_size': board_size,
            'num_cases': num_cases,
            'successful_cases': successful_cases,
            'execution_time': end_time - start_time,
            'timestamp': datetime.now().isoformat(),
            'test_file_used': self.config.test_file is not None,
            'results': asdict(results),
            'token_usage': {
                'total_tokens': total_tokens,
                'total_reasoning_tokens': total_reasoning_tokens,
                'total_output_tokens': total_output_tokens,
                'avg_tokens_per_case': total_tokens / num_cases if num_cases > 0 else 0,
                'avg_reasoning_per_case': total_reasoning_tokens / num_cases if num_cases > 0 else 0,
                'avg_output_per_case': total_output_tokens / num_cases if num_cases > 0 else 0
            }
        }
        
        self.logger.info(f"Test completed. Accuracy: {results.accuracy:.2%}, "
                        f"Time: {end_time - start_time:.2f}s, "
                        f"Total Tokens: {total_tokens}")
        
        return test_result
    
    def _create_error_result(
        self, 
        model_config: ModelConfig, 
        board_size: int, 
        num_cases: int, 
        error_msg: str
    ) -> Dict[str, Any]:
        """Create error result when model fails to load"""
        return {
            'model_name': model_config.model_name,
            'model_type': model_config.model_type,
            'board_size': board_size,
            'num_cases': num_cases,
            'successful_cases': 0,
            'execution_time': 0,
            'timestamp': datetime.now().isoformat(),
            'error': error_msg,
            'test_file_used': self.config.test_file is not None,
            'results': asdict(BenchmarkResults()),  # Empty results
            'token_usage': {
                'total_tokens': 0,
                'total_reasoning_tokens': 0,
                'total_output_tokens': 0,
                'avg_tokens_per_case': 0,
                'avg_reasoning_per_case': 0,
                'avg_output_per_case': 0
            }
        }
    
    def run_benchmark(
        self,
        model_configs: List[ModelConfig],
        board_sizes: List[int] = None,
        test_counts: List[int] = None
    ) -> Dict[str, Any]:
        """Run complete benchmark across multiple configurations"""
        
        if board_sizes is None:
            board_sizes = self.config.board_sizes
        if test_counts is None:
            test_counts = self.config.test_counts
        
        # If using test file, adjust board sizes accordingly
        if self.test_file_data is not None:
            board_sizes = list(self.test_file_data.keys())
            
            self.logger.info(f"Using test file, available board sizes: {board_sizes}")
        
        self.logger.info("Starting benchmark run")
        self.logger.info(f"Models: {[cfg.model_name for cfg in model_configs]}")
        self.logger.info(f"Board sizes: {board_sizes}")
        self.logger.info(f"Test counts: {test_counts}")
        if self.config.test_file:
            self.logger.info(f"Using test file: {self.config.test_file}")
        if self.config.save_response:
            self.logger.info(f"Response logging enabled: {self.config.save_response}")
        
        all_results = []
        
        try:
            for model_config in model_configs:
                for board_size in board_sizes:
                    for num_cases in test_counts:
                        try:
                            result = self.run_single_model_test(model_config, board_size, num_cases)
                            all_results.append(result)
                            
                            # Save individual result
                            result_filename = (f"{model_config.model_name.replace('/', '_')}_{board_size}x{board_size}_"
                                             f"{num_cases}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                            result_path = os.path.join(self.config.output_dir, result_filename)
                            
                            with open(result_path, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                            
                        except Exception as e:
                            self.logger.error(f"Unexpected error in benchmark: {e}")
                            if self.config.verbose:
                                import traceback
                                traceback.print_exc()
                            error_result = self._create_error_result(
                                model_config, board_size, num_cases, str(e)
                            )
                            all_results.append(error_result)
            
            # Create summary report
            summary = self._create_summary_report(all_results)
            
            # Save response logs if enabled
            if self.response_logger:
                self.response_logger.save()
                log_summary = self.response_logger.get_summary()
                
                self.logger.info("\n" + "="*60)
                self.logger.info("Response Log Summary")
                self.logger.info("="*60)
                self.logger.info(f"Total cases logged: {log_summary.get('total_cases', 0)}")
                self.logger.info(f"Correct cases: {log_summary.get('correct_cases', 0)}")
                self.logger.info(f"Accuracy: {log_summary.get('accuracy', 0):.2%}")
                self.logger.info(f"Total tokens used: {log_summary.get('total_tokens_used', 0):,}")
                self.logger.info(f"Total reasoning tokens: {log_summary.get('total_reasoning_tokens', 0):,}")
                self.logger.info(f"Total output tokens: {log_summary.get('total_output_tokens', 0):,}")
                self.logger.info(f"Avg tokens per case: {log_summary.get('avg_tokens_per_case', 0):.1f}")
                self.logger.info(f"Avg reasoning per case: {log_summary.get('avg_reasoning_tokens_per_case', 0):.1f}")
                self.logger.info(f"Avg output per case: {log_summary.get('avg_output_tokens_per_case', 0):.1f}")
                self.logger.info("="*60)
                
                # Add log summary to complete results
                summary['response_log_summary'] = log_summary
            
            # Save complete results
            complete_results = {
                'summary': summary,
                'detailed_results': all_results,
                'benchmark_config': {
                    'board_sizes': board_sizes,
                    'test_counts': test_counts,
                    'test_file': self.config.test_file,
                    'save_response': self.config.save_response,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            summary_filename = f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            summary_path = os.path.join(self.config.output_dir, summary_filename)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(complete_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Benchmark completed. Summary saved to {summary_path}")
            
            return complete_results
            
        except Exception as e:
            # Save logs even if there's an error
            if self.response_logger:
                self.logger.info("Saving response logs due to error...")
                self.response_logger.save()
            raise
    
    def _create_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary report from all results"""
        if not results:
            return {}
        
        # Group results by model
        model_summaries = {}
        
        for result in results:
            model_name = result['model_name']
            if model_name not in model_summaries:
                model_summaries[model_name] = {
                    'model_type': result['model_type'],
                    'test_configurations': [],
                    'average_accuracy': 0,
                    'total_cases': 0,
                    'successful_cases': 0,
                    'total_time': 0,
                    'total_tokens': 0,
                    'total_reasoning_tokens': 0,
                    'total_output_tokens': 0
                }
            
            model_summary = model_summaries[model_name]
            
            # Extract token usage from result
            token_usage = result.get('token_usage', {})
            
            model_summary['test_configurations'].append({
                'board_size': result['board_size'],
                'num_cases': result['num_cases'],
                'accuracy': result.get('results', {}).get('accuracy', 0),
                'execution_time': result.get('execution_time', 0),
                'tokens': token_usage.get('total_tokens', 0),
                'reasoning_tokens': token_usage.get('total_reasoning_tokens', 0),
                'output_tokens': token_usage.get('total_output_tokens', 0)
            })
            
            model_summary['total_cases'] += result['num_cases']
            model_summary['successful_cases'] += result['successful_cases']
            model_summary['total_time'] += result.get('execution_time', 0)
            model_summary['total_tokens'] += token_usage.get('total_tokens', 0)
            model_summary['total_reasoning_tokens'] += token_usage.get('total_reasoning_tokens', 0)
            model_summary['total_output_tokens'] += token_usage.get('total_output_tokens', 0)

        # Calculate average accuracies and token statistics
        for model_name, summary in model_summaries.items():
            if summary['test_configurations']:
                summary['average_accuracy'] = sum(
                    config['accuracy'] for config in summary['test_configurations']
                ) / len(summary['test_configurations'])
            
            if summary['total_cases'] > 0:
                summary['avg_tokens_per_case'] = summary['total_tokens'] / summary['total_cases']
                summary['avg_reasoning_per_case'] = summary['total_reasoning_tokens'] / summary['total_cases']
                summary['avg_output_per_case'] = summary['total_output_tokens'] / summary['total_cases']
            else:
                summary['avg_tokens_per_case'] = 0
                summary['avg_reasoning_per_case'] = 0
                summary['avg_output_per_case'] = 0
            
            # Calculate thinking ratio (reasoning tokens / total tokens)
            if summary['total_tokens'] > 0:
                summary['thinking_ratio'] = summary['total_reasoning_tokens'] / summary['total_tokens']
            else:
                summary['thinking_ratio'] = 0
        
        # Find best model by accuracy
        best_model = None
        best_accuracy = 0
        if model_summaries:
            best_model, best_summary = max(
                model_summaries.items(),
                key=lambda x: x[1]['average_accuracy'],
                default=(None, None)
            )
            if best_summary:
                best_accuracy = best_summary['average_accuracy']
        
        # Find most efficient model (highest accuracy per token)
        most_efficient_model = None
        best_efficiency = 0
        for model_name, summary in model_summaries.items():
            if summary['avg_tokens_per_case'] > 0:
                efficiency = summary['average_accuracy'] / (summary['avg_tokens_per_case'] / 1000)
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    most_efficient_model = model_name
        
        return {
            'total_models_tested': len(model_summaries),
            'total_test_cases': sum(result['num_cases'] for result in results),
            'model_summaries': model_summaries,
            'best_model': best_model,
            'best_accuracy': best_accuracy,
            'most_efficient_model': most_efficient_model,
            'best_efficiency_score': best_efficiency
        }