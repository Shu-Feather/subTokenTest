from typing import List, Dict, Any, Optional
import json
import time
from tqdm import tqdm
import logging

from typewriter.models.base_model import BaseModel
from typewriter.tasks.task1_typewriter import Task1TypewriterEffect
from typewriter.tasks.task2_backspace import Task2BackspaceHandling
from typewriter.evaluation.metrics import BenchmarkMetrics
from typewriter.data.test_cases import TestCaseLoader, TestCaseGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TypewriterBenchmarkEvaluator:
    """Main evaluator for the typewriter benchmark with usage tracking"""
    
    def __init__(self, restricted_reasoning: bool = False):
        self.task1 = Task1TypewriterEffect()
        self.task2 = Task2BackspaceHandling()
        self.test_loader = TestCaseLoader()
        self.test_generator = TestCaseGenerator()
        self.metrics_calculator = BenchmarkMetrics()
        self.restricted_reasoning = restricted_reasoning
    
    def evaluate_model(self, model: BaseModel, 
                      test_cases: Optional[Dict[str, List]] = None,
                      prompt_type: str = "system",
                      use_batch: bool = False) -> Dict[str, Any]:
        """Evaluate a model on both tasks with usage tracking"""
        # if not model.is_available():
        #     raise ValueError(f"Model {model.name} is not available")
        
        logger.info(f"Evaluating model: {model.name}")
        
        # Check if model supports batch generation
        supports_batch = hasattr(model, 'generate_batch') and use_batch
        if use_batch and not supports_batch:
            logger.warning(f"Model {model.name} does not support batch generation")
        
        # Check if model supports usage tracking
        supports_usage = model.supports_usage_tracking()
        if supports_usage:
            logger.info("Model supports usage tracking - will record token counts")
        
        if test_cases is None:
            logger.warning("No test cases provided, using legacy generator")
            test_cases = self.test_generator.generate_all_test_cases()
        
        # Evaluate Task 1
        logger.info("Evaluating Task 1: Typewriter Effect")
        if supports_batch:
            task1_results = self._evaluate_task1_batch(model, test_cases['task1'], prompt_type)
        else:
            task1_results = self._evaluate_task1(model, test_cases['task1'], prompt_type)
        
        # Evaluate Task 2
        logger.info("Evaluating Task 2: Backspace Handling")
        if supports_batch:
            task2_results = self._evaluate_task2_batch(model, test_cases['task2'], prompt_type)
        else:
            task2_results = self._evaluate_task2(model, test_cases['task2'], prompt_type)
        
        # Calculate overall metrics
        overall_metrics = self.metrics_calculator.calculate_overall_metrics(
            task1_results, task2_results
        )
        
        # Calculate total usage across all tasks
        total_usage = self._calculate_total_usage(task1_results + task2_results)
        
        # Compile final results
        evaluation_result = {
            'model_info': model.get_model_info(),
            'evaluation_timestamp': time.time(),
            'prompt_type': prompt_type,
            'used_batch': supports_batch,
            'supports_usage_tracking': supports_usage,
            'task1_results': task1_results,
            'task2_results': task2_results,
            'metrics': overall_metrics,
            'total_usage': total_usage  # NEW: Total token usage
        }
        
        return evaluation_result
    
    def _calculate_total_usage(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate total usage across all test cases"""
        total_usage = {
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'reasoning_tokens': 0,
            'samples_with_usage': 0,
            'total_samples': len(all_results)
        }
        
        for result in all_results:
            usage = result.get('usage_info')
            if usage:
                total_usage['total_tokens'] += usage.get('total_tokens', 0)
                total_usage['input_tokens'] += usage.get('input_tokens', 0)
                total_usage['output_tokens'] += usage.get('output_tokens', 0)
                total_usage['reasoning_tokens'] += usage.get('reasoning_tokens', 0)
                total_usage['samples_with_usage'] += 1
        
        # Calculate averages
        if total_usage['samples_with_usage'] > 0:
            total_usage['avg_tokens_per_sample'] = total_usage['total_tokens'] / total_usage['samples_with_usage']
            total_usage['avg_input_per_sample'] = total_usage['input_tokens'] / total_usage['samples_with_usage']
            total_usage['avg_output_per_sample'] = total_usage['output_tokens'] / total_usage['samples_with_usage']
        
        return total_usage
    
    def _evaluate_task1(self, model: BaseModel, test_cases: List[str], 
                       prompt_type: str) -> List[Dict[str, Any]]:
        """Evaluate model on Task 1 with usage tracking"""
        results = []
        
        for word in tqdm(test_cases, desc="Task 1"):
            try:
                # Generate prompt
                if prompt_type == "system":
                    system_prompt = self.task1.get_system_prompt(self.restricted_reasoning)
                    user_prompt = self.task1.get_user_prompt(word, self.restricted_reasoning)
                    full_prompt = f"{system_prompt}\n\n{user_prompt}"
                else:
                    full_prompt = self.task1.get_few_shot_prompt(word, self.restricted_reasoning)
                
                # Get model response (with or without usage)
                response, usage_info = model.generate_with_usage(full_prompt)
                
                # Evaluate response
                evaluation = self.task1.evaluate_response(word, response)
                evaluation['input'] = word
                evaluation['response'] = response
                evaluation['prompt_type'] = prompt_type
                
                # Add usage info if available
                if usage_info is not None:
                    evaluation['usage_info'] = usage_info
                
                results.append(evaluation)
                
            except Exception as e:
                logger.error(f"Error evaluating word '{word}': {str(e)}")
                results.append({
                    'input': word,
                    'response': f"ERROR: {str(e)}",
                    'exact_match': False,
                    'score': 0.0,
                    'error': str(e),
                    'usage_info': None
                })
        
        return results
    
    def _evaluate_task2(self, model: BaseModel, test_cases: List[str], 
                       prompt_type: str) -> List[Dict[str, Any]]:
        """Evaluate model on Task 2 with usage tracking"""
        results = []
        
        for typing_log in tqdm(test_cases, desc="Task 2"):
            try:
                # Generate prompt
                if prompt_type == "system":
                    system_prompt = self.task2.get_system_prompt(self.restricted_reasoning)
                    user_prompt = self.task2.get_user_prompt(typing_log, self.restricted_reasoning)
                    full_prompt = f"{system_prompt}\n\n{user_prompt}"
                else:
                    full_prompt = self.task2.get_few_shot_prompt(typing_log, self.restricted_reasoning)
                
                # Get model response (with or without usage)
                response, usage_info = model.generate_with_usage(full_prompt)
                
                # Evaluate response
                evaluation = self.task2.evaluate_response(typing_log, response)
                evaluation['input'] = typing_log
                evaluation['response'] = response
                evaluation['prompt_type'] = prompt_type
                
                # Add usage info if available
                if usage_info is not None:
                    evaluation['usage_info'] = usage_info
                
                results.append(evaluation)
                
            except Exception as e:
                logger.error(f"Error evaluating typing log '{typing_log}': {str(e)}")
                results.append({
                    'input': typing_log,
                    'response': f"ERROR: {str(e)}",
                    'exact_match': False,
                    'score': 0.0,
                    'error': str(e),
                    'usage_info': None
                })
        
        return results
    
    def _evaluate_task1_batch(self, model: BaseModel, test_cases: List[str], 
                             prompt_type: str) -> List[Dict[str, Any]]:
        """Evaluate model on Task 1 (batch mode - no individual usage tracking)"""
        results = []
        
        # Prepare all prompts
        prompts = []
        for word in test_cases:
            if prompt_type == "system":
                system_prompt = self.task1.get_system_prompt(self.restricted_reasoning)
                user_prompt = self.task1.get_user_prompt(word, self.restricted_reasoning)
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
            else:
                full_prompt = self.task1.get_few_shot_prompt(word, self.restricted_reasoning)
            prompts.append(full_prompt)
        
        try:
            logger.info(f"Batch generating {len(prompts)} responses...")
            responses = model.generate_batch(prompts)
            
            # Evaluate all responses
            for word, response in tqdm(zip(test_cases, responses), total=len(test_cases), desc="Task 1"):
                evaluation = self.task1.evaluate_response(word, response)
                evaluation['input'] = word
                evaluation['response'] = response
                evaluation['prompt_type'] = prompt_type
                evaluation['usage_info'] = None  # Batch mode doesn't track individual usage
                results.append(evaluation)
                
        except Exception as e:
            logger.error(f"Batch generation error: {str(e)}, falling back to sequential")
            return self._evaluate_task1(model, test_cases, prompt_type)
        
        return results
    
    def _evaluate_task2_batch(self, model: BaseModel, test_cases: List[str], 
                             prompt_type: str) -> List[Dict[str, Any]]:
        """Evaluate model on Task 2 (batch mode - no individual usage tracking)"""
        results = []
        
        # Prepare all prompts
        prompts = []
        for typing_log in test_cases:
            if prompt_type == "system":
                system_prompt = self.task2.get_system_prompt(self.restricted_reasoning)
                user_prompt = self.task2.get_user_prompt(typing_log, self.restricted_reasoning)
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
            else:
                full_prompt = self.task2.get_few_shot_prompt(typing_log, self.restricted_reasoning)
            prompts.append(full_prompt)
        
        try:
            logger.info(f"Batch generating {len(prompts)} responses...")
            responses = model.generate_batch(prompts)
            
            # Evaluate all responses
            for typing_log, response in tqdm(zip(test_cases, responses), total=len(test_cases), desc="Task 2"):
                evaluation = self.task2.evaluate_response(typing_log, response)
                evaluation['input'] = typing_log
                evaluation['response'] = response
                evaluation['prompt_type'] = prompt_type
                evaluation['usage_info'] = None  # Batch mode doesn't track individual usage
                results.append(evaluation)
                
        except Exception as e:
            logger.error(f"Batch generation error: {str(e)}, falling back to sequential")
            return self._evaluate_task2(model, test_cases, prompt_type)
        
        return results
    
    def evaluate_multiple_models(self, model_names: List[str], 
                                test_cases: Optional[Dict[str, List]] = None,
                                prompt_types: List[str] = ["system", "few_shot"],
                                use_vllm: bool = False,
                                use_batch: bool = False) -> Dict[str, Any]:
        """Evaluate multiple models with usage tracking"""
        from typewriter.models import create_model
        
        all_results = {}
        
        for model_name in model_names:
            try:
                logger.info(f"Creating model: {model_name}")
                model = create_model(model_name, use_vllm=use_vllm)
                
                model_results = {}
                for prompt_type in prompt_types:
                    result = self.evaluate_model(model, test_cases, prompt_type, use_batch=use_batch)
                    model_results[prompt_type] = result
                
                all_results[model_name] = model_results
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {str(e)}")
                all_results[model_name] = {"error": str(e)}
        
        return all_results
