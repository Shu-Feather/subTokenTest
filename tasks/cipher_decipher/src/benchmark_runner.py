"""
Run benchmarks for cipher and decipher tasks using various language models.
"""

import logging
import sys
import os
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime
import random
from pathlib import Path

from configs.locator import resolve_config_path

from .ciphers.morse_code import MorseCode
from .ciphers.caesar_cipher import CaesarCipher
from .data_generation.text_generator import TextGenerator
from .models.base_model import BaseModel
from .models import create_model
from .models.vllm_model import VLLMModel
from .evaluation.evaluator import CipherEvaluator, TaskType
from .evaluation.metrics import MetricsCalculator, ResultsExporter
from .utils.prompts import PromptTemplates
from .utils.logger import ResponseLogger

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

class BenchmarkRunner:
    """Main benchmark runner class."""
    
    def __init__(self, config_path: str = "config.yaml", 
                 save_responses: bool = False, 
                 response_log_file: str = None,
                 verbose: bool = False,
                 experiment_id: str = None,
                 restricted_reasoning: bool = False,
                 allowed_models: set = None):
        """
        Initialize benchmark runner.
        
        Args:
            config_path (str): Path to configuration file
            save_responses (bool): Whether to save prompts and responses
            response_log_file (str): Path to response log file
            verbose (bool): Enable verbose logging
        """
        if not Path(config_path).is_absolute():
            config_path = resolve_config_path("cipher_decipher", config_path)

        self.config = self.load_config(config_path)
        self.verbose = verbose
        
        # Store experiment_id for unified naming
        from datetime import datetime
        self.experiment_id = experiment_id or datetime.now().strftime("%m%d")

        # Initialize text generator
        self.text_generator = TextGenerator()
        
        self.evaluator = CipherEvaluator(self.config.get('evaluation', {}))
        self.results_exporter = ResultsExporter(self.config.get('output', {}).get('results_dir', 'results'))
        self.prompt_templates = PromptTemplates()
        
        # Load prompt settings
        self.prompt_settings = self.config.get('prompt_settings', {})
        self.default_prompt_style = self.prompt_settings.get('default_style', 'detailed')
        self.per_task_styles = self.prompt_settings.get('per_task_styles', {})
        self.restricted_reasoning = restricted_reasoning or self.prompt_settings.get('restricted_reasoning', False)
        
        # Initialize response logger if needed
        self.save_responses = save_responses
        self.response_logger = None
        if save_responses:
            if response_log_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                response_log_file = f"logs/responses_{timestamp}.json"
            self.response_logger = ResponseLogger(response_log_file)
        
        # Optional allowlist to avoid loading all heavy local models when user requested a subset
        self.allowed_models = allowed_models

        # Initialize models
        self.models = {}
        self._initialize_models()
    
    def get_prompt_style_for_task(self, task_type: TaskType) -> str:
        """
        Get the appropriate prompt style for a specific task.
        
        Args:
            task_type (TaskType): The task type
            
        Returns:
            str: Prompt style to use
        """
        # Check if there's a specific style for this task
        task_name = task_type.value
        if task_name in self.per_task_styles:
            return self.per_task_styles[task_name]
        
        # Fall back to default style
        return self.default_prompt_style
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to config file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Expand environment variables
            config = self._expand_env_vars(config)
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def _expand_env_vars(self, obj):
        """Recursively expand environment variables in config."""
        if isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        else:
            return obj
    
    def _initialize_models(self):
        """Initialize available models based on configuration using factory pattern."""
        model_configs = self.config.get('models', {})
        
        for model_name, model_config in model_configs.items():
            if self.allowed_models is not None and model_name not in self.allowed_models:
                logger.info(f"Skipping model {model_name} (not in requested --models list)")
                continue
            try:
                # Ensure model config has a 'type' field
                if 'type' not in model_config:
                    # Try to infer type from model name if not specified
                    model_config['type'] = model_name
                
                # Use factory function to create model
                model = create_model(model_config)
                
                if model.is_available():
                    self.models[model_name] = model
                    logger.info(f"Initialized model: {model_name} (type: {model_config.get('type')})")
                else:
                    logger.warning(f"Model {model_name} not available (check API keys or dependencies)")
                    
            except ValueError as e:
                logger.error(f"Failed to initialize {model_name}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error initializing {model_name}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
    
    def generate_test_data(self, num_samples: int, test_file: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate test data for all enabled tasks with difficulty levels.
        Can load from file or generate new data.
        
        Args:
            num_samples (int): Number of samples per task per difficulty
            test_file (str): Optional path to JSON file with pre-generated test data
            
        Returns:
            Dict[str, List[Dict]]: Test data organized by task type
        """
        # Generate new test data
        logger.info(f"Generating {num_samples} test samples per task per difficulty level")
        
        test_config = self.config.get('test_config', {})
        tasks = test_config.get('tasks', {})
        caesar_shifts = test_config.get('caesar_shifts', [1, 3, 5, 7, 13, 25])
        
        # Get difficulty distribution
        data_gen_config = self.config.get('data_generation', {})
        difficulty_dist = data_gen_config.get('difficulty_distribution', {
            'easy': 0.33,
            'medium': 0.34,
            'hard': 0.33
        })
        
        # Calculate samples per difficulty
        samples_per_difficulty = {}
        for difficulty, ratio in difficulty_dist.items():
            samples_per_difficulty[difficulty] = int(num_samples * ratio)
        
        # Generate text samples by difficulty
        logger.info("Generating text samples by difficulty level...")
        total_samples_needed = sum(samples_per_difficulty.values())
        text_samples_by_difficulty = self.text_generator.generate_samples_by_difficulty(
            total_samples_needed
        )
        
        test_data = {}
        
        # Helper function to get text sample by difficulty
        def get_sample_by_difficulty(difficulty: str, index: int) -> str:
            difficulty_samples = text_samples_by_difficulty.get(difficulty, [])
            if difficulty_samples:
                return difficulty_samples[index % len(difficulty_samples)]
            else:
                # Fallback
                return self.text_generator.generate_random_text(20, 100)
        
        # Morse code encoding tasks
        if tasks.get('morse_encode', False):
            test_data['morse_encode'] = []
            for difficulty, count in samples_per_difficulty.items():
                for i in range(count):
                    text = get_sample_by_difficulty(difficulty, i)
                    test_data['morse_encode'].append({
                        'input': text,
                        'task_type': TaskType.MORSE_ENCODE,
                        'difficulty': difficulty
                    })
        
        # Morse code decoding tasks
        if tasks.get('morse_decode', False):
            test_data['morse_decode'] = []
            for difficulty, count in samples_per_difficulty.items():
                for i in range(count):
                    text = get_sample_by_difficulty(difficulty, i)
                    morse_code = MorseCode.encode(text)
                    test_data['morse_decode'].append({
                        'input': morse_code,
                        'task_type': TaskType.MORSE_DECODE,
                        'original_text': text,
                        'difficulty': difficulty
                    })
        
        # Caesar cipher encoding tasks
        if tasks.get('caesar_encode', False):
            test_data['caesar_encode'] = []
            for difficulty, count in samples_per_difficulty.items():
                for i in range(count):
                    text = get_sample_by_difficulty(difficulty, i)
                    shift = random.choice(caesar_shifts)
                    test_data['caesar_encode'].append({
                        'input': text,
                        'shift': shift,
                        'task_type': TaskType.CAESAR_ENCODE,
                        'difficulty': difficulty
                    })
        
        # Caesar cipher decoding tasks
        if tasks.get('caesar_decode', False):
            test_data['caesar_decode'] = []
            for difficulty, count in samples_per_difficulty.items():
                for i in range(count):
                    text = get_sample_by_difficulty(difficulty, i)
                    shift = random.choice(caesar_shifts)
                    encrypted_text = CaesarCipher.encode(text, shift)
                    test_data['caesar_decode'].append({
                        'input': encrypted_text,
                        'shift': shift,
                        'task_type': TaskType.CAESAR_DECODE,
                        'original_text': text,
                        'difficulty': difficulty
                    })
        
        total_samples = sum(len(v) for v in test_data.values())
        logger.info(f"Generated test data: {total_samples} total samples")

        # Log difficulty distribution
        for task_name, task_samples in test_data.items():
            difficulty_counts = {}
            for sample in task_samples:
                diff = sample.get('difficulty', 'unknown')
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            logger.info(f"{task_name}: {difficulty_counts}")
        
        return test_data
    
    def load_test_data_from_file(self, test_file: str, num_samples: int) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load test data from a JSON file generated by generate_dataset.py.
        
        Args:
            test_file (str): Path to JSON file with test data
            num_samples (int): Number of samples per task per difficulty
            
        Returns:
            Dict[str, List[Dict]]: Test data organized by task type
        """
        import json
        
        logger.info(f"Loading test data from file: {test_file}")
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            # Validate dataset structure
            if not isinstance(dataset, dict):
                raise ValueError("Invalid dataset format: expected dictionary")
            
            # Check for required difficulty levels
            for difficulty in ['easy', 'medium', 'hard']:
                if difficulty not in dataset:
                    logger.warning(f"Missing difficulty level '{difficulty}' in dataset")
                    dataset[difficulty] = []
            
            logger.info(f"Loaded dataset with {sum(len(v) for v in dataset.values())} total texts")
            for difficulty, texts in dataset.items():
                logger.info(f"  {difficulty}: {len(texts)} texts")
            
            # Generate test data using loaded texts
            test_config = self.config.get('test_config', {})
            tasks = test_config.get('tasks', {})
            caesar_shifts = test_config.get('caesar_shifts', [1, 3, 5, 7, 13, 25])
            
            # Get difficulty distribution
            data_gen_config = self.config.get('data_generation', {})
            difficulty_dist = data_gen_config.get('difficulty_distribution', {
                'easy': 0.33,
                'medium': 0.34,
                'hard': 0.33
            })
            
            # Calculate samples per difficulty
            samples_per_difficulty = {}
            for difficulty, ratio in difficulty_dist.items():
                samples_per_difficulty[difficulty] = int(num_samples * ratio)
            
            test_data = {}
            
            # Helper function to get text sample by difficulty
            def get_sample_by_difficulty(difficulty: str, index: int) -> str:
                difficulty_texts = dataset.get(difficulty, [])
                if not difficulty_texts:
                    logger.warning(f"No texts available for difficulty '{difficulty}', using fallback")
                    return self.text_generator.generate_random_text(20, 100)
                return difficulty_texts[index % len(difficulty_texts)]
            
            # Morse code encoding tasks
            if tasks.get('morse_encode', False):
                test_data['morse_encode'] = []
                for difficulty, count in samples_per_difficulty.items():
                    for i in range(count):
                        text = get_sample_by_difficulty(difficulty, i)
                        test_data['morse_encode'].append({
                            'input': text,
                            'task_type': TaskType.MORSE_ENCODE,
                            'difficulty': difficulty
                        })
            
            # Morse code decoding tasks
            if tasks.get('morse_decode', False):
                test_data['morse_decode'] = []
                for difficulty, count in samples_per_difficulty.items():
                    for i in range(count):
                        text = get_sample_by_difficulty(difficulty, i)
                        morse_code = MorseCode.encode(text)
                        test_data['morse_decode'].append({
                            'input': morse_code,
                            'task_type': TaskType.MORSE_DECODE,
                            'original_text': text,
                            'difficulty': difficulty
                        })
            
            # Caesar cipher encoding tasks
            if tasks.get('caesar_encode', False):
                test_data['caesar_encode'] = []
                for difficulty, count in samples_per_difficulty.items():
                    for i in range(count):
                        text = get_sample_by_difficulty(difficulty, i)
                        shift = random.choice(caesar_shifts)
                        test_data['caesar_encode'].append({
                            'input': text,
                            'shift': shift,
                            'task_type': TaskType.CAESAR_ENCODE,
                            'difficulty': difficulty
                        })
            
            # Caesar cipher decoding tasks
            if tasks.get('caesar_decode', False):
                test_data['caesar_decode'] = []
                for difficulty, count in samples_per_difficulty.items():
                    for i in range(count):
                        text = get_sample_by_difficulty(difficulty, i)
                        shift = random.choice(caesar_shifts)
                        encrypted_text = CaesarCipher.encode(text, shift)
                        test_data['caesar_decode'].append({
                            'input': encrypted_text,
                            'shift': shift,
                            'task_type': TaskType.CAESAR_DECODE,
                            'original_text': text,
                            'difficulty': difficulty
                        })
            
            total_samples = sum(len(v) for v in test_data.values())
            logger.info(f"Generated {total_samples} test cases from loaded dataset")
            
            # Log difficulty distribution
            for task_name, task_samples in test_data.items():
                difficulty_counts = {}
                for sample in task_samples:
                    diff = sample.get('difficulty', 'unknown')
                    difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
                logger.info(f"{task_name}: {difficulty_counts}")
            
            return test_data
            
        except FileNotFoundError:
            logger.error(f"Test file not found: {test_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading test data from file: {e}")
            raise
    
    def _get_expected_output(self, task_data: Dict[str, Any]) -> str:
        """
        Calculate the expected/golden output for a task.
        
        Args:
            task_data (Dict): Task data
            
        Returns:
            str: Expected output
        """
        task_type = task_data['task_type']
        
        if task_type == TaskType.MORSE_ENCODE:
            return MorseCode.encode(task_data['input'])
        elif task_type == TaskType.MORSE_DECODE:
            return task_data.get('original_text', '')
        elif task_type == TaskType.CAESAR_ENCODE:
            return CaesarCipher.encode(task_data['input'], task_data['shift'])
        elif task_type == TaskType.CAESAR_DECODE:
            return task_data.get('original_text', '')
        else:
            return ''
    
    async def run_model_on_task(self, model: BaseModel, task_data: Dict[str, Any], 
                           task_index: int = 0) -> tuple[Optional[str], str, Dict[str, Any]]:
        """
        Run a model on a single task.
        
        Args:
            model (BaseModel): Model to run
            task_data (Dict): Task data
            task_index (int): Index of current task (for logging)
            
        Returns:
            tuple[Optional[str], str, Dict[str, Any]]: (Model response, Prompt used, Token usage info)
        """
        task_type = task_data['task_type']
        
        try:
            # Get the appropriate prompt style for this task
            style = self.get_prompt_style_for_task(task_type)
            prompt = self._build_prompt(task_data, style, task_index)
            
            # Get model response with token usage info
            response, usage_info = await model.generate_response(prompt)
            
            # Verbose logging - print response and token usage
            if self.verbose:
                print("\n" + "-"*80)
                print("MODEL RESPONSE:")
                print("-"*80)
                print(response if response else "[NO RESPONSE]")
                print("\n" + "-"*80)
                print("TOKEN USAGE:")
                print("-"*80)
                print(f"Total Tokens: {usage_info.get('total_tokens', 0)}")
                print(f"Input Tokens: {usage_info.get('input_tokens', 0)}")
                print(f"Output Tokens: {usage_info.get('output_tokens', 0)}")
                print(f"Reasoning Tokens: {usage_info.get('reasoning_tokens', 0)}")
            
            return response, prompt, usage_info
            
        except Exception as e:
            logger.error(f"Error running model {model.model_name} on {task_type}: {e}")
            return None, "", {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0}

    def _build_prompt(self, task_data: Dict[str, Any], style: str, task_index: int = 0) -> str:
        """Build a prompt for a task with optional verbose logging."""
        task_type = task_data['task_type']
        if task_type == TaskType.MORSE_ENCODE:
            prompt = self.prompt_templates.get_morse_encode_prompt(
                task_data['input'], style, restricted_reasoning=self.restricted_reasoning
            )
        elif task_type == TaskType.MORSE_DECODE:
            prompt = self.prompt_templates.get_morse_decode_prompt(
                task_data['input'], style, restricted_reasoning=self.restricted_reasoning
            )
        elif task_type == TaskType.CAESAR_ENCODE:
            prompt = self.prompt_templates.get_caesar_encode_prompt(
                task_data['input'], task_data['shift'], style,
                restricted_reasoning=self.restricted_reasoning
            )
        elif task_type == TaskType.CAESAR_DECODE:
            prompt = self.prompt_templates.get_caesar_decode_prompt(
                task_data['input'], task_data['shift'], style,
                restricted_reasoning=self.restricted_reasoning
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        if self.verbose:
            print("\n" + "="*80)
            print(f"TASK #{task_index + 1} - {task_type.value.upper()}")
            print("="*80)
            print(f"\nDIFFICULTY: {task_data.get('difficulty', 'N/A')}")
            if 'shift' in task_data:
                print(f"SHIFT: {task_data['shift']}")
            print(f"\nPROMPT STYLE: {style}")
            print("\n" + "-"*80)
            print("PROMPT SENT TO MODEL:")
            print("-"*80)
            print(prompt)
        return prompt
    
    async def benchmark_model(self, model_name: str, test_data: Dict[str, List[Dict]]) -> List:
        """
        Benchmark a single model on all tasks.
        
        Args:
            model_name (str): Name of model to benchmark
            test_data (Dict): Test data for all tasks
            
        Returns:
            List: Raw results for evaluation
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not available")
            return []
        model = self.models[model_name]
        logger.info(f"Starting benchmark for model: {model_name}")
        
        raw_results = []
        total_tasks = sum(len(tasks) for tasks in test_data.values())
        completed_tasks = 0
        
        # Track total token usage for this model
        total_token_usage = {
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'reasoning_tokens': 0
        }
        # Prefer batch generation for vLLM to reuse a single loaded model
        if isinstance(model, VLLMModel):
            batch_size = max(1, int(getattr(model, "batch_size", 1) or 1))
            for task_category, tasks in test_data.items():
                logger.info(f"Running {len(tasks)} {task_category} tasks (batched)")

                prompts: List[str] = []
                expected_outputs: List[str] = []
                additionals: List[Dict[str, Any]] = []
                task_meta: List[Dict[str, Any]] = []

                for task_data in tasks:
                    style = self.get_prompt_style_for_task(task_data['task_type'])
                    prompt = self._build_prompt(task_data, style, completed_tasks + len(prompts))
                    prompts.append(prompt)
                    expected_outputs.append(self._get_expected_output(task_data))
                    info = {}
                    if 'shift' in task_data:
                        info['shift'] = task_data['shift']
                    if 'difficulty' in task_data:
                        info['difficulty'] = task_data['difficulty']
                    additionals.append(info)
                    task_meta.append(task_data)

                for i in range(0, len(prompts), batch_size):
                    chunk_prompts = prompts[i:i + batch_size]
                    chunk_expected = expected_outputs[i:i + batch_size]
                    chunk_info = additionals[i:i + batch_size]
                    chunk_tasks = task_meta[i:i + batch_size]

                    # vLLM batch generate (synchronous)
                    try:
                        batch_outputs = model.generate_batch(chunk_prompts)
                    except Exception as e:
                        logger.error(f"Batch generation failed: {e}")
                        batch_outputs = [("", {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0}) for _ in chunk_prompts]

                    for task_data, prompt, expected_output, additional_info, (response, usage_info) in zip(
                        chunk_tasks, chunk_prompts, chunk_expected, chunk_info, batch_outputs
                    ):
                        usage_info = usage_info or {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0}
                        total_token_usage['total_tokens'] += usage_info.get('total_tokens', 0)
                        total_token_usage['input_tokens'] += usage_info.get('input_tokens', 0)
                        total_token_usage['output_tokens'] += usage_info.get('output_tokens', 0)
                        total_token_usage['reasoning_tokens'] += usage_info.get('reasoning_tokens', 0)

                        raw_results.append((
                            task_data['task_type'],
                            task_data['input'],
                            response or "",
                            additional_info
                        ))

                        if response:
                            eval_result = self.evaluator.evaluate_single(
                                task_data['task_type'],
                                task_data['input'],
                                response,
                                additional_info
                            )
                            extracted_answer = eval_result.model_output
                            is_correct = eval_result.is_correct
                        else:
                            extracted_answer = ""
                            is_correct = False

                        from difflib import SequenceMatcher
                        similarity = SequenceMatcher(
                            None,
                            extracted_answer.strip().lower(),
                            expected_output.strip().lower()
                        ).ratio()

                        if self.verbose:
                            print("\n" + "-"*80)
                            print("GOLDEN ANSWER:")
                            print("-"*80)
                            print(expected_output)
                            print("\n" + "-"*80)
                            print("EXTRACTED ANSWER:")
                            print("-"*80)
                            print(extracted_answer)
                            print("\n" + "-"*80)
                            print(f"RESULT: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                            print("="*80 + "\n")

                        if self.response_logger:
                            self.response_logger.log_interaction(
                                task_type=task_data['task_type'].value,
                                prompt=prompt,
                                response=response or "",
                                expected_output=expected_output,
                                extracted_answer=extracted_answer,
                                is_correct=is_correct,
                                difficulty=task_data.get('difficulty'),
                                usage_info=usage_info,
                                additional_info=additional_info,
                                similarity=similarity
                            )

                        completed_tasks += 1
                        if completed_tasks % 10 == 0:
                            logger.info(f"Completed {completed_tasks}/{total_tasks} tasks for {model_name}")
        else:
            for task_category, tasks in test_data.items():
                logger.info(f"Running {len(tasks)} {task_category} tasks")
                
                for task_data in tasks:
                    # Get expected output (golden answer)
                    expected_output = self._get_expected_output(task_data)
                    
                    # Run model on task (now returns usage_info as well)
                    response, prompt, usage_info = await self.run_model_on_task(model, task_data, completed_tasks)
                    
                    # Accumulate token usage
                    total_token_usage['total_tokens'] += usage_info.get('total_tokens', 0)
                    total_token_usage['input_tokens'] += usage_info.get('input_tokens', 0)
                    total_token_usage['output_tokens'] += usage_info.get('output_tokens', 0)
                    total_token_usage['reasoning_tokens'] += usage_info.get('reasoning_tokens', 0)
                    
                    # Prepare result data for evaluation
                    additional_info = {}
                    if 'shift' in task_data:
                        additional_info['shift'] = task_data['shift']
                    if 'difficulty' in task_data:
                        additional_info['difficulty'] = task_data['difficulty']
                    
                    raw_results.append((
                        task_data['task_type'],
                        task_data['input'],
                        response or "",
                        additional_info
                    ))
                    
                    # Evaluate this single result to get extracted answer
                    if response:
                        eval_result = self.evaluator.evaluate_single(
                            task_data['task_type'],
                            task_data['input'],
                            response,
                            additional_info
                        )
                        extracted_answer = eval_result.model_output
                        is_correct = eval_result.is_correct
                    else:
                        extracted_answer = ""
                        is_correct = False

                    # Calculate similarity between extracted answer and golden answer
                    from difflib import SequenceMatcher
                    similarity = SequenceMatcher(
                        None, 
                        extracted_answer.strip().lower(), 
                        expected_output.strip().lower()
                    ).ratio()

                    # Verbose logging - print golden answer and extracted answer
                    if self.verbose:
                        print("\n" + "-"*80)
                        print("GOLDEN ANSWER:")
                        print("-"*80)
                        print(expected_output)
                        print("\n" + "-"*80)
                        print("EXTRACTED ANSWER:")
                        print("-"*80)
                        print(extracted_answer)
                        print("\n" + "-"*80)
                        print(f"RESULT: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                        print("="*80 + "\n")
                    
                    # Log to response logger if enabled (now includes usage_info)
                    if self.response_logger:
                        self.response_logger.log_interaction(
                            task_type=task_data['task_type'].value,
                            prompt=prompt,
                            response=response or "",
                            expected_output=expected_output,
                            extracted_answer=extracted_answer,
                            is_correct=is_correct,
                            difficulty=task_data.get('difficulty'),
                            usage_info=usage_info,
                            additional_info=additional_info,
                            similarity=similarity  
                        )
                    
                    completed_tasks += 1
                    if completed_tasks % 10 == 0:
                        logger.info(f"Completed {completed_tasks}/{total_tasks} tasks for {model_name}")
        
        logger.info(f"Completed benchmark for model: {model_name}")
        logger.info(f"Total token usage for {model_name}: {total_token_usage}")
        
        # Save response log if enabled
        if self.response_logger:
            self.response_logger.save_to_file('json')
            
            # Log token usage summary
            usage_summary = self.response_logger.get_token_usage_summary()
            logger.info(f"Token usage summary for {model_name}:")
            logger.info(f"  Total tokens: {usage_summary['total_tokens']}")
            logger.info(f"  Average tokens per interaction: {usage_summary['avg_tokens_per_interaction']:.2f}")
        
        return raw_results
    
    async def run_benchmark(self, models: List[str] = None, num_samples: int = None, 
                       test_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete benchmark.
        
        Args:
            models (List[str]): List of model names to benchmark (None for all available)
            num_samples (int): Number of samples per task (None to use config default)
            test_file (str): Optional path to JSON file with pre-generated test data
            
        Returns:
            Dict[str, Any]: Complete benchmark results
        """
        # Use config defaults if not specified
        if models is None:
            models = list(self.models.keys())
        if num_samples is None:
            num_samples = self.config.get('test_config', {}).get('num_samples', 10)
        
        # Generate test data (or load from file)
        if test_file:
            test_data = self.load_test_data_from_file(test_file, num_samples)
        else:
            test_data = self.generate_test_data(num_samples, test_file)
        
        # Run benchmarks for each model
        all_results = {}
        
        for model_name in models:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not available, skipping")
                continue
            
            logger.info(f"Benchmarking model: {model_name}")
            
            # Run benchmark
            raw_results = await self.benchmark_model(model_name, test_data)
            
            # Evaluate results
            evaluations = self.evaluator.evaluate_batch(raw_results)
            
            # Calculate metrics
            metrics = MetricsCalculator.calculate_comprehensive_metrics(evaluations)
            
            # Get token usage summary if response logger is enabled
            token_usage_summary = None
            if self.response_logger:
                token_usage_summary = self.response_logger.get_token_usage_summary()
            
            # Store results
            all_results[model_name] = {
                'evaluations': evaluations,
                'metrics': metrics,
                'model_stats': self.models[model_name].get_stats(),
                'token_usage': token_usage_summary
            }
            
            # Export results
            if self.config.get('output', {}).get('save_results', True):
                self.results_exporter.export_all_formats(evaluations, model_name)
            
            logger.info(f"Model {model_name} - Accuracy: {metrics.get('overall_accuracy', 0):.2f}%")
            if token_usage_summary:
                logger.info(f"Model {model_name} - Total tokens used: {token_usage_summary['total_tokens']}")

        return all_results
        
    def print_summary(self, results: Dict[str, Any]):
        """
        Print a summary of benchmark results including token usage.
        
        Args:
            results (Dict[str, Any]): Benchmark results
        """
        print("\n" + "="*100)
        print("CIPHER & DECIPHER BENCHMARK RESULTS")
        print("="*100)
        
        for model_name, model_results in results.items():
            metrics = model_results['metrics']
            model_stats = model_results['model_stats']
            token_usage = model_results.get('token_usage')
            
            print(f"\nModel: {model_name}")
            print("-" * 100)
            print(f"  Overall Accuracy: {metrics.get('overall_accuracy', 0):.2f}%")
            print(f"  Total Samples: {metrics.get('total_samples', 0)}")
            print(f"  Success Rate: {model_stats.get('success_rate', 0):.1f}%")
            print(f"  Avg Response Time: {model_stats.get('avg_response_time', 0):.2f}s")
            
            # Token usage summary
            if token_usage:
                print(f"\n  Token Usage:")
                print(f"    Total Tokens: {token_usage.get('total_tokens', 0):,}")
                print(f"    Input Tokens: {token_usage.get('total_input_tokens', 0):,}")
                print(f"    Output Tokens: {token_usage.get('total_output_tokens', 0):,}")
                print(f"    Reasoning Tokens: {token_usage.get('total_reasoning_tokens', 0):,}")
                print(f"    Avg Tokens/Interaction: {token_usage.get('avg_tokens_per_interaction', 0):.2f}")
            
            # Per-task breakdown
            per_task = metrics.get('per_task_accuracy', {})
            if per_task:
                print("\n  Per-task Accuracy:")
                for task, accuracy in per_task.items():
                    task_display = task.replace('_', ' ').title()
                    print(f"    {task_display}: {accuracy:.1f}%", end='')
                    
                    # Add token usage per task if available
                    if token_usage and 'by_task_type' in token_usage:
                        task_tokens = token_usage['by_task_type'].get(task, {})
                        avg_tokens = task_tokens.get('avg_total_tokens', 0)
                        if avg_tokens > 0:
                            print(f" (Avg tokens: {avg_tokens:.1f})", end='')
                    print()
            
            # Per-difficulty breakdown
            per_difficulty = metrics.get('per_difficulty_accuracy', {})
            if per_difficulty:
                print("\n  Per-difficulty Accuracy:")
                for difficulty, accuracy in per_difficulty.items():
                    print(f"    {difficulty.title()}: {accuracy:.1f}%", end='')
                    
                    # Add token usage per difficulty if available
                    if token_usage and 'by_difficulty' in token_usage:
                        diff_tokens = token_usage['by_difficulty'].get(difficulty, {})
                        avg_tokens = diff_tokens.get('avg_total_tokens', 0)
                        if avg_tokens > 0:
                            print(f" (Avg tokens: {avg_tokens:.1f})", end='')
                    print()
            
            print()
        
        # Print detailed token usage comparison table
        print("\n" + "="*100)
        print("TOKEN USAGE COMPARISON")
        print("="*100)
        
        # Collect all models with token usage data
        models_with_tokens = [(name, res.get('token_usage')) for name, res in results.items() 
                            if res.get('token_usage')]
        
        if models_with_tokens:
            print(f"\n{'Model':<30} {'Total':<15} {'Input':<15} {'Output':<15} {'Reasoning':<15}")
            print("-" * 100)
            
            for model_name, token_usage in models_with_tokens:
                total = token_usage.get('total_tokens', 0)
                input_tok = token_usage.get('total_input_tokens', 0)
                output_tok = token_usage.get('total_output_tokens', 0)
                reasoning = token_usage.get('total_reasoning_tokens', 0)
                
                print(f"{model_name:<30} {total:>13,}{'':<2} {input_tok:>13,}{'':<2} "
                    f"{output_tok:>13,}{'':<2} {reasoning:>13,}{'':<2}")
            
            # Print by-task token usage breakdown
            print("\n" + "="*100)
            print("TOKEN USAGE BY TASK TYPE")
            print("="*100)
            
            for model_name, token_usage in models_with_tokens:
                by_task = token_usage.get('by_task_type', {})
                if by_task:
                    print(f"\n{model_name}:")
                    print(f"  {'Task Type':<30} {'Count':<10} {'Avg Tokens':<15} {'Total Tokens':<15}")
                    print("  " + "-" * 95)
                    
                    for task_type, task_stats in by_task.items():
                        count = task_stats.get('count', 0)
                        avg_tokens = task_stats.get('avg_total_tokens', 0)
                        total_tokens = task_stats.get('total_tokens', 0)
                        task_display = task_type.replace('_', ' ').title()
                        
                        print(f"  {task_display:<30} {count:<10} {avg_tokens:>13.2f}{'':<2} {total_tokens:>13,}{'':<2}")
            
            # Print by-difficulty token usage breakdown
            print("\n" + "="*100)
            print("TOKEN USAGE BY DIFFICULTY LEVEL")
            print("="*100)
            
            for model_name, token_usage in models_with_tokens:
                by_difficulty = token_usage.get('by_difficulty', {})
                if by_difficulty:
                    print(f"\n{model_name}:")
                    print(f"  {'Difficulty':<30} {'Count':<10} {'Avg Tokens':<15} {'Total Tokens':<15}")
                    print("  " + "-" * 95)
                    
                    for difficulty, diff_stats in by_difficulty.items():
                        count = diff_stats.get('count', 0)
                        avg_tokens = diff_stats.get('avg_total_tokens', 0)
                        total_tokens = diff_stats.get('total_tokens', 0)
                        
                        print(f"  {difficulty.title():<30} {count:<10} {avg_tokens:>13.2f}{'':<2} {total_tokens:>13,}{'':<2}")
        else:
            print("\nNo token usage data available.")
        
        print("\n" + "="*100)
