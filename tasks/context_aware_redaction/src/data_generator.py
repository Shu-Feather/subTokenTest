"""
Data generator for creating synthetic test cases
"""

import os
import random
import string
import re
import time
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm

from .utils import save_json, create_ground_truth


class DataGenerator:
    """
    Generate synthetic data for Context-Aware Redaction benchmark
    """
    
    # Minimum placeholder requirements for each difficulty
    MIN_PLACEHOLDERS = {
        'short': 3,
        'medium': 4,
        'long': 6
    }
    
    def __init__(self, config: Dict[str, Any], api_key: str, verbose: bool = False):
        """
        Initialize data generator
        
        Args:
            config: Configuration dictionary
            api_key: OpenAI API key
            verbose: Whether to print verbose output
        """
        self.config = config
        self.verbose = verbose
        self.client = OpenAI(api_key=api_key)
        self.model = config['data_generation']['gpt_model']
        self.temperature = config['data_generation']['temperature']
        self.output_dir = config['data_generation']['output_dir']
        
    def generate_id_card(self) -> str:
        """Generate random 18-digit ID card number (no spaces)"""
        # Always generate without spaces to distinguish from credit cards
        return ''.join(random.choices(string.digits, k=18))
    
    def generate_phone(self) -> str:
        """Generate random phone number with area code"""
        area_codes = ['+86', '+1', '+44', '+81', '+82']
        area_code = random.choice(area_codes)
        phone_number = ''.join(random.choices(string.digits, k=11))
        return f"{area_code} {phone_number}"
    
    def generate_credit_card(self) -> str:
        """Generate random credit card number with spaces (16 digits)"""
        # Always generate with spaces to distinguish from ID cards
        parts = [''.join(random.choices(string.digits, k=4)) for _ in range(4)]
        return ' '.join(parts)  # This ensures credit cards always have spaces
    
    def count_placeholders(self, text: str) -> Dict[str, int]:
        """
        Count the number of placeholders in the text
        
        Args:
            text: Text containing placeholders
            
        Returns:
            Dictionary with counts for each placeholder type and total
        """
        phone_count = text.count('[PHONE]')
        id_card_count = text.count('[ID_CARD]')
        credit_card_count = text.count('[CREDIT_CARD]')
        total_count = phone_count + id_card_count + credit_card_count
        
        return {
            'phone': phone_count,
            'id_card': id_card_count,
            'credit_card': credit_card_count,
            'total': total_count
        }
    
    def validate_placeholder_count(self, text: str, difficulty: str, 
                                   sensitive_types: List[str]) -> tuple[bool, str]:
        """
        Validate that the text has sufficient placeholders
        
        Args:
            text: Text containing placeholders
            difficulty: Difficulty level
            sensitive_types: Expected types of sensitive information
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        counts = self.count_placeholders(text)
        min_required = self.MIN_PLACEHOLDERS[difficulty]
        
        # Check total count meets minimum
        if counts['total'] < min_required:
            msg = f"Insufficient placeholders: {counts['total']} < {min_required}"
            return False, msg
        
        # Check that all requested types are present
        for stype in sensitive_types:
            type_key = stype  # 'phone', 'id_card', 'credit_card'
            if counts[type_key] == 0:
                msg = f"Missing placeholder type: {stype}"
                return False, msg
        
        return True, f"Valid: {counts}"
    
    def generate_context_prompt(self, difficulty: str, sensitive_types: List[str]) -> str:
        """
        Generate prompt for GPT to create context
        
        Args:
            difficulty: Difficulty level (short/medium/long)
            sensitive_types: Types of sensitive information to include
            
        Returns:
            Prompt string
        """
        # Define requirements based on difficulty
        requirements = {
            "short": {
                "length": "2-4 sentences",
                "min_placeholders": 3,
                "target_placeholders": "exactly 3 or 4"
            },
            "medium": {
                "length": "1 short paragraph (5-8 sentences)",
                "min_placeholders": 4,
                "target_placeholders": "exactly 4 to 6"
            },
            "long": {
                "length": "2-3 paragraphs (10-15 sentences)",
                "min_placeholders": 6,
                "target_placeholders": "exactly 6 to 9"
            }
        }
        
        req = requirements[difficulty]
        
        # Map internal names to placeholder names
        type_mapping = {
            'phone': '[PHONE]',
            'id_card': '[ID_CARD]',
            'credit_card': '[CREDIT_CARD]'
        }
        
        placeholder_list = [type_mapping[t] for t in sensitive_types]
        types_str = ", ".join(placeholder_list)
        
        prompt = f"""Generate a realistic text scenario containing sensitive information placeholders.

CRITICAL REQUIREMENTS - YOU MUST FOLLOW THESE EXACTLY:

1. **Placeholder Count**: Include {req['target_placeholders']} placeholders in total
   - MINIMUM required: You MUST generate AT LEAST{req['min_placeholders']} placeholders
   - You MUST use these exact placeholders: {types_str}
   - Each type must appear at least once
   - You can repeat types to reach the target count

2. **Text Length**: {req['length']}

3. **Exact Placeholder Format** (copy these exactly):
   - For phone numbers: [PHONE]
   - For ID cards: [ID_CARD]
   - For credit cards: [CREDIT_CARD]

4. **Context**: Add detailed context to make it more realistic (customer record, form, email, report, etc.)

5. **NO real data**: Use ONLY the placeholders above, NO actual numbers

NOW generate a DIFFERENT scenario following these rules. Output ONLY the text, no explanations."""

        return prompt
    
    def generate_single_context(self, difficulty: str, sensitive_types: List[str], 
                                max_retries: int = 8) -> str:
        """
        Generate a single context using GPT with validation and retries
        
        Args:
            difficulty: Difficulty level
            sensitive_types: Types of sensitive information
            max_retries: Maximum number of generation attempts
            
        Returns:
            Generated context with placeholders (validated)
        """
        for attempt in range(max_retries):
            try:
                prompt = self.generate_context_prompt(difficulty, sensitive_types)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a precise assistant that generates text samples with exact placeholder counts. Always follow the placeholder requirements exactly. Generate plain text without markdown formatting, titles, or labels."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=min(self.temperature + attempt * 0.1, 1.0),
                    max_tokens=1000
                )
                
                context = response.choices[0].message.content.strip()
                
                # Clean up the generated text
                context = self.clean_generated_text(context)
                
                # Validate placeholder count
                is_valid, msg = self.validate_placeholder_count(context, difficulty, sensitive_types)
                
                if is_valid:
                    if self.verbose or attempt > 0:
                        counts = self.count_placeholders(context)
                        print(f"\n{'='*80}")
                        print(f"✓ Generated valid context ({difficulty}) on attempt {attempt + 1}")
                        print(f"Placeholders: {counts}")
                        print(f"{'='*80}")
                        print(context)
                        print(f"{'='*80}\n")
                    return context
                else:
                    if self.verbose or attempt >= 3:
                        print(f"  Attempt {attempt + 1}/{max_retries}: {msg}")
                    
                    # Small delay between retries
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                    
            except Exception as e:
                print(f"  Error on attempt {attempt + 1}/{max_retries}: {e}")
                time.sleep(1)
                continue
        
        print(f"\n⚠ WARNING: Failed to generate valid {difficulty} context after {max_retries} attempts")
        print(f"  Required types: {sensitive_types}")
        print(f"  Minimum placeholders: {self.MIN_PLACEHOLDERS[difficulty]}")
        return ""
    
    def fill_placeholders(self, context: str) -> str:
        """
        Replace placeholders with actual sensitive information
        
        Args:
            context: Context with placeholders
            
        Returns:
            Context with actual sensitive information
        """
        # Replace placeholders with generated sensitive info
        while "[PHONE]" in context:
            context = context.replace("[PHONE]", self.generate_phone(), 1)
        
        while "[ID_CARD]" in context:
            context = context.replace("[ID_CARD]", self.generate_id_card(), 1)
        
        while "[CREDIT_CARD]" in context:
            context = context.replace("[CREDIT_CARD]", self.generate_credit_card(), 1)
        
        return context
    
    def generate_dataset(self, num_samples_per_difficulty: int) -> List[Dict[str, Any]]:
        """
        Generate complete dataset with all difficulties
        
        Args:
            num_samples_per_difficulty: Number of samples per difficulty level
            
        Returns:
            List of test samples
        """
        difficulties = ['short', 'medium', 'long']
        return self.generate_dataset_by_difficulty(
            difficulties=difficulties,
            num_samples_per_difficulty=num_samples_per_difficulty,
            extra_samples=0
        )
    
    def generate_dataset_by_difficulty(self, difficulties: List[str], 
                                      num_samples_per_difficulty: int,
                                      extra_samples: int = 0) -> List[Dict[str, Any]]:
        """
        Generate dataset with specified difficulties
        
        Args:
            difficulties: List of difficulty levels to generate (e.g., ['short', 'medium', 'long'])
            num_samples_per_difficulty: Number of samples per difficulty level
            extra_samples: Extra samples to add to the first difficulty (for even distribution)
            
        Returns:
            List of test samples
        """
        dataset = []
        
        # Use combinations that guarantee all types are included
        sensitive_types_options = [
            ['phone', 'id_card', 'credit_card'],  # All three types
            ['phone', 'id_card'],
            ['phone', 'credit_card'],
            ['id_card', 'credit_card'],
        ]
        
        total_samples = num_samples_per_difficulty * len(difficulties) + extra_samples
        print(f"Generating dataset with {total_samples} total samples...")
        print(f"Difficulties: {', '.join(difficulties)}")
        print(f"\nMinimum placeholder requirements:")
        for diff in difficulties:
            print(f"  {diff}: >= {self.MIN_PLACEHOLDERS[diff]} placeholders")
        print()
        
        for difficulty_idx, difficulty in enumerate(difficulties):
            # Add extra samples to first difficulty if needed
            num_samples = num_samples_per_difficulty
            if difficulty_idx == 0:
                num_samples += extra_samples
            
            print(f"\nGenerating {num_samples} {difficulty} difficulty samples...")
            
            successful_samples = 0
            failed_attempts = 0
            # max_failed = num_samples * 2  # Allow some failures but not too many
            
            with tqdm(total=num_samples, desc=f"{difficulty.capitalize()} samples") as pbar:
                # while successful_samples < num_samples and failed_attempts < max_failed:
                while successful_samples < num_samples:
                    # Randomly select which types of sensitive info to include
                    # Prefer combinations with all three types for better coverage
                    if random.random() < 0.6:
                        sensitive_types = sensitive_types_options[0]  # All three types
                    else:
                        sensitive_types = random.choice(sensitive_types_options)
                    
                    # Generate context with placeholders
                    context_with_placeholders = self.generate_single_context(
                        difficulty, sensitive_types
                    )
                    
                    if not context_with_placeholders:
                        failed_attempts += 1
                        print(f"  ⚠ Failed attempt {failed_attempts}")
                        continue
                    
                    # Fill placeholders with actual sensitive information
                    original_context = self.fill_placeholders(context_with_placeholders)
                    
                    # Create ground truth redacted version
                    try:
                        redacted_context, sensitive_info = create_ground_truth(original_context)
                    except Exception as e:
                        print(f"  ⚠ Error creating ground truth: {e}")
                        failed_attempts += 1
                        continue
                    
                    # Validate that we have the expected number of sensitive items
                    if len(sensitive_info) < self.MIN_PLACEHOLDERS[difficulty]:
                        print(f"  ⚠ Sample has only {len(sensitive_info)} sensitive items (need {self.MIN_PLACEHOLDERS[difficulty]}), skipping")
                        failed_attempts += 1
                        continue
                    
                    sample = {
                        "id": f"{difficulty}_{successful_samples}",
                        "difficulty": difficulty,
                        "original_context": original_context,
                        "redacted_context": redacted_context,
                        "sensitive_info": sensitive_info,
                        "sensitive_types": sensitive_types,
                        "num_sensitive_items": len(sensitive_info)
                    }
                    
                    dataset.append(sample)
                    successful_samples += 1
                    pbar.update(1)
                    
                    # Reset failed attempts counter on success
                    failed_attempts = max(0, failed_attempts - 1)
            
            if successful_samples < num_samples:
                print(f"\n⚠ WARNING: Only generated {successful_samples}/{num_samples} valid samples for {difficulty}")
                print(f"  Consider:")
                print(f"    1. Checking your OpenAI API key and quota")
                print(f"    2. Using a different GPT model (gpt-4 works better than gpt-3.5)")
                print(f"    3. Running with --verbose to see detailed errors")
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str = "dataset.json") -> None:
        """
        Save generated dataset to file
        
        Args:
            dataset: Dataset to save
            filename: Output filename
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        save_json(dataset, filepath)
        print(f"\nDataset saved to {filepath}")
        print(f"Total samples: {len(dataset)}")
        
        if len(dataset) == 0:
            print("\n⚠ WARNING: No samples were generated!")
            print("  Possible causes:")
            print("    1. OpenAI API issues (check your API key and quota)")
            print("    2. Model not generating valid placeholders")
            print("    3. Network connectivity issues")
            return
        
        # Print detailed statistics
        difficulties = {}
        sensitive_counts = {}
        type_distribution = {}
        
        for sample in dataset:
            diff = sample['difficulty']
            difficulties[diff] = difficulties.get(diff, 0) + 1
            
            if diff not in sensitive_counts:
                sensitive_counts[diff] = []
            sensitive_counts[diff].append(sample['num_sensitive_items'])
            
            # Track type distribution
            for info in sample['sensitive_info']:
                info_type = info['type']
                if diff not in type_distribution:
                    type_distribution[diff] = {}
                type_distribution[diff][info_type] = type_distribution[diff].get(info_type, 0) + 1
        
        print("\n" + "="*80)
        print("Dataset Statistics")
        print("="*80)
        
        for diff in sorted(difficulties.keys()):
            count = difficulties[diff]
            counts_list = sensitive_counts[diff]
            avg_sensitive = sum(counts_list) / len(counts_list) if counts_list else 0
            min_sensitive = min(counts_list) if counts_list else 0
            max_sensitive = max(counts_list) if counts_list else 0
            
            print(f"\n{diff.upper()}:")
            print(f"  Total samples: {count}")
            print(f"  Sensitive items per sample:")
            print(f"    Average: {avg_sensitive:.1f}")
            print(f"    Range: {min_sensitive} - {max_sensitive}")
            
            if diff in type_distribution:
                print(f"  Type distribution:")
                for info_type, type_count in sorted(type_distribution[diff].items()):
                    print(f"    {info_type}: {type_count}")
        
        print("\n" + "="*80)

    def clean_generated_text(self, text: str) -> str:
        """
        Clean up generated text to remove markdown and unnecessary formatting
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned text
        """
        # Remove markdown code blocks
        text = re.sub(r'^```.*?\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n```$', '', text, flags=re.MULTILINE)
        
        # Remove markdown bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **text** -> text
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *text* -> text
        
        # Remove scenario labels and headers
        text = re.sub(r'^\*\*Scenario\*\*:?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^Scenario:?\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'^\*\*.*?\*\*:?\s*\n', '', text, flags=re.MULTILINE)
        
        # Remove extra newlines and spaces
        text = re.sub(r'\n\s*\n', '\n', text)
        text = text.strip()
        
        return text