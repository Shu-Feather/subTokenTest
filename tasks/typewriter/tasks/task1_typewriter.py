from typing import List, Dict, Any
import json
import re

class Task1TypewriterEffect:
    """Task 1: Progressive typing simulation - generate step-by-step typing sequence"""
    
    def __init__(self):
        self.task_name = "Task 1: Typewriter Effect Simulation"
        self.description = "Generate progressive typing sequence for given input word"
        
    def get_system_prompt(self, restricted_reasoning: bool = False) -> str:
        """Get the system prompt for Task 1"""
        prompt = """You are a typewriter effect simulator. Your task is to show the progressive typing sequence of a given word.

TASK DESCRIPTION:
Given an input word, you need to generate the complete sequence showing how the word appears character by character as if someone is typing it.

RULES:
1. Start with the first character
2. Add one character at a time until the complete word is formed
3. Use "→" (arrow) to separate each step
4. Do not include any extra characters, spaces, or explanations
5. Only output the progressive sequence

EXAMPLES:
Input: "hello"
Output: h→he→hel→hell→hello

Input: "cat"
Output: c→ca→cat

Input: "a"
Output: a

Input: "test123"
Output: t→te→tes→test→test1→test12→test123

IMPORTANT:
- Your response should ONLY contain the progressive sequence wrapped inside <answer> and </answer>
- No explanations, no additional text
- Follow the exact format shown in examples"""
        if restricted_reasoning:
            prompt += "\n\nAnswer directly without extra thinking."
        return prompt

    def get_user_prompt(self, word: str, restricted_reasoning: bool = False) -> str:
        """Generate user prompt for a specific word"""
        prompt = f'Input: "{word}"\nOutput (wrap in <answer></answer>):'
        if restricted_reasoning:
            prompt += " (answer directly without extended reasoning)"
        return prompt
    
    def get_few_shot_prompt(self, word: str, restricted_reasoning: bool = False) -> str:
        """Generate few-shot prompt with examples"""
        examples = [
            ('hello', '<answer>h→he→hel→hell→hello</answer>'),
            ('cat', '<answer>c→ca→cat</answer>'),
            ('test', '<answer>t→te→tes→test</answer>')
        ]
        
        prompt = """TASK: Typewriter Effect Simulation
Generate the progressive typing sequence for the given word.

Examples:
"""
        for example_word, example_output in examples:
            prompt += f'Input: "{example_word}"\nOutput: {example_output}\n\n'
            
        prompt += f'Input: "{word}"\nOutput (wrap in <answer></answer>):'
        if restricted_reasoning:
            prompt += " (answer directly without extended reasoning)"
        return prompt
    
    def evaluate_response(self, word: str, response: str) -> Dict[str, Any]:
        """Evaluate the model's response for Task 1"""
        expected = self.generate_expected_output(word)
        parsed_answer = self._extract_answer(response)

        # Check exact match
        exact_match = parsed_answer == expected
        
        # Check if it contains the expected sequence
        contains_expected = expected in parsed_answer
        
        # Check format (contains arrows)
        correct_format = "→" in parsed_answer
        
        # Check if all progressive steps are present
        expected_steps = expected.split("→")
        response_steps = parsed_answer.split("→") if parsed_answer else []
        
        steps_match = len(expected_steps) == len(response_steps)
        if steps_match:
            steps_match = all(exp.strip() == resp.strip() for exp, resp in zip(expected_steps, response_steps))
        
        return {
            "exact_match": exact_match,
            "contains_expected": contains_expected,
            "correct_format": correct_format,
            "steps_match": steps_match,
            "expected": expected,
            "actual": parsed_answer,
            "score": 1.0 if exact_match else (0.8 if steps_match else (0.5 if contains_expected else 0.0))
        }

    @staticmethod
    def _extract_answer(response: str) -> str:
        """Extract content strictly inside <answer> tags; missing tags -> empty string."""
        match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return match.group(1).strip()
    
    def generate_expected_output(self, word: str) -> str:
        """Generate the expected output for a given word"""
        if not word:
            return ""
        
        steps = []
        for i in range(1, len(word) + 1):
            steps.append(word[:i])
        
        return "→".join(steps)
