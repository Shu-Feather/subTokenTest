from typing import List, Dict, Any
import json
import re

class Task2BackspaceHandling:
    """Task 2: Handle backspace operations in typing log"""
    
    def __init__(self):
        self.task_name = "Task 2: Backspace Operation Handling"
        self.description = "Parse typing log with backspace operations and output final result"
        
    def get_system_prompt(self, restricted_reasoning: bool = False) -> str:
        """Get the system prompt for Task 2"""
        prompt = """You are a typing log parser. Your task is to process a typing log that includes backspace operations and output the final result.

TASK DESCRIPTION:
Given a typing log with characters and backspace operations, simulate the typing process and return the final text.

RULES:
1. Regular characters are added to the current text
2. The "←" symbol represents a backspace operation (delete the last character)
3. If backspace is used when text is empty, ignore it
4. Spaces in the log represent actual space characters being typed
5. Output only the final resulting text after all operations

EXAMPLES:
Input: "h e l l o ← ← k o"
Process: h→he→hel→hell→hello→hell→hel→helk→helko
Output: helko

Input: "a b c ← d"
Process: a→ab→abc→ab→abd
Output: abd

Input: "← a b ←"
Process: (empty)→a→ab→a
Output: a

Input: "h i ← ← ← j"
Process: h→hi→h→(empty)→(empty)→j
Output: j

IMPORTANT:
- Your response should ONLY contain the final result text wrapped inside <answer> and </answer>
- No explanations, no process steps, no additional text
- Handle edge cases (multiple backspaces, backspace on empty text)"""
        if restricted_reasoning:
            prompt += "\n\nAnswer directly without extra thinking."
        return prompt

    def get_user_prompt(self, typing_log: str, restricted_reasoning: bool = False) -> str:
        """Generate user prompt for a specific typing log"""
        prompt = f'Input: "{typing_log}"\nOutput (wrap in <answer></answer>):'
        if restricted_reasoning:
            prompt += " (answer directly without extended reasoning)"
        return prompt
    
    def get_few_shot_prompt(self, typing_log: str, restricted_reasoning: bool = False) -> str:
        """Generate few-shot prompt with examples"""
        examples = [
            ("h e l l o ← ← k o", "<answer>helko</answer>"),
            ("a b c ← d", "<answer>abd</answer>"),
            ("← a b ←", "<answer>a</answer>"),
            ("t e s t ← ← i n g", "<answer>tesing</answer>")
        ]
        
        prompt = """TASK: Typing Log Parser with Backspace Operations
Process the typing log and return the final text after all operations.

Examples:
"""
        for example_input, example_output in examples:
            prompt += f'Input: "{example_input}"\nOutput: {example_output}\n\n'
            
        prompt += f'Input: "{typing_log}"\nOutput (wrap in <answer></answer>):'
        if restricted_reasoning:
            prompt += " (answer directly without extended reasoning)"
        return prompt
    
    def evaluate_response(self, typing_log: str, response: str) -> Dict[str, Any]:
        """Evaluate the model's response for Task 2"""
        expected = self.generate_expected_output(typing_log)
        parsed_answer = self._extract_answer(response)

        # Remove quotes if present
        if parsed_answer.startswith('"') and parsed_answer.endswith('"'):
            parsed_answer = parsed_answer[1:-1]
        
        # Check exact match
        exact_match = parsed_answer == expected
        
        # Calculate character-level accuracy
        char_accuracy = self.calculate_char_accuracy(expected, parsed_answer)
        
        return {
            "exact_match": exact_match,
            "char_accuracy": char_accuracy,
            "expected": expected,
            "actual": parsed_answer,
            "score": 1.0 if exact_match else char_accuracy
        }

    @staticmethod
    def _extract_answer(response: str) -> str:
        """Extract content strictly inside <answer> tags; missing tags -> empty string."""
        match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return match.group(1).strip()
    
    def generate_expected_output(self, typing_log: str) -> str:
        """Generate the expected output for a given typing log"""
        tokens = typing_log.strip().split()
        result = ""
        
        for token in tokens:
            if token == "←":  # Backspace operation
                if len(result) > 0:
                    result = result[:-1]
            else:
                result += token
        
        return result
    
    def calculate_char_accuracy(self, expected: str, actual: str) -> float:
        """Calculate character-level accuracy between expected and actual output"""
        if not expected and not actual:
            return 1.0
        
        if not expected or not actual:
            return 0.0
        
        # Calculate edit distance
        max_len = max(len(expected), len(actual))
        if max_len == 0:
            return 1.0
        
        # Simple character match ratio
        matches = sum(1 for i in range(min(len(expected), len(actual))) 
                     if expected[i] == actual[i])
        
        return matches / max_len
