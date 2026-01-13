"""
Quick test script for a single model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typewriter.models import create_model
from typewriter.tasks.task1_typewriter import Task1TypewriterEffect
from typewriter.tasks.task2_backspace import Task2BackspaceHandling
import argparse

def test_model(model_name: str, verbose: bool = False):
    """Quick test of a model on sample inputs"""
    
    print("="*60)
    print(f"Testing Model: {model_name}")
    print("="*60)
    
    # Create model
    try:
        print(f"\n[1/4] Loading model...")
        model = create_model(model_name)
        print(f"✓ Model loaded successfully")
        print(f"  Type: {model.config.model_type}")
        if hasattr(model.config, 'local_path'):
            print(f"  Path: {model.config.local_path}")
        
    except Exception as e:
        print(f"✗ Failed to load model: {str(e)}")
        return
    
    # Check availability
    print(f"\n[2/4] Checking model availability...")
    if not model.is_available():
        print(f"✗ Model is not available")
        return
    
    print("Model Greeting Test:")
    try:
        greeting = model.generate("Hello, how are you?")
        print(f"Response: '{greeting}'")
    except Exception as e:
        print(f"✗ Model generation error: {str(e)}")
        return
    print(f"✓ Model is available")
    
    # Test Task 1
    print(f"\n[3/4] Testing Task 1: Typewriter Effect")
    task1 = Task1TypewriterEffect()
    test_word = "hello"
    
    try:
        prompt = task1.get_few_shot_prompt(test_word)
        if verbose:
            print(f"\nPrompt:\n{prompt}\n")
        
        print(f"Input: '{test_word}'")
        response = model.generate(prompt)
        print(f"Response: '{response}'")
        
        evaluation = task1.evaluate_response(test_word, response)
        print(f"Expected: '{evaluation['expected']}'")
        print(f"Exact Match: {evaluation['exact_match']}")
        print(f"Score: {evaluation['score']:.3f}")
        
        if evaluation['exact_match']:
            print("✓ Task 1 PASSED")
        else:
            print("✗ Task 1 FAILED")
        
    except Exception as e:
        print(f"✗ Task 1 error: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
    
    # Test Task 2
    print(f"\n[4/4] Testing Task 2: Backspace Handling")
    task2 = Task2BackspaceHandling()
    test_log = "h e l l o ← ← k o"
    
    try:
        prompt = task2.get_few_shot_prompt(test_log)
        if verbose:
            print(f"\nPrompt:\n{prompt}\n")
        
        print(f"Input: '{test_log}'")
        response = model.generate(prompt)
        print(f"Response: '{response}'")
        
        evaluation = task2.evaluate_response(test_log, response)
        print(f"Expected: '{evaluation['expected']}'")
        print(f"Exact Match: {evaluation['exact_match']}")
        print(f"Score: {evaluation['score']:.3f}")
        
        if evaluation['exact_match']:
            print("✓ Task 2 PASSED")
        else:
            print("✗ Task 2 FAILED")
        
    except Exception as e:
        print(f"✗ Task 2 error: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test completed")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Quick model test")
    parser.add_argument('model', help='Model name to test')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show detailed output including prompts')
    
    args = parser.parse_args()
    test_model(args.model, args.verbose)

if __name__ == "__main__":
    main()
