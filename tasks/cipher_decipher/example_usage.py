"""
Example usage of the Cipher & Decipher Benchmark.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ciphers.morse_code import MorseCode
from src.ciphers.caesar_cipher import CaesarCipher
from src.data_generation.text_generator import TextGenerator
from src.utils.prompts import PromptTemplates


def demonstrate_ciphers():
    """Demonstrate cipher functionality."""
    print("=" * 60)
    print("CIPHER & DECIPHER EXAMPLES")
    print("=" * 60)
    print("1. Configure your API keys in configs/cipher_decipher/config.yaml or environment variables")

    # Morse Code Examples
    print("\n1. MORSE CODE:")
    print("-" * 20)
    
    text = "Hello World"
    morse_encoded = MorseCode.encode(text)
    morse_decoded = MorseCode.decode(morse_encoded)
    
    print(f"Original text: {text}")
    print(f"Morse encoded: {morse_encoded}")
    print(f"Morse decoded: {morse_decoded}")
    print(f"Roundtrip success: {text.upper() == morse_decoded}")
    
    # Caesar Cipher Examples
    print("\n2. CAESAR CIPHER:")
    print("-" * 20)
    
    text = "Hello World"
    shift = 3
    caesar_encoded = CaesarCipher.encode(text, shift)
    caesar_decoded = CaesarCipher.decode(caesar_encoded, shift)
    
    print(f"Original text: {text}")
    print(f"Caesar encoded (shift {shift}): {caesar_encoded}")
    print(f"Caesar decoded: {caesar_decoded}")
    print(f"Roundtrip success: {text == caesar_decoded}")


def demonstrate_text_generation():
    """Demonstrate text generation."""
    print("\n3. TEXT GENERATION:")
    print("-" * 20)
    
    generator = TextGenerator()
    
    # Generate some sample texts
    samples = generator.generate_samples(5, min_length=20, max_length=50)
    
    print("Generated text samples:")
    for i, sample in enumerate(samples, 1):
        print(f"{i}. {sample}")
    
    # Generate alphanumeric text
    alphanumeric = generator.generate_alphanumeric_text(30)
    print(f"\nAlphanumeric sample: {alphanumeric}")
    
    # Generate text with punctuation
    punctuated = generator.generate_punctuation_text("Hello world this is a test")
    print(f"With punctuation: {punctuated}")


def demonstrate_prompts():
    """Demonstrate prompt generation."""
    print("\n4. PROMPT TEMPLATES:")
    print("-" * 20)
    
    templates = PromptTemplates()
    
    # Morse encode prompt
    text = "Hello World"
    morse_prompt = templates.get_morse_encode_prompt(text, style='detailed')
    print("Morse Encode Prompt:")
    print(morse_prompt)
    print()
    
    # Caesar decode prompt
    encrypted = "Khoor Zruog"
    shift = 3
    caesar_prompt = templates.get_caesar_decode_prompt(encrypted, shift, style='detailed')
    print("Caesar Decode Prompt:")
    print(caesar_prompt)


def demonstrate_evaluation_format():
    """Demonstrate the expected evaluation format."""
    print("\n5. EXPECTED LLM RESPONSE FORMAT:")
    print("-" * 35)
    
    example_response = """<think>: I need to convert "Hello" to Morse code. Let me go through each letter:
H = ....
E = .
L = .-..
L = .-..
O = ---
Combining with spaces between letters: .... . .-.. .-.. ---
</think>
<answer>: .... . .-.. .-.. ---</answer>"""
    
    print("Example LLM Response:")
    print(example_response)
    print("\nThe benchmark will extract the content from the <answer> tags for evaluation.")


async def demonstrate_benchmark_flow():
    """Demonstrate the benchmark evaluation flow."""
    print("\n6. BENCHMARK EVALUATION FLOW:")
    print("-" * 35)
    
    from src.evaluation.evaluator import CipherEvaluator, TaskType
    
    # Create evaluator
    config = {
        'strict_match': True,
        'case_sensitive': False,
        'ignore_punctuation': True
    }
    evaluator = CipherEvaluator(config)
    
    # Example evaluation
    input_text = "Hello"
    model_output = """<think>: Converting to Morse code... </think>
<answer>: .... . .-.. .-.. ---</answer>"""
    
    result = evaluator.evaluate_morse_encode(input_text, model_output)
    
    print(f"Input text: {result.input_text}")
    print(f"Expected output: {result.expected_output}")
    print(f"Model output: {result.model_output}")
    print(f"Is correct: {result.is_correct}")
    print(f"Similarity score: {result.similarity_score:.3f}")


def main():
    """Run all demonstrations."""
    demonstrate_ciphers()
    demonstrate_text_generation()
    demonstrate_prompts()
    demonstrate_evaluation_format()
    
    # Run async demonstration
    asyncio.run(demonstrate_benchmark_flow())
    
    print("\n" + "=" * 60)
    print("READY TO RUN BENCHMARK!")
    print("=" * 60)
    print("To run the full benchmark:")
    print("1. Configure your API keys in configs/cipher_decipher/config.yaml or environment variables")
    print("2. Run: python main.py")
    print("3. Check results in the results/directory")
    print("\nFor help: python main.py --help")


if __name__ == "__main__":
    main()
