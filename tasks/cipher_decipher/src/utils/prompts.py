"""
Prompt templates for cipher and decipher tasks.
"""

from typing import Dict, Any


class PromptTemplates:
    """Collection of prompt templates for different cipher tasks."""

    MORSE_ENCODE_RULES = """Encode the given text to Morse code using the exact mappings and rules below:

Mapping Rules:

Letters:
A .-    B -...  C -.-.  D -..   E .     F ..-.  G --.   H ....  I ..    J .---
K -.-   L .-..  M --    N -.    O ---   P .--.  Q --.-  R .-.   S ...   T -
U ..-   V ...-  W .--   X -..-  Y -.--  Z --..

Digits:
0 -----  1 .----  2 ..---  3 ...--  4 ....-  5 .....  
6 -....  7 --...  8 ---..  9 ----.

Punctuation:
. .-.-.-    , --..--    ? ..--..    ' .----.    ! -.-.-- 
/ -..-.     ( -.--.     ) -.--.-    & .-...     : ---...
; -.-.-.    = -...-     + .-.-.     - -....-    _ ..--.- 
" .-..-.    $ ...-..-   @ .--.-.
"""

    # Morse Code Encoding Prompts
    MORSE_ENCODE_PROMPTS = {
        'basic': """Convert the following text to Morse code. Use dots (.) and dashes (-) for the Morse code symbols, and separate each letter with a space. Use forward slash (/) for spaces between words.

Encode the given text to Morse code using the exact mappings and rules below:

Mapping Rules:

Letters:
A .-    B -...  C -.-.  D -..   E .     F ..-.  G --.   H ....  I ..    J .---
K -.-   L .-..  M --    N -.    O ---   P .--.  Q --.-  R .-.   S ...   T -
U ..-   V ...-  W .--   X -..-  Y -.--  Z --..

Digits:
0 -----  1 .----  2 ..---  3 ...--  4 ....-  5 .....  
6 -....  7 --...  8 ---..  9 ----.

Punctuation:
. .-.-.-    , --..--    ? ..--..    ' .----.    ! -.-.-- 
/ -..-.     ( -.--.     ) -.--.-    & .-...     : ---...
; -.-.-.    = -...-     + .-.-.     - -....-    _ ..--.- 
" .-..-.    $ ...-..-   @ .--.-.

The text to encode is below:

{text}

Please provide your response in the following format:
[Optional: Your reasoning or analysis]
<answer>Your answer here</answer>

Be careful, the words between <answer> and </answer> tags should be your hand-in final answer without any extra explanation.""",
        
        'detailed': """You are a Morse code expert. Convert the given English text to Morse code following these rules:
1. Use dots (.) and dashes (-) only
2. Separate each letter's Morse code with a single space
3. Use forward slash (/) to represent spaces between words
4. Convert all letters to uppercase before encoding
5. Only encode letters, numbers, and basic punctuation

Encode the given text to Morse code using the exact mappings and rules below:

Mapping Rules:

Letters:
A .-    B -...  C -.-.  D -..   E .     F ..-.  G --.   H ....  I ..    J .---
K -.-   L .-..  M --    N -.    O ---   P .--.  Q --.-  R .-.   S ...   T -
U ..-   V ...-  W .--   X -..-  Y -.--  Z --..

Digits:
0 -----  1 .----  2 ..---  3 ...--  4 ....-  5 .....  
6 -....  7 --...  8 ---..  9 ----.

Punctuation:
. .-.-.-    , --..--    ? ..--..    ' .----.    ! -.-.-- 
/ -..-.     ( -.--.     ) -.--.-    & .-...     : ---...
; -.-.-.    = -...-     + .-.-.     - -....-    _ ..--.- 
" .-..-.    $ ...-..-   @ .--.-.

The text to encode is below:

{text}

Please provide your response in the following format:
[Optional: Your reasoning or analysis]
<answer>Your answer here</answer>

Be careful, the words between <answer> and </answer> tags should be your hand-in final answer without any extra explanation.""",
        
        'step_by_step': """Convert the text to Morse code step by step:

Step 1: Convert each character to its Morse code equivalent
Step 2: Separate letters with spaces and words with forward slashes

Encode the given text to Morse code using the exact mappings and rules below:

Mapping Rules:

Letters:
A .-    B -...  C -.-.  D -..   E .     F ..-.  G --.   H ....  I ..    J .---
K -.-   L .-..  M --    N -.    O ---   P .--.  Q --.-  R .-.   S ...   T -
U ..-   V ...-  W .--   X -..-  Y -.--  Z --..

Digits:
0 -----  1 .----  2 ..---  3 ...--  4 ....-  5 .....  
6 -....  7 --...  8 ---..  9 ----.

Punctuation:
. .-.-.-    , --..--    ? ..--..    ' .----.    ! -.-.-- 
/ -..-.     ( -.--.     ) -.--.-    & .-...     : ---...
; -.-.-.    = -...-     + .-.-.     - -....-    _ ..--.- 
" .-..-.    $ ...-..-   @ .--.-.

The text to encode is below:

{text}

Please provide your response in the following format:
[Optional: Your reasoning or analysis]
<answer>Your answer here</answer>

Be careful, the words between <answer> and </answer> tags should be your hand-in final answer without any extra explanation.""",
    }
    
    # Morse Code Decoding Prompts
    MORSE_DECODE_PROMPTS = {
        'basic': """Convert the following Morse code back to English text. The Morse code uses dots (.) and dashes (-), with spaces separating letters and forward slashes (/) representing spaces between words.
Be careful, as morse code DO NOT distinguish between uppercase and lowercase letters, your all answer words should be UPPERCASE.

Decode the given text to Morse code using the exact mappings and rules below:

Mapping Rules:

Letters:
A .-    B -...  C -.-.  D -..   E .     F ..-.  G --.   H ....  I ..    J .---
K -.-   L .-..  M --    N -.    O ---   P .--.  Q --.-  R .-.   S ...   T -
U ..-   V ...-  W .--   X -..-  Y -.--  Z --..

Digits:
0 -----  1 .----  2 ..---  3 ...--  4 ....-  5 .....  
6 -....  7 --...  8 ---..  9 ----.

Punctuation:
. .-.-.-    , --..--    ? ..--..    ' .----.    ! -.-.-- 
/ -..-.     ( -.--.     ) -.--.-    & .-...     : ---...
; -.-.-.    = -...-     + .-.-.     - -....-    _ ..--.- 
" .-..-.    $ ...-..-   @ .--.-.

The morse code to decode is below: 

{morse_code}

Please provide your response in the following format:
[Optional: Your reasoning or analysis]
<answer>Your answer here</answer>

Be careful, the words between <answer> and </answer> tags should be your hand-in final answer without any extra explanation.""",
        
        'detailed': """You are a Morse code expert. Decode the given Morse code to English text following these rules:
1. Each sequence of dots and dashes separated by spaces represents one letter
2. Forward slashes (/) represent spaces between words
3. Convert the decoded result to readable English text
4. As morse code DO NOT distinguish between uppercase and lowercase letters, your all answer words should be UPPERCASE.

Decode the given text to Morse code using the exact mappings and rules below:

Mapping Rules:

Letters:
A .-    B -...  C -.-.  D -..   E .     F ..-.  G --.   H ....  I ..    J .---
K -.-   L .-..  M --    N -.    O ---   P .--.  Q --.-  R .-.   S ...   T -
U ..-   V ...-  W .--   X -..-  Y -.--  Z --..

Digits:
0 -----  1 .----  2 ..---  3 ...--  4 ....-  5 .....  
6 -....  7 --...  8 ---..  9 ----.

Punctuation:
. .-.-.-    , --..--    ? ..--..    ' .----.    ! -.-.-- 
/ -..-.     ( -.--.     ) -.--.-    & .-...     : ---...
; -.-.-.    = -...-     + .-.-.     - -....-    _ ..--.- 
" .-..-.    $ ...-..-   @ .--.-.

The morse code to decode is below: 

{morse_code}

Please provide your response in the following format:
[Optional: Your reasoning or analysis]
<answer>Your answer here</answer>

Be careful, the words between <answer> and </answer> tags should be your hand-in final answer without any extra explanation.""",
        
        'step_by_step': """Decode the Morse code step by step:

Step 1: Split the Morse code by spaces to get individual letters
Step 2: Convert each Morse sequence to its corresponding letter
Step 3: Handle forward slashes as word separators
Be careful, as morse code DO NOT distinguish between uppercase and lowercase letters, your all answer words should be UPPERCASE.

Decode the given text to Morse code using the exact mappings and rules below:

Mapping Rules:

Letters:
A .-    B -...  C -.-.  D -..   E .     F ..-.  G --.   H ....  I ..    J .---
K -.-   L .-..  M --    N -.    O ---   P .--.  Q --.-  R .-.   S ...   T -
U ..-   V ...-  W .--   X -..-  Y -.--  Z --..

Digits:
0 -----  1 .----  2 ..---  3 ...--  4 ....-  5 .....  
6 -....  7 --...  8 ---..  9 ----.

Punctuation:
. .-.-.-    , --..--    ? ..--..    ' .----.    ! -.-.-- 
/ -..-.     ( -.--.     ) -.--.-    & .-...     : ---...
; -.-.-.    = -...-     + .-.-.     - -....-    _ ..--.- 
" .-..-.    $ ...-..-   @ .--.-.

The morse code to decode is below: 

{morse_code}

Please provide your response in the following format:
[Optional: Your reasoning or analysis]
<answer>Your answer here</answer>

Be careful, the words between <answer> and </answer> tags should be your hand-in final answer without any extra explanation.""",
    }
    
    # Caesar Cipher Encoding Prompts
    CAESAR_ENCODE_PROMPTS = {
        'basic': """Encrypt the following text using a Caesar cipher with a shift of {shift}. 

Be careful: 
Only shift alphabetic characters, keep other characters unchanged; Wrap around the alphabet (Z shifts to A, z shifts to a).

**Encrypt reference**:

For uppercase: new_index = (index(A..Z) + shift_value) mod 26
For lowercase: new_index = (index(a..z) + shift_value) mod 26

The text to caesar encode is below:

{text}

You have to shift: {shift}

Please provide your response in the following format:
[Optional: Your reasoning or analysis]
<answer>Your answer here</answer>

Be careful, the words between <answer> and </answer> tags should be your hand-in final answer without any extra explanation.""",
        
        'detailed': """You are a cryptography expert. Encrypt the given text using Caesar cipher with the following specifications:
1. Shift each letter by {shift} positions in the alphabet
2. Preserve the case of letters (uppercase stays uppercase, lowercase stays lowercase)
3. Keep non-alphabetic characters (numbers, punctuation, spaces) unchanged
4. Wrap around the alphabet (Z shifts to A, z shifts to a)

**Encrypt reference**:

For uppercase: new_index = (index(A..Z) + shift_value) mod 26
For lowercase: new_index = (index(a..z) + shift_value) mod 26

The text to caesar encode is below:

{text}

You have to shift: {shift}

Please provide your response in the following format:
[Optional: Your reasoning or analysis]
<answer>Your answer here</answer>

Be careful, the words between <answer> and </answer> tags should be your hand-in final answer without any extra explanation.""",
        
        'step_by_step': """Encrypt using Caesar cipher step by step:

Step 1: For each alphabetic character, shift it {shift} positions forward in the alphabet
Step 2: Keep non-alphabetic characters unchanged
Step 3: Maintain original case

**Encrypt reference**:

For uppercase: new_index = (index(A..Z) + shift_value) mod 26
For lowercase: new_index = (index(a..z) + shift_value) mod 26

Be careful: 
Only shift alphabetic characters, keep other characters unchanged; Wrap around the alphabet (Z shifts to A, z shifts to a).

The text to caesar encode is below:

{text}

You have to shift: {shift}

Please provide your response in the following format:
[Optional: Your reasoning or analysis]
<answer>Your answer here</answer>

Be careful, the words between <answer> and </answer> tags should be your hand-in final answer without any extra explanation.""",
    }
    
    # Caesar Cipher Decoding Prompts
    CAESAR_DECODE_PROMPTS = {
        'basic': """Decrypt the following Caesar cipher encrypted text. The original text was shifted by {shift} positions. Shift each letter back by {shift} positions to get the original text.

**Decode reference**:

For uppercase: original_index = (index(A..Z) - shift_value) mod 26
For lowercase: original_index = (index(a..z) - shift_value) mod 26

The encrypted text to decode is:

{encrypted_text}

The shift value used is: {shift}

Be careful: 
Only shift alphabetic characters, keep other characters unchanged; Wrap around the alphabet (A shifts to Z, a shifts to z when going backwards).

Please provide your response in the following format:
[Optional: Your reasoning or analysis]
<answer>Your answer here</answer>

Be careful, the words between <answer> and </answer> tags should be your hand-in final answer without any extra explanation.""",
        
        'detailed': """You are a cryptography expert. Decrypt the given Caesar cipher text with the following specifications:
1. The text was encrypted with a shift of {shift} positions
2. Shift each letter back by {shift} positions in the alphabet
3. Preserve the case of letters (uppercase stays uppercase, lowercase stays lowercase)
4. Keep non-alphabetic characters unchanged
5. Wrap around the alphabet (A shifts to Z, a shifts to z when going backwards)

**Decode reference**:

For uppercase: original_index = (index(A..Z) - shift_value) mod 26
For lowercase: original_index = (index(a..z) - shift_value) mod 26

The encrypted text to decode is:

{encrypted_text}

The shift value used is: {shift}

Please provide your response in the following format:
<think>: Your thinking process here. </think>
<answer>: Your answer here. </answer>

Be careful, the words between <answer> and </answer> tags should be your hand-in final answer without any extra explanation.""",
        
        'step_by_step': """Decrypt Caesar cipher step by step:

Step 1: For each alphabetic character, shift it {shift} positions backward in the alphabet
Step 2: Keep non-alphabetic characters unchanged
Step 3: Maintain original case

**Decode reference**:

For uppercase: original_index = (index(A..Z) - shift_value) mod 26
For lowercase: original_index = (index(a..z) - shift_value) mod 26

The encrypted text to decode is:

{encrypted_text}

The shift value used is: {shift}

Be careful: 
Only shift alphabetic characters, keep other characters unchanged; Wrap around the alphabet (A shifts to Z, a shifts to z when going backwards).

Please provide your response in the following format:
[Optional: Your reasoning or analysis]
<answer>Your answer here</answer>

Be careful, the words between <answer> and </answer> tags should be your hand-in final answer without any extra explanation.""",
    }
    
    @classmethod
    def _add_restriction(self, prompt: str, restricted_reasoning: bool) -> str:
        """Append restricted reasoning guidance if requested."""
        if not restricted_reasoning:
            return prompt
        return (
            f"{prompt}\n\nAnswer directly after <answer> tags without thinking or reasoning. Begin your answer now: <answer>"
        )

    def get_morse_encode_prompt(self, text: str, style: str = 'detailed', restricted_reasoning: bool = False) -> str:
        """
        Get Morse code encoding prompt.
        
        Args:
            text (str): Text to encode
            style (str): Prompt style ('basic', 'detailed', 'step_by_step')
            
        Returns:
            str: Formatted prompt
        """
        if style not in self.MORSE_ENCODE_PROMPTS:
            style = 'detailed'
        prompt = self.MORSE_ENCODE_PROMPTS[style].format(text=text)
        return self._add_restriction(prompt, restricted_reasoning)
    
    @classmethod
    def get_morse_decode_prompt(self, morse_code: str, style: str = 'detailed', restricted_reasoning: bool = False) -> str:
        """
        Get Morse code decoding prompt.
        
        Args:
            morse_code (str): Morse code to decode
            style (str): Prompt style ('basic', 'detailed', 'step_by_step')
            
        Returns:
            str: Formatted prompt
        """
        if style not in self.MORSE_DECODE_PROMPTS:
            style = 'detailed'
        prompt = self.MORSE_DECODE_PROMPTS[style].format(morse_code=morse_code)
        return self._add_restriction(prompt, restricted_reasoning)
    
    @classmethod
    def get_caesar_encode_prompt(self, text: str, shift: int, style: str = 'detailed', restricted_reasoning: bool = False) -> str:
        """
        Get Caesar cipher encoding prompt.
        
        Args:
            text (str): Text to encrypt
            shift (int): Shift value
            style (str): Prompt style ('basic', 'detailed', 'step_by_step')
            
        Returns:
            str: Formatted prompt
        """
        if style not in self.CAESAR_ENCODE_PROMPTS:
            style = 'detailed'
        prompt = self.CAESAR_ENCODE_PROMPTS[style].format(text=text, shift=shift)
        return self._add_restriction(prompt, restricted_reasoning)
    
    @classmethod
    def get_caesar_decode_prompt(self, encrypted_text: str, shift: int, style: str = 'detailed', restricted_reasoning: bool = False) -> str:
        """
        Get Caesar cipher decoding prompt.
        
        Args:
            encrypted_text (str): Encrypted text to decrypt
            shift (int): Original shift value
            style (str): Prompt style ('basic', 'detailed', 'step_by_step')
            
        Returns:
            str: Formatted prompt
        """
        if style not in self.CAESAR_DECODE_PROMPTS:
            style = 'detailed'
        prompt = self.CAESAR_DECODE_PROMPTS[style].format(encrypted_text=encrypted_text, shift=shift)
        return self._add_restriction(prompt, restricted_reasoning)
    
    @classmethod
    def get_available_styles(self) -> Dict[str, list]:
        """
        Get available prompt styles for each task.
        
        Returns:
            Dict[str, list]: Available styles for each task type
        """
        return {
            'morse_encode': list(self.MORSE_ENCODE_PROMPTS.keys()),
            'morse_decode': list(self.MORSE_DECODE_PROMPTS.keys()),
            'caesar_encode': list(self.CAESAR_ENCODE_PROMPTS.keys()),
            'caesar_decode': list(self.CAESAR_DECODE_PROMPTS.keys()),
        }


class PromptValidator:
    """Validates and processes prompts for consistency."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and prepare text for prompt inclusion.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        cleaned = ' '.join(text.split())
        
        # Remove potentially problematic characters for prompts
        cleaned = cleaned.replace('"', "'").replace('\n', ' ').replace('\r', ' ')
        
        return cleaned.strip()
    
    @staticmethod
    def validate_shift(shift: int) -> int:
        """
        Validate and normalize Caesar cipher shift value.
        
        Args:
            shift (int): Shift value
            
        Returns:
            int: Normalized shift value (0-25)
        """
        if not isinstance(shift, int):
            raise ValueError("Shift must be an integer")
        
        # Normalize to 0-25 range
        return shift % 26
    
    @staticmethod
    def format_morse_code(morse_code: str) -> str:
        """
        Format Morse code for consistent presentation.
        
        Args:
            morse_code (str): Raw Morse code
            
        Returns:
            str: Formatted Morse code
        """
        if not morse_code:
            return ""
        
        # Clean up spacing and ensure consistent format
        cleaned = ' '.join(morse_code.split())
        
        # Ensure forward slashes for word breaks are properly spaced
        cleaned = cleaned.replace(' / ', ' / ').replace('/', ' / ')
        
        return cleaned.strip()
