"""
Morse code encoder and decoder utilities.
"""

class MorseCode:
    """Morse code encoder and decoder."""
    
    # Morse code mapping
    MORSE_CODE_DICT = {
        'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
        'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
        'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
        'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
        'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---',
        '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...',
        '8': '---..', '9': '----.', ' ': '/', '.': '.-.-.-', ',': '--..--',
        '?': '..--..', "'": '.----.', '!': '-.-.--', '/': '-..-.', '(': '-.--.',
        ')': '-.--.-', '&': '.-...', ':': '---...', ';': '-.-.-.', '=': '-...-',
        '+': '.-.-.', '-': '-....-', '_': '..--.-', '"': '.-..-.', '$': '...-..-',
        '@': '.--.-.'
    }
    
    # Reverse mapping for decoding
    REVERSE_MORSE_CODE_DICT = {v: k for k, v in MORSE_CODE_DICT.items()}
    
    @classmethod
    def encode(cls, text: str) -> str:
        """
        Encode text to Morse code.
        
        Args:
            text (str): Input text to encode
            
        Returns:
            str: Morse code representation
        """
        text = text.upper()
        morse_code = []
        
        for char in text:
            if char in cls.MORSE_CODE_DICT:
                morse_code.append(cls.MORSE_CODE_DICT[char])
            else:
                # Skip unknown characters
                continue
        
        return ' '.join(morse_code)
    
    @classmethod
    def decode(cls, morse_code: str) -> str:
        """
        Decode Morse code to text.
        
        Args:
            morse_code (str): Morse code to decode
            
        Returns:
            str: Decoded text
        """
        morse_code = morse_code.strip()
        morse_symbols = morse_code.split(' ')
        decoded_text = []
        
        for symbol in morse_symbols:
            if symbol in cls.REVERSE_MORSE_CODE_DICT:
                decoded_text.append(cls.REVERSE_MORSE_CODE_DICT[symbol])
            elif symbol == '':
                # Handle multiple spaces
                continue
            else:
                # Skip unknown symbols
                continue
        
        return ''.join(decoded_text)
    
    @classmethod
    def is_valid_morse(cls, morse_code: str) -> bool:
        """
        Check if the given string is valid Morse code.
        
        Args:
            morse_code (str): String to validate
            
        Returns:
            bool: True if valid Morse code, False otherwise
        """
        if not morse_code or not morse_code.strip():
            return False
        
        # Split by spaces and check each symbol
        symbols = morse_code.strip().split(' ')
        for symbol in symbols:
            if symbol and symbol not in cls.REVERSE_MORSE_CODE_DICT:
                return False
        
        return True