"""
Caesar cipher encoder and decoder utilities.
"""

class CaesarCipher:
    """Caesar cipher encoder and decoder."""
    
    @staticmethod
    def encode(text: str, shift: int) -> str:
        """
        Encode text using Caesar cipher with given shift.
        
        Args:
            text (str): Input text to encode
            shift (int): Number of positions to shift (0-25)
            
        Returns:
            str: Encoded text
        """
        result = []
        shift = shift % 26  # Normalize shift to 0-25 range
        
        for char in text:
            if char.isalpha():
                # Determine if uppercase or lowercase
                ascii_offset = ord('A') if char.isupper() else ord('a')
                # Shift the character
                shifted_char = chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
                result.append(shifted_char)
            else:
                # Keep non-alphabetic characters unchanged
                result.append(char)
        
        return ''.join(result)
    
    @staticmethod
    def decode(encrypted_text: str, shift: int) -> str:
        """
        Decode Caesar cipher encrypted text with given shift.
        
        Args:
            encrypted_text (str): Encrypted text to decode
            shift (int): Number of positions that were shifted during encoding
            
        Returns:
            str: Decoded text
        """
        # To decode, shift in the opposite direction
        return CaesarCipher.encode(encrypted_text, -shift)
    
    @staticmethod
    def find_shift(original_text: str, encrypted_text: str) -> int:
        """
        Find the shift value used to encrypt the text.
        
        Args:
            original_text (str): Original plain text
            encrypted_text (str): Encrypted text
            
        Returns:
            int: Shift value used, or -1 if not found
        """
        if len(original_text) != len(encrypted_text):
            return -1
        
        # Try all possible shifts (0-25)
        for shift in range(26):
            if CaesarCipher.encode(original_text, shift).lower() == encrypted_text.lower():
                return shift
        
        return -1
    
    @staticmethod
    def brute_force_decode(encrypted_text: str) -> dict:
        """
        Try all possible shifts to decode the text.
        
        Args:
            encrypted_text (str): Encrypted text to decode
            
        Returns:
            dict: Dictionary with shift as key and decoded text as value
        """
        results = {}
        for shift in range(26):
            results[shift] = CaesarCipher.decode(encrypted_text, shift)
        return results
    
    @staticmethod
    def is_valid_shift(shift: int) -> bool:
        """
        Check if the shift value is valid.
        
        Args:
            shift (int): Shift value to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return isinstance(shift, int) and 0 <= shift <= 25