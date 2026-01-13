"""
Perturbation module for generating adversarial prompt variants.
"""

import random
from typing import List, Dict


class PerturbationEngine:
    """Engine for applying various perturbations to text."""
    
    LEET_SPEAK_MAP = {
        'a': ['а', 'α'],          # Cyrillic a, Greek alpha
        'b': ['в', 'β'],          # Cyrillic ve, Greek beta
        'c': ['с', 'ϲ', 'ς'],     # Cyrillic es, Greek lunate sigma
        'd': ['ԁ'],               # Cyrillic komi de
        'e': ['е', 'ε'],          # Cyrillic ie, Greek epsilon
        'f': ['ғ'],               # Cyrillic ge with stroke
        'g': ['ɡ'],               # Latin letter small g 
        'h': ['һ', 'н', 'ح'],     # Cyrillic shha, Cyrillic en, Arabic ح
        'i': ['і', 'ι', 'ı'],     # Cyrillic i, Greek iota, dotless i 
        'j': ['ј', 'ʝ'],          # Cyrillic je, Latin j-like
        'k': ['к', 'κ'],          # Cyrillic ka, Greek kappa
        'l': ['ⅼ', 'ӏ', 'ا'],     # Roman numeral fifty, Cyrillic el, Arabic alif
        'm': ['м'],               # Cyrillic em
        'n': ['п', 'η'],          # Cyrillic pe, Greek eta
        'o': ['о', 'ο', '٠', 'و'],# Cyrillic o, Greek omicron, Arabic-Indic 0, Arabic waw
        'p': ['р', 'ρ'],          # Cyrillic er, Greek rho
        'r': ['г'],               # Cyrillic ge
        's': ['ѕ', 'ς'],          # Cyrillic dze, Greek final sigma
        't': ['т', 'τ'],          # Cyrillic te, Greek tau
        'u': ['υ', 'ս'],          # Greek upsilon, Armenian se
        'v': ['ν', 'ѵ'],          # Greek nu, Cyrillic izhitsa
        'w': ['ѡ', 'ω'],          # Cyrillic omega, Greek omega
        'x': ['х', 'χ'],          # Cyrillic ha, Greek chi
        'y': ['у', 'γ'],          # Cyrillic u, Greek gamma
        'z': ['ᴢ', 'ζ'],          # Latin small capital Z, Greek zeta
    }
    
    def __init__(self, config: Dict):
        """
        Initialize perturbation engine.
        
        Args:
            config: Configuration dictionary with perturbation settings
        """
        self.config = config
        self.leet_ratio = config.get('leet_speak_ratio', 0.3)
        self.insertion_chars = config.get('insertion_chars', ['9', '_', '-'])
        self.insertion_ratio = config.get('insertion_ratio', 0.3)
        
    def apply_leet_speak(self, text: str) -> str:
        """
        Apply leet speak transformation.
        
        Args:
            text: Original text
            
        Returns:
            Perturbed text with leet speak
        """
        result = list(text.lower())
        
        for i, char in enumerate(result):
            if char in self.LEET_SPEAK_MAP and random.random() < self.leet_ratio:
                result[i] = random.choice(self.LEET_SPEAK_MAP[char])
        
        return ''.join(result)
    
    def apply_insertion(self, text: str) -> str:
        """
        Apply character insertion perturbation.
        
        Args:
            text: Original text
            
        Returns:
            Perturbed text with inserted characters
        """
        words = text.split()
        result = []
        
        for word in words:
            chars = list(word.lower())
            perturbed_chars = []
            
            for char in chars:
                perturbed_chars.append(char)
                if random.random() < self.insertion_ratio:
                    perturbed_chars.append(random.choice(self.insertion_chars))
            
            result.append(''.join(perturbed_chars))
        
        # Join with insertion characters
        if random.random() < 0.5:
            return random.choice(self.insertion_chars).join(result)
        else:
            return ''.join(result)
    
    def apply_mixed(self, text: str) -> str:
        """
        Apply mixed perturbations (leet speak + insertion).
        
        Args:
            text: Original text
            
        Returns:
            Perturbed text with mixed perturbations
        """
        # First apply leet speak
        text = self.apply_leet_speak(text)
        # Then apply insertion
        text = self.apply_insertion(text)
        return text
    
    def perturb(self, text: str, method: str) -> str:
        """
        Apply specified perturbation method.
        
        Args:
            text: Original text
            method: Perturbation method name
            
        Returns:
            Perturbed text
        """
        if method == "leet_speak":
            return self.apply_leet_speak(text)
        elif method == "insertion":
            return self.apply_insertion(text)
        elif method == "mixed":
            return self.apply_mixed(text)
        else:
            raise ValueError(f"Unknown perturbation method: {method}")