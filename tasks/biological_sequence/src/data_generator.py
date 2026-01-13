"""
Data generator for biological sequence manipulation benchmark tasks.
"""

import random
from typing import List, Tuple, Dict
import json

class BiologicalSequenceGenerator:
    """Generator for biological sequence test data."""
    
    def __init__(self):
        # DNA base pairs
        self.dna_complement = {
            'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'
        }
        
        # RNA base pairs (A pairs with U in RNA)
        self.rna_complement = {
            'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'
        }
        
        # Protein amino acid mappings
        self.aa_three_to_one = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        
        self.aa_one_to_three = {v: k for k, v in self.aa_three_to_one.items()}
        
        # Available bases and amino acids
        self.dna_bases = list(self.dna_complement.keys())
        self.rna_bases = list(self.rna_complement.keys())
        self.aa_three_letter = list(self.aa_three_to_one.keys())
        self.aa_one_letter = list(self.aa_one_to_three.keys())
    
    def generate_dna_sequence(self, length: int) -> str:
        """Generate a random DNA sequence of specified length."""
        return ''.join(random.choices(self.dna_bases, k=length))
    
    def generate_rna_sequence(self, length: int) -> str:
        """Generate a random RNA sequence of specified length."""
        return ''.join(random.choices(self.rna_bases, k=length))
    
    def generate_protein_three_letter(self, length: int) -> str:
        """Generate a random protein sequence in three-letter format."""
        amino_acids = random.choices(self.aa_three_letter, k=length)
        return '-'.join(amino_acids)
    
    def generate_protein_one_letter(self, length: int) -> str:
        """Generate a random protein sequence in one-letter format."""
        amino_acids = random.choices(self.aa_one_letter, k=length)
        return ''.join(amino_acids)  # No separators for one-letter format
    
    def get_dna_complement(self, sequence: str) -> str:
        """Get complement DNA sequence."""
        return ''.join(self.dna_complement[base] for base in sequence)
    
    def get_rna_complement(self, sequence: str) -> str:
        """Get complement RNA sequence."""
        return ''.join(self.rna_complement[base] for base in sequence)
    
    def convert_protein_three_to_one(self, sequence: str) -> str:
        """Convert protein sequence from three-letter to one-letter format."""
        amino_acids = sequence.split('-')
        return ''.join(self.aa_three_to_one[aa] for aa in amino_acids)  # No separators
    
    def convert_protein_one_to_three(self, sequence: str) -> str:
        """Convert protein sequence from one-letter to three-letter format."""
        # For one-letter sequences, each character is an amino acid (no separators)
        amino_acids = list(sequence)
        return '-'.join(self.aa_one_to_three[aa] for aa in amino_acids)
    
    def generate_test_cases(self, task_type: str, num_cases: int, 
                           sequence_length_range: Tuple[int, int] = (5, 15)) -> List[Dict]:
        """
        Generate test cases for specified task type.
        
        Args:
            task_type: Type of task ('dna_complement', 'rna_complement', 
                      'protein_three_to_one', 'protein_one_to_three')
            num_cases: Number of test cases to generate
            sequence_length_range: Range of sequence lengths
        
        Returns:
            List of test cases with input and expected output
        """
        test_cases = []
        min_len, max_len = sequence_length_range
        
        for _ in range(num_cases):
            length = random.randint(min_len, max_len)
            
            if task_type == 'dna_complement':
                input_seq = self.generate_dna_sequence(length)
                expected_output = self.get_dna_complement(input_seq)
                
            elif task_type == 'rna_complement':
                input_seq = self.generate_rna_sequence(length)
                expected_output = self.get_rna_complement(input_seq)
                
            elif task_type == 'protein_three_to_one':
                input_seq = self.generate_protein_three_letter(length)
                expected_output = self.convert_protein_three_to_one(input_seq)
                
            elif task_type == 'protein_one_to_three':
                input_seq = self.generate_protein_one_letter(length)
                expected_output = self.convert_protein_one_to_three(input_seq)
                
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            test_cases.append({
                'task_type': task_type,
                'input': input_seq,
                'expected_output': expected_output,
                'sequence_length': length
            })
        
        return test_cases
    
    def generate_full_benchmark(self, config: Dict) -> Dict:
        """
        Generate complete benchmark dataset based on configuration.
        
        Args:
            config: Configuration dictionary with task settings
        
        Returns:
            Dictionary containing all test cases organized by task
        """
        benchmark_data = {}
        
        for task_type, task_config in config.items():
            if task_config.get('enabled', True):
                num_cases = task_config.get('num_cases', 50)
                length_range = task_config.get('sequence_length_range', (5, 15))
                
                test_cases = self.generate_test_cases(
                    task_type, num_cases, length_range
                )
                benchmark_data[task_type] = test_cases
        
        return benchmark_data


def main():
    """Example usage of the data generator."""
    generator = BiologicalSequenceGenerator()
    
    # Example configuration
    config = {
        'dna_complement': {
            'enabled': True,
            'num_cases': 10,
            'sequence_length_range': (8, 12)
        },
        'rna_complement': {
            'enabled': True,
            'num_cases': 10,
            'sequence_length_range': (8, 12)
        },
        'protein_three_to_one': {
            'enabled': True,
            'num_cases': 10,
            'sequence_length_range': (5, 8)
        },
        'protein_one_to_three': {
            'enabled': True,
            'num_cases': 10,
            'sequence_length_range': (5, 8)
        }
    }
    
    # Generate benchmark data
    benchmark_data = generator.generate_full_benchmark(config)
    
    # Print sample data
    for task_type, cases in benchmark_data.items():
        print(f"\n{task_type.upper()} - Sample case:")
        print(f"Input: {cases[0]['input']}")
        print(f"Expected: {cases[0]['expected_output']}")


if __name__ == "__main__":
    main()