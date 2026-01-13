"""
Data generator for creating benchmark test cases.
"""

import random
import json
from typing import List, Dict, Optional
from pathlib import Path
from src.perturbation import PerturbationEngine


class BenchmarkDataGenerator:
    """Generator for benchmark test data."""
    
    def __init__(self, config: Dict, seed: int = 42):
        """
        Initialize data generator.
        
        Args:
            config: Configuration dictionary
            seed: Random seed for reproducibility
        """
        self.config = config
        self.perturbation_engine = PerturbationEngine(config['perturbation'])
        random.seed(seed)
        self.generated_contexts = None
        
    def load_generated_contexts(self, contexts_file: str):
        """
        Load GPT-generated contexts from file.
        
        Args:
            contexts_file: Path to JSON file with generated contexts
        """
        with open(contexts_file, 'r') as f:
            self.generated_contexts = json.load(f)
        print(f"Loaded generated contexts from: {contexts_file}")
    
    def generate_samples(self, num_samples: int) -> List[Dict]:
        """
        Generate benchmark samples.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of sample dictionaries
        """
        use_generated = self.config['benchmark'].get('use_generated_contexts', False)
        
        if use_generated and self.generated_contexts is None:
            contexts_file = self.config['benchmark'].get('generated_contexts_file')
            if contexts_file and Path(contexts_file).exists():
                self.load_generated_contexts(contexts_file)
            else:
                print("Warning: use_generated_contexts=True but no contexts loaded. "
                      "Using template-based generation.")
                use_generated = False
        
        if use_generated and self.generated_contexts:
            return self._generate_from_contexts(num_samples)
        else:
            return self._generate_from_templates(num_samples)
    
    def _generate_from_templates(self, num_samples: int) -> List[Dict]:
        """
        Generate samples from predefined templates.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        categories = self.config['data']['categories']
        perturbation_types = self.config['perturbation']['types']
        
        for i in range(num_samples):
            # Select category and template
            category = random.choice(categories)
            template = random.choice(category['templates'])
            
            # Select perturbation type
            perturbation_type = random.choice(perturbation_types)
            
            # Generate perturbed text
            perturbed_text = self.perturbation_engine.perturb(
                template, perturbation_type
            )
            
            sample = {
                'id': i,
                'category': category['name'],
                'difficulty': category.get('difficulty', 'unknown'),
                'original': template,
                'perturbed': perturbed_text,
                'perturbation_type': perturbation_type,
                'source': 'template'
            }
            
            samples.append(sample)
        
        return samples
    
    def _generate_from_contexts(self, num_samples: int) -> List[Dict]:
        """
        Generate samples from GPT-generated contexts.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        perturbation_types = self.config['perturbation']['types']
        
        # Flatten all contexts with their metadata
        all_contexts = []
        for category, difficulty_dict in self.generated_contexts.items():
            for difficulty, contexts in difficulty_dict.items():
                for context in contexts:
                    all_contexts.append({
                        'category': category,
                        'difficulty': difficulty,
                        'text': context
                    })
        
        # Sample from contexts
        if len(all_contexts) < num_samples:
            # If not enough unique contexts, allow repeats
            selected = random.choices(all_contexts, k=num_samples)
        else:
            selected = random.sample(all_contexts, num_samples)
        
        for i, context_info in enumerate(selected):
            # Select perturbation type
            perturbation_type = random.choice(perturbation_types)
            
            # Generate perturbed text
            perturbed_text = self.perturbation_engine.perturb(
                context_info['text'], perturbation_type
            )
            
            sample = {
                'id': i,
                'category': context_info['category'],
                'difficulty': context_info['difficulty'],
                'original': context_info['text'],
                'perturbed': perturbed_text,
                'perturbation_type': perturbation_type,
                'source': 'generated'
            }
            
            samples.append(sample)
        
        return samples
    
    def get_statistics(self, samples: List[Dict]) -> Dict:
        """
        Get statistics about generated samples.
        
        Args:
            samples: List of generated samples
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_samples': len(samples),
            'by_category': {},
            'by_perturbation': {},
            'by_difficulty': {},
            'by_source': {}
        }
        
        for sample in samples:
            # Count by category
            category = sample['category']
            stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
            
            # Count by perturbation type
            pert_type = sample['perturbation_type']
            stats['by_perturbation'][pert_type] = stats['by_perturbation'].get(pert_type, 0) + 1
            
            # Count by difficulty
            difficulty = sample.get('difficulty', 'unknown')
            stats['by_difficulty'][difficulty] = stats['by_difficulty'].get(difficulty, 0) + 1
            
            # Count by source
            source = sample.get('source', 'unknown')
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
        
        return stats