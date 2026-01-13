"""
Script to generate original contexts using GPT
Location: scripts/generate_contexts.py
"""

import os
import json
import argparse
from typing import List, Dict
from openai import OpenAI


class ContextGenerator:
    """Generate table contexts using GPT."""
    
    def __init__(self, model: str = "gpt-4", api_key: str = None):
        """
        Initialize context generator.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key
        """
        self.model = model
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
    
    def generate_context(self, num_rows: int, num_cols: int, 
                        entity_type: str = None) -> Dict:
        """
        Generate a single table context.
        
        Args:
            num_rows: Number of rows in the table
            num_cols: Number of columns in the table
            entity_type: Type of entity (optional)
            
        Returns:
            Dictionary with table data and context
        """
        prompt = self._build_generation_prompt(num_rows, num_cols, entity_type)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates realistic table data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content
        
        # Parse the response
        parsed_data = self._parse_gpt_response(content, num_rows, num_cols)
        
        return parsed_data
    
    def _build_generation_prompt(self, num_rows: int, num_cols: int, 
                                 entity_type: str = None) -> str:
        """Build prompt for context generation."""
        entity_hint = ""
        if entity_type:
            entity_hint = f"The entities should be {entity_type}."
        
        prompt = f"""Generate realistic table data with the following specifications:
- Number of rows: {num_rows}
- Number of columns: {num_cols}
- The first column should contain entity names (e.g., countries, companies, products, etc.)
- The remaining columns should contain relevant attributes for those entities
{entity_hint}

Please provide:
1. A brief description of what the table represents
2. The table data in the following format:
   row 1: value1 | value2 | value3, ...
   row 2: value1 | value2 | value3, ...
   etc.

Make sure the data is realistic and meaningful. **You should vary the length of cell values to make it diverse**.

Format your response as:
DESCRIPTION: [Your description here]
DATA:
row 1: [values]
row 2: [values]
...

IMPORTANT: Ensure there are exactly {num_rows} rows and {num_cols} columns in the data.
"""
        return prompt
    
    def _parse_gpt_response(self, response: str, num_rows: int, 
                           num_cols: int) -> Dict:
        """Parse GPT response into structured data."""
        lines = response.strip().split('\n')
        
        description = ""
        table_data = []
        
        in_data_section = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            if line.startswith('DESCRIPTION:'):
                description = line.replace('DESCRIPTION:', '').strip()
            elif line.startswith('DATA:'):
                in_data_section = True
            elif in_data_section and line.startswith('row'):
                # Extract row data
                if ':' in line:
                    # Get content after "row X:"
                    content = line.split(':', 1)[1].strip()
                    # Split by pipe separator as specified in the prompt
                    cells = [cell.strip() for cell in content.split('|')]
                    table_data.append(cells)
        
        # Build context description
        context_parts = [description]
        for idx, row in enumerate(table_data, 1):
            row_desc = f"row {idx}: " + ", ".join(row)
            context_parts.append(row_desc)
        
        context = " ".join(context_parts)
        
        return {
            'description': description,
            'table_data': table_data,
            'context': context,
            'num_rows': len(table_data),
            'num_cols': len(table_data[0]) if table_data else 0
        }
    
    def generate_batch(self, specifications: List[Dict]) -> List[Dict]:
        """
        Generate multiple contexts.
        
        Args:
            specifications: List of dicts with 'num_rows', 'num_cols', 'entity_type'
            
        Returns:
            List of generated context dictionaries
        """
        contexts = []
        
        for spec in specifications:
            try:
                context = self.generate_context(
                    num_rows=spec['num_rows'],
                    num_cols=spec['num_cols'],
                    entity_type=spec.get('entity_type')
                )
                contexts.append(context)
                print(f"Generated context {len(contexts)}/{len(specifications)}")
            except Exception as e:
                print(f"Error generating context: {e}")
                continue
        
        return contexts


def main():
    """Main function for context generation script."""
    parser = argparse.ArgumentParser(description='Generate table contexts using GPT')
    parser.add_argument('--model', type=str, default='gpt-4',
                       help='OpenAI model to use')
    parser.add_argument('--num_contexts', type=int, default=100,
                       help='Number of contexts to generate')
    parser.add_argument('--output', type=str, default='generated_contexts.json',
                       help='Output JSON file')
    parser.add_argument('--min_rows', type=int, default=3,
                       help='Minimum number of rows')
    parser.add_argument('--max_rows', type=int, default=8,
                       help='Maximum number of rows')
    parser.add_argument('--min_cols', type=int, default=3,
                       help='Minimum number of columns')
    parser.add_argument('--max_cols', type=int, default=6,
                       help='Maximum number of columns')
    
    args = parser.parse_args()
    
    # Create generator
    generator = ContextGenerator(model=args.model)
    
    # Generate specifications
    import random
    entity_types = ['countries', 'companies', 'movies', 'books', 
                   'universities', 'products', 'athletes', 'cities']
    
    specifications = []
    for i in range(args.num_contexts):
        spec = {
            'num_rows': random.randint(args.min_rows, args.max_rows),
            'num_cols': random.randint(args.min_cols, args.max_cols),
            'entity_type': random.choice(entity_types)
        }
        specifications.append(spec)
    
    print(f"Generating {args.num_contexts} contexts...")
    
    # Generate contexts
    contexts = generator.generate_batch(specifications)
    
    # Save to file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(contexts, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(contexts)} contexts to {args.output}")


if __name__ == '__main__':
    main()