"""
Validate and clean generated contexts
Location: scripts/validate_contexts.py
"""

import json
import argparse
from typing import List, Dict


def validate_and_clean_contexts(input_file: str, output_file: str = None,
                                fix_issues: bool = True) -> List[Dict]:
    """
    Validate and optionally clean context data.
    
    Args:
        input_file: Input JSON file
        output_file: Output JSON file (optional)
        fix_issues: Whether to fix issues automatically
        
    Returns:
        List of validated/cleaned contexts
    """
    # Load contexts
    with open(input_file, 'r', encoding='utf-8') as f:
        contexts = json.load(f)
    
    print(f"Loaded {len(contexts)} contexts from {input_file}")
    print("\nValidating...")
    
    cleaned_contexts = []
    issues_found = 0
    issues_fixed = 0
    
    for idx, context in enumerate(contexts):
        has_issues = False
        
        # Check required fields
        if 'table_data' not in context:
            print(f"Context {idx}: Missing 'table_data' field - SKIPPING")
            issues_found += 1
            continue
        
        table_data = context['table_data']
        
        if not table_data:
            print(f"Context {idx}: Empty table_data - SKIPPING")
            issues_found += 1
            continue
        
        # Check column consistency
        col_counts = [len(row) for row in table_data]
        if len(set(col_counts)) > 1:
            has_issues = True
            issues_found += 1
            print(f"Context {idx}: Inconsistent column counts: {col_counts}")
            
            if fix_issues:
                # Fix by normalizing to the most common column count
                from collections import Counter
                most_common_cols = Counter(col_counts).most_common(1)[0][0]
                
                fixed_table = []
                for row in table_data:
                    if len(row) < most_common_cols:
                        # Pad with empty strings
                        fixed_row = row + [''] * (most_common_cols - len(row))
                    elif len(row) > most_common_cols:
                        # Truncate
                        fixed_row = row[:most_common_cols]
                    else:
                        fixed_row = row
                    fixed_table.append(fixed_row)
                
                context['table_data'] = fixed_table
                context['num_cols'] = most_common_cols
                issues_fixed += 1
                print(f"  → Fixed: normalized to {most_common_cols} columns")
        
        # Update metadata
        context['num_rows'] = len(context['table_data'])
        if 'num_cols' not in context or not has_issues:
            context['num_cols'] = len(context['table_data'][0])
        
        cleaned_contexts.append(context)
    
    print(f"\nValidation Summary:")
    print(f"  Total contexts: {len(contexts)}")
    print(f"  Valid contexts: {len(cleaned_contexts)}")
    print(f"  Issues found: {issues_found}")
    print(f"  Issues fixed: {issues_fixed}")
    print(f"  Contexts skipped: {len(contexts) - len(cleaned_contexts)}")
    
    # Save cleaned data
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_contexts, f, indent=2, ensure_ascii=False)
        print(f"\nCleaned contexts saved to: {output_file}")
    
    return cleaned_contexts


def main():
    parser = argparse.ArgumentParser(
        description='Validate and clean generated contexts'
    )
    parser.add_argument('input', type=str,
                       help='Input JSON file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (default: input_cleaned.json)')
    parser.add_argument('--no-fix', action='store_true',
                       help='Only validate, do not fix issues')
    
    args = parser.parse_args()
    
    # Set default output filename
    if args.output is None:
        args.output = args.input.replace('.json', '_cleaned.json')
    
    validate_and_clean_contexts(
        args.input,
        args.output,
        fix_issues=not args.no_fix
    )


if __name__ == '__main__':
    main()