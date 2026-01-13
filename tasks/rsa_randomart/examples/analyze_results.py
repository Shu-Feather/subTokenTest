"""
Script to analyze benchmark results
Path: examples/analyze_results.py
"""

import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath: str) -> dict:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_error_patterns(results: dict):
    """Analyze common error patterns"""
    print("\n" + "="*60)
    print("Error Pattern Analysis")
    print("="*60)
    
    individual_results = results.get('individual_results', [])
    
    # Categorize samples by performance
    perfect = []
    goo