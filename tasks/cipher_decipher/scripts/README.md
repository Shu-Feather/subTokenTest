# Advanced Benchmark Analysis Script

## Overview

This script provides comprehensive statistical analysis and visualization for cipher benchmark results. It evaluates model performance on cipher encoding/decoding tasks with advanced metrics including similarity scores, error patterns, length correlations, and multi-model comparisons.

## Features

### 📊 Core Analytics
- **Performance Metrics**: Overall accuracy, task-wise breakdown, difficulty analysis
- **Similarity Analysis**: Character-level similarity scoring with statistical distributions
- **Length Correlation**: Spearman correlation analysis between input length and performance
- **Error Analysis**: Pattern detection, length differences, and sample error cases

### 📈 Visualizations
- **Overview Dashboard**: 4-panel performance summary with pie charts and bar plots
- **Correlation Plots**: Scatter plots showing length vs accuracy/similarity
- **Multi-Model Comparison**: Side-by-side comparison of multiple models

### 📝 Reporting
- **Text Reports**: Comprehensive analysis reports with statistics and insights
- **JSON Exports**: Machine-readable summaries for further processing
- **Comparison Reports**: Detailed multi-model rankings and breakdowns

## Installation

### Requirements

```bash
pip install matplotlib seaborn pandas numpy scipy
```

### Dependencies
- Python 3.7+
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- pandas >= 1.2.0
- numpy >= 1.19.0
- scipy >= 1.6.0

## Usage

### Basic Analysis (Single Model)

```bash
python advanced_analysis.py model_responses.json
```

**Output:**
- `model_overview.png` - Performance dashboard
- `model_correlation.png` - Length correlation analysis
- `model_report.txt` - Comprehensive text report
- `model_summary.json` - JSON data export

### Multi-Model Comparison

```bash
python advanced_analysis.py model1.json model2.json model3.json --compare
```

**Additional Output:**
- `multi_model_comparison.png` - Comparative visualization
- `comparison_report.txt` - Ranking and detailed comparison

### Command-Line Options

```
positional arguments:
  log_files             Path(s) to response log JSON file(s)

optional arguments:
  -h, --help           Show help message and exit
  -o, --output DIR     Output directory (default: analysis_results)
  -c, --compare        Generate multi-model comparison
  --no-plots           Skip generating visualizations
  --no-reports         Skip generating text reports
```

### Examples

```bash
# Analyze single model with custom output directory
python advanced_analysis.py gpt4_responses.json -o gpt4_results

# Compare multiple models
python advanced_analysis.py gpt4.json claude.json llama.json --compare

# Generate reports only (no plots)
python advanced_analysis.py model.json --no-plots

# Generate plots only (no reports)
python advanced_analysis.py model.json --no-reports
```

## Input Format

The script expects JSON log files with the following structure:

```json
[
  {
    "task_type": "morse_encode",
    "difficulty": "easy",
    "golden_answer": ".... . .-.. .-.. ---",
    "model_response": "The morse code is: .... . .-.. .-.. ---",
    "extracted_answer": ".... . .-.. .-.. ---",
    "is_correct": true
  },
  {
    "task_type": "caesar_decode",
    "difficulty": "medium",
    "golden_answer": "HELLO",
    "model_response": "The decoded text is HELLO",
    "extracted_answer": "HELLO",
    "is_correct": true
  }
]
```

**Required Fields:**
- `task_type`: Task identifier (e.g., morse_encode, caesar_decode)
- `golden_answer`: Expected correct answer
- `model_response`: Raw model output (can be empty)
- `extracted_answer`: Parsed answer from model response
- `is_correct`: Boolean indicating correctness
- `difficulty`: Task difficulty level (easy/medium/hard)

## Output Files

### Visualizations

#### 1. Overview Dashboard (`{model}_overview.png`)
- **Panel 1**: Pie chart of correct/incorrect distribution
- **Panel 2**: Task-wise accuracy bar chart
- **Panel 3**: Difficulty-wise accuracy comparison
- **Panel 4**: Similarity scores by task

#### 2. Correlation Analysis (`{model}_correlation.png`)
- **Left**: Accuracy vs input length scatter plot with trend line
- **Right**: Similarity vs input length colored by correctness

#### 3. Multi-Model Comparison (`multi_model_comparison.png`)
- **Panel 1**: Overall accuracy comparison
- **Panel 2**: Task-wise accuracy breakdown
- **Panel 3**: Difficulty-wise performance

### Reports

#### Text Report (`{model}_report.txt`)

```
================================================================================
BENCHMARK ANALYSIS REPORT - gpt4_turbo
================================================================================

1. OVERALL PERFORMANCE
--------------------------------------------------------------------------------
Total Tasks:      200
Correct:          178
Overall Accuracy: 89.00%

Performance by Task:
  morse_encode              45/50  (90.00%)
  morse_decode              42/50  (84.00%)
  caesar_encode             46/50  (92.00%)
  caesar_decode             45/50  (90.00%)

2. SIMILARITY ANALYSIS
--------------------------------------------------------------------------------
Mean:    0.9234
Median:  0.9567
Std Dev: 0.1245
Range:   0.3421 - 1.0000

3. LENGTH CORRELATION
--------------------------------------------------------------------------------
Accuracy vs Length:
  Spearman ρ: -0.2341
  p-value:    0.0012
  Significant: Yes

4. ERROR ANALYSIS
--------------------------------------------------------------------------------
Total Errors: 22

Sample Errors:
  #1 (morse_decode - hard):
    Expected: THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG...
    Actual:   THE QUICK BROWN FOX JUMP OVER THE LAZY DOG...
    Similarity: 0.9567
```

#### JSON Summary (`{model}_summary.json`)

```json
{
  "stats": {
    "total_tasks": 200,
    "correct_tasks": 178,
    "overall_accuracy": 89.0,
    "per_task": {
      "morse_encode": {
        "total": 50,
        "correct": 45,
        "accuracy": 90.0
      }
    }
  },
  "similarity": {
    "overall": {
      "mean": 0.9234,
      "median": 0.9567,
      "std": 0.1245,
      "min": 0.3421,
      "max": 1.0
    }
  },
  "correlation": {
    "accuracy_vs_length": {
      "coefficient": -0.2341,
      "p_value": 0.0012,
      "significant": true
    }
  }
}
```

#### Comparison Report (`comparison_report.txt`)

```
================================================================================
MULTI-MODEL COMPARISON REPORT
================================================================================

OVERALL RANKINGS
--------------------------------------------------------------------------------
1. gpt4_turbo                  89.00%
2. claude_3_opus               85.50%
3. llama_3_70b                 78.00%

DETAILED COMPARISON
--------------------------------------------------------------------------------
Metric                   gpt4_turbo claude_3_o llama_3_70b
--------------------------------------------------------------------------------
Overall Accuracy (%)          89.00      85.50      78.00

Task-wise Accuracy (%):
  morse_encode                90.00      88.00      75.00
  morse_decode                84.00      82.00      74.00
  caesar_encode               92.00      87.00      80.00
  caesar_decode               90.00      85.00      83.00
```

## Key Metrics Explained

### Similarity Score
- **Range**: 0.0 to 1.0
- **Calculation**: Character-level similarity using normalized Levenshtein distance
- **Interpretation**: 
  - 1.0 = Perfect match
  - > 0.9 = Very close
  - 0.5-0.9 = Partial match
  - < 0.5 = Poor match

### Spearman Correlation (ρ)
- **Range**: -1.0 to 1.0
- **Interpretation**:
  - ρ > 0: Positive correlation (longer inputs → better/worse performance)
  - ρ < 0: Negative correlation (longer inputs → worse/better performance)
  - |ρ| < 0.1: Negligible
  - |ρ| 0.1-0.3: Weak
  - |ρ| 0.3-0.5: Moderate
  - |ρ| > 0.5: Strong
- **p-value < 0.05**: Statistically significant

## Console Output

### Single Model Analysis

```
============================================================
Summary: gpt4_turbo
============================================================
Total:    200
Correct:  178
Accuracy: 89.00%
Similarity: 0.9234

✓ Completed: gpt4_turbo
```

### Multi-Model Comparison

```
============================================================
MODEL COMPARISON
============================================================
1. gpt4_turbo                  89.00%
2. claude_3_opus               85.50%
3. llama_3_70b                 78.00%

============================================================
KEY INSIGHTS
============================================================

Top Model: gpt4_turbo (89.00%)
Performance Gap: 11.00%

Task Champions:
  morse_encode         gpt4_turbo (90.0%)
  morse_decode         gpt4_turbo (84.0%)
  caesar_encode        gpt4_turbo (92.0%)
  caesar_decode        gpt4_turbo (90.0%)
```

## Interpretation Guide

### Performance Insights

1. **Overall Accuracy < 70%**: Model struggles with cipher tasks
2. **Large Difficulty Gap**: Check if model handles complexity appropriately
3. **Negative Length Correlation**: Performance degrades with longer inputs
4. **Low Similarity on Incorrect**: Model produces random/unrelated outputs

### Warning Signs

- ⚠️ **Significant negative length correlation**: Model may have context limitations
- ⚠️ **High variance in task performance**: Inconsistent understanding
- ⚠️ **Low similarity even on "correct" answers**: Answer extraction issues

## Troubleshooting

### Common Issues

**1. Empty/Missing Responses**
```
WARNING: Filtered X invalid/empty responses from log file
```
**Solution**: Ensure `model_response` field is populated in JSON

**2. No Correlation Data**
```
No length correlation data available
```
**Solution**: Ensure JSON contains at least 2 valid entries

**3. Import Errors**
```
ModuleNotFoundError: No module named 'src.evaluation.evaluator'
```
**Solution**: Run script from project root directory or adjust sys.path

### File Structure

```
project/
├── src/
│   └── evaluation/
│       └── evaluator.py
├── scripts/
│   └── advanced_analysis.py
├── logs/
│   ├── model1_responses.json
│   └── model2_responses.json
└── analysis_results/
    ├── model1_overview.png
    ├── model1_report.txt
    └── ...
```

## Advanced Usage

### Batch Processing

```bash
# Process all JSON files in logs directory
for file in logs/*.json; do
    python advanced_analysis.py "$file" -o "results/$(basename $file .json)"
done
```

### Comparison with Filtering

```python
# Custom filtering in script
analyzer = BenchmarkAnalyzer(log_file)
# Filter specific difficulties
hard_only = [d for d in analyzer.data if d.get('difficulty') == 'hard']
```





