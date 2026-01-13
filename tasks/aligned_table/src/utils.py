"""
Utility functions for the Aligned-Table Benchmark
Location: src/utils.py
"""

import re
from typing import List, Tuple, Optional


def parse_answer(response: str) -> Optional[str]:
    """
    Extract the answer content between <answer> and </answer> tags.
    
    Args:
        response: The LLM's response string
        
    Returns:
        The extracted answer content, or None if tags not found
    """
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None


def parse_table_data(input_text: str) -> List[List[str]]:
    """
    Parse the input text format into a 2D list of table data.
    
    Args:
        input_text: Input text in format "row 1: A, B, C\nrow 2: D, E, F"
        
    Returns:
        2D list of table cell values
    """
    rows = []
    lines = input_text.strip().split('\n')
    
    for line in lines:
        if ':' in line:
            # Extract content after colon
            content = line.split(':', 1)[1].strip()
            # Split by comma and strip whitespace
            cells = [cell.strip() for cell in content.split(',')]
            rows.append(cells)
    
    return rows


def normalize_table_data(data: List[List[str]]) -> List[List[str]]:
    """
    Normalize table data to ensure all rows have the same number of columns.
    Pads short rows with empty strings and truncates long rows.
    
    Args:
        data: 2D list of table cell values
        
    Returns:
        Normalized 2D list with consistent column count
    """
    if not data:
        return []
    
    # Find the maximum number of columns
    max_cols = max(len(row) for row in data)
    
    # Normalize all rows to have max_cols columns
    normalized = []
    for row in data:
        if len(row) < max_cols:
            # Pad with empty strings
            normalized_row = row + [''] * (max_cols - len(row))
        elif len(row) > max_cols:
            # Truncate (shouldn't happen if max_cols is correct)
            normalized_row = row[:max_cols]
        else:
            normalized_row = row
        normalized.append(normalized_row)
    
    return normalized


def get_column_widths(data: List[List[str]]) -> List[int]:
    """
    Calculate the maximum width needed for each column.
    
    Args:
        data: 2D list of table cell values
        
    Returns:
        List of maximum widths for each column
    """
    if not data:
        return []
    
    # Normalize data first to ensure consistent columns
    normalized_data = normalize_table_data(data)
    
    num_cols = len(normalized_data[0]) if normalized_data else 0
    widths = [0] * num_cols
    
    for row in normalized_data:
        for i in range(min(len(row), num_cols)):
            widths[i] = max(widths[i], len(str(row[i])))
    
    return widths


def format_latex_table(data: List[List[str]]) -> str:
    """
    Format data as an aligned LaTeX table.
    
    Args:
        data: 2D list of table cell values
        
    Returns:
        Formatted LaTeX table string
    """
    if not data:
        return ""
    
    # Normalize data first
    normalized_data = normalize_table_data(data)
    widths = get_column_widths(normalized_data)
    num_cols = len(normalized_data[0]) if normalized_data else 0
    
    lines = []
    lines.append("\\begin{table}[]")
    lines.append("\\begin{tabular}{|" + "l|" * num_cols + "}")
    lines.append("\\hline")
    
    for row in normalized_data:
        formatted_cells = []
        for i in range(num_cols):
            cell = row[i] if i < len(row) else ''
            # Left-align with proper spacing
            formatted_cell = str(cell).ljust(widths[i])
            formatted_cells.append(formatted_cell)
        
        line = " & ".join(formatted_cells) + " \\\\ \\hline"
        lines.append(line)
    
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def format_text_table(data: List[List[str]]) -> str:
    """
    Format data as an aligned text table.
    
    Args:
        data: 2D list of table cell values
        
    Returns:
        Formatted text table string
    """
    if not data:
        return ""
    
    # Normalize data first
    normalized_data = normalize_table_data(data)
    widths = get_column_widths(normalized_data)
    
    def create_separator():
        parts = ['+']
        for width in widths:
            parts.append('-' * (width + 2))
            parts.append('+')
        return ''.join(parts)
    
    lines = []
    separator = create_separator()
    lines.append(separator)
    
    for row in normalized_data:
        parts = ['|']
        for i, width in enumerate(widths):
            cell = row[i] if i < len(row) else ''
            # Add space, content (left-aligned), space
            formatted_cell = ' ' + str(cell).ljust(width) + ' '
            parts.append(formatted_cell)
            parts.append('|')
        lines.append(''.join(parts))
        lines.append(separator)
    
    return "\n".join(lines)


def format_markdown_table(data: List[List[str]]) -> str:
    """
    Format data as an aligned Markdown table.
    
    Args:
        data: 2D list of table cell values
        
    Returns:
        Formatted Markdown table string
    """
    if not data:
        return ""
    
    # Normalize data first
    normalized_data = normalize_table_data(data)
    widths = get_column_widths(normalized_data)
    
    lines = []
    
    for idx, row in enumerate(normalized_data):
        parts = ['|']
        for i, width in enumerate(widths):
            cell = row[i] if i < len(row) else ''
            # Add space, content (left-aligned), space
            formatted_cell = ' ' + str(cell).ljust(width) + ' '
            parts.append(formatted_cell)
            parts.append('|')
        lines.append(''.join(parts))
        
        # Add separator after first row
        if idx == 0:
            sep_parts = ['|']
            for width in widths:
                sep_parts.append('-' * (width + 2))
                sep_parts.append('|')
            lines.append(''.join(sep_parts))
    
    return "\n".join(lines)


def format_table(data: List[List[str]], table_format: str) -> str:
    """
    Format table data according to specified format.
    
    Args:
        data: 2D list of table cell values
        table_format: One of 'latex', 'text', or 'markdown'
        
    Returns:
        Formatted table string
    """
    format_map = {
        'latex': format_latex_table,
        'text': format_text_table,
        'markdown': format_markdown_table
    }
    
    formatter = format_map.get(table_format.lower())
    if formatter:
        return formatter(data)
    else:
        raise ValueError(f"Unsupported table format: {table_format}")


def extract_table_content(table_str: str, table_format: str) -> List[List[str]]:
    """
    Extract cell content from a formatted table string.
    
    Args:
        table_str: Formatted table string
        table_format: One of 'latex', 'text', or 'markdown'
        
    Returns:
        2D list of extracted cell values
    """
    if table_format.lower() == 'latex':
        return extract_latex_content(table_str)
    elif table_format.lower() == 'text':
        return extract_text_content(table_str)
    elif table_format.lower() == 'markdown':
        return extract_markdown_content(table_str)
    else:
        raise ValueError(f"Unsupported table format: {table_format}")


def extract_latex_content(table_str: str) -> List[List[str]]:
    """Extract content from LaTeX table."""
    rows = []
    lines = table_str.split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for lines with & (table rows)
        if '&' in line and '\\\\' in line:
            # Remove trailing \\ \hline
            line = re.sub(r'\s*\\\\\s*\\hline\s*$', '', line)
            line = re.sub(r'\s*\\\\\s*$', '', line)
            
            # Split by & and strip whitespace
            cells = [cell.strip() for cell in line.split('&')]
            rows.append(cells)
    
    return rows


def extract_text_content(table_str: str) -> List[List[str]]:
    """Extract content from text table."""
    rows = []
    lines = table_str.split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for lines with | (table rows, not separators)
        if line.startswith('|') and not line.startswith('+-'):
            # Remove leading and trailing |
            line = line.strip('|')
            # Split by | and strip whitespace
            cells = [cell.strip() for cell in line.split('|')]
            rows.append(cells)
    
    return rows


def extract_markdown_content(table_str: str) -> List[List[str]]:
    """Extract content from Markdown table."""
    rows = []
    lines = table_str.split('\n')
    
    for line in lines:
        line = line.strip()
        # Skip separator lines (contain only |, -, and spaces)
        if re.match(r'^[\|\-\s]+$', line):
            continue
            
        # Look for lines with |
        if line.startswith('|'):
            # Remove leading and trailing |
            line = line.strip('|')
            # Split by | and strip whitespace
            cells = [cell.strip() for cell in line.split('|')]
            rows.append(cells)
    
    return rows


def check_alignment(table_str: str, table_format: str) -> Tuple[bool, float]:
    """
    Check if the table delimiters are properly aligned.
    
    Args:
        table_str: Formatted table string
        table_format: One of 'latex', 'text', or 'markdown'
        
    Returns:
        Tuple of (is_aligned, alignment_score)
    """
    if table_format.lower() == 'latex':
        return check_latex_alignment(table_str)
    elif table_format.lower() == 'text':
        return check_text_alignment(table_str)
    elif table_format.lower() == 'markdown':
        return check_markdown_alignment(table_str)
    else:
        return False, 0.0


def check_latex_alignment(table_str: str) -> Tuple[bool, float]:
    """Check alignment for LaTeX tables (& symbols)."""
    lines = table_str.split('\n')
    data_lines = []
    
    for line in lines:
        if '&' in line and '\\\\' in line:
            data_lines.append(line)
    
    if len(data_lines) < 2:
        return True, 1.0
    
    # Find positions of & symbols
    positions_list = []
    for line in data_lines:
        positions = [i for i, char in enumerate(line) if char == '&']
        positions_list.append(positions)
    
    # Check if all rows have same number of &
    num_separators = len(positions_list[0])
    if not all(len(pos) == num_separators for pos in positions_list):
        return False, 0.0
    
    # Check if & positions match across rows
    total_checks = 0
    aligned_checks = 0
    
    for col_idx in range(num_separators):
        expected_pos = positions_list[0][col_idx]
        for row_positions in positions_list[1:]:
            total_checks += 1
            if row_positions[col_idx] == expected_pos:
                aligned_checks += 1
    
    if total_checks == 0:
        return True, 1.0
    
    alignment_score = aligned_checks / total_checks
    is_aligned = alignment_score == 1.0
    
    return is_aligned, alignment_score


def check_text_alignment(table_str: str) -> Tuple[bool, float]:
    """Check alignment for text tables (| and + symbols)."""
    lines = table_str.split('\n')
    
    if not lines:
        return True, 1.0
    
    # Find positions of | or + in each line
    positions_list = []
    for line in lines:
        positions = [i for i, char in enumerate(line) if char in '|+']
        if positions:
            positions_list.append(positions)
    
    if len(positions_list) < 2:
        return True, 1.0
    
    # All lines should have same | or + positions
    reference_positions = positions_list[0]
    total_checks = len(positions_list) - 1
    aligned_checks = 0
    
    for positions in positions_list[1:]:
        if positions == reference_positions:
            aligned_checks += 1
    
    alignment_score = aligned_checks / total_checks if total_checks > 0 else 1.0
    is_aligned = alignment_score == 1.0
    
    return is_aligned, alignment_score


def check_markdown_alignment(table_str: str) -> Tuple[bool, float]:
    """Check alignment for Markdown tables (| symbols)."""
    lines = table_str.split('\n')
    
    if not lines:
        return True, 1.0
    
    # Find positions of | in each line
    positions_list = []
    for line in lines:
        positions = [i for i, char in enumerate(line) if char == '|']
        if positions:
            positions_list.append(positions)
    
    if len(positions_list) < 2:
        return True, 1.0
    
    # All lines should have same | positions
    reference_positions = positions_list[0]
    total_checks = len(positions_list) - 1
    aligned_checks = 0
    
    for positions in positions_list[1:]:
        if positions == reference_positions:
            aligned_checks += 1
    
    alignment_score = aligned_checks / total_checks if total_checks > 0 else 1.0
    is_aligned = alignment_score == 1.0
    
    return is_aligned, alignment_score