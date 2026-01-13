"""
Prompt builder for generating LLM prompts
Location: src/prompt_builder.py
"""

from typing import Dict, List


class PromptBuilder:
    """Build prompts for the aligned-table benchmark."""
    
    LATEX_EXAMPLE = """\\begin{table}[]
\\begin{tabular}{|l|l|l|l|l|}
\\hline
Name    & Age & Occupation & City        & Country   \\\\ \\hline
Alice   & 28  & Engineer   & Boston      & USA       \\\\ \\hline
Bob     & 35  & Teacher    & London      & UK        \\\\ \\hline
Charlie & 42  & Doctor     & Sydney      & Australia \\\\ \\hline
Diana   & 31  & Artist     & Paris       & France    \\\\ \\hline
\\end{tabular}
\\end{table}"""
    
    TEXT_EXAMPLE = """+---------+-----+------------+-----------+-----------+
| Name    | Age | Occupation | City      | Country   |
+---------+-----+------------+-----------+-----------+
| Alice   | 28  | Engineer   | Boston    | USA       |
+---------+-----+------------+-----------+-----------+
| Bob     | 35  | Teacher    | London    | UK        |
+---------+-----+------------+-----------+-----------+
| Charlie | 42  | Doctor     | Sydney    | Australia |
+---------+-----+------------+-----------+-----------+
| Diana   | 31  | Artist     | Paris     | France    |
+---------+-----+------------+-----------+-----------+"""
    
    MARKDOWN_EXAMPLE = """| Name    | Age | Occupation | City      | Country   |
|---------|-----|------------|-----------|-----------|
| Alice   | 28  | Engineer   | Boston    | USA       |
| Bob     | 35  | Teacher    | London    | UK        |
| Charlie | 42  | Doctor     | Sydney    | Australia |
| Diana   | 31  | Artist     | Paris     | France    |"""
    
    def __init__(self, restricted_reasoning: bool = False):
        """Initialize the prompt builder."""
        self.restricted_reasoning = restricted_reasoning

    def _format_task_input(self, test_case: Dict) -> str:
        """
        Format the task input from test case data.
        Ensures consistent formatting with example.
        
        Args:
            test_case: Test case dictionary
            
        Returns:
            Formatted task input string
        """
        # Check if context already contains description
        context = test_case.get('context', '')
        table_data = test_case.get('table_data', [])
        
        # If we have table_data, format it properly
        if table_data:
            formatted_input = self.build_input_format(table_data)
            
            # Extract description from context if present
            # Context format: "description\nrow 1: ...\nrow 2: ..."
            if '\n' in context and context.startswith('This table'):
                # Split context to get description
                lines = context.split('\n')
                description = lines[0]
                return f"{description}\n{formatted_input}"
            else:
                # Just return formatted table data
                return formatted_input
        else:
            # Fallback to original context
            return context
    
    def build_prompt(self, test_case: Dict) -> str:
        """
        Build a prompt for a test case.
        
        Args:
            test_case: Test case dictionary containing context and format info
            
        Returns:
            Formatted prompt string
        """
        table_format = test_case['table_format']
        
        # Get format-specific instructions and example
        format_instructions = self._get_format_instructions(table_format)
        format_example = self._get_format_example(table_format)
        
        # Format the task input
        task_input = self._format_task_input(test_case)
        
        prompt = f"""You are tasked with creating a well-formatted table based on the given information.

**Task Description:**
Given a description of table content, you need to output a properly formatted {table_format.upper()} table.

**Format Requirements:**
{format_instructions}

**Example Input Content:**
row 1: 'Name', 'Age', 'Occupation', 'City', 'Country'
row 2: 'Alice', '28', 'Engineer', 'Boston', 'USA'
row 3: 'Bob', '35', 'Teacher', 'London', 'UK'
row 4: 'Charlie', '42', 'Doctor', 'Sydney', 'Australia'
row 5: 'Diana', '31', 'Artist', 'Paris', 'France'

**Example Golden Answer ({table_format.upper()}):**
<answer>
{format_example}
</answer>

**Important Alignment Rules:**
1. All delimiter symbols ({self._get_delimiter_description(table_format)}) MUST be vertically aligned across all rows.
2. Within each column, there should be exactly ONE space between the content and the delimiter on both sides.
3. Content in each cell should be left-aligned.
4. Each column width is determined by the longest content in that column plus one space on each side.

**Input Format:**
- Each row is specified as: row N: 'element1', 'element2', 'element3', ...
- Elements are enclosed in single quotes and separated by commas

**Your Task:**
{task_input}

Please create a properly formatted {table_format.upper()} table based on the information above.

**Output Instructions:**
- Place your final answer between <answer> and </answer> tags.
- Ensure all delimiters are perfectly aligned.
- Follow the spacing rules strictly.

Your response:"""

        if self.restricted_reasoning:
            prompt += (
                "\n\nAnswer directly after <answer> tags without thinking or reasoning. Begin your answer now: <answer>"
            )
        
        return prompt
    
    def _get_format_instructions(self, table_format: str) -> str:
        """Get format-specific instructions."""
        instructions = {
            'latex': """For LaTeX tables:
- Use \\begin{table}[] and \\begin{tabular}{|l|l|...} structure
- Use & as column separator
- Use \\\\ \\hline at the end of each row
- All & symbols must be vertically aligned
- Use \\hline for horizontal lines""",
            
            'text': """For text tables:
- Use + for corners and intersections
- Use - for horizontal lines
- Use | for vertical separators
- All | and + symbols must be vertically aligned
- Each cell has one space padding on both sides""",
            
            'markdown': """For Markdown tables:
- Use | as column separator
- Add a separator row after the header (first row) using dashes
- All | symbols must be vertically aligned
- Each cell has one space padding on both sides"""
        }
        
        return instructions.get(table_format.lower(), "")
    
    def _get_format_example(self, table_format: str) -> str:
        """Get format-specific example."""
        examples = {
            'latex': self.LATEX_EXAMPLE,
            'text': self.TEXT_EXAMPLE,
            'markdown': self.MARKDOWN_EXAMPLE
        }
        
        return examples.get(table_format.lower(), "")
    
    def _get_delimiter_description(self, table_format: str) -> str:
        """Get description of delimiters for each format."""
        delimiters = {
            'latex': '& symbols',
            'text': '| and + symbols',
            'markdown': '| symbols'
        }
        
        return delimiters.get(table_format.lower(), "delimiters")
    
    def build_input_format(self, table_data: List[List[str]]) -> str:
        """
        Build input format string from table data.
        
        Args:
            table_data: 2D list of table cell values
            
        Returns:
            Formatted input string
        """
        lines = []
        for idx, row in enumerate(table_data, 1):
            # Wrap each element in single quotes
            quoted_elements = [f"'{element}'" for element in row]
            row_str = f"row {idx}: " + ", ".join(quoted_elements)
            lines.append(row_str)
        
        return "\n".join(lines)
