def get_sokoban_prompt(map_str: str, question: str, restricted_reasoning: bool = False) -> str:
    """
    Generate prompt for Sokoban tasks.
    
    Args:
        map_str: String representation of the map
        question: The specific question to ask
        
    Returns:
        Complete prompt string
    """
    system_prompt = """You are an expert at spatial reasoning and map navigation. You will be given a Sokoban environment and asked questions about it.

**Environment Description:**
- The environment is an N x N grid
- Coordinate system: Top-left corner is (0, 0), x-axis increases to the right (horizontal), y-axis increases downward (vertical)
- Each cell has a size of 1

**Element Definitions:**
- `#`: Wall - An impassable obstacle
- `_`: Empty space - A walkable area
- `P`: Player - The controllable character (only 1 in the map)
- `X`: Box - An object that can be pushed (only 1 in the map)
- `O`: Goal - The target location for the box (only 1 in the map)

**Important Notes:**
1. Coordinates are given as (x, y) where x is the column and y is the row
2. The 8 surrounding directions are: up, down, left, right, up-left, up-right, down-left, down-right
3. Relative position (dx, dy) means: to go from position A to position B, move dx units in x-direction and dy units in y-direction

**Instructions:**
- Analyze the map carefully
- Answer the question accurately
- Put your final answer between <answer> and </answer> tags
- For coordinate answers, use format: (x, y)
- For surrounding elements, provide a JSON object with directions as keys
- Be precise and concise"""

    user_prompt = f"""Here is the Sokoban environment map:

```
{map_str}
```

**Question:** {question}

Please analyze the map and provide your answer. Remember to put your final answer between <answer> and </answer> tags."""

    if restricted_reasoning:
        user_prompt = (
            f"{user_prompt}\n\nAnswer directly after <answer> tags without thinking or reasoning. Begin your answer now: <answer>"
        )

    return system_prompt + "\n\n" + user_prompt


def get_sokoban_system_prompt(restricted_reasoning: bool = False) -> str:
    """Get system prompt for Sokoban tasks."""
    prompt = """You are an expert at spatial reasoning and map navigation. You will be given a Sokoban environment and asked questions about it.

**Environment Description:**
- The environment is an N x N grid
- Coordinate system: Top-left corner is (0, 0), x-axis increases to the right (horizontal), y-axis increases downward (vertical)
- Each cell has a size of 1

**Element Definitions:**
- `#`: Wall - An impassable obstacle
- `_`: Empty space - A walkable area
- `P`: Player - The controllable character (only 1 in the map)
- `X`: Box - An object that can be pushed (only 1 in the map)
- `O`: Goal - The target location for the box (only 1 in the map)

**Important Notes:**
1. Coordinates are given as (x, y) where x is the column and y is the row
2. The 8 surrounding directions are: up, down, left, right, up-left, up-right, down-left, down-right
3. Relative position (dx, dy) means: to go from position A to position B, move dx units in x-direction and dy units in y-direction

**Instructions:**
- Analyze the map carefully
- Answer the question accurately
- Put your final answer between <answer> and </answer> tags
- For coordinate answers, use format: (x, y)
- For surrounding elements, provide a JSON object with directions as keys
- Be precise and concise"""
    if restricted_reasoning:
        prompt += "\n\nKeep reasoning minimal and answer directly inside <answer> tags."
    return prompt


def get_sokoban_user_prompt(map_str: str, question: str, restricted_reasoning: bool = False) -> str:
    """Get user prompt for Sokoban tasks."""
    prompt = f"""Here is the Sokoban environment map:

```
{map_str}
```

**Question:** {question}

Please analyze the map and provide your answer. Remember to put your final answer between <answer> and </answer> tags."""
    if restricted_reasoning:
        prompt += "\n\nAnswer directly after <answer> tags without thinking or reasoning. Begin your answer now: <answer>"
    return prompt
