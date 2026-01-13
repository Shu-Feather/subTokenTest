def get_frozenlake_prompt(map_str: str, question: str, restricted_reasoning: bool = False) -> str:
    """
    Generate prompt for FrozenLake tasks.
    
    Args:
        map_str: String representation of the map
        question: The specific question to ask
        
    Returns:
        Complete prompt string
    """
    system_prompt = """You are an expert at spatial reasoning and map navigation. You will be given a FrozenLake environment and asked questions about it.

**Environment Description:**
- The environment is an N x N grid representing a frozen lake
- Coordinate system: Top-left corner is (0, 0), x-axis increases to the right (horizontal), y-axis increases downward (vertical)
- Each cell has a size of 1

**Element Definitions:**
- `_`: Ice - A safe, walkable frozen surface
- `O`: Hole - A dangerous hole in the ice (there can be multiple holes)
- `P`: Player - The starting position (only 1 in the map)
- `G`: Goal - The target destination/gift (only 1 in the map)

**Important Notes:**
1. Coordinates are given as (x, y) where x is the column and y is the row
2. The 8 surrounding directions are: up, down, left, right, up-left, up-right, down-left, down-right
3. Relative position (dx, dy) means: to go from position A to position B, move dx units in x-direction and dy units in y-direction
4. There can be multiple holes (O) in the environment

**Instructions:**
- Analyze the map carefully
- Answer the question accurately
- Put your final answer between <answer> and </answer> tags
- For coordinate answers, use format: (x, y)
- For surrounding elements, provide a JSON object with directions as keys
- For counting questions, provide a single number
- Be precise and concise"""

    user_prompt = f"""Here is the FrozenLake environment map:

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


def get_frozenlake_system_prompt(restricted_reasoning: bool = False) -> str:
    """Get system prompt for FrozenLake tasks."""
    prompt = """You are an expert at spatial reasoning and map navigation. You will be given a FrozenLake environment and asked questions about it.

**Environment Description:**
- The environment is an N x N grid representing a frozen lake
- Coordinate system: Top-left corner is (0, 0), x-axis increases to the right (horizontal), y-axis increases downward (vertical)
- Each cell has a size of 1

**Element Definitions:**
- `_`: Ice - A safe, walkable frozen surface
- `O`: Hole - A dangerous hole in the ice (there can be multiple holes)
- `P`: Player - The starting position (only 1 in the map)
- `G`: Goal - The target destination/gift (only 1 in the map)

**Important Notes:**
1. Coordinates are given as (x, y) where x is the column and y is the row
2. The 8 surrounding directions are: up, down, left, right, up-left, up-right, down-left, down-right
3. Relative position (dx, dy) means: to go from position A to position B, move dx units in x-direction and dy units in y-direction
4. There can be multiple holes (O) in the environment

**Instructions:**
- Analyze the map carefully
- Answer the question accurately
- Put your final answer between <answer> and </answer> tags
- For coordinate answers, use format: (x, y)
- For surrounding elements, provide a JSON object with directions as keys
- For counting questions, provide a single number
- Be precise and concise"""
    if restricted_reasoning:
        prompt += "\n\nKeep reasoning minimal and answer directly inside <answer> tags."
    return prompt


def get_frozenlake_user_prompt(map_str: str, question: str, restricted_reasoning: bool = False) -> str:
    """Get user prompt for FrozenLake tasks."""
    prompt = f"""Here is the FrozenLake environment map:

```
{map_str}
```

**Question:** {question}

Please analyze the map and provide your answer. Remember to put your final answer between <answer> and </answer> tags."""
    if restricted_reasoning:
        prompt += "\n\nAnswer directly after <answer> tags without thinking or reasoning. Begin your answer now: <answer>"
    return prompt
