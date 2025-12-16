"""Prompt loading functionality for AI Researcher."""

from pathlib import Path

# Prompts folder location (same directory as this file)
PROMPTS_FOLDER = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt from the prompts folder.

    Args:
        name: Name of the prompt file (without .txt extension)

    Returns:
        The prompt content as a string
    """
    prompt_path = PROMPTS_FOLDER / f"{name}.txt"
    with open(prompt_path, encoding="utf-8") as f:
        return f.read()
