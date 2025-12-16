"""Cost calculation for Anthropic API usage."""

# Anthropic pricing per 1M tokens (USD) - Claude Sonnet 4.5
# https://www.anthropic.com/pricing
ANTHROPIC_PRICING = {
    "claude-sonnet-4-5-20250929": {
        "input": 3.00,  # $3 per 1M input tokens
        "output": 15.00,  # $15 per 1M output tokens
    },
    "claude-sonnet-4-20250514": {
        "input": 3.00,
        "output": 15.00,
    },
    # Default fallback for unknown models
    "default": {
        "input": 3.00,
        "output": 15.00,
    },
}


def calculate_cost(
    input_tokens: int, output_tokens: int, model: str = "default"
) -> float:
    """
    Calculate the cost in USD based on token usage.

    Args:
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
        model: Model name for pricing lookup

    Returns:
        Cost in USD
    """
    pricing = ANTHROPIC_PRICING.get(model, ANTHROPIC_PRICING["default"])
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
