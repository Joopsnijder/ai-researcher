"""Language detection for research reports."""


def detect_language(text: str) -> str:
    """
    Detect language of text using simple heuristics.

    Args:
        text: Text to analyze

    Returns:
        "nl" for Dutch, "en" for English
    """
    dutch_words = [
        "de",
        "het",
        "een",
        "van",
        "voor",
        "met",
        "zijn",
        "worden",
        "naar",
        "door",
        "aan",
        "op",
        "is",
        "dat",
        "dit",
        "heeft",
        "nieuwe",
        "bij",
    ]
    words = text.lower().split()[:100]
    dutch_count = sum(1 for w in words if w in dutch_words)
    # Lower threshold: 3+ Dutch words suggests Dutch text
    return "nl" if dutch_count >= 3 else "en"
