"""Extract research findings from agent messages."""


def extract_research_from_messages(messages):
    """
    Extract research findings from agent messages.

    Args:
        messages: List of agent messages

    Returns:
        str: Concatenated research content
    """
    research_findings = []

    # Patterns to skip (internal agent messages, not actual research)
    skip_patterns = [
        "You are",  # System prompts
        "Successfully",  # Tool confirmations
        "Error",  # Error messages
        "Updated todo list",  # Todo updates
        "Remember to start",  # Planning reminders
        "write_todos",  # Todo tool mentions
        "{'content':",  # Raw todo JSON
        "Now I have comprehensive",  # Internal thinking
        "Let me compile",  # Internal thinking
        "I'll start by",  # Internal thinking
        "I need to",  # Internal thinking
    ]

    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            content = msg.content.strip()

            # Skip short messages
            if len(content) < 200:
                continue

            # Skip messages matching skip patterns
            should_skip = False
            for pattern in skip_patterns:
                if pattern in content[:100]:  # Check start of message
                    should_skip = True
                    break

            if should_skip:
                continue

            # Only include messages that look like actual research content
            # (have proper paragraphs, not just lists of actions)
            if content.count("\n") > 3 and not content.startswith("["):
                research_findings.append(content)

    return "\n\n---\n\n".join(research_findings) if research_findings else ""
