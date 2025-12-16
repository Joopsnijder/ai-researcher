"""Server-Sent Events utilities."""

import json
from typing import Any


def format_sse_event(event_type: str, data: Any) -> str:
    """
    Format data as a Server-Sent Events message.

    Args:
        event_type: The event type (e.g., "status", "search", "complete")
        data: Data to serialize as JSON

    Returns:
        Formatted SSE message string
    """
    # Serialize data to JSON
    if isinstance(data, dict):
        json_data = json.dumps(data, default=str)
    else:
        json_data = json.dumps({"value": data}, default=str)

    # Format as SSE
    lines = [
        f"event: {event_type}",
        f"data: {json_data}",
        "",  # Empty line to end the event
    ]
    return "\n".join(lines) + "\n"


def format_keepalive() -> str:
    """Format a keepalive/heartbeat message."""
    return ": keepalive\n\n"
