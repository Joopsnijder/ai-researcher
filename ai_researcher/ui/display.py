"""Display functions for updating the live UI."""

from .console import console
from .panels import create_todo_panel


def display_todos(tracker, search_display, todos):
    """Display TODO list - uses in-place update if Live is active."""
    # Store todos for combined display
    tracker.current_todos = todos

    # Notify callback for web interface
    tracker.notify_update("todos")

    # If no Live display, print panel normally (fallback)
    if tracker.live_display is None:
        panel = create_todo_panel(todos)
        if panel:
            console.print(panel)
    # Note: Live display auto-refreshes via LiveStatusRenderable


def update_search_display(tracker, search_display):
    """Update the live display with current search status."""
    # Notify callback for web interface
    tracker.notify_update("search")
    # Note: Live display auto-refreshes via LiveStatusRenderable


def update_agent_status(tracker, search_display):
    """Update the live display with current agent status.

    This function is called frequently during agent execution to show
    real-time progress even when no searches are happening.
    """
    # Notify callback for web interface
    tracker.notify_update("status")
    # Note: Live display auto-refreshes via LiveStatusRenderable
