"""Display functions for updating the live UI."""

from .console import console
from .panels import create_todo_panel, create_combined_status_panel


def display_todos(tracker, search_display, todos):
    """Display TODO list - uses in-place update if Live is active."""
    # Store todos for combined display
    tracker.current_todos = todos

    # Notify callback for web interface
    tracker.notify_update("todos")

    # Update Live display with new content
    if tracker.live_display is not None:
        combined = create_combined_status_panel(search_display, tracker, todos)
        tracker.live_display.update(combined)
    else:
        # Fallback: print todo panel normally
        panel = create_todo_panel(todos)
        if panel:
            console.print(panel)


def update_search_display(tracker, search_display):
    """Update the live display with current search status."""
    # Notify callback for web interface
    tracker.notify_update("search")

    # Update Live display with new content
    if tracker.live_display is not None:
        combined = create_combined_status_panel(
            search_display, tracker, tracker.current_todos
        )
        tracker.live_display.update(combined)


def update_agent_status(tracker, search_display):
    """Update the live display with current agent status.

    This function is called frequently during agent execution to show
    real-time progress even when no searches are happening.
    """
    # Notify callback for web interface
    tracker.notify_update("status")

    # Update Live display with new content
    if tracker.live_display is not None:
        combined = create_combined_status_panel(
            search_display, tracker, tracker.current_todos
        )
        tracker.live_display.update(combined)
