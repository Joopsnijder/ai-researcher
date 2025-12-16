"""Display functions for updating the live UI."""

from .console import console
from .panels import create_todo_panel, create_combined_status_panel


def display_todos(tracker, search_display, todos):
    """Display TODO list - uses in-place update if Live is active."""
    # Store todos for combined display
    tracker.current_todos = todos

    # If Live display is active, update it with combined panel
    if tracker.live_display is not None:
        combined = create_combined_status_panel(search_display, todos)
        tracker.live_display.update(combined)
    else:
        # Fallback: print todo panel normally
        panel = create_todo_panel(todos)
        if panel:
            console.print(panel)


def update_search_display(tracker, search_display):
    """Update the live display with current search status."""
    if tracker.live_display is not None:
        combined = create_combined_status_panel(search_display, tracker.current_todos)
        tracker.live_display.update(combined)
