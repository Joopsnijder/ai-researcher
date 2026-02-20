"""Display functions for updating the live UI.

Uses Live with auto_refresh=False to avoid Rich 14 cursor-tracking bugs.
Each function explicitly calls live.update() with freshly built content.
"""

from .console import console
from .panels import create_todo_panel, create_combined_status_panel


def _refresh_live(tracker, search_display):
    """Rebuild and push the combined panel to the Live display."""
    if tracker.live_display is not None:
        combined = create_combined_status_panel(
            search_display, tracker, tracker.current_todos
        )
        tracker.live_display.update(combined)


def display_todos(tracker, search_display, todos):
    """Display TODO list — pushes update to Live display."""
    tracker.current_todos = todos
    tracker.notify_update("todos")

    if tracker.live_display is not None:
        _refresh_live(tracker, search_display)
    else:
        panel = create_todo_panel(todos)
        if panel:
            console.print(panel)


def update_search_display(tracker, search_display):
    """Update search status — pushes update to Live display."""
    tracker.notify_update("search")
    _refresh_live(tracker, search_display)


def update_agent_status(tracker, search_display):
    """Update agent status — pushes update to Live display.

    Called frequently during agent execution so elapsed time stays current.
    """
    tracker.notify_update("status")
    _refresh_live(tracker, search_display)
