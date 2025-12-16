"""Display functions for updating the live UI.

The Live display uses LiveStatusRenderable which auto-refreshes via
refresh_per_second=4. We do NOT call live_display.update() because that
would replace the LiveStatusRenderable with a static Group, breaking
the auto-refresh behavior.

Instead, these functions just update the underlying data (tracker, search_display)
and the LiveStatusRenderable picks up the changes on its next automatic refresh.
"""

from .console import console
from .panels import create_todo_panel


def display_todos(tracker, search_display, todos):
    """Display TODO list - updates data for Live display auto-refresh."""
    # Store todos for combined display - LiveStatusRenderable will pick this up
    tracker.current_todos = todos

    # Notify callback for web interface
    tracker.notify_update("todos")

    # If Live is not active, fall back to direct printing
    if tracker.live_display is None:
        panel = create_todo_panel(todos)
        if panel:
            console.print(panel)
    # Note: No update() call - LiveStatusRenderable auto-refreshes


def update_search_display(tracker, search_display):
    """Update search status - data is auto-refreshed by LiveStatusRenderable."""
    # Notify callback for web interface
    tracker.notify_update("search")
    # Note: No update() call - LiveStatusRenderable auto-refreshes
    # search_display.recent_searches is already updated by the caller


def update_agent_status(tracker, search_display):
    """Update agent status - data is auto-refreshed by LiveStatusRenderable.

    This function is called frequently during agent execution. The tracker's
    current_status is updated by the caller before this function is called.
    The LiveStatusRenderable picks up all data changes on its next refresh.
    """
    # Notify callback for web interface
    tracker.notify_update("status")
    # Note: No update() call - LiveStatusRenderable auto-refreshes
    # tracker.current_status is already updated by the caller
