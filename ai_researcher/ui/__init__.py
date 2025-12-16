"""UI module for Rich console display."""

from .console import console
from .panels import create_todo_panel, create_combined_status_panel
from .display import display_todos, update_search_display

__all__ = [
    "console",
    "create_todo_panel",
    "create_combined_status_panel",
    "display_todos",
    "update_search_display",
]
