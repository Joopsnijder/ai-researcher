"""Rich panel creation for UI display."""

from rich.panel import Panel
from rich.table import Table
from rich.console import Group


def create_todo_panel(todos):
    """Create a todo panel (returns renderable, doesn't print)."""
    if not todos:
        return None

    todo_table = Table(show_header=False, box=None, padding=(0, 1), expand=False)
    todo_table.add_column("Status", style="bold", width=3)
    todo_table.add_column("Task", style="")

    for todo in todos:
        status = todo.get("status", "pending")
        content = todo.get("content", "")

        # Choose icon and color based on status
        if status == "completed":
            icon = "[green]✓[/green]"
            task_style = "[dim green]"
        elif status == "in_progress":
            icon = "[yellow]▶[/yellow]"
            task_style = "[bold yellow]"
        else:  # pending
            icon = "[dim]○[/dim]"
            task_style = "[dim]"

        todo_table.add_row(icon, f"{task_style}{content}[/]")

    return Panel(
        todo_table,
        title="[bold cyan]📋 Taken[/bold cyan]",
        border_style="cyan",
        padding=(0, 1),
    )


def create_combined_status_panel(search_display, todos=None):
    """Create a combined panel showing both search activity and todos."""
    panels = []

    # Search activity panel (always show during research)
    if search_display.recent_searches:
        search_lines = []
        for s in search_display.recent_searches:
            cache_mark = " [green]✓[/green]" if s["cached"] else ""
            line = f"[cyan]#{s['num']:3d}[/cyan] {s['query']} [green]→ {s['results']}[/green] [dim]({s['provider']})[/dim]{cache_mark}"
            search_lines.append(line)
        search_content = "\n".join(search_lines)
    else:
        search_content = "[dim]Wachten op zoekopdrachten...[/dim]"

    search_panel = Panel(
        search_content,
        title=f"[bold cyan]🔍 Zoekactiviteit[/bold cyan] [dim](laatste {search_display.max_history})[/dim]",
        border_style="cyan",
        padding=(0, 1),
    )
    panels.append(search_panel)

    # Todo panel (if there are todos)
    if todos:
        todo_panel = create_todo_panel(todos)
        if todo_panel:
            panels.append(todo_panel)

    # Return a group of panels stacked vertically
    return Group(*panels)
