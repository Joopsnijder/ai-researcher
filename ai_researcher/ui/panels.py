"""Rich panel creation for UI display."""

from rich.panel import Panel
from rich.table import Table
from rich.console import Group, RenderableType


class LiveStatusRenderable:
    """A renderable that refreshes on each render cycle.

    This class creates a fresh panel on each render, ensuring the
    elapsed time updates even when no agent events are occurring.
    """

    def __init__(self, search_display, tracker):
        self.search_display = search_display
        self.tracker = tracker

    def __rich__(self) -> RenderableType:
        """Called by Rich on each render cycle."""
        return create_combined_status_panel(
            self.search_display,
            self.tracker,
            self.tracker.current_todos if self.tracker else None,
        )


def create_progress_bar(
    progress_pct: float, threshold_pct: float = 85, width: int = 20
) -> str:
    """Create a text-based progress bar with threshold indicator.

    Args:
        progress_pct: Current progress percentage (0-100)
        threshold_pct: Threshold percentage to mark (default 85%)
        width: Total width of the bar in characters
    """
    filled = int(width * progress_pct / 100)
    threshold_pos = int(width * threshold_pct / 100)

    bar = ""
    for i in range(width):
        if i < filled:
            if i >= threshold_pos:
                bar += "[red]█[/red]"
            else:
                bar += "[green]█[/green]"
        elif i == threshold_pos:
            bar += "[yellow]│[/yellow]"
        else:
            bar += "[dim]░[/dim]"

    return f"[{bar}]"


def create_agent_status_panel(tracker) -> Panel:
    """Create agent status panel with live metrics."""
    # Elapsed time
    elapsed = tracker.get_elapsed_time()

    # Iteration progress
    if tracker.recursion_limit > 0:
        progress_pct = (tracker.iteration_count / tracker.recursion_limit) * 100
    else:
        progress_pct = 0
    bar = create_progress_bar(progress_pct, threshold_pct=85)

    # Token counts (formatted in K)
    input_k = tracker.total_input_tokens / 1000
    output_k = tracker.total_output_tokens / 1000

    # Cost
    cost = tracker.get_total_cost()

    # Build content lines
    lines = [
        f"[dim]⏱️  Verstreken:[/dim] [bold]{elapsed}[/bold]",
        f"[dim]🔄 Iteratie:[/dim] [bold]{tracker.iteration_count}[/bold]/{tracker.recursion_limit} ({progress_pct:.0f}%)  {bar}",
        f"[dim]📊 Tokens:[/dim] [bold]{input_k:.1f}K[/bold] in / [bold]{output_k:.1f}K[/bold] out",
        f"[dim]💰 Kosten:[/dim] [bold]${cost:.2f}[/bold]",
        f"[dim]🤖 Status:[/dim] [bold cyan]{tracker.current_status}[/bold cyan]",
    ]

    return Panel(
        "\n".join(lines),
        title="[bold cyan]Agent Status[/bold cyan]",
        border_style="cyan",
        padding=(0, 1),
    )


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


def create_combined_status_panel(search_display, tracker=None, todos=None):
    """Create a combined panel showing agent status, search activity, and todos."""
    panels = []

    # Agent status panel (shown when session is active)
    if tracker and tracker.start_time is not None:
        panels.append(create_agent_status_panel(tracker))

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
