"""Search status display for live UI updates."""

from rich.live import Live
from rich.panel import Panel

from ..ui.console import console


# Live search status display configuration
MAX_SEARCH_HISTORY = 5  # Number of recent searches to show in live display


class SearchStatusDisplay:
    """Manages a live-updating display of recent search activity."""

    def __init__(self, max_history: int = MAX_SEARCH_HISTORY):
        self.max_history = max_history
        self.recent_searches: list[dict] = []
        self.live: Live | None = None
        self.enabled = True

    def start(self):
        """Start the live display."""
        if self.enabled and self.live is None:
            self.live = Live(
                self._render(),
                console=console,
                refresh_per_second=4,
                transient=True,  # Remove display when stopped
            )
            self.live.start()

    def stop(self):
        """Stop the live display."""
        if self.live is not None:
            self.live.stop()
            self.live = None

    def add_search(
        self,
        search_num: int,
        query: str,
        results_count: int,
        provider: str,
        cached: bool = False,
    ):
        """Add a search result to the display."""
        # Truncate query for display
        display_query = query[:55] + "..." if len(query) > 55 else query

        self.recent_searches.append(
            {
                "num": search_num,
                "query": display_query,
                "results": results_count,
                "provider": provider,
                "cached": cached,
            }
        )

        # Keep only the most recent searches
        if len(self.recent_searches) > self.max_history:
            self.recent_searches = self.recent_searches[-self.max_history :]

        # Update live display if active
        if self.live is not None:
            self.live.update(self._render())

    def _render(self):
        """Render the search status panel."""
        if not self.recent_searches:
            content = "[dim]Wachten op zoekopdrachten...[/dim]"
        else:
            lines = []
            for s in self.recent_searches:
                cache_mark = " [green][/green]" if s["cached"] else ""
                line = f"[cyan]#{s['num']:3d}[/cyan] {s['query']} [green] {s['results']}[/green] [dim]({s['provider']})[/dim]{cache_mark}"
                lines.append(line)
            content = "\n".join(lines)

        return Panel(
            content,
            title=f"[bold cyan]Zoekactiviteit[/bold cyan] [dim](laatste {self.max_history})[/dim]",
            border_style="cyan",
            padding=(0, 1),
        )
