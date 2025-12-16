"""Agent state tracking for research sessions."""

import time

from .costs import calculate_cost


class AgentTracker:
    """Global state tracker for agent execution metrics."""

    def __init__(self):
        self.current_step = None
        self.searches_count = 0
        self.cache_hits = 0
        self.messages_count = 0
        self.file_operations = []
        self.current_todos = []
        self.debug_mode = False
        self.live_display = None  # Rich Live instance for in-place updates
        # Iteration tracking for report guarantee
        self.iteration_count = 0
        self.recursion_limit = 100
        self.report_triggered = False
        # Token and cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.model_name = "claude-sonnet-4-5-20250929"
        # Session timing and status
        self.start_time: float | None = None
        self.current_status: str = "Initialiseren..."

    def add_token_usage(self, input_tokens: int, output_tokens: int):
        """Add token usage from an API call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def get_total_cost(self) -> float:
        """Calculate total cost based on accumulated token usage."""
        return calculate_cost(
            self.total_input_tokens, self.total_output_tokens, self.model_name
        )

    def reset_token_tracking(self):
        """Reset token tracking for a new research session."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def start_session(self):
        """Start timing for a research session."""
        self.start_time = time.time()
        self.current_status = "Onderzoek starten..."

    def get_elapsed_time(self) -> str:
        """Return formatted elapsed time string."""
        if self.start_time is None:
            return "0m 0s"
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes}m {seconds}s"
