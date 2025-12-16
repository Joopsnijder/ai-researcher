"""Configuration constants and utility functions for AI Researcher."""

import contextlib
import io
import os

# Research output folder (keeps reports out of git)
RESEARCH_FOLDER = "research"

# Report guarantee constants - ensures final report is always written
REPORT_RESERVED_ITERATIONS = 20  # Iterations reserved for final report writing
REPORT_TRIGGER_THRESHOLD = 0.85  # Trigger early report at 85% of limit


def ensure_research_folder():
    """Create research folder if it doesn't exist."""
    from .ui.console import console

    if not os.path.exists(RESEARCH_FOLDER):
        os.makedirs(RESEARCH_FOLDER)
        console.print(f"[dim]Created {RESEARCH_FOLDER}/ folder[/dim]")


@contextlib.contextmanager
def suppress_output():
    """Suppress stdout and stderr output from external libraries."""
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        yield
