import argparse
import contextlib
import glob
import io
import os
import time
from typing import Literal

import dotenv

from tavily import TavilyClient
from multi_search_api import SmartSearchTool
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.prompt import Prompt
from rich.live import Live

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

dotenv.load_dotenv()

# Initialize rich console
console = Console()

# Research output folder (keeps reports out of git)
RESEARCH_FOLDER = "research"

# Prompts folder (external prompt files)
PROMPTS_FOLDER = "prompts"

# Report guarantee constants - ensures final report is always written
REPORT_RESERVED_ITERATIONS = 20  # Iterations reserved for final report writing
REPORT_TRIGGER_THRESHOLD = 0.85  # Trigger early report at 85% of limit


def load_prompt(name: str) -> str:
    """
    Load a prompt from the prompts folder.

    Args:
        name: Name of the prompt file (without .txt extension)

    Returns:
        The prompt content as a string
    """
    prompt_path = os.path.join(PROMPTS_FOLDER, f"{name}.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def ensure_research_folder():
    """Create research folder if it doesn't exist"""
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


# Hybrid Search Tool supporting multiple providers
class HybridSearchTool:
    """Supports both Tavily and Multi-Search API providers"""

    def __init__(self, provider="multi-search"):
        self.provider = provider
        self.provider_usage = {}  # Track which providers were used

        # Initialize clients
        if provider in ["tavily", "auto"]:
            self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        if provider in ["multi-search", "auto"]:
            self.multi_search = SmartSearchTool(
                serper_api_key=os.getenv("SERPER_API_KEY"),
                brave_api_key=os.getenv("BRAVE_API_KEY"),
                enable_cache=True,  # Thread-safe since multi-search-api v0.1.1
            )

    def normalize_multi_search_response(self, response):
        """Convert multi-search-api format to Tavily-compatible format"""
        return {
            "query": response.get("query", ""),
            "results": [
                {
                    "title": result.get("title", ""),
                    "content": result.get("snippet", ""),  # snippet -> content
                    "url": result.get("link", ""),  # link -> url
                    "score": 0.9,  # Multi-search doesn't provide scores
                }
                for result in response.get("results", [])
            ],
            "_provider": response.get("provider", "unknown"),
            "_cache_hit": response.get("cache_hit", False),
        }

    def search(self, query: str, max_results: int = 7, topic: str = "general"):
        """Execute search with selected provider"""

        if self.provider == "tavily":
            result = self.tavily.search(query, max_results=max_results, topic=topic)
            self.provider_usage["Tavily"] = self.provider_usage.get("Tavily", 0) + 1
            result["_actual_provider"] = "Tavily"
            return result

        elif self.provider == "multi-search":
            with suppress_output():
                response = self.multi_search.search(
                    query=query, num_results=max_results
                )
            normalized = self.normalize_multi_search_response(response)
            provider_name = normalized.get("_provider", "Unknown")
            self.provider_usage[provider_name] = (
                self.provider_usage.get(provider_name, 0) + 1
            )
            normalized["_actual_provider"] = provider_name
            return normalized

        elif self.provider == "auto":
            # Try multi-search first (free), fallback to Tavily
            try:
                with suppress_output():
                    response = self.multi_search.search(
                        query=query, num_results=max_results
                    )
                normalized = self.normalize_multi_search_response(response)
                provider_name = normalized.get("_provider", "Unknown")
                self.provider_usage[provider_name] = (
                    self.provider_usage.get(provider_name, 0) + 1
                )
                normalized["_actual_provider"] = provider_name
                return normalized
            except Exception as e:
                console.print(
                    f"[yellow]Multi-search failed, using Tavily: {e}[/yellow]"
                )
                result = self.tavily.search(query, max_results=max_results, topic=topic)
                self.provider_usage["Tavily (fallback)"] = (
                    self.provider_usage.get("Tavily (fallback)", 0) + 1
                )
                result["_actual_provider"] = "Tavily (fallback)"
                return result

    def clear_cache(self):
        """Clear the search cache (multi-search only)"""
        if self.provider in ["multi-search", "auto"] and hasattr(self, "multi_search"):
            self.multi_search.clear_cache()
            console.print("[green]✓[/green] Cache cleared successfully")
        else:
            console.print("[yellow]Cache not available for this provider[/yellow]")

    def get_cache_stats(self):
        """Get cache statistics (multi-search only)"""
        if self.provider in ["multi-search", "auto"] and hasattr(self, "multi_search"):
            status = self.multi_search.get_status()
            cache_info = status.get("cache", {})
            return {
                "total_entries": cache_info.get("total_entries", 0),
                "oldest_entry": cache_info.get("oldest_entry", "N/A"),
                "newest_entry": cache_info.get("newest_entry", "N/A"),
                "cache_file": cache_info.get("cache_file", "N/A"),
            }
        return None

    def display_cache_stats(self):
        """Display cache statistics in a nice format"""
        stats = self.get_cache_stats()
        if stats is None:
            console.print(
                "[yellow]Cache stats not available for this provider[/yellow]"
            )
            return

        table = Table(title="Cache Statistics", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Entries", str(stats["total_entries"]))
        table.add_row("Oldest Entry", stats["oldest_entry"])
        table.add_row("Newest Entry", stats["newest_entry"])
        table.add_row("Cache Location", stats["cache_file"])

        console.print(table)


# Global search tool (will be initialized after provider selection)
search_tool = None


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
                cache_mark = " [green]✓[/green]" if s["cached"] else ""
                line = f"[cyan]#{s['num']:3d}[/cyan] {s['query']} [green]→ {s['results']}[/green] [dim]({s['provider']})[/dim]{cache_mark}"
                lines.append(line)
            content = "\n".join(lines)

        return Panel(
            content,
            title=f"[bold cyan]🔍 Zoekactiviteit[/bold cyan] [dim](laatste {self.max_history})[/dim]",
            border_style="cyan",
            padding=(0, 1),
        )


# Global search status display
search_display = SearchStatusDisplay()


# Global state to track agent activity
class AgentTracker:
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


tracker = AgentTracker()


def should_trigger_early_report() -> bool:
    """
    Check if we should trigger early report writing to guarantee completion.

    Returns True if:
    - We've used >= 85% of iterations AND
    - Report hasn't been triggered yet AND
    - Final report doesn't exist yet
    """
    if tracker.report_triggered:
        return False

    if tracker.recursion_limit <= 0:
        return False

    iterations_used_ratio = tracker.iteration_count / tracker.recursion_limit

    # Check if we're approaching the limit
    if iterations_used_ratio >= REPORT_TRIGGER_THRESHOLD:
        # Verify report doesn't exist yet
        final_report_path = os.path.join(RESEARCH_FOLDER, "final_report.md")
        if not os.path.exists(final_report_path):
            return True

    return False


def create_finalize_instruction() -> str:
    """
    Create an urgent instruction for the agent to finalize the report immediately.
    """
    return """
URGENT: You are approaching the iteration limit. You MUST write the final report NOW.

CRITICAL INSTRUCTIONS:
1. STOP all research activities immediately
2. Use all findings gathered so far (even if incomplete)
3. Write the final report to `research/final_report.md` using the write_file tool
4. Follow the standard report structure (metadata header, Management Samenvatting, etc.)
5. Include all citations gathered so far

Do NOT:
- Start new research-agent tasks
- Call the critique-agent
- Perform additional searches

BEGIN WRITING THE REPORT IMMEDIATELY.
"""


def create_todo_panel(todos):
    """Create a todo panel (returns renderable, doesn't print)"""
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


def create_combined_status_panel(todos=None):
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
    from rich.console import Group

    return Group(*panels)


def display_todos(todos):
    """Display TODO list - uses in-place update if Live is active"""
    # Store todos for combined display
    tracker.current_todos = todos

    # If Live display is active, update it with combined panel
    if tracker.live_display is not None:
        combined = create_combined_status_panel(todos)
        tracker.live_display.update(combined)
    else:
        # Fallback: print todo panel normally
        panel = create_todo_panel(todos)
        if panel:
            console.print(panel)


def update_search_display():
    """Update the live display with current search status."""
    if tracker.live_display is not None:
        combined = create_combined_status_panel(tracker.current_todos)
        tracker.live_display.update(combined)


# Search tool to use to do research
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    tracker.searches_count += 1
    search_num = tracker.searches_count

    # Use the global search_tool
    search_docs = search_tool.search(query, max_results=max_results, topic=topic)

    # Extract search result info
    if isinstance(search_docs, dict) and "results" in search_docs:
        results_count = len(search_docs["results"])
        provider_name = search_docs.get("_actual_provider", "Unknown")
        cache_hit = search_docs.get("_cache_hit", False)
        if cache_hit:
            tracker.cache_hits += 1
    else:
        results_count = 0
        provider_name = "Unknown"
        cache_hit = False

    # Update the search display state
    search_display.add_search(
        search_num=search_num,
        query=query,
        results_count=results_count,
        provider=provider_name,
        cached=cache_hit,
    )

    # Update the live display with new search info
    update_search_display()

    return search_docs


def write_file(filename: str, content: str):
    """Write content to a file (for Quick Research mode)"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        console.print(f"[bold green]✍️  Wrote file:[/bold green] {filename}")
        return f"Successfully wrote {len(content)} characters to {filename}"
    except Exception as e:
        console.print(f"[bold red]❌ Error writing {filename}:[/bold red] {e}")
        return f"Error: {e}"


# Track file operations for debugging
def track_file_operation(operation: str, filename: str):
    """Log file operations for debugging"""
    tracker.file_operations.append({"operation": operation, "file": filename})
    console.print(f"[bold green]📝 File {operation}:[/bold green] {filename}")


research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
    "system_prompt": load_prompt("research_agent"),
    "tools": [internet_search],
}

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Used to critique the final report. Give this agent some information about how you want it to critique the report.",
    "system_prompt": load_prompt("critique_agent"),
}


# ══════════════════════════════════════════════════════════════
# DEEP RESEARCH MODE (Agentic with Sub-agents)
# Thorough research with planning, parallel agents, and critique
# ══════════════════════════════════════════════════════════════

# Load prompts from external files
quick_research_prompt = load_prompt("quick_research")
research_instructions = load_prompt("deep_research")

# Create the agent with FilesystemBackend to write reports to disk
agent = create_deep_agent(
    tools=[internet_search],
    system_prompt=research_instructions,
    subagents=[critique_sub_agent, research_sub_agent],
    backend=FilesystemBackend(),  # Write to actual filesystem, not in-memory
)


# ══════════════════════════════════════════════════════════════
# DETERMINISTIC REPORT GENERATION GUARANTEE
# These functions ensure final_report.md ALWAYS exists after agent runs
# ══════════════════════════════════════════════════════════════


def extract_research_from_messages(messages):
    """
    Extract research findings from agent messages

    Args:
        messages: List of agent messages

    Returns:
        str: Concatenated research content
    """
    research_findings = []

    # Patterns to skip (internal agent messages, not actual research)
    skip_patterns = [
        "You are",  # System prompts
        "Successfully",  # Tool confirmations
        "Error",  # Error messages
        "Updated todo list",  # Todo updates
        "Remember to start",  # Planning reminders
        "write_todos",  # Todo tool mentions
        "{'content':",  # Raw todo JSON
        "Now I have comprehensive",  # Internal thinking
        "Let me compile",  # Internal thinking
        "I'll start by",  # Internal thinking
        "I need to",  # Internal thinking
    ]

    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            content = msg.content.strip()

            # Skip short messages
            if len(content) < 200:
                continue

            # Skip messages matching skip patterns
            should_skip = False
            for pattern in skip_patterns:
                if pattern in content[:100]:  # Check start of message
                    should_skip = True
                    break

            if should_skip:
                continue

            # Only include messages that look like actual research content
            # (have proper paragraphs, not just lists of actions)
            if content.count("\n") > 3 and not content.startswith("["):
                research_findings.append(content)

    return "\n\n---\n\n".join(research_findings) if research_findings else ""


def create_emergency_report(question, research_content, partial=False):
    """
    Create a markdown report from available research

    Args:
        question: The research question
        research_content: Available research findings
        partial: Whether this is a partial/incomplete report

    Returns:
        str: Markdown formatted report
    """
    status = (
        "Partial Research Report" if partial else "Research Report (Auto-Generated)"
    )
    warning = ""

    if partial:
        warning = """
> ⚠️ **Warning**: This report was automatically generated because the research process
> was interrupted or incomplete (recursion limit, timeout, or manual stop).
"""
    else:
        warning = """
> ℹ️ **Note**: This report was automatically generated because the AI agent did not
> create the final report file. The content below was extracted from the agent's research.
"""

    report = f"""# {status}

## Research Question

{question}

## Status

{warning}

## Research Findings

{research_content if research_content.strip() else "*No research content was captured during agent execution.*"}

---

*This report was auto-generated by the deterministic safety net in `run_research()`*
*Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}*
"""

    return report


def detect_language(text: str) -> str:
    """
    Detect language of text using simple heuristics.

    Args:
        text: Text to analyze

    Returns:
        "nl" for Dutch, "en" for English
    """
    dutch_words = [
        "de",
        "het",
        "een",
        "van",
        "voor",
        "met",
        "zijn",
        "worden",
        "naar",
        "door",
        "aan",
        "op",
        "is",
        "dat",
        "dit",
        "heeft",
        "nieuwe",
        "bij",
    ]
    words = text.lower().split()[:100]
    dutch_count = sum(1 for w in words if w in dutch_words)
    # Lower threshold: 3+ Dutch words suggests Dutch text
    return "nl" if dutch_count >= 3 else "en"


def refine_emergency_report_with_llm(
    question: str, raw_findings: str, language: str = "nl"
) -> str | None:
    """
    Use Claude to restructure raw emergency content into a professional report.

    Args:
        question: The original research question
        raw_findings: Raw extracted research content
        language: Language for the report (nl/en)

    Returns:
        Structured markdown report, or None if refinement fails
    """
    from anthropic import Anthropic

    # Skip if no substantial content
    if not raw_findings or len(raw_findings.strip()) < 500:
        return None

    client = Anthropic()

    if language == "nl":
        lang_instruction = """Schrijf in het Nederlands.
BELANGRIJK: Gebruik Nederlandse titel-casing voor koppen (alleen eerste woord met hoofdletter, niet elk woord).
Correct: "De veranderende rol van leiderschap"
Fout: "De Veranderende Rol van Leiderschap" """
    else:
        lang_instruction = (
            "Write in English (use standard English title case for headings)."
        )

    # Load prompt template and fill in variables
    refinement_prompt = load_prompt("emergency_refinement").format(
        lang_instruction=lang_instruction,
        question=question,
        raw_findings=raw_findings[:80000],
    )

    try:
        console.print(
            "\n[bold yellow]📝 Genereren van gestructureerd rapport uit research...[/bold yellow]"
        )
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=12000,
            messages=[{"role": "user", "content": refinement_prompt}],
        )
        refined = response.content[0].text
        console.print("[green]✓ Rapport succesvol gestructureerd met LLM[/green]")
        return refined
    except Exception as e:
        console.print(
            f"[yellow]⚠ LLM refinement failed: {e}, using raw content[/yellow]"
        )
        return None


def ensure_report_exists(question, result, partial=False):
    """
    GUARANTEE: Ensures final_report.md exists after agent execution

    This function is called deterministically (not by the LLM) to ensure
    a report file is always created, even if the agent failed to do so.

    Args:
        question: The research question
        result: Agent execution result (can be None)
        partial: Whether this is a partial report (error/interrupt case)
    """
    # Ensure research folder exists
    ensure_research_folder()
    final_report_path = os.path.join(RESEARCH_FOLDER, "final_report.md")

    # Check if report already exists
    if os.path.exists(final_report_path):
        console.print(
            f"[dim]✓ {final_report_path} already exists (created by agent)[/dim]"
        )
        return

    # Check if agent wrote to a different file (common mistake)
    possible_reports = (
        glob.glob("*.md")
        + glob.glob("/tmp/*.md")
        + glob.glob(f"{RESEARCH_FOLDER}/*.md")
    )
    skip_files = ["README.md", "CLAUDE.md", "requirements.md", "final_report.md"]
    recent_md_files = [
        f
        for f in possible_reports
        if os.path.isfile(f)
        and os.path.getmtime(f) > time.time() - 300  # Last 5 minutes
        and os.path.basename(f) not in skip_files
        and not os.path.basename(f).startswith("test_")
    ]

    # If we found a recent .md file, use it instead of generating emergency report
    if recent_md_files:
        # Sort by modification time, newest first
        recent_md_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        found_report = recent_md_files[0]

        console.print(
            f"\n[bold yellow]⚠️  Agent schreef naar verkeerde locatie: {found_report}[/bold yellow]"
        )
        console.print(f"[green]   → Kopiëren naar {final_report_path}...[/green]")

        # Copy the found report to final_report.md
        with open(found_report, "r", encoding="utf-8") as src:
            content = src.read()
        with open(final_report_path, "w", encoding="utf-8") as dst:
            dst.write(content)

        console.print(f"[green]✓ Rapport gekopieerd van {found_report}[/green]")
        return

    # No recent file found - generate emergency report
    console.print(
        f"\n[bold yellow]⚠️  Agent did not create {final_report_path} - generating emergency report...[/bold yellow]"
    )
    console.print("[yellow]   Mogelijke oorzaken:[/yellow]")
    console.print(
        "[dim]   • Agent dacht klaar te zijn zonder bestand te schrijven[/dim]"
    )
    console.print(
        "[dim]   • Recursion limit bereikt voordat rapport werd geschreven[/dim]"
    )

    # Extract any research from agent messages
    research_content = ""
    if result and "messages" in result:
        research_content = extract_research_from_messages(result["messages"])

    # Try LLM refinement first for better quality reports
    language = detect_language(question)
    refined_content = refine_emergency_report_with_llm(
        question, research_content, language
    )

    if refined_content:
        # Wrap refined content in report template
        report = f"""# Onderzoeksrapport (Automatisch Gegenereerd)

> ℹ️ **Note**: Dit rapport is automatisch gegenereerd en gestructureerd uit verzamelde
> research bevindingen omdat de agent het eindrapport niet zelf heeft geschreven.

{refined_content}

---
*Rapport gestructureerd met Claude op: {time.strftime("%Y-%m-%d %H:%M:%S")}*
"""
    else:
        # Fallback to old method if LLM fails
        report = create_emergency_report(question, research_content, partial)

    # Write report to file
    with open(final_report_path, "w") as f:
        f.write(report)

    console.print("[green]✓ Emergency report created from available research[/green]")
    if not research_content.strip():
        console.print("[dim]  (Note: Limited research content was available)[/dim]")


def generate_report_filename(question: str) -> str:
    """
    Generate a safe filename based on the question

    Args:
        question: The research question

    Returns:
        str: Safe filename like "what-is-quantum-computing.md"
    """
    import re

    # Take first 50 chars of question
    safe_question = question[:50]

    # Remove special characters and convert to lowercase
    safe_question = re.sub(r"[^\w\s-]", "", safe_question)
    safe_question = re.sub(r"[-\s]+", "-", safe_question)
    safe_question = safe_question.strip("-").lower()

    # Ensure it's not empty
    if not safe_question:
        safe_question = "research-report"

    return f"{safe_question}.md"


def rename_final_report(question: str) -> str:
    """
    Rename final_report.md to a question-based filename in research folder

    Args:
        question: The research question

    Returns:
        str: The new filename (full path), or None if rename failed
    """
    ensure_research_folder()
    final_report_path = os.path.join(RESEARCH_FOLDER, "final_report.md")

    if not os.path.exists(final_report_path):
        return None

    base_filename = generate_report_filename(question)
    new_filename = os.path.join(RESEARCH_FOLDER, base_filename)

    # If file already exists, add a number
    base_name = new_filename[:-3]  # Remove .md
    counter = 1
    while os.path.exists(new_filename):
        new_filename = f"{base_name}-{counter}.md"
        counter += 1

    try:
        os.rename(final_report_path, new_filename)
        return new_filename
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not rename report: {e}[/yellow]")
        return final_report_path


# ══════════════════════════════════════════════════════════════
# QUICK RESEARCH IMPLEMENTATION
# ══════════════════════════════════════════════════════════════


def run_quick_research(question: str, max_searches: int = 5):
    """
    Quick research using direct LLM calls (no agentic overhead)

    This mode is faster and cheaper than deep research:
    - No agents, no sub-agents, no planning overhead
    - Direct LLM interaction with search tools
    - Suitable for simple questions, facts, overviews
    - Completes in 1-3 minutes vs 10-30 minutes for deep research

    Args:
        question: The research question
        max_searches: Maximum number of web searches (default: 5)
    """
    from langchain_anthropic import ChatAnthropic

    # Start timing
    start_time = time.time()

    # Clean up files from previous runs to avoid confusion
    ensure_research_folder()
    cleanup_files = ["question.txt", os.path.join(RESEARCH_FOLDER, "final_report.md")]
    for old_file in cleanup_files:
        if os.path.exists(old_file):
            os.remove(old_file)
            console.print(f"[dim]Removed old {old_file}[/dim]")

    # Track existing files before starting
    existing_files = set(os.listdir("."))

    # Print header
    console.print("\n")
    console.print(
        Panel.fit(
            f"[bold white]{question}[/bold white]",
            title="[bold cyan]🚀 AI Quick Research[/bold cyan]",
            border_style="cyan",
        )
    )

    console.print("\n[yellow]Starting quick research (direct LLM mode)...[/yellow]\n")

    try:
        # Initialize Claude model
        model = ChatAnthropic(
            model_name="claude-sonnet-4-5-20250929",
            max_tokens=8000,
        )

        # Define available tools for the model
        tools = [internet_search, write_file]
        model_with_tools = model.bind_tools(tools)

        # Create initial message with system prompt and question
        messages = [
            {"role": "system", "content": quick_research_prompt},
            {"role": "user", "content": question},
        ]

        console.print("[dim]💭 Analyzing question and planning research...[/dim]\n")

        # Interaction loop with tool calling
        searches_performed = 0
        max_iterations = 20  # Safety limit

        for iteration in range(max_iterations):
            # Call model
            response = model_with_tools.invoke(messages)

            # Add response to conversation
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.tool_calls
                    if hasattr(response, "tool_calls")
                    else [],
                }
            )

            # Check if model wants to use tools
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Execute tool calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_input = tool_call["args"]

                    if (
                        tool_name == "internet_search"
                        and searches_performed < max_searches
                    ):
                        # Execute search
                        result = internet_search(**tool_input)
                        searches_performed += 1

                        # Add tool result to conversation
                        messages.append(
                            {
                                "role": "tool",
                                "content": str(result),
                                "tool_call_id": tool_call["id"],
                            }
                        )
                    elif tool_name == "write_file":
                        # Execute file write
                        result = write_file(**tool_input)

                        # Add tool result to conversation
                        messages.append(
                            {
                                "role": "tool",
                                "content": str(result),
                                "tool_call_id": tool_call["id"],
                            }
                        )
                    else:
                        # Search limit reached or unknown tool
                        messages.append(
                            {
                                "role": "tool",
                                "content": "Search limit reached"
                                if tool_name == "internet_search"
                                else "Tool not available",
                                "tool_call_id": tool_call["id"],
                            }
                        )
            else:
                # No more tool calls - model is done
                console.print(
                    "\n[dim]📝 Finalizing research and creating report...[/dim]\n"
                )
                break

        # Calculate duration
        end_time = time.time()
        duration_seconds = end_time - start_time

        # Format duration
        minutes, seconds = divmod(int(duration_seconds), 60)
        duration_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

        # Create a mock result for ensure_report_exists
        mock_result = {"messages": messages}

        # DETERMINISTIC GUARANTEE: Ensure report exists
        ensure_report_exists(question, mock_result, partial=False)

        # Print summary
        console.print("\n")
        console.print(Rule("[bold green]Klaar![/bold green]"))

        stats_table = Table(show_header=False, box=None, padding=(0, 2))
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow")

        stats_table.add_row("⏱️  Duur", duration_str)
        stats_table.add_row("🔍 Aantal zoekopdrachten", str(searches_performed))

        # Add cache statistics
        if tracker.cache_hits > 0:
            cache_percentage = (
                (tracker.cache_hits / tracker.searches_count * 100)
                if tracker.searches_count > 0
                else 0
            )
            stats_table.add_row(
                "💾 Cache hits", f"{tracker.cache_hits} ({cache_percentage:.0f}%)"
            )
            api_calls_saved = tracker.cache_hits
            stats_table.add_row("✨ API calls bespaard", str(api_calls_saved))

        stats_table.add_row(
            "💬 LLM interactions",
            str(len([m for m in messages if m.get("role") == "assistant"])),
        )
        stats_table.add_row("🚀 Mode", "Quick Research (Direct LLM)")

        console.print(
            Panel(stats_table, title="[bold]Statistieken[/bold]", border_style="green")
        )

        # Show newly created files
        new_files = set(os.listdir(".")) - existing_files
        if new_files:
            console.print("\n[bold]Nieuwe files aangemaakt:[/bold]")
            for file in sorted(new_files):
                if not file.startswith("."):
                    console.print(f"  • [green]{file}[/green]")

        # Rename report to question-based filename
        final_filename = rename_final_report(question)

        # Show report location
        if final_filename and os.path.exists(final_filename):
            console.print(
                f"\n[bold green]✓ Rapport opgeslagen als:[/bold green] [link=file://{final_filename}]{final_filename}[/link]"
            )

        return {
            "messages": messages,
            "searches": searches_performed,
            "report_file": final_filename,
        }

    except KeyboardInterrupt:
        console.print(
            "\n\n[bold red]⚠️  Onderzoek onderbroken door gebruiker[/bold red]"
        )
        ensure_report_exists(question, None, partial=True)
        final_filename = rename_final_report(question)
        if final_filename:
            console.print(f"[dim]Partial report saved as: {final_filename}[/dim]")
        return None

    except Exception as e:
        console.print(f"\n\n[bold red]❌ Fout opgetreden:[/bold red] {str(e)}")
        ensure_report_exists(question, None, partial=True)
        final_filename = rename_final_report(question)
        if final_filename:
            console.print(f"[dim]Partial report saved as: {final_filename}[/dim]")
        raise


# ══════════════════════════════════════════════════════════════
# DEEP RESEARCH IMPLEMENTATION (Agentic)
# ══════════════════════════════════════════════════════════════


def run_research(question: str, recursion_limit: int = 100):
    """Run the research agent with rich UI"""

    # Start timing
    start_time = time.time()

    # Reset tracker for new research session
    tracker.iteration_count = 0
    tracker.recursion_limit = recursion_limit
    tracker.report_triggered = False

    # Clean up files from previous runs to avoid confusion
    ensure_research_folder()
    cleanup_files = ["question.txt", os.path.join(RESEARCH_FOLDER, "final_report.md")]
    for old_file in cleanup_files:
        if os.path.exists(old_file):
            os.remove(old_file)
            console.print(f"[dim]Removed old {old_file}[/dim]")

    # Track existing files before starting
    existing_files = set(os.listdir("."))

    # Enhance question with planning reminder for better TODO structure
    enhanced_question = f"""{question}

Remember to start by creating a detailed TODO plan using write_todos before beginning research."""

    # Print header
    console.print("\n")
    console.print(
        Panel.fit(
            f"[bold white]{question}[/bold white]",
            title="[bold cyan]🔬 AI Research Agent[/bold cyan]",
            border_style="cyan",
        )
    )

    console.print("\n[yellow]Agent gestart...[/yellow]\n")

    # Reset search display for new research session
    search_display.recent_searches = []

    # Start Live display for in-place status updates (search + todos)
    tracker.live_display = Live(
        create_combined_status_panel(),
        console=console,
        refresh_per_second=4,
        transient=False,
    )
    tracker.live_display.start()

    try:
        # Stream the agent's work with recursion limit
        # This prevents infinite loops by limiting the number of agent iterations
        # Collect the final result during streaming to avoid running the agent twice
        result = {"messages": []}

        for event in agent.stream(
            {"messages": [{"role": "user", "content": enhanced_question}]},
            {"recursion_limit": recursion_limit},  # Maximum number of agent steps
            stream_mode="updates",
        ):
            # Track agent steps
            if event:
                for node_name, node_data in event.items():
                    # Skip if node_data is None
                    if node_data is None:
                        continue

                    # Track iteration count for early report trigger
                    tracker.iteration_count += 1

                    # Check if we need to trigger early report
                    if should_trigger_early_report():
                        console.print(
                            "\n[bold yellow]>>> Iteratie-limiet nadert - "
                            "rapport schrijven wordt geforceerd[/bold yellow]\n"
                        )
                        tracker.report_triggered = True
                        # Add finalize instruction to the result messages
                        # This will be processed by ensure_report_exists if agent doesn't respond
                        finalize_msg = {
                            "role": "system",
                            "content": create_finalize_instruction(),
                        }
                        result["messages"].append(finalize_msg)

                    # Collect messages from all nodes for the final result
                    if "messages" in node_data:
                        messages = node_data["messages"]
                        # Handle Overwrite objects from deepagents
                        if hasattr(messages, "value"):
                            messages = messages.value
                        # Only extend if it's actually a list
                        if isinstance(messages, list):
                            result["messages"].extend(messages)

                    # Check for TODO updates
                    if "todos" in node_data:
                        new_todos = node_data["todos"]
                        result["todos"] = new_todos
                        # Only display if todos changed
                        if new_todos != tracker.current_todos:
                            tracker.current_todos = new_todos
                            # In-place update via Live display
                            display_todos(new_todos)

                    if node_name == "model":
                        # Model is thinking
                        if "messages" in node_data:
                            msgs = node_data["messages"]
                            # Handle Overwrite objects from deepagents
                            if hasattr(msgs, "value"):
                                msgs = msgs.value
                            if not isinstance(msgs, list):
                                continue
                            for msg in msgs:
                                if hasattr(msg, "content") and msg.content:
                                    # Check if it's a tool call
                                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                                        # Only show tool calls in debug mode
                                        if tracker.debug_mode:
                                            for tool_call in msg.tool_calls:
                                                tool_name = tool_call.get(
                                                    "name", "unknown"
                                                )
                                                # Don't show write_todos tool calls (we show the result instead)
                                                if tool_name != "write_todos":
                                                    console.print(
                                                        f"\n[bold magenta]🛠️  Tool aangeroepen:[/bold magenta] {tool_name}"
                                                    )
                                    # Check for text content (only in debug mode)
                                    elif tracker.debug_mode and (
                                        isinstance(msg.content, str)
                                        and msg.content.strip()
                                    ):
                                        # Don't print system messages
                                        if not msg.content.startswith("You are"):
                                            console.print(
                                                "\n[bold yellow]💭 Agent denkt...[/bold yellow]"
                                            )
                                            # Show preview of thinking (first 150 chars)
                                            preview = (
                                                msg.content[:150] + "..."
                                                if len(msg.content) > 150
                                                else msg.content
                                            )
                                            console.print(f"[dim]{preview}[/dim]")

                    elif "research-agent" in node_name or "critique-agent" in node_name:
                        # Only show sub-agent activity in debug mode
                        if tracker.debug_mode:
                            agent_type = (
                                "Research" if "research" in node_name else "Critique"
                            )
                            console.print(
                                f"\n[bold blue]🤖 {agent_type} Sub-agent actief[/bold blue]"
                            )

        # Stop Live display before final output
        if tracker.live_display is not None:
            tracker.live_display.stop()
            tracker.live_display = None

        # ==================================================================
        # DETERMINISTIC GUARANTEE: Ensure report exists after agent finishes
        # ==================================================================
        ensure_report_exists(question, result, partial=False)

        # Calculate duration
        end_time = time.time()
        duration_seconds = end_time - start_time

        # Format duration nicely
        hours, remainder = divmod(int(duration_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            duration_str = f"{hours}u {minutes}m {seconds}s"
        elif minutes > 0:
            duration_str = f"{minutes}m {seconds}s"
        else:
            duration_str = f"{seconds}s"

        # Print summary statistics
        console.print("\n")
        console.print(Rule("[bold green]Klaar![/bold green]"))

        stats_table = Table(show_header=False, box=None, padding=(0, 2))
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow")

        stats_table.add_row("⏱️  Duur", duration_str)
        stats_table.add_row("🔍 Aantal zoekopdrachten", str(tracker.searches_count))

        # Add cache statistics
        if tracker.cache_hits > 0:
            cache_percentage = (
                (tracker.cache_hits / tracker.searches_count * 100)
                if tracker.searches_count > 0
                else 0
            )
            stats_table.add_row(
                "💾 Cache hits", f"{tracker.cache_hits} ({cache_percentage:.0f}%)"
            )
            api_calls_saved = tracker.cache_hits
            stats_table.add_row("✨ API calls bespaard", str(api_calls_saved))

        stats_table.add_row("💬 Aantal berichten", str(len(result.get("messages", []))))
        stats_table.add_row(
            "🔄 Iteraties gebruikt", f"{tracker.iteration_count}/{recursion_limit}"
        )
        if tracker.report_triggered:
            stats_table.add_row("⚡ Early report trigger", "Ja (limiet naderde)")

        # Add provider usage statistics
        if search_tool and search_tool.provider_usage:
            provider_stats = ", ".join(
                [
                    f"{name}: {count}"
                    for name, count in search_tool.provider_usage.items()
                ]
            )
            stats_table.add_row("🌐 Gebruikte providers", provider_stats)

        console.print(
            Panel(stats_table, title="[bold]Statistieken[/bold]", border_style="green")
        )

        # Show newly created files
        new_files = set(os.listdir(".")) - existing_files
        if new_files:
            console.print("\n[bold]Nieuwe files aangemaakt:[/bold]")
            for file in sorted(new_files):
                if not file.startswith("."):  # Skip hidden files
                    console.print(f"  • [green]{file}[/green]")

        # Rename report to question-based filename
        final_filename = rename_final_report(question)

        # Check if final report was created
        if final_filename and os.path.exists(final_filename):
            console.print(
                f"\n[bold green]✓ Rapport opgeslagen als:[/bold green] [link=file://{final_filename}]{final_filename}[/link]"
            )
        else:
            console.print(
                "\n[bold red]⚠️  WAARSCHUWING: Geen rapport gevonden![/bold red]"
            )
            console.print(
                "[yellow]De agent heeft het onderzoek gedaan maar geen rapport geschreven.[/yellow]"
            )
            console.print("[yellow]Dit kan betekenen:[/yellow]")
            console.print(
                "  [dim]• Recursion limit bereikt voordat rapport werd geschreven[/dim]"
            )
            console.print("  [dim]• Agent heeft file write permission issues[/dim]")
            console.print(
                "  [dim]• Bug in agent logic - TODOs gemarkeerd als complete zonder daadwerkelijk werk[/dim]"
            )

        return result

    except KeyboardInterrupt:
        # Stop Live display
        if tracker.live_display is not None:
            tracker.live_display.stop()
            tracker.live_display = None
        console.print(
            "\n\n[bold red]⚠️  Onderzoek onderbroken door gebruiker[/bold red]"
        )
        # Still try to salvage research into a report
        ensure_report_exists(question, None, partial=True)
        final_filename = rename_final_report(question)
        if final_filename:
            console.print(f"[dim]Partial report saved as: {final_filename}[/dim]")
        return None
    except Exception as e:
        # Stop Live display
        if tracker.live_display is not None:
            tracker.live_display.stop()
            tracker.live_display = None
        # Check for recursion limit error
        if "GraphRecursionError" in str(type(e).__name__) or "Recursion limit" in str(
            e
        ):
            console.print("\n\n[bold red]❌ Recursion limit bereikt[/bold red]")
            console.print(
                f"[yellow]De agent heeft het maximum aantal iteraties ({recursion_limit}) bereikt.[/yellow]"
            )
            console.print("[yellow]Dit kan betekenen:[/yellow]")
            console.print(
                "  [dim]• Het onderzoek is te complex voor de huidige limiet[/dim]"
            )
            console.print("  [dim]• De sub-agents hebben te veel iteraties nodig[/dim]")
            console.print("  [dim]• Er is mogelijk een oneindige loop[/dim]")
            console.print(
                "\n[cyan]💡 Tip:[/cyan] Probeer het opnieuw met een hogere recursion limit (bijv. 300-500)"
            )
            # Still try to salvage research into a report
            ensure_report_exists(question, None, partial=True)
            final_filename = rename_final_report(question)
            if final_filename:
                console.print(f"[dim]Partial report saved as: {final_filename}[/dim]")
            return None
        else:
            console.print(f"\n\n[bold red]❌ Fout opgetreden:[/bold red] {str(e)}")
            raise


def parse_args():
    """Parse command-line arguments for non-interactive mode."""
    parser = argparse.ArgumentParser(
        description="AI Research Agent - Deep research powered by Claude & DeepAgents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python research.py                           # Interactive mode
  python research.py "Your question here"      # Quick mode with question
  python research.py -d "Your question"        # Deep research mode
  python research.py -d -i 300 "Your question" # Deep mode with 300 iterations
        """,
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Research question (if not provided, runs in interactive mode)",
    )
    parser.add_argument(
        "-d",
        "--deep",
        action="store_true",
        help="Use deep research mode (default: quick mode)",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=200,
        help="Max iterations for deep research (default: 200, range: 50-500)",
    )
    parser.add_argument(
        "-p",
        "--provider",
        choices=["tavily", "multi-search", "auto"],
        default="auto",
        help="Search provider (default: auto)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    return parser.parse_args()


def run_interactive():
    """Run in interactive mode with prompts."""
    # Print welcome banner
    console.print("\n")
    console.print(
        Panel.fit(
            "[bold cyan]AI Research Agent[/bold cyan]\n"
            "[dim]Powered by Claude & DeepAgents[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    console.print(
        "\n[bold yellow]Welkom![/bold yellow] Deze AI research agent kan onderzoek doen naar je vraag.\n"
    )

    # Research Mode Selection
    console.print("[bold]Kies een research mode:[/bold]\n")
    console.print(
        "  [cyan]1.[/cyan] Quick Research   [dim](1-3 min, direct LLM, 3-5 searches)[/dim]"
    )
    console.print(
        "                      [dim]→ Geschikt voor: feiten, overzichten, snelle antwoorden[/dim]"
    )
    console.print(
        "  [cyan]2.[/cyan] Deep Research    [dim](10-30 min, agentic, 50-200 searches)[/dim]"
    )
    console.print(
        "                      [dim]→ Geschikt voor: complexe analyses, diepgaand onderzoek[/dim]"
    )

    mode_choice = Prompt.ask(
        "\n[bold cyan]Research mode[/bold cyan]", choices=["1", "2"], default="1"
    )

    is_quick_mode = mode_choice == "1"

    if is_quick_mode:
        console.print(
            "\n[green]✓[/green] Quick Research mode geselecteerd (snel & efficiënt)\n"
        )
    else:
        console.print(
            "\n[green]✓[/green] Deep Research mode geselecteerd (diepgaand & grondig)\n"
        )

    # Recursion limit configuration (only for deep research)
    recursion_limit = 200
    if not is_quick_mode:
        console.print("[bold]Agent configuratie:[/bold]")
        console.print(
            "[dim]Let op: Het recursion limit wordt gedeeld tussen de hoofd-agent en sub-agents.[/dim]"
        )
        console.print(
            "[dim]Voor complexe onderzoeken zijn vaak 150-300 iteraties nodig.[/dim]\n"
        )

        recursion_limit_choice = Prompt.ask(
            "[cyan]Maximaal aantal agent iteraties[/cyan] [dim](voorkomt oneindige loops)[/dim]",
            default="200",
        )
        try:
            recursion_limit = int(recursion_limit_choice)
            if recursion_limit < 50:
                console.print(
                    "[yellow]⚠️  Minimum 50 iteraties aanbevolen voor sub-agents. Instellen op 50.[/yellow]"
                )
                recursion_limit = 50
            elif recursion_limit > 500:
                console.print("[yellow]Maximum 500 iteraties ingesteld[/yellow]")
                recursion_limit = 500
        except ValueError:
            console.print("[yellow]Ongeldige invoer, gebruik standaard (200)[/yellow]")
            recursion_limit = 200

        console.print(f"[dim]Recursion limit: {recursion_limit}[/dim]\n")

    # Provider selection (only for Deep Research mode)
    global search_tool
    if is_quick_mode:
        # Quick mode: automatically use Multi-Search (free tier, good enough for quick queries)
        selected_provider = "multi-search"
        search_tool = HybridSearchTool(provider=selected_provider)
        console.print(
            "[dim]Using Multi-Search API (gratis tier) for quick research[/dim]\n"
        )
    else:
        # Deep mode: let user choose provider
        console.print("[bold]Kies een search provider:[/bold]\n")
        console.print(
            "  [cyan]1.[/cyan] Tavily          [dim](betaald, hoogste kwaliteit, AI-optimized)[/dim]"
        )
        console.print(
            "  [cyan]2.[/cyan] Multi-Search   [dim](gratis tier, auto-fallback, meerdere providers)[/dim]"
        )
        console.print(
            "  [cyan]3.[/cyan] Auto           [dim](slim kiezen: Multi-Search eerst, Tavily als fallback)[/dim]"
        )

        provider_choice = Prompt.ask(
            "\n[bold cyan]Provider[/bold cyan]", choices=["1", "2", "3"], default="2"
        )

        # Map choice to provider
        provider_map = {"1": "tavily", "2": "multi-search", "3": "auto"}
        selected_provider = provider_map[provider_choice]

        # Initialize search tool with selected provider
        search_tool = HybridSearchTool(provider=selected_provider)

        # Show confirmation
        provider_names = {
            "tavily": "Tavily",
            "multi-search": "Multi-Search API (gratis tier)",
            "auto": "Auto (hybrid modus)",
        }
        console.print(
            f"\n[green]✓[/green] {provider_names[selected_provider]} geactiveerd\n"
        )

    # Check for debug mode via environment variable
    tracker.debug_mode = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")

    # Cache management menu (optional, for dev mode)
    debug_indicator = " [dim](DEBUG AAN)[/dim]" if tracker.debug_mode else ""
    console.print(
        f"[dim]Dev tools: [c] Cache stats, [x] Clear cache, [d] Debug toggle{debug_indicator}[/dim]"
    )
    utility_choice = Prompt.ask(
        "\n[bold cyan]Doorgaan of dev tool gebruiken?[/bold cyan]",
        choices=["go", "c", "x", "d"],
        default="go",
    )

    if utility_choice == "c":
        # Show cache statistics
        search_tool.display_cache_stats()
        console.print()
    elif utility_choice == "x":
        # Clear cache
        if (
            Prompt.ask(
                "[yellow]Weet je zeker dat je de cache wilt wissen?[/yellow]",
                choices=["ja", "nee"],
                default="nee",
            )
            == "ja"
        ):
            search_tool.clear_cache()
        console.print()
    elif utility_choice == "d":
        # Toggle debug mode
        tracker.debug_mode = not tracker.debug_mode
        status = "[green]AAN[/green]" if tracker.debug_mode else "[red]UIT[/red]"
        console.print(f"\n[bold]Debug mode:[/bold] {status}")
        console.print(
            "[dim]In debug mode worden tool calls en agent activiteit getoond.[/dim]\n"
        )

    # Get question from user
    question = Prompt.ask(
        "[bold cyan]Wat wil je onderzoeken?[/bold cyan]",
        default="What are the latest advancements in Explainable AI as of 2025?",
    )

    # Confirm before starting
    console.print(f"\n[dim]Je vraag: {question}[/dim]")
    console.print(
        f"[dim]Mode: {'Quick Research' if is_quick_mode else 'Deep Research'}[/dim]"
    )

    if (
        Prompt.ask(
            "\n[bold]Start onderzoek?[/bold]", choices=["ja", "nee"], default="ja"
        )
        == "ja"
    ):
        if is_quick_mode:
            run_quick_research(question, max_searches=5)
        else:
            run_research(question, recursion_limit=recursion_limit)
    else:
        console.print("\n[yellow]Onderzoek geannuleerd.[/yellow]\n")


def run_cli(args):
    """Run in CLI mode with command-line arguments."""
    global search_tool

    # Validate iterations
    iterations = max(50, min(500, args.iterations))

    # Initialize search tool
    search_tool = HybridSearchTool(provider=args.provider)

    # Set debug mode
    tracker.debug_mode = args.debug or os.getenv("DEBUG", "").lower() in (
        "1",
        "true",
        "yes",
    )

    # Print header
    mode = "Deep Research" if args.deep else "Quick Research"
    console.print("\n")
    console.print(
        Panel.fit(
            f"[bold cyan]AI Research Agent[/bold cyan]\n"
            f"[dim]{mode} | Provider: {args.provider} | Iterations: {iterations}[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    console.print(f"\n[bold]Vraag:[/bold] {args.question}\n")

    # Run research
    if args.deep:
        run_research(args.question, recursion_limit=iterations)
    else:
        run_quick_research(args.question, max_searches=5)


if __name__ == "__main__":
    args = parse_args()

    if args.question:
        # CLI mode - run with provided question
        run_cli(args)
    else:
        # Interactive mode
        run_interactive()
