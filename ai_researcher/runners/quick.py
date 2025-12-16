"""Quick research mode - direct LLM without agents."""

import os
import time
from typing import Literal

from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

from ..config import RESEARCH_FOLDER, ensure_research_folder
from ..ui.console import console
from ..report.finalization import ensure_report_exists, finalize_report
from ..prompts import load_prompt


def run_quick_research(
    question: str,
    max_searches: int = 5,
    tracker=None,
    search_tool=None,
    search_display=None,
):
    """
    Quick research using direct LLM calls (no agentic overhead).

    This mode is faster and cheaper than deep research:
    - No agents, no sub-agents, no planning overhead
    - Direct LLM interaction with search tools
    - Suitable for simple questions, facts, overviews
    - Completes in 1-3 minutes vs 10-30 minutes for deep research

    Args:
        question: The research question
        max_searches: Maximum number of web searches (default: 5)
        tracker: AgentTracker instance (optional, creates new if None)
        search_tool: HybridSearchTool instance (optional)
        search_display: SearchStatusDisplay instance (optional)
    """
    from langchain_anthropic import ChatAnthropic

    from ..tracking import AgentTracker
    from ..search import HybridSearchTool, SearchStatusDisplay
    from ..ui.display import update_search_display

    # Initialize dependencies if not provided
    if tracker is None:
        tracker = AgentTracker()
    if search_tool is None:
        search_tool = HybridSearchTool(provider="multi-search")
    if search_display is None:
        search_display = SearchStatusDisplay()

    # Load quick research prompt
    quick_research_prompt = load_prompt("quick_research")

    # Start timing
    start_time = time.time()

    # Reset tracker for this session
    tracker.reset_token_tracking()
    tracker.searches_count = 0
    tracker.cache_hits = 0

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
            title="[bold cyan]AI Quick Research[/bold cyan]",
            border_style="cyan",
        )
    )

    console.print("\n[yellow]Starting quick research (direct LLM mode)...[/yellow]\n")

    # Define internet_search function with access to tracker and search_tool
    def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
    ):
        """Run a web search."""
        tracker.searches_count += 1
        search_num = tracker.searches_count

        # Use the search_tool
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
        update_search_display(tracker, search_display)

        return search_docs

    def write_file(filename: str, content: str):
        """Write content to a file (for Quick Research mode)."""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            console.print(f"[bold green]Wrote file:[/bold green] {filename}")
            return f"Successfully wrote {len(content)} characters to {filename}"
        except Exception as e:
            console.print(f"[bold red]Error writing {filename}:[/bold red] {e}")
            return f"Error: {e}"

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

        console.print("[dim]Analyzing question and planning research...[/dim]\n")

        # Interaction loop with tool calling
        searches_performed = 0
        max_iterations = 20  # Safety limit

        for iteration in range(max_iterations):
            # Call model
            response = model_with_tools.invoke(messages)

            # Track token usage from response
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                tracker.add_token_usage(input_tokens, output_tokens)

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
                    "\n[dim]Finalizing research and creating report...[/dim]\n"
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

        stats_table.add_row("Duur", duration_str)
        stats_table.add_row("Aantal zoekopdrachten", str(searches_performed))

        # Add cache statistics
        if tracker.cache_hits > 0:
            cache_percentage = (
                (tracker.cache_hits / tracker.searches_count * 100)
                if tracker.searches_count > 0
                else 0
            )
            stats_table.add_row(
                "Cache hits", f"{tracker.cache_hits} ({cache_percentage:.0f}%)"
            )
            api_calls_saved = tracker.cache_hits
            stats_table.add_row("API calls bespaard", str(api_calls_saved))

        stats_table.add_row(
            "LLM interactions",
            str(len([m for m in messages if m.get("role") == "assistant"])),
        )
        stats_table.add_row("Mode", "Quick Research (Direct LLM)")

        # Add token usage and cost statistics
        if tracker.total_input_tokens > 0 or tracker.total_output_tokens > 0:
            total_tokens = tracker.total_input_tokens + tracker.total_output_tokens
            stats_table.add_row(
                "Tokens gebruikt",
                f"{total_tokens:,} (in: {tracker.total_input_tokens:,}, out: {tracker.total_output_tokens:,})",
            )
            total_cost = tracker.get_total_cost()
            stats_table.add_row("Geschatte kosten", f"${total_cost:.4f}")

        console.print(
            Panel(stats_table, title="[bold]Statistieken[/bold]", border_style="green")
        )

        # Show newly created files
        new_files = set(os.listdir(".")) - existing_files
        if new_files:
            console.print("\n[bold]Nieuwe files aangemaakt:[/bold]")
            for file in sorted(new_files):
                if not file.startswith("."):
                    console.print(f"  [green]{file}[/green]")

        # Finalize report: rename and post-process
        final_filename = finalize_report(question)

        # Show report location
        if final_filename and os.path.exists(final_filename):
            console.print(
                f"\n[bold green]Rapport opgeslagen als:[/bold green] [link=file://{final_filename}]{final_filename}[/link]"
            )

        return {
            "messages": messages,
            "searches": searches_performed,
            "report_file": final_filename,
        }

    except KeyboardInterrupt:
        console.print("\n\n[bold red]Onderzoek onderbroken door gebruiker[/bold red]")
        ensure_report_exists(question, None, partial=True)
        final_filename = finalize_report(question)
        if final_filename:
            console.print(f"[dim]Partial report saved as: {final_filename}[/dim]")
        return None

    except Exception as e:
        console.print(f"\n\n[bold red]Fout opgetreden:[/bold red] {str(e)}")
        ensure_report_exists(question, None, partial=True)
        final_filename = finalize_report(question)
        if final_filename:
            console.print(f"[dim]Partial report saved as: {final_filename}[/dim]")
        raise
