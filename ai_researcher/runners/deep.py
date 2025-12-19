"""Deep research mode - agentic research with sub-agents."""

import os
import random
import time
from typing import Literal

from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from ..config import RESEARCH_FOLDER, ensure_research_folder
from ..ui.console import console
from ..ui.panels import LiveStatusRenderable
from ..ui.display import display_todos, update_search_display, update_agent_status
from ..report.finalization import ensure_report_exists, finalize_report
from ..report.language import detect_language
from ..prompts import load_prompt
from ..templates import load_template, get_template_prompt, TemplateNotFoundError
from .helpers import should_trigger_early_report, create_finalize_instruction


def get_dynamic_status(tool_name: str | None, tracker) -> str:
    """Generate dynamic status based on current activity.

    Args:
        tool_name: Name of the tool being called, or None if just thinking
        tracker: AgentTracker with current_todos

    Returns:
        Human-readable status string
    """
    # Tool-specific status messages
    tool_statuses = {
        "internet_search": "🔍 Zoeken op internet...",
        "search": "🔍 Zoeken...",
        "write_file": "📝 Bestand schrijven...",
        "read_file": "📖 Bestand lezen...",
        "write_todos": "📋 Taken bijwerken...",
    }

    if tool_name:
        if tool_name in tool_statuses:
            return tool_statuses[tool_name]
        # Generic tool status
        return f"🔧 {tool_name}..."

    # Check current in-progress todo for context
    if tracker.current_todos:
        for todo in tracker.current_todos:
            if todo.get("status") == "in_progress":
                # Use activeForm if available, otherwise content
                active_form = todo.get("activeForm") or todo.get("content", "")
                if active_form:
                    return f"⏳ {active_form}"

    # Rotating default thinking statuses for visual feedback
    thinking_statuses = [
        "🤔 Agent analyseert...",
        "💭 Informatie verwerken...",
        "🧠 Aan het nadenken...",
        "📚 Bronnen evalueren...",
        "🔬 Gegevens onderzoeken...",
        "✨ Inzichten verzamelen...",
        "🎯 Strategie bepalen...",
        "📊 Data analyseren...",
        "🔗 Verbanden leggen...",
        "💡 Conclusies vormen...",
    ]
    return random.choice(thinking_statuses)


# Load prompts
research_instructions = load_prompt("deep_research")


# Global state for agent - will be initialized on first use
_agent_state = {
    "agent": None,
    "tracker": None,
    "search_tool": None,
    "search_display": None,
}


def _create_internet_search():
    """Create internet_search function that uses global state."""

    def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
    ):
        """Run a web search."""
        from ..config import suppress_output

        tracker = _agent_state["tracker"]
        search_tool = _agent_state["search_tool"]
        search_display = _agent_state["search_display"]

        tracker.searches_count += 1
        search_num = tracker.searches_count

        # Use the search_tool - suppress noisy API error messages
        with suppress_output():
            search_docs = search_tool.search(
                query, max_results=max_results, topic=topic
            )

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

    return internet_search


# Create the global internet_search function
_internet_search = _create_internet_search()

# Create sub-agent configs
_research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
    "system_prompt": load_prompt("research_agent"),
    "tools": [_internet_search],
}

_critique_sub_agent = {
    "name": "critique-agent",
    "description": "Used to critique the final report. Give this agent some information about how you want it to critique the report.",
    "system_prompt": load_prompt("critique_agent"),
}

# Create the global agent with FilesystemBackend (default template)
# Use virtual_mode=True so paths like "/research/final_report.md" are resolved
# relative to cwd, not as absolute filesystem paths
_agent = create_deep_agent(
    tools=[_internet_search],
    system_prompt=research_instructions,
    subagents=[_critique_sub_agent, _research_sub_agent],
    backend=FilesystemBackend(virtual_mode=True),
)


def _create_agent_with_template(template_name: str, language: str = "en"):
    """Create an agent with a custom template."""
    # Load the template
    template = load_template(template_name)
    template_prompt = get_template_prompt(template, language)

    # Inject template sections into the research instructions
    # Replace {{TEMPLATE_SECTIONS}} placeholder with actual sections
    customized_instructions = research_instructions.replace(
        "{{TEMPLATE_SECTIONS}}", template_prompt
    )

    # Create a new agent with the customized prompt
    # Use virtual_mode=True so paths are resolved relative to cwd
    return create_deep_agent(
        tools=[_internet_search],
        system_prompt=customized_instructions,
        subagents=[_critique_sub_agent, _research_sub_agent],
        backend=FilesystemBackend(virtual_mode=True),
    )


def create_agent(
    search_tool, tracker, search_display, template: str = None, language: str = "en"
):
    """Initialize global agent state and return the agent."""
    # Update global state that internet_search uses
    _agent_state["tracker"] = tracker
    _agent_state["search_tool"] = search_tool
    _agent_state["search_display"] = search_display

    # Use template-specific agent if template is specified
    if template and template != "default":
        agent = _create_agent_with_template(template, language)
        return agent, _internet_search

    # For default template, remove the placeholder from the prompt
    # and use the global agent
    return _agent, _internet_search


def run_research(
    question: str,
    recursion_limit: int = 100,
    tracker=None,
    search_tool=None,
    search_display=None,
    template: str = None,
):
    """
    Run the research agent with rich UI.

    Args:
        question: The research question
        recursion_limit: Maximum number of agent iterations
        tracker: AgentTracker instance (optional, creates new if None)
        search_tool: HybridSearchTool instance (optional)
        search_display: SearchStatusDisplay instance (optional)
        template: Template name for report structure (optional)
    """
    from ..tracking import AgentTracker
    from ..search import HybridSearchTool, SearchStatusDisplay

    # Initialize dependencies if not provided
    if tracker is None:
        tracker = AgentTracker()
    if search_tool is None:
        search_tool = HybridSearchTool(provider="multi-search")
    if search_display is None:
        search_display = SearchStatusDisplay()

    # Detect language first (needed for template)
    detected_lang = detect_language(question)

    # Load and validate template if specified
    template_info = None
    if template:
        try:
            template_info = load_template(template)
            console.print(
                f"[dim]Template: {template_info.get('name', template)} - {template_info.get('description', '')}[/dim]"
            )
        except TemplateNotFoundError as e:
            console.print(f"[bold red]Template error:[/bold red] {e}")
            return None

    # Create agent with dependencies (and template if specified)
    agent, internet_search = create_agent(
        search_tool, tracker, search_display, template=template, language=detected_lang
    )

    # Start timing (both for final stats and live display)
    start_time = time.time()
    tracker.start_session()

    # Reset tracker for new research session
    tracker.iteration_count = 0
    tracker.recursion_limit = recursion_limit
    tracker.report_triggered = False
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

    # Build language instruction (language was already detected above)
    lang_instruction = (
        "BELANGRIJK: De onderzoeksvraag is in het NEDERLANDS gesteld. "
        "Het eindrapport MOET volledig in het NEDERLANDS geschreven worden. "
        "Alle secties, titels, en inhoud moeten Nederlands zijn."
        if detected_lang == "nl"
        else "IMPORTANT: The research question is in ENGLISH. "
        "The final report MUST be written entirely in ENGLISH."
    )

    # Enhance question with language instruction and planning reminder
    enhanced_question = f"""{question}

{lang_instruction}

Remember to start by creating a detailed TODO plan using write_todos before beginning research.
Note in your TODO plan: Report language = {"Nederlands" if detected_lang == "nl" else "English"}."""

    # Print header
    console.print("\n")
    console.print(
        Panel.fit(
            f"[bold white]{question}[/bold white]",
            title="[bold cyan]AI Research Agent[/bold cyan]",
            border_style="cyan",
        )
    )

    console.print("\n[yellow]Agent gestart...[/yellow]\n")

    # Reset search display for new research session
    search_display.recent_searches = []

    # Start Live display for in-place status updates (agent status, search, todos)
    # Use LiveStatusRenderable so elapsed time updates on each render cycle,
    # even when no agent events are occurring (e.g., during sub-agent work)
    live_renderable = LiveStatusRenderable(search_display, tracker)
    tracker.live_display = Live(
        live_renderable,
        console=console,
        refresh_per_second=4,
        transient=False,
        vertical_overflow="visible",
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

                    # Determine current tool being called (if any)
                    current_tool = None
                    if "messages" in node_data:
                        msgs = node_data["messages"]
                        if hasattr(msgs, "value"):
                            msgs = msgs.value
                        if isinstance(msgs, list):
                            for msg in msgs:
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    # Get the first tool being called
                                    current_tool = msg.tool_calls[0].get("name")
                                    break

                    # Update status based on node type and activity
                    if "research-agent" in node_name:
                        tracker.current_status = "🔬 Research sub-agent actief..."
                    elif "critique-agent" in node_name:
                        tracker.current_status = "📊 Critique sub-agent actief..."
                    else:
                        # Use dynamic status based on tool or current todo
                        tracker.current_status = get_dynamic_status(
                            current_tool, tracker
                        )

                    # Update the live display with current status
                    update_agent_status(tracker, search_display)

                    # Check if we need to trigger early report
                    if should_trigger_early_report(tracker):
                        tracker.current_status = "⚠️ Limiet nadert - rapport afronden..."
                        update_agent_status(tracker, search_display)
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
                            display_todos(tracker, search_display, new_todos)

                            # SAFEGUARD: Detect when all todos are completed but report doesn't exist
                            # This catches the case where agent marks tasks complete without writing
                            if new_todos and all(
                                t.get("status") == "completed" for t in new_todos
                            ):
                                report_path = os.path.join(
                                    RESEARCH_FOLDER, "final_report.md"
                                )
                                if not os.path.exists(report_path):
                                    console.print(
                                        "\n[bold yellow]⚠️ Alle taken zijn voltooid maar "
                                        "final_report.md bestaat nog niet![/bold yellow]"
                                    )
                                    console.print(
                                        "[yellow]De agent zou nu het rapport moeten "
                                        "schrijven...[/yellow]\n"
                                    )
                                    # Inject a reminder message into the result
                                    reminder_msg = {
                                        "role": "system",
                                        "content": (
                                            "CRITICAL REMINDER: All todos are marked completed "
                                            "but research/final_report.md does NOT exist yet! "
                                            "You MUST call write_file to create the report NOW. "
                                            "Do not stop until the file exists."
                                        ),
                                    }
                                    result["messages"].append(reminder_msg)

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
                                # Track token usage from response metadata
                                if (
                                    hasattr(msg, "usage_metadata")
                                    and msg.usage_metadata
                                ):
                                    usage = msg.usage_metadata
                                    input_tokens = usage.get("input_tokens", 0)
                                    output_tokens = usage.get("output_tokens", 0)
                                    tracker.add_token_usage(input_tokens, output_tokens)
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
                                                        f"\n[bold magenta]Tool aangeroepen:[/bold magenta] {tool_name}"
                                                    )
                                    # Check for text content (only in debug mode)
                                    elif tracker.debug_mode and (
                                        isinstance(msg.content, str)
                                        and msg.content.strip()
                                    ):
                                        # Don't print system messages
                                        if not msg.content.startswith("You are"):
                                            console.print(
                                                "\n[bold yellow]Agent denkt...[/bold yellow]"
                                            )
                                            # Show preview of thinking (first 150 chars)
                                            preview = (
                                                msg.content[:150] + "..."
                                                if len(msg.content) > 150
                                                else msg.content
                                            )
                                            console.print(f"[dim]{preview}[/dim]")

                    elif "research-agent" in node_name or "critique-agent" in node_name:
                        # Track token usage from sub-agents
                        if "messages" in node_data:
                            sub_msgs = node_data["messages"]
                            if hasattr(sub_msgs, "value"):
                                sub_msgs = sub_msgs.value
                            if isinstance(sub_msgs, list):
                                for sub_msg in sub_msgs:
                                    if (
                                        hasattr(sub_msg, "usage_metadata")
                                        and sub_msg.usage_metadata
                                    ):
                                        usage = sub_msg.usage_metadata
                                        input_tokens = usage.get("input_tokens", 0)
                                        output_tokens = usage.get("output_tokens", 0)
                                        tracker.add_token_usage(
                                            input_tokens, output_tokens
                                        )
                        # Only show sub-agent activity in debug mode
                        if tracker.debug_mode:
                            agent_type = (
                                "Research" if "research" in node_name else "Critique"
                            )
                            console.print(
                                f"\n[bold blue]{agent_type} Sub-agent actief[/bold blue]"
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

        stats_table.add_row("Duur", duration_str)
        stats_table.add_row("Aantal zoekopdrachten", str(tracker.searches_count))

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

        stats_table.add_row("Aantal berichten", str(len(result.get("messages", []))))
        stats_table.add_row(
            "Iteraties gebruikt", f"{tracker.iteration_count}/{recursion_limit}"
        )
        if tracker.report_triggered:
            stats_table.add_row("Early report trigger", "Ja (limiet naderde)")

        # Add provider usage statistics
        if search_tool and search_tool.provider_usage:
            provider_stats = ", ".join(
                [
                    f"{name}: {count}"
                    for name, count in search_tool.provider_usage.items()
                ]
            )
            stats_table.add_row("Gebruikte providers", provider_stats)

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
                if not file.startswith("."):  # Skip hidden files
                    console.print(f"  [green]{file}[/green]")

        # Finalize report: rename and post-process
        final_filename = finalize_report(question)

        # Check if final report was created
        if final_filename and os.path.exists(final_filename):
            console.print(
                f"\n[bold green]Rapport opgeslagen als:[/bold green] [link=file://{final_filename}]{final_filename}[/link]"
            )
        else:
            console.print("\n[bold red]WAARSCHUWING: Geen rapport gevonden![/bold red]")
            console.print(
                "[yellow]De agent heeft het onderzoek gedaan maar geen rapport geschreven.[/yellow]"
            )
            console.print("[yellow]Dit kan betekenen:[/yellow]")
            console.print(
                "  [dim]Recursion limit bereikt voordat rapport werd geschreven[/dim]"
            )
            console.print("  [dim]Agent heeft file write permission issues[/dim]")
            console.print(
                "  [dim]Bug in agent logic - TODOs gemarkeerd als complete zonder daadwerkelijk werk[/dim]"
            )

        return result

    except KeyboardInterrupt:
        # Stop Live display
        if tracker.live_display is not None:
            tracker.live_display.stop()
            tracker.live_display = None
        console.print("\n\n[bold red]Onderzoek onderbroken door gebruiker[/bold red]")
        # Still try to salvage research into a report
        ensure_report_exists(question, None, partial=True)
        final_filename = finalize_report(question)
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
            console.print("\n\n[bold red]Recursion limit bereikt[/bold red]")
            console.print(
                f"[yellow]De agent heeft het maximum aantal iteraties ({recursion_limit}) bereikt.[/yellow]"
            )
            console.print("[yellow]Dit kan betekenen:[/yellow]")
            console.print(
                "  [dim]Het onderzoek is te complex voor de huidige limiet[/dim]"
            )
            console.print("  [dim]De sub-agents hebben te veel iteraties nodig[/dim]")
            console.print("  [dim]Er is mogelijk een oneindige loop[/dim]")
            console.print(
                "\n[cyan]Tip:[/cyan] Probeer het opnieuw met een hogere recursion limit (bijv. 300-500)"
            )
            # Still try to salvage research into a report
            ensure_report_exists(question, None, partial=True)
            final_filename = finalize_report(question)
            if final_filename:
                console.print(f"[dim]Partial report saved as: {final_filename}[/dim]")
            return None
        else:
            console.print(f"\n\n[bold red]Fout opgetreden:[/bold red] {str(e)}")
            raise
