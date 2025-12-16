"""
AI Research Agent - Deep research powered by Claude & DeepAgents

This module provides backwards-compatible imports from the ai_researcher package.
For new code, prefer importing directly from ai_researcher submodules.
"""
# ruff: noqa: E402 (imports after dotenv.load_dotenv is intentional)

import dotenv

dotenv.load_dotenv()

# Re-export all public API items from the package
# ruff: noqa: F401 (unused imports are re-exports)
from ai_researcher import (
    # Config
    RESEARCH_FOLDER,
    REPORT_RESERVED_ITERATIONS,
    REPORT_TRIGGER_THRESHOLD,
    ensure_research_folder,
    suppress_output,
    # Prompts
    PROMPTS_FOLDER,
    load_prompt,
    # Tracking
    ANTHROPIC_PRICING,
    calculate_cost,
    AgentTracker,
    # Search
    HybridSearchTool,
    SearchStatusDisplay,
    MAX_SEARCH_HISTORY,
    # UI
    console,
    create_todo_panel,
    create_combined_status_panel,
    display_todos,
    update_search_display,
    # Report
    detect_language,
    generate_report_filename,
    extract_research_from_messages,
    create_emergency_report,
    refine_emergency_report_with_llm,
    ensure_report_exists,
    rename_final_report,
    finalize_report,
    postprocess_report,
    _fix_report_title,
    _fix_report_date,
    _fix_sources_section,
    _fix_inline_references,
    _extract_title_from_url,
    # Runners
    should_trigger_early_report as _should_trigger_early_report,
    create_finalize_instruction,
    run_quick_research as _run_quick_research,
    run_research as _run_research,
    research_instructions,
    # CLI
    parse_args,
    run_interactive,
    run_cli,
    main,
)

# Global instances for backwards compatibility
tracker = AgentTracker()
search_tool = None  # Will be initialized at runtime
search_display = SearchStatusDisplay()

# Quick research prompt (loaded for backwards compatibility)
quick_research_prompt = load_prompt("quick_research")

# Create agent for deep research (for tests that import it)
from ai_researcher.runners.deep import create_agent as _create_agent


def _get_agent():
    """Lazy initialization of the agent."""
    global search_tool
    if search_tool is None:
        search_tool = HybridSearchTool(provider="multi-search")
    agent, _ = _create_agent(search_tool, tracker, search_display)
    return agent


# Lazy agent property - only created when accessed
class _AgentProxy:
    _agent = None

    def __getattr__(self, name):
        if self._agent is None:
            self._agent = _get_agent()
        return getattr(self._agent, name)


agent = _AgentProxy()


# Sub-agent configs for backwards compatibility
research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions.",
    "system_prompt": load_prompt("research_agent"),
}

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Used to critique the final report.",
    "system_prompt": load_prompt("critique_agent"),
}


# Legacy function wrappers that use global state
def internet_search(query, max_results=5, topic="general", include_raw_content=False):
    """Run a web search (legacy wrapper using global search_tool)."""
    global search_tool
    if search_tool is None:
        search_tool = HybridSearchTool(provider="multi-search")

    tracker.searches_count += 1
    search_num = tracker.searches_count

    search_docs = search_tool.search(query, max_results=max_results, topic=topic)

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

    search_display.add_search(
        search_num=search_num,
        query=query,
        results_count=results_count,
        provider=provider_name,
        cached=cache_hit,
    )

    update_search_display(tracker, search_display)

    return search_docs


def write_file(filename, content):
    """Write content to a file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        console.print(f"[bold green]Wrote file:[/bold green] {filename}")
        return f"Successfully wrote {len(content)} characters to {filename}"
    except Exception as e:
        console.print(f"[bold red]Error writing {filename}:[/bold red] {e}")
        return f"Error: {e}"


def track_file_operation(operation, filename):
    """Log file operations for debugging."""
    tracker.file_operations.append({"operation": operation, "file": filename})
    console.print(f"[bold green]File {operation}:[/bold green] {filename}")


# Wrapper functions for backwards compatibility (use global state)
def should_trigger_early_report():
    """Backwards-compatible wrapper using global tracker."""
    return _should_trigger_early_report(tracker)


def run_quick_research(question, max_searches=5):
    """Backwards-compatible wrapper using global state."""
    global search_tool
    if search_tool is None:
        search_tool = HybridSearchTool(provider="multi-search")
    return _run_quick_research(
        question, max_searches, tracker, search_tool, search_display
    )


def run_research(question, recursion_limit=100):
    """Backwards-compatible wrapper using global state."""
    global search_tool
    if search_tool is None:
        search_tool = HybridSearchTool(provider="multi-search")
    return _run_research(
        question, recursion_limit, tracker, search_tool, search_display
    )


if __name__ == "__main__":
    main()
