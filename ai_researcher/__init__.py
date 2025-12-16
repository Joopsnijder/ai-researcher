"""AI Researcher - Deep research powered by Claude & DeepAgents."""

# Config
from .config import (
    RESEARCH_FOLDER,
    REPORT_RESERVED_ITERATIONS,
    REPORT_TRIGGER_THRESHOLD,
    ensure_research_folder,
    suppress_output,
)

# Prompts
from .prompts import PROMPTS_FOLDER, load_prompt

# Tracking
from .tracking import ANTHROPIC_PRICING, calculate_cost, AgentTracker

# Search
from .search import HybridSearchTool, SearchStatusDisplay, MAX_SEARCH_HISTORY

# UI
from .ui import (
    console,
    create_todo_panel,
    create_combined_status_panel,
    display_todos,
    update_search_display,
)

# Report
from .report import (
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
)

# Runners
from .runners import (
    should_trigger_early_report,
    create_finalize_instruction,
    run_quick_research,
    run_research,
    research_instructions,
)

# CLI
from .cli import parse_args, run_interactive, run_cli, main

__all__ = [
    # Config
    "RESEARCH_FOLDER",
    "REPORT_RESERVED_ITERATIONS",
    "REPORT_TRIGGER_THRESHOLD",
    "ensure_research_folder",
    "suppress_output",
    # Prompts
    "PROMPTS_FOLDER",
    "load_prompt",
    # Tracking
    "ANTHROPIC_PRICING",
    "calculate_cost",
    "AgentTracker",
    # Search
    "HybridSearchTool",
    "SearchStatusDisplay",
    "MAX_SEARCH_HISTORY",
    # UI
    "console",
    "create_todo_panel",
    "create_combined_status_panel",
    "display_todos",
    "update_search_display",
    # Report
    "detect_language",
    "generate_report_filename",
    "extract_research_from_messages",
    "create_emergency_report",
    "refine_emergency_report_with_llm",
    "ensure_report_exists",
    "rename_final_report",
    "finalize_report",
    "postprocess_report",
    "_fix_report_title",
    "_fix_report_date",
    "_fix_sources_section",
    "_fix_inline_references",
    "_extract_title_from_url",
    # Runners
    "should_trigger_early_report",
    "create_finalize_instruction",
    "run_quick_research",
    "run_research",
    "research_instructions",
    # CLI
    "parse_args",
    "run_interactive",
    "run_cli",
    "main",
]
