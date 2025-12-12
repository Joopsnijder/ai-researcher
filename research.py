import glob
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


def ensure_research_folder():
    """Create research folder if it doesn't exist"""
    if not os.path.exists(RESEARCH_FOLDER):
        os.makedirs(RESEARCH_FOLDER)
        console.print(f"[dim]Created {RESEARCH_FOLDER}/ folder[/dim]")


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

    def search(self, query: str, max_results: int = 5, topic: str = "general"):
        """Execute search with selected provider"""

        if self.provider == "tavily":
            result = self.tavily.search(query, max_results=max_results, topic=topic)
            self.provider_usage["Tavily"] = self.provider_usage.get("Tavily", 0) + 1
            result["_actual_provider"] = "Tavily"
            return result

        elif self.provider == "multi-search":
            response = self.multi_search.search(query=query, num_results=max_results)
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


tracker = AgentTracker()


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


def display_todos(todos):
    """Display TODO list - uses in-place update if Live is active"""
    panel = create_todo_panel(todos)
    if panel is None:
        return

    # If Live display is active, update it in-place
    if tracker.live_display is not None:
        tracker.live_display.update(panel)
    else:
        # Fallback: print normally
        console.print(panel)


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

    # Show compact search result on one line
    if isinstance(search_docs, dict) and "results" in search_docs:
        results_count = len(search_docs["results"])
        # Truncate query if too long
        display_query = query[:60] + "..." if len(query) > 60 else query
        provider_name = search_docs.get("_actual_provider", "Unknown")
        cache_hit = search_docs.get("_cache_hit", False)
        if cache_hit:
            tracker.cache_hits += 1
        cache_indicator = " [green]✓ CACHED[/green]" if cache_hit else ""
        console.print(
            f"[cyan]🔍 [#{search_num}][/cyan] {display_query} [green]→ {results_count} resultaten[/green] [dim]({provider_name})[/dim]{cache_indicator}"
        )
    else:
        display_query = query[:60] + "..." if len(query) > 60 else query
        console.print(
            f"[cyan]🔍 [#{search_num}][/cyan] {display_query} [yellow]→ geen resultaten[/yellow]"
        )

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


sub_research_prompt = """You are a dedicated researcher. Your job is to research ONE specific subtopic.

IMPORTANT OUTPUT FORMAT:
- Return ONLY raw research findings (facts, statistics, sources)
- Do NOT write a complete report or add markdown headers like "# ONDERZOEKSRAPPORT" or "## Section"
- Keep your response CONCISE (maximum 1000 words)
- Format as bullet points with inline citations
- Include source URLs where available

Your findings will be integrated into a larger report by the main agent. Do NOT structure it as a standalone report.

Example of GOOD output:
- AI agents have a 95% failure rate in enterprise deployments [1]
- Average implementation cost: $50,000 - $200,000 for enterprise [2]
- Only 25% of organizations achieve expected ROI [3]

Sources:
[1] [MIT Research](https://example.com/mit-research)
[2] [Ampcome 2025](https://ampcome.com/ai-costs)
[3] [BCG Report](https://bcg.com/ai-roi)

IMPORTANT FORMATTING:
- Use standard markdown bullet points (- or *), NOT special characters like •
- Use numbered inline citations [1], [2], [3] in the text
- List sources at the end with markdown links: [1] [Title](URL)

Example of BAD output (DO NOT DO THIS):
# ONDERZOEKSRAPPORT: AI Agent Challenges
## Executive Summary
Dit rapport behandelt...
## Hoofdstuk 1
..."""

research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
    "system_prompt": sub_research_prompt,
    "tools": [internet_search],
}

sub_critique_prompt = """You are a dedicated editor. You are being tasked to critique a report.

You can find the report at `research/final_report.md`.
You can find the question/topic for this report at `question.txt`.

You can use the search tool to search for information, if that will help you critique the report.
Do not write to the `research/final_report.md` yourself.

Things to check:
- Check that each section is appropriately named
- Check that the report is written as you would find in an essay or a textbook - it should be text heavy, do not let it just be a list of bullet points!
- Check that the report is comprehensive. If any paragraphs or sections are short, or missing important details, point it out.
- Check that the article covers key areas of the industry, ensures overall understanding, and does not omit important parts.
- Check that the article deeply analyzes causes, impacts, and trends, providing valuable insights
- Check that the article closely follows the research topic and directly answers questions
- Check that the article has a clear structure, fluent language, and is easy to understand.
- Check that EVERY fact or claim has an inline citation [1], [2], etc.

IMPORTANT: You MUST format your critique using the following priority structure:

## HIGH Priority
Issues that MUST be resolved before the report is acceptable:
- Factual errors or incorrect information
- Missing essential information that is critical to the topic
- Structural problems that make the report hard to understand
- Missing citations for important claims

List each issue as a bullet point with a specific, actionable description.
Maximum 5 items. If no HIGH priority issues, write "None identified."

## MEDIUM Priority
Issues that SHOULD be resolved to improve quality:
- Unclear or confusing formulations
- Sections that need more depth or detail
- Missing context or background information
- Weak or missing source citations

List each issue as a bullet point with a specific, actionable description.
Maximum 5 items. If no MEDIUM priority issues, write "None identified."

## LOW Priority
Minor improvements (these will be IGNORED, but list them for completeness):
- Stylistic improvements
- Minor wording suggestions
- Optional enhancements

List each issue briefly. Maximum 3 items. If none, write "None identified."

CRITICAL FORMATTING RULES:
1. You MUST ALWAYS use the exact format with ## HIGH Priority, ## MEDIUM Priority, and ## LOW Priority headers
2. NEVER deviate from this format, even for severe issues like topic mismatches
3. If the report addresses the wrong topic, list it as the FIRST item under ## HIGH Priority
4. Do NOT use custom headers like "CRITICAL MISMATCH" - everything must fit in HIGH/MEDIUM/LOW
5. Your response must start with "## HIGH Priority" (no preamble or introduction)

Example of correct format for a topic mismatch:
## HIGH Priority
- TOPIC MISMATCH: The report discusses [wrong topic] but question.txt asks about [correct topic]. The entire report needs to be rewritten to address the correct question.

## MEDIUM Priority
None identified.

## LOW Priority
None identified.
"""

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Used to critique the final report. Give this agent some information about how you want it to critique the report.",
    "system_prompt": sub_critique_prompt,
}


# ══════════════════════════════════════════════════════════════
# QUICK RESEARCH MODE (Direct LLM, No Agents)
# Fast research for simple questions without agentic overhead
# ══════════════════════════════════════════════════════════════

quick_research_prompt = """You are an expert researcher conducting QUICK, efficient research.

Your task is to gather information and write a comprehensive report DIRECTLY.
This is NOT deep research with agents - work efficiently and write the report yourself.

WORKFLOW:
1. Analyze the research question carefully
2. Generate 3-5 targeted search queries to gather information
3. Use the internet_search tool to execute searches
4. Review and synthesize the search results
5. Write a comprehensive report to `research/final_report.md` using write_file tool

SEARCH STRATEGY:
- Generate diverse queries covering different aspects
- Focus on authoritative, recent sources
- Use specific terminology related to the topic
- Cover multiple angles (technical, practical, historical, future)

REPORT REQUIREMENTS:
- Write directly to `research/final_report.md` (don't just describe it!)
- Use clear markdown formatting
- Include an overview, key findings, and conclusion
- Cite sources with URLs in a Sources section
- Match the language of the question (English question → English report)

REPORT STRUCTURE:
# [Clear Title Based on Question]

## Overview
[2-3 paragraph introduction setting context]

## Key Findings
[Well-organized sections with subheadings covering main aspects]
[Each finding should reference sources]

## Recent Developments (if applicable)
[Latest updates, trends, or changes]

## Conclusion
[Summary of main points and implications]

## Sources
1. [Title](URL)
2. [Title](URL)
...

IMPORTANT:
- Be thorough but efficient - this is quick research (target: 3-5 searches)
- Synthesize information clearly and concisely
- MUST write to research/final_report.md using write_file tool
- Include proper citations
- Focus on accuracy and relevance
"""


# ══════════════════════════════════════════════════════════════
# DEEP RESEARCH MODE (Agentic with Sub-agents)
# Thorough research with planning, parallel agents, and critique
# ══════════════════════════════════════════════════════════════

# Prompt prefix to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

PLANNING APPROACH:
At the very start of each research task, IMMEDIATELY use the write_todos tool to create a structured plan:

1. Analyze the research question and identify 3-5 key subtopics
2. Create a TODO for each research area (use research-agent for each)
3. Add a TODO for writing initial report
4. Add a TODO for critique (use critique-agent)
5. Add a TODO for follow-up research on HIGH + MEDIUM priority feedback
6. Add a TODO for finalizing the report
7. Note what language the final report should be written in

Example TODO structure:
[ ] Research subtopic A
[ ] Research subtopic B
[ ] Research subtopic C
[ ] Write initial report to research/final_report.md
[ ] Get critique (HIGH/MEDIUM/LOW priorities)
[ ] Follow-up research for HIGH + MEDIUM issues
[ ] Update and finalize report in [language]

The first thing you should do after planning is to write the original user question to `question.txt`.

COMPACT REPORT RULES:
- Target length: 3000-4000 words (DO NOT exceed 5000 words)
- Research sub-agents return RAW FINDINGS (bullet points), not complete reports
- YOU (the main agent) integrate all findings into ONE cohesive report
- Remove redundancy - don't repeat the same information from different sub-agents
- Focus on answering the question directly and concisely
- Use clear, efficient language - avoid filler text and unnecessary elaboration

RESEARCH WORKFLOW:

PHASE 1 - Initial Research:
Use the research-agent to conduct deep research on each subtopic. The research-agent will return RAW FINDINGS (bullet points with sources), NOT a complete report.

After receiving findings from each research-agent:
- Collect and organize the raw findings
- Wait until you have findings from ALL subtopics before writing the report
- DO NOT write partial reports after each sub-agent - collect all findings first

PHASE 2 - Critique:
After initial research is complete, call the critique-agent. The critique will be structured as:
- HIGH Priority: Issues that MUST be fixed (factual errors, missing critical info)
- MEDIUM Priority: Issues that SHOULD be fixed (unclear sections, weak citations)
- LOW Priority: Minor issues (will be IGNORED)

PHASE 3 - Follow-up Research:
Based on the critique, do additional research ONLY for HIGH and MEDIUM priority items:
- Use research-agent to gather more information on the identified gaps
- Ignore LOW priority feedback completely
- Focus specifically on what the critique identified as missing or incorrect

PHASE 4 - Write Final Report:
After collecting all research findings (from PHASE 1) and addressing critique feedback (PHASE 3):
- Write ONE integrated report to research/final_report.md
- Synthesize all raw findings into cohesive prose
- Remove duplicate information that appears in multiple sub-agent findings
- Ensure all claims have inline citations [1], [2], etc.
- Verify the Sources section matches the inline citations
- Stay within the 3000-4000 word limit

CRITICAL REQUIREMENT - FINAL REPORT:
You MUST ALWAYS write a final report to `research/final_report.md` before you finish. This is NOT optional!

IMPORTANT - ONE INTEGRATED REPORT:
- Write ONE cohesive report, not multiple concatenated reports
- DO NOT copy-paste sub-agent responses directly into the report
- Transform raw findings into well-structured prose with proper sections
- Each section header should appear only ONCE in the entire report

IMPORTANT - REPORT CONTENT RULES:
The final report must contain ONLY the polished research findings. NEVER include:
- TODO lists or task status updates
- Internal planning notes or thinking
- References to "write_todos" or task management
- JSON or dictionary structures like {'content': ...}
- The original question repeated as plain text

FORBIDDEN AGENT STATEMENTS (never write these in the report):
- "Now I'll compile..." / "Now I will..."
- "Let me analyze..." / "Let me summarize..."
- "I will now..." / "I'm going to..."
- "Based on my research..." / "In my analysis..."
- "I found that..." / "I discovered..."
- "The research shows..." (use passive: "Research shows...")
- Any first-person references to yourself as the writer
- Meta-commentary about your writing process
- Incomplete sentences or placeholder text like "[to be added]"

The report should read like a professional article written by a human researcher.
Write as if you ARE a subject matter expert, not an AI describing what it found.

Only edit the file once at a time (if you call this tool in parallel, there may be conflicts).

REMINDER: Before you finish your work, you MUST have created research/final_report.md. Do not mark your work as complete until this file exists!

Here are instructions for writing the final report:

<report_instructions>

CRITICAL: Make sure the answer is written in the same language as the human messages! If you make a todo plan - you should note in the plan what language the report should be in so you dont forget!
Note: the language the report should be in is the language the QUESTION is in, not the language/country that the question is ABOUT.

MANDATORY REPORT STRUCTURE:
Every report MUST follow this exact structure:

# [Titel gebaseerd op onderzoeksvraag]

> **Onderzoeksvraag:** [De originele vraag hier]
> **Type:** AI-gegenereerd onderzoeksrapport
> **Bronnen geraadpleegd:** [tel het aantal unieke bronnen] bronnen
> **Datum:** [huidige datum in YYYY-MM-DD format]

## Management Samenvatting

[Write 2-3 paragraphs summarizing the key findings and conclusions. This is the
first thing readers see - make it impactful and concise. NO bullet points here,
only flowing prose. Highlight the most important insights.]

## [Content sections follow...]

[The rest of the report with well-structured sections]

## Bronnen

[Numbered source list with markdown links]

---

Please create a detailed answer to the overall research brief that:
1. ALWAYS starts with the metadata header and Management Samenvatting as shown above
2. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
3. Includes specific facts and insights from the research
4. References relevant sources using inline citations [1], [2], etc.
5. Provides a balanced, thorough analysis
6. Includes a "Bronnen" section at the end with all referenced links

You can structure your report in a number of different ways. Here are some examples:

To answer a question that asks you to compare two things, you might structure your report like this:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

To answer a question that asks you to return a list of things, you might only need a single section which is the entire list.
1/ list of things or table of things
Or, you could choose to make each item in the list a separate section in the report. When asked for lists, you don't need an introduction or conclusion.
1/ item 1
2/ item 2
3/ item 3

To answer a question that asks you to summarize a topic, give a report, or give an overview, you might structure your report like this:
1/ overview of topic
2/ concept 1
3/ concept 2
4/ concept 3
5/ conclusion

If you think you can answer the question with a single section, you can do that too!
1/ answer

REMEMBER: Section is a VERY fluid and loose concept. You can structure your report however you think is best, including in ways that are not listed above!
Make sure that your sections are cohesive, and make sense for the reader.

WRITING STYLE - PROSE, NOT BULLET LISTS:
- Write in PROSE (flowing paragraphs), NOT bullet point lists
- Bullet points are ONLY acceptable for:
  - Short lists of exactly 3-5 items maximum
  - Technical specifications or step-by-step processes
- Default format is ALWAYS paragraphs with clear topic sentences
- Each paragraph should develop ONE idea fully
- Use transition words to connect ideas between paragraphs

BAD EXAMPLE (avoid this):
- Challenge 1: Cost issues affect many organizations
- Challenge 2: Technical complexity is high
- Challenge 3: Skill gaps are common

GOOD EXAMPLE (do this instead):
"The most significant challenge facing AI agent implementations is cost
management. Organizations frequently underestimate the total cost of
ownership, which extends far beyond initial licensing fees. Technical
complexity compounds these cost issues, as integrating agents with
existing systems requires specialized expertise that many teams lack."

For each section of the report:
- Use simple, clear language
- Use ## for section titles (Markdown format)
- Do NOT refer to yourself as the writer - this is a professional report
- Do not describe what you are doing - just write the content directly
- Each section should be comprehensive with flowing paragraphs
- Transform research findings into cohesive narrative prose

REMEMBER:
The brief and research may be in English, but you need to translate this information to the right language when writing the final answer.
Make sure the final answer report is in the SAME language as the human messages in the message history.

Format the report in clear markdown with proper structure and include source references where appropriate.

<Citation Rules>
INLINE CITATIONS ARE MANDATORY! Every fact, statistic, claim, or piece of information from a source MUST have an inline citation.

How to cite inline:
- Use numbered references like [1], [2], [3] directly after the relevant sentence or paragraph
- Example: "LangGraph supports cyclic workflows and state management [1]. CrewAI focuses on role-based agent collaboration [2]."

How to format the Sources section:
- End with ## Bronnen (or ## Sources in English reports)
- Number sources sequentially: [1], [2], [3], etc.
- Format each source as: [number] Title - URL
- Example:
  [1] LangGraph Documentation - https://langchain.com/langgraph
  [2] CrewAI GitHub - https://github.com/crewai/crewai

CRITICAL REQUIREMENTS:
- EVERY paragraph that contains researched information must have at least one citation
- Do NOT write facts without citing where they came from
- If you cannot cite a source for a claim, do not include that claim
- Match inline citation numbers [1], [2] to the Sources list at the end
- Citations show credibility and allow readers to verify information
</Citation Rules>
</report_instructions>

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

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

    # Create emergency report
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

    # Start Live display for in-place todo updates
    tracker.live_display = Live(console=console, refresh_per_second=4, transient=False)
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


if __name__ == "__main__":
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
