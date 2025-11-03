import os
import dotenv
from typing import Literal
import time
from datetime import timedelta

from tavily import TavilyClient
from multi_search_api import SmartSearchTool
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.live import Live
from rich.table import Table
from rich.markdown import Markdown
from rich import print as rprint
from rich.rule import Rule
from rich.text import Text
from rich.prompt import Prompt

from deepagents import create_deep_agent

dotenv.load_dotenv()

# Initialize rich console
console = Console()


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
                enable_cache=False  # Disable cache to avoid thread-safety issues with parallel searches
            )

    def normalize_multi_search_response(self, response):
        """Convert multi-search-api format to Tavily-compatible format"""
        return {
            "query": response.get("query", ""),
            "results": [
                {
                    "title": result.get("title", ""),
                    "content": result.get("snippet", ""),  # snippet -> content
                    "url": result.get("link", ""),         # link -> url
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
            self.provider_usage[provider_name] = self.provider_usage.get(provider_name, 0) + 1
            normalized["_actual_provider"] = provider_name
            return normalized

        elif self.provider == "auto":
            # Try multi-search first (free), fallback to Tavily
            try:
                response = self.multi_search.search(query=query, num_results=max_results)
                normalized = self.normalize_multi_search_response(response)
                provider_name = normalized.get("_provider", "Unknown")
                self.provider_usage[provider_name] = self.provider_usage.get(provider_name, 0) + 1
                normalized["_actual_provider"] = provider_name
                return normalized
            except Exception as e:
                console.print(f"[yellow]Multi-search failed, using Tavily: {e}[/yellow]")
                result = self.tavily.search(query, max_results=max_results, topic=topic)
                self.provider_usage["Tavily (fallback)"] = self.provider_usage.get("Tavily (fallback)", 0) + 1
                result["_actual_provider"] = "Tavily (fallback)"
                return result


# Global search tool (will be initialized after provider selection)
search_tool = None


# Global state to track agent activity
class AgentTracker:
    def __init__(self):
        self.current_step = None
        self.searches_count = 0
        self.messages_count = 0
        self.file_operations = []
        self.current_todos = []

tracker = AgentTracker()


def display_todos(todos):
    """Display TODO list in a nice panel"""
    if not todos:
        return

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

    console.print(Panel(
        todo_table,
        title="[bold cyan]📋 Taken[/bold cyan]",
        border_style="cyan",
        padding=(0, 1)
    ))


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
    if isinstance(search_docs, dict) and 'results' in search_docs:
        results_count = len(search_docs['results'])
        # Truncate query if too long
        display_query = query[:60] + "..." if len(query) > 60 else query
        provider_name = search_docs.get("_actual_provider", "Unknown")
        cache_hit = search_docs.get("_cache_hit", False)
        cache_indicator = " [dim](cache)[/dim]" if cache_hit else ""
        console.print(f"[cyan]🔍 [#{search_num}][/cyan] {display_query} [green]→ {results_count} resultaten[/green] [dim]({provider_name}){cache_indicator}[/dim]")
    else:
        display_query = query[:60] + "..." if len(query) > 60 else query
        console.print(f"[cyan]🔍 [#{search_num}][/cyan] {display_query} [yellow]→ geen resultaten[/yellow]")

    return search_docs


sub_research_prompt = """You are a dedicated researcher. Your job is to conduct research based on the users questions.

Conduct thorough research and then reply to the user with a detailed answer to their question

only your FINAL answer will be passed on to the user. They will have NO knowledge of anything except your final message, so your final report should be your final message!"""

research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
    "system_prompt": sub_research_prompt,
    "tools": [internet_search],
}

sub_critique_prompt = """You are a dedicated editor. You are being tasked to critique a report.

You can find the report at `final_report.md`.

You can find the question/topic for this report at `question.txt`.

The user may ask for specific areas to critique the report in. Respond to the user with a detailed critique of the report. Things that could be improved.

You can use the search tool to search for information, if that will help you critique the report

Do not write to the `final_report.md` yourself.

Things to check:
- Check that each section is appropriately named
- Check that the report is written as you would find in an essay or a textbook - it should be text heavy, do not let it just be a list of bullet points!
- Check that the report is comprehensive. If any paragraphs or sections are short, or missing important details, point it out.
- Check that the article covers key areas of the industry, ensures overall understanding, and does not omit important parts.
- Check that the article deeply analyzes causes, impacts, and trends, providing valuable insights
- Check that the article closely follows the research topic and directly answers questions
- Check that the article has a clear structure, fluent language, and is easy to understand.
"""

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Used to critique the final report. Give this agent some information about how you want it to critique the report.",
    "system_prompt": sub_critique_prompt,
}


# Prompt prefix to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

PLANNING APPROACH:
At the very start of each research task, IMMEDIATELY use the write_todos tool to create a structured plan. Break down the research into clear, actionable steps:

1. Analyze the research question and identify 3-5 key subtopics or research areas
2. Create a TODO for each research area (use research-agent for each)
3. Add a TODO for compiling findings into initial report
4. Add a TODO for critique and verification (use critique-agent)
5. Add a TODO for addressing critique feedback and finalizing report
6. Note what language the final report should be written in

Example TODO structure for "What are the latest advancements in Explainable AI?":
[ ] Research XAI interpretability techniques (2025)
[ ] Research XAI evaluation metrics and benchmarks
[ ] Research XAI application domains and case studies
[ ] Compile initial report with all findings
[ ] Critique report for completeness and accuracy
[ ] Address feedback and finalize report in English

This planning ensures thorough coverage and helps track progress throughout the research process.

The first thing you should do after planning is to write the original user question to `question.txt` so you have a record of it.

Use the research-agent to conduct deep research. It will respond to your questions/topics with a detailed answer.

When you think you enough information to write a final report, write it to `final_report.md`

You can call the critique-agent to get a critique of the final report. After that (if needed) you can do more research and edit the `final_report.md`
You can do this however many times you want until are you satisfied with the result.

Only edit the file once at a time (if you call this tool in parallel, there may be conflicts).

Here are instructions for writing the final report:

<report_instructions>

CRITICAL: Make sure the answer is written in the same language as the human messages! If you make a todo plan - you should note in the plan what language the report should be in so you dont forget!
Note: the language the report should be in is the language the QUESTION is in, not the language/country that the question is ABOUT.

Please create a detailed answer to the overall research brief that:
1. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
2. Includes specific facts and insights from the research
3. References relevant sources using [Title](URL) format
4. Provides a balanced, thorough analysis. Be as comprehensive as possible, and include all information that is relevant to the overall research question. People are using you for deep research and will expect detailed, comprehensive answers.
5. Includes a "Sources" section at the end with all referenced links

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

For each section of the report, do the following:
- Use simple, clear language
- Use ## for section title (Markdown format) for each section of the report
- Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language. 
- Do not say what you are doing in the report. Just write the report without any commentary from yourself.
- Each section should be as long as necessary to deeply answer the question with the information you have gathered. It is expected that sections will be fairly long and verbose. You are writing a deep research report, and users will expect a thorough answer.
- Use bullet points to list out information when appropriate, but by default, write in paragraph form.

REMEMBER:
The brief and research may be in English, but you need to translate this information to the right language when writing the final answer.
Make sure the final answer report is in the SAME language as the human messages in the message history.

Format the report in clear markdown with proper structure and include source references where appropriate.

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Each source should be a separate line item in a list, so that in markdown it is rendered as a list.
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
- Citations are extremely important. Make sure to include these, and pay a lot of attention to getting these right. Users will often use these citations to look into more information.
</Citation Rules>
</report_instructions>

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

# Create the agent
agent = create_deep_agent(
    tools=[internet_search],
    system_prompt=research_instructions,
    subagents=[critique_sub_agent, research_sub_agent],
)


def run_research(question: str, recursion_limit: int = 100):
    """Run the research agent with rich UI"""

    # Start timing
    start_time = time.time()

    # Enhance question with planning reminder for better TODO structure
    enhanced_question = f"""{question}

Remember to start by creating a detailed TODO plan using write_todos before beginning research."""

    # Print header
    console.print("\n")
    console.print(Panel.fit(
        f"[bold white]{question}[/bold white]",
        title="[bold cyan]🔬 AI Research Agent[/bold cyan]",
        border_style="cyan"
    ))

    console.print("\n[yellow]Agent gestart...[/yellow]\n")

    try:
        # Stream the agent's work with recursion limit
        # This prevents infinite loops by limiting the number of agent iterations
        for event in agent.stream(
            {"messages": [{"role": "user", "content": enhanced_question}]},
            {"recursion_limit": recursion_limit},  # Maximum number of agent steps
            stream_mode="updates"
        ):
            # Track agent steps
            if event:
                for node_name, node_data in event.items():
                    # Skip if node_data is None
                    if node_data is None:
                        continue

                    # Check for TODO updates
                    if "todos" in node_data:
                        new_todos = node_data["todos"]
                        # Only display if todos changed
                        if new_todos != tracker.current_todos:
                            tracker.current_todos = new_todos
                            console.print("\n")
                            display_todos(new_todos)
                            console.print()

                    if node_name == "model":
                        # Model is thinking
                        if "messages" in node_data:
                            for msg in node_data["messages"]:
                                if hasattr(msg, "content") and msg.content:
                                    # Check if it's a tool call
                                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                                        for tool_call in msg.tool_calls:
                                            tool_name = tool_call.get('name', 'unknown')
                                            # Don't show write_todos tool calls (we show the result instead)
                                            if tool_name != "write_todos":
                                                console.print(f"\n[bold magenta]🛠️  Tool aangeroepen:[/bold magenta] {tool_name}")
                                    # Check for text content
                                    elif isinstance(msg.content, str) and msg.content.strip():
                                        # Don't print system messages
                                        if not msg.content.startswith("You are"):
                                            console.print("\n[bold yellow]💭 Agent denkt...[/bold yellow]")
                                            # Show preview of thinking (first 150 chars)
                                            preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                                            console.print(f"[dim]{preview}[/dim]")

                    elif "research-agent" in node_name or "critique-agent" in node_name:
                        agent_type = "Research" if "research" in node_name else "Critique"
                        console.print(f"\n[bold blue]🤖 {agent_type} Sub-agent actief[/bold blue]")

        # Get final result (with same recursion limit)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": enhanced_question}]},
            {"recursion_limit": recursion_limit}
        )

        # Calculate duration
        end_time = time.time()
        duration_seconds = end_time - start_time
        duration_td = timedelta(seconds=int(duration_seconds))

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
        stats_table.add_row("💬 Aantal berichten", str(len(result.get("messages", []))))

        # Add provider usage statistics
        if search_tool and search_tool.provider_usage:
            provider_stats = ", ".join([f"{name}: {count}" for name, count in search_tool.provider_usage.items()])
            stats_table.add_row("🌐 Gebruikte providers", provider_stats)

        console.print(Panel(stats_table, title="[bold]Statistieken[/bold]", border_style="green"))

        # Check if final report was created
        if os.path.exists("final_report.md"):
            console.print("\n[bold green]✓ Rapport opgeslagen in:[/bold green] [link=file://final_report.md]final_report.md[/link]")

            # Show preview of report
            with open("final_report.md", "r") as f:
                content = f.read()
                preview = content[:500] + "\n\n[dim]...(zie final_report.md voor volledig rapport)[/dim]" if len(content) > 500 else content

                console.print("\n")
                console.print(Panel(
                    Markdown(preview),
                    title="[bold cyan]📄 Rapport Preview[/bold cyan]",
                    border_style="cyan"
                ))

        return result

    except KeyboardInterrupt:
        console.print("\n\n[bold red]⚠️  Onderzoek onderbroken door gebruiker[/bold red]")
        return None
    except Exception as e:
        # Check for recursion limit error
        if "GraphRecursionError" in str(type(e).__name__) or "Recursion limit" in str(e):
            console.print("\n\n[bold red]❌ Recursion limit bereikt[/bold red]")
            console.print(f"[yellow]De agent heeft het maximum aantal iteraties ({recursion_limit}) bereikt.[/yellow]")
            console.print("[yellow]Dit kan betekenen:[/yellow]")
            console.print("  [dim]• Het onderzoek is te complex voor de huidige limiet[/dim]")
            console.print("  [dim]• De sub-agents hebben te veel iteraties nodig[/dim]")
            console.print("  [dim]• Er is mogelijk een oneindige loop[/dim]")
            console.print("\n[cyan]💡 Tip:[/cyan] Probeer het opnieuw met een hogere recursion limit (bijv. 300-500)")
            return None
        else:
            console.print(f"\n\n[bold red]❌ Fout opgetreden:[/bold red] {str(e)}")
            raise


if __name__ == "__main__":
    # Print welcome banner
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]AI Research Agent[/bold cyan]\n"
        "[dim]Powered by DeepAgents[/dim]",
        border_style="cyan",
        padding=(1, 2)
    ))

    console.print("\n[bold yellow]Welkom![/bold yellow] Deze AI research agent kan diepgaand onderzoek doen naar je vraag.\n")

    # Recursion limit configuration
    console.print("[bold]Agent configuratie:[/bold]")
    console.print("[dim]Let op: Het recursion limit wordt gedeeld tussen de hoofd-agent en sub-agents.[/dim]")
    console.print("[dim]Voor complexe onderzoeken zijn vaak 150-300 iteraties nodig.[/dim]\n")

    recursion_limit_choice = Prompt.ask(
        "[cyan]Maximaal aantal agent iteraties[/cyan] [dim](voorkomt oneindige loops)[/dim]",
        default="200"
    )
    try:
        recursion_limit = int(recursion_limit_choice)
        if recursion_limit < 50:
            console.print("[yellow]⚠️  Minimum 50 iteraties aanbevolen voor sub-agents. Instellen op 50.[/yellow]")
            recursion_limit = 50
        elif recursion_limit > 500:
            console.print("[yellow]Maximum 500 iteraties ingesteld[/yellow]")
            recursion_limit = 500
    except ValueError:
        console.print("[yellow]Ongeldige invoer, gebruik standaard (200)[/yellow]")
        recursion_limit = 200

    console.print(f"[dim]Recursion limit: {recursion_limit}[/dim]\n")

    # Provider selection
    console.print("[bold]Kies een search provider:[/bold]\n")
    console.print("  [cyan]1.[/cyan] Tavily          [dim](betaald, hoogste kwaliteit, AI-optimized)[/dim]")
    console.print("  [cyan]2.[/cyan] Multi-Search   [dim](gratis tier, auto-fallback, meerdere providers)[/dim]")
    console.print("  [cyan]3.[/cyan] Auto           [dim](slim kiezen: Multi-Search eerst, Tavily als fallback)[/dim]")

    provider_choice = Prompt.ask(
        "\n[bold cyan]Provider[/bold cyan]",
        choices=["1", "2", "3"],
        default="2"
    )

    # Map choice to provider
    provider_map = {
        "1": "tavily",
        "2": "multi-search",
        "3": "auto"
    }
    selected_provider = provider_map[provider_choice]

    # Initialize search tool with selected provider
    search_tool = HybridSearchTool(provider=selected_provider)

    # Show confirmation
    provider_names = {
        "tavily": "Tavily",
        "multi-search": "Multi-Search API (gratis tier)",
        "auto": "Auto (hybrid modus)"
    }
    console.print(f"\n[green]✓[/green] {provider_names[selected_provider]} geactiveerd\n")

    # Get question from user
    question = Prompt.ask(
        "[bold cyan]Wat wil je onderzoeken?[/bold cyan]",
        default="What are the latest advancements in Explainable AI as of 2025?"
    )

    # Confirm before starting
    console.print(f"\n[dim]Je vraag: {question}[/dim]")

    if Prompt.ask("\n[bold]Start onderzoek?[/bold]", choices=["ja", "nee"], default="ja") == "ja":
        run_research(question, recursion_limit=recursion_limit)
    else:
        console.print("\n[yellow]Onderzoek geannuleerd.[/yellow]\n")