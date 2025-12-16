"""Command-line interface for AI Researcher."""

import argparse
import os

from rich.panel import Panel
from rich.prompt import Prompt

from .ui.console import console
from .tracking import AgentTracker
from .search import HybridSearchTool, SearchStatusDisplay
from .runners import run_quick_research, run_research
from .templates import get_template_info


def display_templates():
    """Display available templates in a nice format."""
    from rich.table import Table

    templates = get_template_info()

    console.print("\n")
    console.print(
        Panel.fit(
            "[bold cyan]Available Report Templates[/bold cyan]",
            border_style="cyan",
        )
    )

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Template", style="green")
    table.add_column("Description")

    for t in templates:
        table.add_row(t["name"], t["description"])

    console.print(table)
    console.print(
        '\n[dim]Usage: python research.py -d -t <template> "Your question"[/dim]'
    )
    console.print(
        '[dim]Example: python research.py -d -t swot "Analyse SWOT voor AI in healthcare"[/dim]\n'
    )


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
    parser.add_argument(
        "-t",
        "--template",
        type=str,
        default=None,
        help="Report template to use (e.g., swot, comparison, market)",
    )
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List available report templates and exit",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start web interface instead of CLI",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for web interface (default: 8000)",
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
        "                      [dim]-> Geschikt voor: feiten, overzichten, snelle antwoorden[/dim]"
    )
    console.print(
        "  [cyan]2.[/cyan] Deep Research    [dim](10-30 min, agentic, 50-200 searches)[/dim]"
    )
    console.print(
        "                      [dim]-> Geschikt voor: complexe analyses, diepgaand onderzoek[/dim]"
    )

    mode_choice = Prompt.ask(
        "\n[bold cyan]Research mode[/bold cyan]", choices=["1", "2"], default="1"
    )

    is_quick_mode = mode_choice == "1"

    if is_quick_mode:
        console.print(
            "\n[green][/green] Quick Research mode geselecteerd (snel & efficient)\n"
        )
    else:
        console.print(
            "\n[green][/green] Deep Research mode geselecteerd (diepgaand & grondig)\n"
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
                    "[yellow]Minimum 50 iteraties aanbevolen voor sub-agents. Instellen op 50.[/yellow]"
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
            f"\n[green][/green] {provider_names[selected_provider]} geactiveerd\n"
        )

    # Template selection (only for Deep Research mode)
    selected_template = None
    if not is_quick_mode:
        templates = get_template_info()
        console.print("[bold]Kies een rapport template:[/bold]\n")

        # Build choice options
        template_choices = ["1"]  # Default is always option 1
        console.print(
            "  [cyan]1.[/cyan] Standaard     [dim](flexibele structuur, agent bepaalt secties)[/dim]"
        )

        for i, t in enumerate(templates, 2):
            if t["name"] != "default":
                template_choices.append(str(i))
                console.print(
                    f"  [cyan]{i}.[/cyan] {t['name'].capitalize():12} [dim]({t['description']})[/dim]"
                )

        template_choice = Prompt.ask(
            "\n[bold cyan]Template[/bold cyan]", choices=template_choices, default="1"
        )

        # Map choice to template name
        if template_choice != "1":
            idx = int(template_choice) - 2
            non_default = [t for t in templates if t["name"] != "default"]
            if 0 <= idx < len(non_default):
                selected_template = non_default[idx]["name"]
                console.print(
                    f"\n[green][/green] Template '{selected_template}' geselecteerd\n"
                )
        else:
            console.print("\n[green][/green] Standaard template geselecteerd\n")

    # Create tracker and search display
    tracker = AgentTracker()
    search_display = SearchStatusDisplay()

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
            run_quick_research(
                question,
                max_searches=5,
                tracker=tracker,
                search_tool=search_tool,
                search_display=search_display,
            )
        else:
            run_research(
                question,
                recursion_limit=recursion_limit,
                tracker=tracker,
                search_tool=search_tool,
                search_display=search_display,
                template=selected_template,
            )
    else:
        console.print("\n[yellow]Onderzoek geannuleerd.[/yellow]\n")


def run_cli(args):
    """Run in CLI mode with command-line arguments."""
    # Validate iterations
    iterations = max(50, min(500, args.iterations))

    # Initialize dependencies
    search_tool = HybridSearchTool(provider=args.provider)
    tracker = AgentTracker()
    search_display = SearchStatusDisplay()

    # Set debug mode
    tracker.debug_mode = args.debug or os.getenv("DEBUG", "").lower() in (
        "1",
        "true",
        "yes",
    )

    # Prepare template info for header
    template_info = ""
    if args.template:
        template_info = f" | Template: {args.template}"

    # Print header
    mode = "Deep Research" if args.deep else "Quick Research"
    console.print("\n")
    console.print(
        Panel.fit(
            f"[bold cyan]AI Research Agent[/bold cyan]\n"
            f"[dim]{mode} | Provider: {args.provider} | Iterations: {iterations}{template_info}[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    console.print(f"\n[bold]Vraag:[/bold] {args.question}\n")

    # Run research
    if args.deep:
        run_research(
            args.question,
            recursion_limit=iterations,
            tracker=tracker,
            search_tool=search_tool,
            search_display=search_display,
            template=args.template,
        )
    else:
        if args.template:
            console.print(
                "[yellow]Waarschuwing: Templates worden alleen ondersteund in deep research mode (-d)[/yellow]\n"
            )
        run_quick_research(
            args.question,
            max_searches=5,
            tracker=tracker,
            search_tool=search_tool,
            search_display=search_display,
        )


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    # Handle --list-templates
    if args.list_templates:
        display_templates()
        return

    # Handle --web (start web interface)
    if args.web:
        from .web import run_server

        console.print("\n")
        console.print(
            Panel.fit(
                "[bold cyan]AI Research Agent - Web Interface[/bold cyan]\n"
                f"[dim]Starting server on http://127.0.0.1:{args.port}[/dim]",
                border_style="cyan",
            )
        )
        console.print("\n[yellow]Open je browser en ga naar:[/yellow]")
        console.print(f"[bold green]http://127.0.0.1:{args.port}[/bold green]\n")
        console.print("[dim]Druk op Ctrl+C om de server te stoppen[/dim]\n")

        run_server(host="127.0.0.1", port=args.port)
        return

    if args.question:
        # CLI mode - run with provided question
        run_cli(args)
    else:
        # Interactive mode
        run_interactive()


if __name__ == "__main__":
    main()
