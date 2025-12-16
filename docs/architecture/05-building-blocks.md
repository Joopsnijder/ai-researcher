# 5. Building Block View

## 5.1 Package Structuur

```
ai-researcher/
├── research.py              # Entry point (backwards-compatible facade)
├── ai_researcher/           # Main package
│   ├── __init__.py          # Public API exports
│   ├── cli.py               # CLI interface (parse_args, main)
│   ├── config.py            # Constants en configuratie
│   │
│   ├── prompts/             # Prompt templates
│   │   ├── __init__.py      # load_prompt()
│   │   ├── deep_research.txt
│   │   ├── quick_research.txt
│   │   ├── research_agent.txt
│   │   ├── critique_agent.txt
│   │   └── emergency_refinement.txt
│   │
│   ├── tracking/            # Metrics tracking
│   │   ├── __init__.py
│   │   ├── costs.py         # ANTHROPIC_PRICING, calculate_cost()
│   │   └── agent_tracker.py # AgentTracker class
│   │
│   ├── search/              # Search tools
│   │   ├── __init__.py
│   │   ├── tools.py         # HybridSearchTool class
│   │   └── display.py       # SearchStatusDisplay class
│   │
│   ├── ui/                  # Terminal UI
│   │   ├── __init__.py
│   │   ├── console.py       # Rich console instance
│   │   ├── panels.py        # Panel creation functions
│   │   └── display.py       # Live display updates
│   │
│   ├── report/              # Report generation
│   │   ├── __init__.py
│   │   ├── language.py      # detect_language()
│   │   ├── extraction.py    # extract_research_from_messages()
│   │   ├── emergency.py     # Emergency report generation
│   │   ├── postprocessing.py # Title/source/link fixes
│   │   └── finalization.py  # ensure_report_exists(), finalize_report()
│   │
│   └── runners/             # Research execution
│       ├── __init__.py
│       ├── helpers.py       # Utility functions
│       ├── quick.py         # run_quick_research()
│       └── deep.py          # run_research(), agent configuration
│
├── tests/                   # Unit tests
│   ├── test_helpers.py
│   ├── test_prompt_changes.py
│   └── test_report_guarantee.py
│
├── research/                # Output folder (generated reports)
└── docs/                    # Documentation
```

## 5.2 Module Overzicht

### 5.2.1 config.py

Centrale configuratie en constanten.

| Export | Type | Beschrijving |
|--------|------|--------------|
| `RESEARCH_FOLDER` | str | Output directory ("research") |
| `REPORT_TRIGGER_THRESHOLD` | float | 0.85 (85% van iteraties) |
| `REPORT_RESERVED_ITERATIONS` | int | Gereserveerd voor rapport schrijven |

### 5.2.2 cli.py

Command-line interface.

| Functie | Beschrijving |
|---------|--------------|
| `parse_args()` | Argument parsing met argparse |
| `run_interactive()` | Interactieve vraag-modus |
| `run_cli()` | Main CLI flow |
| `main()` | Entry point |

### 5.2.3 prompts/

Prompt templates als package data.

| Bestand | Gebruik |
|---------|---------|
| `deep_research.txt` | Supervisor agent instructies |
| `quick_research.txt` | Quick mode system prompt |
| `research_agent.txt` | Research sub-agent instructies |
| `critique_agent.txt` | Critique sub-agent instructies |
| `emergency_refinement.txt` | LLM refinement voor noodrapport |

### 5.2.4 tracking/

Metrics en kostentracking.

```python
class AgentTracker:
    iteration_count: int      # Huidige iteratie
    recursion_limit: int      # Max iteraties
    report_triggered: bool    # Early report triggered?
    total_input_tokens: int   # Totaal input tokens
    total_output_tokens: int  # Totaal output tokens

    def add_token_usage(input_tokens, output_tokens)
    def get_total_cost() -> float
```

### 5.2.5 search/

Hybrid search implementatie.

```python
class HybridSearchTool:
    provider: str           # "tavily", "multi-search", etc.
    cache: dict            # Query cache voor deduplicatie

    def search(query: str) -> list[dict]

class SearchStatusDisplay:
    recent_searches: list   # Laatste N zoekopdrachten
    max_history: int        # Max aantal te tonen

    def add_search(num, query, results, provider, cached)
```

### 5.2.6 ui/

Rich terminal interface.

| Module | Exports |
|--------|---------|
| `console.py` | `console` (Rich Console instance) |
| `panels.py` | `create_todo_panel()`, `create_combined_status_panel()` |
| `display.py` | `display_todos()`, `update_search_display()` |

### 5.2.7 report/

Rapportgeneratie en post-processing.

| Module | Functies |
|--------|----------|
| `language.py` | `detect_language()` - NL/EN detectie |
| `extraction.py` | `extract_research_from_messages()` |
| `emergency.py` | `create_emergency_report()`, `refine_emergency_report_with_llm()` |
| `postprocessing.py` | Titel, datum, bronnen correcties |
| `finalization.py` | `ensure_report_exists()`, `finalize_report()` |

### 5.2.8 runners/

Research executie modi.

| Module | Functies |
|--------|----------|
| `helpers.py` | `should_trigger_early_report()`, `create_finalize_instruction()` |
| `quick.py` | `run_quick_research()` - Enkele LLM call |
| `deep.py` | `run_research()` - Multi-agent orchestratie |

## 5.3 Dependencies Diagram

```
                    cli.py
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
      runners/     config.py    prompts/
           │
     ┌─────┴─────┐
     ▼           ▼
  search/     report/
     │           │
     └─────┬─────┘
           ▼
      tracking/
           │
           ▼
        ui/
```

Afhankelijkheden vloeien naar beneden - geen circulaire imports.
