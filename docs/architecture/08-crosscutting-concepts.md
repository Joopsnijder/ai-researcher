# 8. Crosscutting Concepts

## 8.1 Configuratie

### Environment Variables

```bash
# Verplicht
ANTHROPIC_API_KEY=sk-ant-...

# Optioneel - Search providers
TAVILY_API_KEY=tvly-...
MULTI_SEARCH_API_KEY=...

# Optioneel - Debug
DEBUG=true
```

### Constanten (config.py)

```python
RESEARCH_FOLDER = "research"
REPORT_TRIGGER_THRESHOLD = 0.85  # 85% van iteraties
REPORT_RESERVED_ITERATIONS = 5   # Gereserveerd voor rapport schrijven
```

## 8.2 Error Handling

### Strategie

1. **Graceful degradation** - Bij API fouten, probeer alternatieve providers
2. **Fallback reports** - Bij agent falen, genereer noodrapport uit berichten
3. **Logging** - Rich console output voor debugging

### Exception Handling Pattern

```python
try:
    result = api_call()
except Exception as e:
    console.print(f"[red]Error: {e}[/red]")
    # Fallback logic
    result = fallback_value
```

## 8.3 Logging en Monitoring

### Console Output (Rich)

```python
from ai_researcher.ui import console

console.print("[green]Success[/green]")
console.print("[red]Error[/red]")
console.print("[dim]Debug info[/dim]")
```

### Live Display

```python
with Live(panel, console=console) as live:
    # Updates tijdens uitvoering
    live.update(new_panel)
```

### Cost Tracking

```python
tracker.add_token_usage(input_tokens, output_tokens)
total_cost = tracker.get_total_cost()
console.print(f"Kosten: ${total_cost:.4f}")
```

## 8.4 Caching

### Search Cache

```python
class HybridSearchTool:
    cache: dict  # query -> results

    def search(self, query):
        if query in self.cache:
            return self.cache[query]  # Cache hit
        results = self._do_search(query)
        self.cache[query] = results
        return results
```

- In-memory cache per sessie
- Voorkomt duplicate API calls
- Geen persistente cache

## 8.5 Internationalisatie

### Taaldetectie

```python
from ai_researcher.report import detect_language

lang = detect_language("Dit is een Nederlandse tekst")
# Returns: "nl"
```

### Ondersteunde Talen

| Taal | Code | Rapport Template |
|------|------|------------------|
| Nederlands | `nl` | Management Samenvatting, Bevindingen, Conclusie |
| Engels | `en` | Executive Summary, Findings, Conclusion |

## 8.6 Dependency Injection

### Pattern

```python
def run_research(
    question: str,
    tracker: AgentTracker | None = None,
    search_tool: HybridSearchTool | None = None,
):
    # Use provided or create default
    if tracker is None:
        tracker = AgentTracker()
    if search_tool is None:
        search_tool = HybridSearchTool()
```

### Voordelen

- Testbaar met mocks
- Flexibele configuratie
- Geen hidden global state

## 8.7 Backwards Compatibiliteit

### Facade Pattern

`research.py` fungeert als facade:

```python
# research.py
from ai_researcher import (
    run_research,
    run_quick_research,
    tracker,
    search_tool,
    # ... alle public exports
)

if __name__ == "__main__":
    from ai_researcher.cli import main
    main()
```

### Imports

```python
# Beide werken:
from research import run_research
from ai_researcher import run_research
```

## 8.8 Testing

### Test Structuur

```
tests/
├── test_helpers.py           # Utility function tests
├── test_prompt_changes.py    # Prompt loading tests
└── test_report_guarantee.py  # Report generation tests
```

### Mocking

```python
from unittest.mock import Mock, patch

with patch("ai_researcher.runners.deep.create_agent") as mock:
    mock.return_value = (mock_agent, Mock())
    run_research("test question")
```

### Uitvoeren

```bash
pytest tests/ -v
```
