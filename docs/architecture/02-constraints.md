# 2. Beperkingen

## 2.1 Technische Beperkingen

| Beperking | Achtergrond |
|-----------|-------------|
| **Python 3.10+** | Vereist voor type hints en match statements |
| **API Keys** | Anthropic API key verplicht, optioneel Tavily en Multi-Search |
| **Geen pip install** | Package draait als standalone, geen setup.py/pyproject.toml |
| **Lokale uitvoering** | CLI tool, geen web interface of API |

## 2.2 Organisatorische Beperkingen

| Beperking | Achtergrond |
|-----------|-------------|
| **Nederlandse rapporten** | Standaard output in het Nederlands (detectie aanwezig) |
| **Markdown formaat** | Rapporten worden als .md bestanden opgeslagen |
| **Research folder** | Output altijd in `research/` directory |

## 2.3 Conventies

| Conventie | Beschrijving |
|-----------|--------------|
| **Code style** | Ruff formatter en linter |
| **Testing** | Pytest voor unit tests |
| **Branching** | Feature branches, PR naar main |

## 2.4 Externe Afhankelijkheden

```
deepagents          # LangGraph-based agent framework
tavily-python       # Tavily Search API client
multi-search-api    # Multi-provider search aggregator
python-dotenv       # Environment variable loading
rich                # Terminal UI formatting
anthropic           # Claude API (indirect via deepagents)
```
