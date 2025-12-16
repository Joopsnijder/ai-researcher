# 12. Glossary

## Termen

| Term | Definitie |
|------|-----------|
| **Agent** | Autonome LLM-gestuurde component die taken uitvoert |
| **Deep Research** | Uitgebreide onderzoeksmodus met meerdere agents en iteraties |
| **Emergency Report** | Automatisch gegenereerd rapport bij agent falen |
| **Facade** | Design pattern: vereenvoudigde interface voor complex systeem |
| **Hybrid Search** | Combinatie van meerdere zoekproviders |
| **Iteration** | Enkele stap in de agent executie loop |
| **LLM** | Large Language Model (bijv. Claude) |
| **Quick Research** | Snelle onderzoeksmodus met enkele LLM call |
| **Recursion Limit** | Maximum aantal iteraties voor agent |
| **Sub-Agent** | Gespecialiseerde agent aangeroepen door supervisor |
| **Supervisor Agent** | Hoofd-agent die onderzoek coördineert |
| **Tracker** | Object dat metrics bijhoudt (iteraties, kosten, tokens) |

## Afkortingen

| Afkorting | Betekenis |
|-----------|-----------|
| **ADR** | Architecture Decision Record |
| **API** | Application Programming Interface |
| **CLI** | Command Line Interface |
| **NL** | Nederlands (taalcode) |
| **EN** | Engels (taalcode) |
| **UI** | User Interface |
| **UX** | User Experience |

## Componenten

| Component | Locatie | Beschrijving |
|-----------|---------|--------------|
| `AgentTracker` | `tracking/agent_tracker.py` | State tracking voor metrics |
| `HybridSearchTool` | `search/tools.py` | Multi-provider zoektool |
| `SearchStatusDisplay` | `search/display.py` | UI voor zoekactiviteit |

## Externe Services

| Service | URL | Gebruik |
|---------|-----|---------|
| Anthropic API | api.anthropic.com | LLM inference (Claude) |
| Tavily Search | api.tavily.com | Web search met AI ranking |
| Multi-Search API | varies | Aggregated search providers |

## Bestanden

| Bestand | Doel |
|---------|------|
| `research.py` | Entry point (facade) |
| `final_report.md` | Tijdelijk rapport (wordt hernoemd) |
| `.env` | Environment variabelen (API keys) |
| `requirements.txt` | Python dependencies |

## Rapport Structuur

| Sectie | Beschrijving |
|--------|--------------|
| **Metadata Header** | YAML frontmatter met datum, vraag, tags |
| **Management Samenvatting** | Executive summary |
| **Inhoudsopgave** | Automatisch gegenereerd |
| **Bevindingen** | Hoofdinhoud met bronvermeldingen |
| **Conclusie** | Samenvatting en aanbevelingen |
| **Bronnen** | Lijst van gebruikte URLs |
