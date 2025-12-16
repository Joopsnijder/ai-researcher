# 4. Oplossingsstrategie

## 4.1 Technologie Keuzes

| Beslissing | Keuze | Rationale |
|------------|-------|-----------|
| **Agent Framework** | DeepAgents (LangGraph) | Proven framework voor multi-agent orchestratie |
| **LLM Provider** | Anthropic Claude | Beste balans kwaliteit/prijs voor onderzoekstaken |
| **Search Strategy** | Hybrid (multi-provider) | Redundantie en bredere dekking |
| **UI Framework** | Rich | Mooie terminal output, Live updates |
| **Package Layout** | Flat (geen pip install) | Eenvoud, direct uitvoerbaar |

## 4.2 Architectuur Aanpak

### Multi-Agent Patroon

Het systeem gebruikt een supervisor-worker patroon:

1. **Supervisor Agent** - Coördineert onderzoek, delegeert taken
2. **Research Sub-Agent** - Voert zoekacties uit, verzamelt informatie
3. **Critique Sub-Agent** - Beoordeelt volledigheid, identificeert gaps

### Rapport Garantie

Om te garanderen dat er ALTIJD een rapport wordt gegenereerd:

1. **Early Trigger** - Bij 85% iteratielimiet: forceer rapport schrijven
2. **Emergency Report** - Bij timeout/fout: genereer rapport uit verzamelde berichten
3. **Post-processing** - Automatische opschoning na generatie

### Modulariteit

```
research.py (facade)
       │
       ▼
ai_researcher/
├── cli.py          # Entry point
├── config.py       # Centrale configuratie
├── runners/        # Quick/Deep research
├── search/         # Search tools
├── report/         # Rapportgeneratie
├── tracking/       # Metrics & costs
└── ui/             # Terminal display
```

## 4.3 Kwaliteitsstrategieën

| Kwaliteit | Strategie |
|-----------|-----------|
| **Betrouwbaarheid** | Fallback mechanismen, exception handling |
| **Testbaarheid** | Dependency injection, mock-vriendelijke structuur |
| **Onderhoudbaarheid** | Modulaire packages, single responsibility |
| **Transparantie** | Live UI updates, kostentracking |
