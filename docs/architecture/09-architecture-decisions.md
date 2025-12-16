# 9. Architectuurbeslissingen

## ADR-001: Flat Package Layout

### Status
Geaccepteerd

### Context
Het project begon als een enkel `research.py` bestand van 2100+ regels. Refactoring was nodig voor onderhoudbaarheid.

### Beslissing
Gebruik een flat package layout (`ai_researcher/`) zonder pip installatie vereiste.

### Consequenties
- **Positief**: Direct uitvoerbaar, geen setup.py nodig
- **Positief**: Eenvoudige deployment
- **Negatief**: Geen versioning via pip
- **Negatief**: Imports vereisen project root in PYTHONPATH

---

## ADR-002: Backwards-Compatible Facade

### Status
Geaccepteerd

### Context
Bestaande scripts en `langgraph.json` refereren naar `research.py`.

### Beslissing
Behoud `research.py` als thin facade die re-exporteert uit `ai_researcher`.

### Consequenties
- **Positief**: Geen breaking changes voor bestaande gebruikers
- **Positief**: `langgraph.json` blijft werken
- **Negatief**: Dubbele entry points

---

## ADR-003: Report Guarantee Mechanism

### Status
Geaccepteerd

### Context
Agents kunnen timeout krijgen of falen voordat ze een rapport schrijven.

### Beslissing
Implementeer drie-laags garantie:
1. Early trigger bij 85% iteratielimiet
2. Emergency report uit agent berichten
3. Post-processing na elke run

### Consequenties
- **Positief**: 100% garantie op rapport output
- **Positief**: Geen verloren onderzoekswerk
- **Negatief**: Emergency reports zijn minder gestructureerd
- **Negatief**: Complexere code flow

---

## ADR-004: Hybrid Search Strategy

### Status
Geaccepteerd

### Context
Enkele search provider heeft beperkingen (rate limits, coverage).

### Beslissing
Ondersteun meerdere providers (Tavily, Multi-Search) met fallback.

### Consequenties
- **Positief**: Betere dekking en redundantie
- **Positief**: Flexibiliteit in API keuze
- **Negatief**: Meer configuratie nodig
- **Negatief**: Inconsistente resultaat formaten

---

## ADR-005: Rich Terminal UI

### Status
Geaccepteerd

### Context
CLI tools hebben vaak saaie output, gebruikers willen voortgang zien.

### Beslissing
Gebruik Rich library voor:
- Live updating panels
- Colored output
- Progress tracking

### Consequenties
- **Positief**: Professionele UI
- **Positief**: Real-time feedback
- **Negatief**: Terminal compatibility issues mogelijk
- **Negatief**: Extra dependency

---

## ADR-006: In-Memory Caching Only

### Status
Geaccepteerd

### Context
Search queries kunnen duplicate zijn binnen een sessie.

### Beslissing
Implementeer simpele in-memory dict cache, geen persistente opslag.

### Consequenties
- **Positief**: Eenvoudig, geen externe dependencies
- **Positief**: Voorkomt duplicate API calls
- **Negatief**: Cache leeg bij nieuwe sessie
- **Negatief**: Geen cross-sessie optimalisatie

---

## ADR-007: Nederlandse Default Taal

### Status
Geaccepteerd

### Context
Primaire gebruikers zijn Nederlandstalig.

### Beslissing
Default rapport output in het Nederlands, met taaldetectie voor input.

### Consequenties
- **Positief**: Betere UX voor doelgroep
- **Negatief**: Internationale gebruikers moeten accepteren
- **Negatief**: Prompts zijn deels Nederlands

---

## ADR-008: DeepAgents Framework

### Status
Geaccepteerd

### Context
Multi-agent orchestratie vereist framework support.

### Beslissing
Gebruik DeepAgents (gebaseerd op LangGraph) voor agent management.

### Consequenties
- **Positief**: Proven framework
- **Positief**: Built-in state management
- **Negatief**: Vendor lock-in
- **Negatief**: Learning curve

---

## ADR-009: Prompt Templates als Package Data

### Status
Geaccepteerd

### Context
Prompts waren externe bestanden, refactoring naar package vereist toegang.

### Beslissing
Verplaats prompts naar `ai_researcher/prompts/` met `load_prompt()` functie.

### Consequenties
- **Positief**: Prompts bundled met code
- **Positief**: Relatieve paden werken altijd
- **Negatief**: Edits vereisen code changes
