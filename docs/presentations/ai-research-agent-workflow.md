---
marp: true
theme: default
paginate: true
backgroundColor: #1a1a2e
color: #eee
style: |
  section {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  h1 { color: #00d4ff; font-size: 2.2em; }
  h2 { color: #00d4ff; font-size: 1.6em; }
  code { background: #16213e; padding: 2px 8px; border-radius: 4px; }
  ul { font-size: 0.95em; }
  li { margin: 0.4em 0; }
  .columns { display: flex; gap: 2em; }
  .col { flex: 1; }
  strong { color: #00d4ff; }
  img[alt~="center"] { display: block; margin: 0 auto; }
  mermaid { background: transparent; }
---

# AI Research Agent
### Autonomous Deep Research met Claude

**Van vraag naar rapport in 6 stappen**

---

# Architectuur Overview

```
┌─────────────────────────────────────────┐
│           MAIN ORCHESTRATOR             │
│         (Planning & Coördinatie)        │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌──────────────┐ ┌──────────────┐
│   RESEARCH   │ │   CRITIQUE   │
│   SUB-AGENT  │ │   SUB-AGENT  │
└──────┬───────┘ └──────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│         HYBRID SEARCH TOOL           │
│  Tavily │ Serper │ Brave │ + Cache   │
└──────────────────────────────────────┘
```

---

# Stap 1: Planning

**Input:** Onderzoeksvraag van gebruiker

**Main Agent maakt TODO-lijst:**

- Identificeer kernconcepten
- Bepaal zoekstrategie
- Plan deelvragen
- Definieer kwaliteitscriteria

```
○ Zoek naar recente ontwikkelingen
○ Analyseer academische bronnen
○ Vergelijk praktijkvoorbeelden
○ Schrijf synthese
```

---

# Stap 2: Research Sub-Agent

**Doel:** Diepgaand onderzoek uitvoeren

De Research Agent:
1. **Zoekt** via HybridSearchTool
2. **Leest** relevante webpagina's
3. **Extraheert** kernpunten
4. **Categoriseert** bevindingen

```
▶ Zoeken naar "AI agents 2025"
  → 8 resultaten gevonden
  → Pagina 1, 3, 5 relevant
```

---

# Stap 3: Zoeken & Selectie

**HybridSearchTool workflow:**

```
Query ──► Cache Check
              │
         Hit? ├── Ja ──► Return cached
              │
              └── Nee ──► API Call
                              │
                         ┌────┴────┐
                         ▼         ▼
                      Tavily    Multi-Search
                         │         │
                         └────┬────┘
                              ▼
                         Cache + Return
```

**Selectiecriteria:** Relevantie, actualiteit, betrouwbaarheid

---

# Stap 4: Concept Rapport

**Research Agent schrijft `final_report.md`:**

```markdown
# Research Report: [Vraag]

## Samenvatting
Kernbevindingen in 3-5 zinnen

## Bevindingen
### Thema 1
- Punt met [bron]
### Thema 2
- Analyse met onderbouwing

## Bronnen
1. [url] - beschrijving
```

---

# Stap 5: Critique Sub-Agent

**Doel:** Kwaliteitscontrole & gaps identificeren

De Critique Agent beoordeelt:

| Aspect | Check |
|--------|-------|
| **Volledigheid** | Alle deelvragen beantwoord? |
| **Bronkwaliteit** | Betrouwbare, recente bronnen? |
| **Consistentie** | Geen tegenstrijdigheden? |
| **Diepgang** | Voldoende detail? |

**Output:** Prioritized verbeterpunten (HIGH/MEDIUM/LOW)

---

# Stap 6: Iteratie & Verbetering

**Feedback loop:**

```
Critique: "HIGH: Mis praktijkvoorbeelden"
    │
    ▼
Main Agent: Update TODO
    │
    ▼
Research Agent: Extra zoekactie
    │
    ▼
Rapport: Sectie toegevoegd
    │
    ▼
Critique: "Rapport voldoet aan criteria" ✓
```

**Max iteraties:** Configureerbaar (default: 200)

---

# Volledige Flow

```
         ┌──────────────────────────────────────┐
         │           USER QUESTION              │
         └──────────────┬───────────────────────┘
                        ▼
         ┌──────────────────────────────────────┐
    ┌───►│         1. PLANNING                  │
    │    └──────────────┬───────────────────────┘
    │                   ▼
    │    ┌──────────────────────────────────────┐
    │    │    2. RESEARCH (zoek + lees)         │
    │    └──────────────┬───────────────────────┘
    │                   ▼
    │    ┌──────────────────────────────────────┐
    │    │    3. CONCEPT RAPPORT                │
    │    └──────────────┬───────────────────────┘
    │                   ▼
    │    ┌──────────────────────────────────────┐
    │    │    4. CRITIQUE                       │◄──┐
    │    └──────────────┬───────────────────────┘   │
    │                   │                           │
    │         Gaps? ────┼─── Ja ────────────────────┘
    │                   │
    │                   └─── Nee
    │                         │
    └─────────────────────────┘
                        ▼
         ┌──────────────────────────────────────┐
         │         FINAL REPORT                 │
         └──────────────────────────────────────┘
```

---

# Key Features

<div class="columns">
<div class="col">

**Intelligent**
- Autonome planning
- Context-aware zoeken
- Kritische evaluatie

**Efficient**
- Thread-safe caching
- 60-90% API reductie
- Parallel processing

</div>
<div class="col">

**Betrouwbaar**
- Bronvermelding
- Fact-checking loop
- Kwaliteitsgarantie

**Flexibel**
- Multi-provider search
- Configureerbare diepte
- PDF export

</div>
</div>

---

# Gebruiken

```bash
# Start research
python research.py

# Kies provider en stel vraag
> Wat zijn de laatste ontwikkelingen in AI agents?

# Export naar PDF
python export_pdf.py
```

**Output:** `research/{vraag}.md` + optioneel PDF

---

# Vragen?

**Repository:** github.com/Joopsnijder/ai-researcher

**Stack:**
- LangChain DeepAgents
- Claude Sonnet 4
- Tavily / Multi-Search API
- Rich Terminal UI
