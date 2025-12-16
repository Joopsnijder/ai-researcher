---
marp: true
theme: default
paginate: true
backgroundColor: #1a1a2e
color: #eaeaea
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }
  h1 {
    color: #00d4ff;
  }
  h2 {
    color: #7b68ee;
  }
  code {
    background-color: #2d2d44;
    color: #00ff88;
  }
  table {
    font-size: 0.8em;
  }
  th {
    background-color: #7b68ee;
    color: white;
  }
  td {
    background-color: #2d2d44;
  }
---

# AI Research Agent
## Tools & capabilities overzicht

---

# Architectuur

```
┌─────────────────────────────────────────┐
│           Main Research Agent           │
│  (planning, synthese, rapportage)       │
├─────────────────────────────────────────┤
│     Sub-agents (parallel execution)     │
│  ┌─────────────┐    ┌────────────────┐  │
│  │ research-   │    │ critique-      │  │
│  │ agent       │    │ agent          │  │
│  └─────────────┘    └────────────────┘  │
└─────────────────────────────────────────┘
```

---

# Directe tools

| Tool | Doel |
|------|------|
| `internet_search` | Web zoeken via Tavily/Multi-Search |
| `write_file` | Bestanden schrijven naar disk |
| `read_file` | Bestanden lezen |
| `edit_file` | Bestaande bestanden aanpassen |
| `write_todos` | TODO-lijst beheren |

---

# internet_search

```python
internet_search(
    query="AI agents 2024",
    max_results=5,
    topic="general",
    include_raw_content=False
)
```

**Providers:**
- Tavily (AI-geoptimaliseerd, betaald)
- Multi-Search API (gratis tier)
- Auto mode (fallback)

---

# Bestandsoperaties

```python
# Schrijven
write_file("research/final_report.md", content)

# Lezen
read_file("question.txt")

# Bewerken
edit_file("report.md", "oude tekst", "nieuwe tekst")
```

**Output folder:** `research/`

---

# TODO management

```python
write_todos([
    {"task": "Research subtopic A", "status": "pending"},
    {"task": "Research subtopic B", "status": "in_progress"},
    {"task": "Write report", "status": "completed"}
])
```

Live weergave tijdens agent executie

---

# Sub-agent: research-agent

**Taak:** Diepgaand onderzoek per subtopic

**Tools:** `internet_search`

**Output format:**
```markdown
- Feit 1 met bron [1]
- Feit 2 met bron [2]

Sources:
[1] [Titel](URL)
[2] [Titel](URL)
```

---

# Sub-agent: critique-agent

**Taak:** Kwaliteitscontrole rapport

**Feedback structuur:**

| Prioriteit | Actie |
|------------|-------|
| HIGH | Moet opgelost |
| MEDIUM | Zou moeten |
| LOW | Wordt genegeerd |

---

# Research modes

## Deep Research
- Multi-fase workflow
- 2 sub-agents parallel
- 10-30 minuten
- 5000-8000 woorden

## Quick Research
- Direct LLM
- Geen sub-agents
- 1-3 minuten
- Beknopter rapport

---

# Workflow: Deep Research

```
1. Plan maken (write_todos)
       ↓
2. Research sub-agents (parallel)
       ↓
3. Critique agent
       ↓
4. Follow-up research (HIGH/MEDIUM)
       ↓
5. Final report schrijven
```

---

# Safety features

**Rapport garantie:**
- Altijd `final_report.md` aanwezig
- Fallback bij agent failure
- Emergency report generatie

**Early trigger:**
- Bij 85% iteraties → rapport afronden
- Voorkomt data verlies

---

# Monitoring

**Live displays:**
- Laatste 5 zoekopdrachten
- TODO status
- Iteratie teller
- Cache statistieken

**AgentTracker:**
- Search count
- Cache hits
- File operations

---

# Configuratie

| Setting | Opties |
|---------|--------|
| Provider | tavily, multi-search, auto |
| Recursion limit | 50-500 (default: 200) |
| Debug mode | Gedetailleerde logging |
| Language | nl (default), en |

---

# Prompt bestanden

```
prompts/
├── deep_research.txt      # Main agent
├── research_agent.txt     # Sub-agent
├── critique_agent.txt     # Critique
├── quick_research.txt     # Quick mode
└── emergency_refinement.txt
```

Makkelijk aanpasbaar zonder code wijzigingen

---

# Samenvatting

**5 directe tools** voor search, files, todos

**2 sub-agents** voor parallel onderzoek

**2 research modes** (deep/quick)

**Veiligheidsnet** voor rapport garantie

**Externe prompts** voor flexibiliteit
