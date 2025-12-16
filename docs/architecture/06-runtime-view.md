# 6. Runtime View

## 6.1 Deep Research Flow

```
┌──────────┐
│  User    │
└────┬─────┘
     │ vraag
     ▼
┌────────────────────────────────────────────────────────────────┐
│ CLI (cli.py)                                                   │
│  1. Parse arguments                                            │
│  2. Load .env                                                  │
│  3. Initialize tracker, search_tool                            │
└────┬───────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│ Deep Runner (runners/deep.py)                                  │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Supervisor Agent                                         │  │
│  │  - Analyseert vraag                                     │  │
│  │  - Delegeert naar sub-agents                            │  │
│  │  - Schrijft final_report.md                             │  │
│  └────────────────────┬────────────────────────────────────┘  │
│                       │                                        │
│         ┌─────────────┴─────────────┐                         │
│         ▼                           ▼                         │
│  ┌──────────────┐           ┌──────────────┐                  │
│  │ Research     │           │ Critique     │                  │
│  │ Sub-Agent    │           │ Sub-Agent    │                  │
│  │              │           │              │                  │
│  │ - Zoekt web  │           │ - Beoordeelt │                  │
│  │ - Verzamelt  │           │ - Identificeert│                │
│  │   bronnen    │           │   gaps        │                 │
│  └──────┬───────┘           └──────────────┘                  │
│         │                                                      │
│         ▼                                                      │
│  ┌──────────────┐                                             │
│  │ HybridSearch │                                             │
│  │ Tool         │──────▶ Tavily / Multi-Search API            │
│  └──────────────┘                                             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│ Report Finalization (report/finalization.py)                   │
│  1. ensure_report_exists()                                     │
│  2. postprocess_report()                                       │
│  3. rename_final_report() → research/[slug]-[date].md          │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────┐
│ Output:  │
│ research/│
│ *.md     │
└──────────┘
```

## 6.2 Quick Research Flow

```
User ─▶ CLI ─▶ run_quick_research() ─▶ Single LLM call ─▶ Print response
```

Simpele flow zonder sub-agents of rapport opslag.

## 6.3 Report Guarantee Mechanism

```
                    Iteratie Loop
                         │
                         ▼
              ┌──────────────────────┐
              │ iteration_count++    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ >= 85% limit?        │
              └──────────┬───────────┘
                    yes/ │ \no
                   ┌─────┘  └─────┐
                   ▼              ▼
        ┌─────────────────┐  (continue)
        │ Inject          │
        │ URGENT message  │
        │ to write report │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Agent writes    │
        │ final_report.md │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ report_triggered│
        │ = True          │
        └─────────────────┘

                    │
         ┌──────────┴──────────┐
         │ Na loop completion  │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │ ensure_report_exists│
         │                     │
         │ if no report:       │
         │   extract messages  │
         │   create emergency  │
         │   report            │
         └─────────────────────┘
```

## 6.4 Search Flow met Caching

```
search(query)
     │
     ▼
┌─────────────────┐
│ In cache?       │
└────────┬────────┘
    yes/ │ \no
   ┌─────┘  └─────┐
   ▼              ▼
Return        ┌─────────────────┐
cached        │ Call provider   │
result        │ (Tavily/Multi)  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Store in cache  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Update display  │
              │ (SearchStatus)  │
              └────────┬────────┘
                       │
                       ▼
                  Return results
```

## 6.5 UI Update Cycle

```
┌───────────────────────────────────────────────────────────┐
│ Rich Live Context                                         │
│                                                           │
│  ┌─────────────┐                                         │
│  │ Agent Event │                                         │
│  └──────┬──────┘                                         │
│         │                                                 │
│    ┌────┴────┬────────────┐                              │
│    ▼         ▼            ▼                              │
│  Search   TODO        Token                              │
│  Result   Update      Usage                              │
│    │         │            │                              │
│    └────┬────┴────────────┘                              │
│         ▼                                                 │
│  ┌─────────────────────────────────────────┐            │
│  │ create_combined_status_panel()          │            │
│  │  - Search activity panel                │            │
│  │  - TODO list panel                      │            │
│  └──────────────────┬──────────────────────┘            │
│                     │                                    │
│                     ▼                                    │
│  ┌─────────────────────────────────────────┐            │
│  │ live_display.update(combined_panel)     │            │
│  └─────────────────────────────────────────┘            │
│                                                          │
└───────────────────────────────────────────────────────────┘
```
