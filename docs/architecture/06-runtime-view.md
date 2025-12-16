# 6. Runtime View

## 6.1 Deep Research Flow

```mermaid
flowchart TB
    User[User]

    subgraph CLI[CLI - cli.py]
        Parse[1. Parse arguments]
        LoadEnv[2. Load .env]
        Init[3. Initialize tracker, search_tool]
        Parse --> LoadEnv --> Init
    end

    subgraph DeepRunner[Deep Runner - runners/deep.py]
        subgraph Supervisor[Supervisor Agent]
            Analyze[Analyseert vraag]
            Delegate[Delegeert naar sub-agents]
            WriteReport[Schrijft final_report.md]
        end

        subgraph SubAgents[Sub-Agents]
            Research[Research Sub-Agent<br/>- Zoekt web<br/>- Verzamelt bronnen]
            Critique[Critique Sub-Agent<br/>- Beoordeelt<br/>- Identificeert gaps]
        end

        HybridSearch[HybridSearch Tool]

        Supervisor --> SubAgents
        Research --> HybridSearch
    end

    subgraph Finalization[Report Finalization]
        Ensure[1. ensure_report_exists]
        PostProcess[2. postprocess_report]
        Rename[3. rename_final_report]
        Ensure --> PostProcess --> Rename
    end

    ExternalAPIs[Tavily / Multi-Search API]
    Output[research/*.md]

    User -->|vraag| CLI
    CLI --> DeepRunner
    HybridSearch --> ExternalAPIs
    DeepRunner --> Finalization
    Finalization --> Output
```

## 6.2 Quick Research Flow

```mermaid
flowchart LR
    User[User] --> CLI[CLI]
    CLI --> Quick[run_quick_research]
    Quick --> LLM[Single LLM call]
    LLM --> Response[Print response]
```

Simpele flow zonder sub-agents of rapport opslag.

## 6.3 Report Guarantee Mechanism

```mermaid
flowchart TB
    Start[Iteratie Loop]
    Increment[iteration_count++]
    Check{>= 85% limit?}
    Continue[Continue research]
    Inject[Inject URGENT message<br/>to write report]
    AgentWrite[Agent writes<br/>final_report.md]
    SetFlag[report_triggered = True]
    LoopEnd[Na loop completion]
    EnsureExists{Report exists?}
    Done[Done]
    Emergency[Extract messages<br/>Create emergency report]

    Start --> Increment
    Increment --> Check
    Check -->|no| Continue
    Check -->|yes| Inject
    Continue --> Start
    Inject --> AgentWrite
    AgentWrite --> SetFlag
    SetFlag --> LoopEnd
    LoopEnd --> EnsureExists
    EnsureExists -->|yes| Done
    EnsureExists -->|no| Emergency
    Emergency --> Done
```

## 6.4 Search Flow met Caching

```mermaid
flowchart TB
    Search[search query]
    CacheCheck{In cache?}
    ReturnCached[Return cached result]
    CallProvider[Call provider<br/>Tavily/Multi-Search]
    StoreCache[Store in cache]
    UpdateDisplay[Update display<br/>SearchStatus]
    ReturnResults[Return results]

    Search --> CacheCheck
    CacheCheck -->|yes| ReturnCached
    CacheCheck -->|no| CallProvider
    CallProvider --> StoreCache
    StoreCache --> UpdateDisplay
    UpdateDisplay --> ReturnResults
```

## 6.5 UI Update Cycle

```mermaid
flowchart TB
    subgraph LiveContext[Rich Live Context]
        Event[Agent Event]

        SearchResult[Search Result]
        TodoUpdate[TODO Update]
        TokenUsage[Token Usage]

        CombinedPanel[create_combined_status_panel<br/>- Search activity panel<br/>- TODO list panel]

        LiveUpdate[live_display.update<br/>combined_panel]

        Event --> SearchResult
        Event --> TodoUpdate
        Event --> TokenUsage

        SearchResult --> CombinedPanel
        TodoUpdate --> CombinedPanel
        TokenUsage --> CombinedPanel

        CombinedPanel --> LiveUpdate
    end
```
