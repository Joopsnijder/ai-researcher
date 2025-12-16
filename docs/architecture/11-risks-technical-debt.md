# 11. Risico's en Technische Schuld

## 11.1 Risico's

### R-1: API Vendor Lock-in

| Aspect | Beschrijving |
|--------|--------------|
| **Risico** | Afhankelijkheid van Anthropic Claude API |
| **Impact** | Hoog - core functionaliteit stopt bij API wijzigingen |
| **Waarschijnlijkheid** | Laag |
| **Mitigatie** | DeepAgents framework ondersteunt andere LLMs |
| **Status** | Geaccepteerd |

### R-2: Search Provider Beschikbaarheid

| Aspect | Beschrijving |
|--------|--------------|
| **Risico** | Tavily of Multi-Search API offline |
| **Impact** | Medium - degraded search quality |
| **Waarschijnlijkheid** | Medium |
| **Mitigatie** | Hybrid search met fallback providers |
| **Status** | Gemitigeerd |

### R-3: Cost Overruns

| Aspect | Beschrijving |
|--------|--------------|
| **Risico** | Onverwacht hoge API kosten |
| **Impact** | Medium - financieel |
| **Waarschijnlijkheid** | Medium |
| **Mitigatie** | Cost tracking, iteratielimieten |
| **Status** | Gemitigeerd |

### R-4: Rate Limiting

| Aspect | Beschrijving |
|--------|--------------|
| **Risico** | API rate limits bereikt tijdens onderzoek |
| **Impact** | Medium - onderzoek onderbroken |
| **Waarschijnlijkheid** | Laag |
| **Mitigatie** | Query caching, error handling |
| **Status** | Geaccepteerd |

## 11.2 Technische Schuld

### TD-1: Geen Versioning

| Aspect | Beschrijving |
|--------|--------------|
| **Item** | Package heeft geen versienummer |
| **Impact** | Lastig om releases te tracken |
| **Effort** | Laag |
| **Prioriteit** | Laag |

### TD-2: Requirements zonder Pinning

| Aspect | Beschrijving |
|--------|--------------|
| **Item** | `requirements.txt` heeft geen versie pins |
| **Impact** | Reproducibility issues mogelijk |
| **Effort** | Laag |
| **Prioriteit** | Medium |

### TD-3: Beperkte Test Coverage

| Aspect | Beschrijving |
|--------|--------------|
| **Item** | ~60% test coverage |
| **Impact** | Regressions mogelijk |
| **Effort** | Medium |
| **Prioriteit** | Medium |

### TD-4: Hardcoded Nederlandse Strings

| Aspect | Beschrijving |
|--------|--------------|
| **Item** | UI strings hardcoded in Nederlands |
| **Impact** | Internationalisatie lastig |
| **Effort** | Medium |
| **Prioriteit** | Laag |

### TD-5: Geen Type Checking

| Aspect | Beschrijving |
|--------|--------------|
| **Item** | Geen mypy of pyright configuratie |
| **Impact** | Type errors niet automatisch gedetecteerd |
| **Effort** | Laag |
| **Prioriteit** | Medium |

## 11.3 Verbeteringsplan

### Korte Termijn

1. Pin versies in `requirements.txt`
2. Voeg mypy configuratie toe
3. Verhoog test coverage naar 80%

### Lange Termijn

1. Overweeg `pyproject.toml` voor proper packaging
2. Voeg versienummer toe
3. Internationalisatie framework

## 11.4 Dependency Risico's

| Dependency | Risico | Alternatief |
|------------|--------|-------------|
| `deepagents` | Framework updates kunnen breken | LangGraph direct |
| `rich` | Laag risico, stable | Standaard print |
| `tavily-python` | Provider kan stoppen | Multi-search alleen |
| `anthropic` | API wijzigingen | Via DeepAgents abstraction |
