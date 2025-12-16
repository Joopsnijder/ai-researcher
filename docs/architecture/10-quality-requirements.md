# 10. Kwaliteitseisen

## 10.1 Quality Tree

```
                        Kwaliteit
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   Betrouwbaarheid    Bruikbaarheid      Onderhoudbaarheid
        │                   │                   │
   ┌────┴────┐        ┌────┴────┐        ┌────┴────┐
   │         │        │         │        │         │
Rapport   Error     CLI      Output   Modular  Testbaar
Garantie  Handling  UX       Kwaliteit Code
```

## 10.2 Kwaliteitsscenario's

### QS-1: Rapport Garantie

| Aspect | Beschrijving |
|--------|--------------|
| **Bron** | Systeem (timeout, error) |
| **Stimulus** | Agent faalt of bereikt iteratielimiet |
| **Omgeving** | Normale operatie |
| **Artifact** | Report module |
| **Response** | Emergency rapport generatie |
| **Maatstaf** | 100% van runs produceert een rapport |

### QS-2: Zoekresultaat Kwaliteit

| Aspect | Beschrijving |
|--------|--------------|
| **Bron** | Gebruiker |
| **Stimulus** | Onderzoeksvraag invoer |
| **Omgeving** | Internet beschikbaar |
| **Artifact** | Search module |
| **Response** | Relevante zoekresultaten |
| **Maatstaf** | >= 5 unieke bronnen per rapport |

### QS-3: Response Time

| Aspect | Beschrijving |
|--------|--------------|
| **Bron** | Gebruiker |
| **Stimulus** | Quick research vraag |
| **Omgeving** | Normale API latency |
| **Artifact** | Quick runner |
| **Response** | Direct antwoord |
| **Maatstaf** | < 30 seconden voor quick mode |

### QS-4: Cost Transparency

| Aspect | Beschrijving |
|--------|--------------|
| **Bron** | Gebruiker |
| **Stimulus** | Research sessie voltooien |
| **Omgeving** | Elke run |
| **Artifact** | Tracking module |
| **Response** | Kostenrapportage |
| **Maatstaf** | Exacte kosten getoond na elke run |

### QS-5: Modulariteit

| Aspect | Beschrijving |
|--------|--------------|
| **Bron** | Ontwikkelaar |
| **Stimulus** | Nieuwe feature toevoegen |
| **Omgeving** | Development |
| **Artifact** | Package structuur |
| **Response** | Lokale wijzigingen |
| **Maatstaf** | Wijziging raakt <= 2 modules |

## 10.3 Kwaliteitsmetrieken

| Metriek | Target | Huidig |
|---------|--------|--------|
| Test coverage | > 70% | ~60% |
| Rapport success rate | 100% | 100% |
| Gemiddelde deep research tijd | < 5 min | 2-4 min |
| Gemiddelde quick research tijd | < 30 sec | 5-15 sec |
| Module coupling | Low | Low |
| Cyclomatic complexity | < 10 per functie | OK |

## 10.4 Test Strategie

### Unit Tests

```bash
pytest tests/ -v
```

Coverage:
- Report generation
- Helper functions
- Prompt loading
- Early trigger logic

### Integration Tests

Handmatig via CLI:
```bash
python research.py "Test vraag" --quick
python research.py "Test vraag" --deep
```

### Smoke Tests

Automatisch in `test_report_guarantee.py`:
- Mock agent execution
- Verify report creation
