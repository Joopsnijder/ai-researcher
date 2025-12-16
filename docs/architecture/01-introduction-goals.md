# 1. Introductie en Doelen

## 1.1 Beschrijving

AI Researcher is een command-line tool voor het uitvoeren van diepgaand onderzoek met behulp van AI-agents. Het systeem combineert meerdere zoekmachines met LLM-gestuurde analyse om uitgebreide onderzoeksrapporten te genereren.

## 1.2 Kwaliteitsdoelen

| Prioriteit | Kwaliteitsdoel | Scenario |
|------------|----------------|----------|
| 1 | **Betrouwbaarheid** | Het systeem genereert ALTIJD een rapport, ook bij fouten of onderbrekingen |
| 2 | **Volledigheid** | Rapporten bevatten bronvermeldingen en zijn gestructureerd volgens een vast formaat |
| 3 | **Gebruiksgemak** | CLI interface met interactieve en non-interactieve modus |
| 4 | **Kostenbeheersing** | Inzicht in API-kosten per onderzoekssessie |

## 1.3 Stakeholders

| Rol | Verwachting |
|-----|-------------|
| **Eindgebruiker** | Snel antwoord op complexe onderzoeksvragen met betrouwbare bronnen |
| **Ontwikkelaar** | Modulaire codebase, makkelijk uit te breiden |
| **Beheerder** | Duidelijke logging, kostenoverzicht, configureerbare limieten |

## 1.4 Belangrijkste Functionaliteiten

1. **Quick Research** - Snelle antwoorden via enkele LLM-call
2. **Deep Research** - Multi-agent onderzoek met sub-agents voor zoeken en kritische analyse
3. **Hybrid Search** - Combineert Tavily, Multi-Search API en andere providers
4. **Rapport Garantie** - Automatische fallback bij timeout of fouten
5. **Post-processing** - Automatische opschoning van titels, bronnen en links
