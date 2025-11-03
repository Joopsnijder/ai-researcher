# AI Research Agent

Een geavanceerde research agent gebouwd met [DeepAgents](https://github.com/langchain-ai/deepagents) die diepgaand onderzoek uitvoert met automatische planning, verificatie en rapportage.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- 🤖 **Intelligent Planning**: Automatische TODO-lijst generatie en tracking
- 🔍 **Multi-Provider Search**: Keuze tussen Tavily (premium) en Multi-Search API (gratis tier)
- 🎨 **Rich Terminal UI**: Real-time visualisatie van agent activiteit
- ✅ **Quality Assurance**: Ingebouwde critique sub-agent voor verificatie
- ⏱️ **Performance Tracking**: Duration metrics en usage statistics
- 🛡️ **Safety**: Configureerbare recursion limits tegen oneindige loops
- 📄 **Professional Reports**: Markdown-formatted research rapporten

## Architecture

```
Main Agent
├── Research Sub-Agent    → Diepgaand onderzoek
├── Critique Sub-Agent    → Validatie & verificatie
└── HybridSearchTool
    ├── Tavily API       → Premium AI-optimized search
    ├── Multi-Search API → Gratis met auto-fallback
    └── Auto mode        → Intelligent provider selection
```

## Installation

### Requirements
- Python 3.11 of hoger
- API keys (optioneel, afhankelijk van provider keuze)

### Setup

1. **Clone repository**
```bash
git clone https://github.com/Joopsnijder/ai-researcher.git
cd ai-researcher
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Op Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API keys**

Maak een `.env` file in de project root:

```env
# Required: Anthropic API key voor DeepAgents
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional: Search providers (kies er minimaal 1)
TAVILY_API_KEY=your_tavily_key_here
SERPER_API_KEY=your_serper_key_here
BRAVE_API_KEY=your_brave_key_here
```

**API Keys verkrijgen:**
- [Anthropic](https://console.anthropic.com/) - Required
- [Tavily](https://tavily.com/) - Premium search (betaald)
- [Serper](https://serper.dev/) - Gratis tier: 2,500 queries/maand
- [Brave Search](https://brave.com/search/api/) - Gratis tier: 2,000 queries/maand

## Usage

### Basic Usage

Start de research agent:

```bash
python research.py
```

Je wordt gevraagd om:
1. **Recursion limit** in te stellen (default: 200)
2. **Search provider** te kiezen:
   - `1` = Tavily (premium, hoogste kwaliteit)
   - `2` = Multi-Search (gratis tier, aanbevolen voor development)
   - `3` = Auto (intelligent switchen)
3. **Onderzoeksvraag** in te voeren

### Example Session

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ AI Research Agent          ┃
┃ Powered by DeepAgents      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Agent configuratie:
Let op: Het recursion limit wordt gedeeld tussen de hoofd-agent en sub-agents.
Voor complexe onderzoeken zijn vaak 150-300 iteraties nodig.

Maximaal aantal agent iteraties (voorkomt oneindige loops) [200]:
Recursion limit: 200

Kies een search provider:
  1. Tavily          (betaald, hoogste kwaliteit, AI-optimized)
  2. Multi-Search   (gratis tier, auto-fallback, meerdere providers)
  3. Auto           (slim kiezen: Multi-Search eerst, Tavily als fallback)

Provider [2]: 2
✓ Multi-Search API (gratis tier) geactiveerd

Wat wil je onderzoeken? [What are the latest advancements in Explainable AI as of 2025?]:

Start onderzoek? [ja/nee] (ja): ja
```

### Terminal UI Features

De Rich UI toont real-time:

- 🔍 **Searches**: `[#1] query → 8 resultaten (provider)`
- 💭 **Agent thinking**: Preview van redenering
- 🛠️ **Tool calls**: Welke tools worden gebruikt
- 🤖 **Sub-agents**: Research/Critique agent activiteit
- 📋 **TODO lijst**:
  - ○ Pending tasks
  - ▶ In progress (geel)
  - ✓ Completed (groen)

### Output

Na afloop krijg je:

1. **Statistics panel**:
   - ⏱️ Totale duur
   - 🔍 Aantal zoekopdrachten
   - 💬 Aantal berichten
   - 🌐 Provider usage

2. **Markdown rapport**: `final_report.md`
   - Gestructureerd onderzoek
   - Bronvermeldingen
   - Conclusies

## Configuration

### Recursion Limits

De recursion limit bepaalt het maximum aantal agent iteraties:

- **50-100**: Snelle, oppervlakkige research
- **150-200**: Standaard (aanbevolen voor meeste vragen)
- **300-500**: Diepgaand onderzoek met veel verificatie

**Let op**: Hoofdagent en sub-agents delen dit budget!

### Provider Selection

#### Tavily (Option 1)
- ✅ AI-optimized results
- ✅ Hoogste kwaliteit
- ❌ Betaald ($)

#### Multi-Search (Option 2) - **Aanbevolen**
- ✅ Gratis tiers
- ✅ Auto-fallback over meerdere providers
- ✅ Stabiel met caching disabled
- ⚠️ Iets lagere kwaliteit

#### Auto Mode (Option 3)
- ✅ Best of both worlds
- ✅ Multi-Search eerst, Tavily bij fouten
- ⚠️ Vereist beide API keys

## Tech Stack

- **[DeepAgents](https://github.com/langchain-ai/deepagents)**: LangChain-based agentic framework
- **[Tavily](https://tavily.com/)**: Premium AI search API
- **[Multi-Search API](https://github.com/yourusername/multi-search-api)**: Free-tier search aggregator
- **[Rich](https://rich.readthedocs.io/)**: Terminal UI library
- **[Python-dotenv](https://github.com/theskumar/python-dotenv)**: Environment management

## Project Structure

```
ai-researcher/
├── research.py              # Main agent implementation
├── requirements.txt         # Python dependencies
├── .env                     # API keys (niet in git!)
├── .gitignore
├── docs/
│   ├── ai-research-agent-presentatie.md    # Marp presentatie
│   └── template-presentation.md            # Marp template
├── scripts/
│   ├── build-presentations.sh              # Build Marp slides
│   └── mermaid-to-images.js               # Mermaid converter
└── final_report.md         # Output (generated)
```

## Troubleshooting

### Recursion Limit Bereikt

**Error**: `GraphRecursionError: Recursion limit of X reached`

**Oplossing**:
- Verhoog recursion limit naar 300-500
- Simplificeer de onderzoeksvraag
- Check of er geen oneindige loop is

### Search Provider Faalt

**Error**: API key errors of geen resultaten

**Oplossing**:
1. Verifieer API keys in `.env`
2. Check quota limits van provider
3. Probeer een andere provider
4. Gebruik 'auto' mode voor fallback

### Thread Safety Issues

**Error**: `RuntimeError: dictionary changed size during iteration`

**Oplossing**: Al gefixt! Cache is disabled in Multi-Search configuratie.

## Development

### Running Tests

```bash
# Gebruik multi-search voor development (gratis tier)
python research.py
# Kies option 2 (Multi-Search)
```

⚠️ **Let op**: Elk run kost API calls. Test niet zonder toestemming!

### Building Presentation

```bash
# Install Marp dependencies (first time only)
npm install

# Build presentation
./scripts/build-presentations.sh docs/ai-research-agent-presentatie.md

# Open result
open docs/ai-research-agent-presentatie-final.html
```

## Roadmap

- [ ] Custom TODO planning via system prompts
- [ ] PDF export voor rapporten
- [ ] Web interface
- [ ] Conversation history
- [ ] Document upload (RAG)
- [ ] Multi-language support
- [ ] Citation verification
- [ ] Knowledge base persistence

## Contributing

Contributions welkom! Open een issue of pull request.

## License

MIT License - zie LICENSE file voor details.

## Credits

Gebouwd met:
- [LangChain DeepAgents](https://github.com/langchain-ai/deepagents)
- [Claude](https://anthropic.com/) (Sonnet 4.5)
- [Rich](https://github.com/Textualize/rich)

---

**🤖 Generated with [Claude Code](https://claude.com/claude-code)**
