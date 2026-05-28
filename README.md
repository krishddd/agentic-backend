# Multi-Agent Orchestrator v6.0 — Deep Research + Desktop Automation

A production multi-agent system using **3 local Ollama LLMs as 6 specialized agents** for universal deep research, **21 desktop automation tools**, and **12 workflow pipelines** with **25+ MCP connectors**. Accepts **any topic** and produces **publication-quality `.md` reports** with strict citation integrity.

---

## System Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              USER REQUEST                    │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │           PROMPT ROUTER (llama3.2)           │
                    │   12 Workflows · LLM + Keyword Fallback      │
                    └──┬───────────┬───────────┬──────────────────┘
                       │           │           │
         ┌─────────────▼──┐  ┌────▼─────┐  ┌──▼──────────────┐
         │   Deep Research  │  │ Financial │  │ Desktop/Browser  │
         │   v6.0 Pipeline  │  │ Pipelines │  │ Agent Pipelines  │
         │   (8-node graph) │  │ (CrewAI)  │  │ (21 tools)       │
         └────────┬────────┘  └────┬─────┘  └──────┬──────────┘
                  │                │                │
         ┌────────▼────────────────▼────────────────▼──────────┐
         │                25+ MCP CONNECTORS                    │
         │  Tavily · DuckDuckGo · Wikipedia · ArXiv · PubMed    │
         │  Yahoo Finance · GitHub · Reddit · YouTube · News    │
         │  Playwright · BrowserUse · Memory · Desktop · n8n    │
         └─────────────────────────┬───────────────────────────┘
                                   │
         ┌─────────────────────────▼───────────────────────────┐
         │              EXPORT & DELIVERY                       │
         │  Reports (.md) · Google Sheets · Gmail · n8n Webhook │
         └─────────────────────────────────────────────────────┘
```

---

## 12 Workflow Pipelines

### Financial & Business (1–8)

| # | Workflow | Description | Agents |
|---|----------|-------------|--------|
| 1 | `due_diligence` | Full pre-investment analysis (financials + SEC + sentiment + quality-looped report) | 6 agents |
| 2 | `competitor_intel` | Side-by-side multi-company comparison | 4 agents |
| 3 | `earnings_monitor` | Earnings/filing analysis + risk-based alerts | 4 agents |
| 4 | `market_pulse` | Real-time news sentiment scan | 3 agents |
| 5 | `lead_gen` | Synthetic lead generation + outreach + scoring | 3 agents |
| 6 | `portfolio_review` | Multi-stock portfolio analysis + rebalancing recs | 5 agents |
| 7 | `deal_pipeline` | Process pending sheet rows through email + scoring | 3 agents |
| 8 | `risk_scan` | SEC 10-K risk factor deep extraction + severity matrix | 4 agents |

### Desktop, Browser & Data (9–12)

| # | Workflow | Description | Tools |
|---|----------|-------------|-------|
| 9 | `browser_research` | AI browser agent for live web research, data extraction, screenshots | Playwright, BrowserUse, Tavily |
| 10 | `desktop_automation` | Execute desktop tasks: file ops, code execution, system commands, PDF | DesktopAgent (21 tools) |
| 11 | `document_pipeline` | Read PDFs → summarize → generate reports → email | DesktopAgent, Memory |
| 12 | `data_analysis` | CSV/Excel → statistics → charts → PDF reports | DesktopAgent |

---

## Deep Research v6.0 Pipeline

### How It Works

1. **Classifier Agent** (llama3.2) — Detects topic domain (financial, scientific, tech, geopolitical, general)
2. **Strategist Agent** (qwen3:8b) — Decomposes query into 5–8 sub-questions with tool assignments
3. **Search Executor** — Runs sub-questions against 20+ MCP tools in parallel (semaphore-gated)
4. **Enricher** — Fetches full page content from source URLs via requests+BeautifulSoup
5. **Semantic Chunker** — Reduces 160K→30K tokens using nomic-embed-text embeddings
6. **Extractor Agent** (llama3.2) — Extracts verified facts with locked source citations
7. **Verifier Agent** (qwen3:4b) — Cross-checks facts, rejects questionable claims
8. **Scorer Agent** (qwen3:4b) — Quality gate with 3-tier fallback (JSON → regex → metric-based)
9. **Composer Agent** (qwen3:8b) — Writes 2500+ word publication-quality report
10. **Export** — Saves report, updates Google Sheets, sends Gmail summary, triggers n8n webhook

### Multi-Agent LLM Assignments

| Agent | LLM Model | Role |
|-------|----------|------|
| **Classifier Agent** | `llama3.2:latest` | Topic detection & domain routing |
| **Strategist Agent** | `qwen3:8b` | Research planning & query decomposition |
| **Extractor Agent** | `llama3.2:latest` | Fact extraction from raw content |
| **Verifier Agent** | `qwen3:4b` | Cross-checks facts & rejects questionable claims |
| **Scorer Agent** | `qwen3:4b` | LLM-based quality assessment & gap analysis |
| **Composer Agent** | `qwen3:8b` | Report synthesis & long-form writing |

### Topic Domains & Tool Routing

| Domain | Primary Tools | Secondary Tools |
|--------|-------------|-----------------| 
| **Financial** | tavily, yahoo_finance, news_headlines | reddit, safe_web_scraper, duckduckgo |
| **Scientific** | arxiv, semantic_scholar, pubmed, tavily | wikipedia, github |
| **Technology** | github, tavily, arxiv | reddit, youtube, news_headlines |
| **Geopolitical** | news_headlines, tavily, wikipedia | reddit, youtube |
| **General** | tavily, duckduckgo, wikipedia | news_headlines, reddit, youtube |

---

## Desktop Agent

The Desktop Agent is an **LLM-powered tool router** that converts natural language instructions into desktop operations. It uses **qwen3:8b** for intent classification with keyword fallback.

### 21 Tools Available

| Category | Tools | Description |
|----------|-------|-------------|
| **File Operations** | `file_list`, `file_read`, `file_write`, `file_search`, `file_info`, `file_delete`, `file_copy`, `file_move` | Full file CRUD with audit trail |
| **Directory** | `dir_tree`, `dir_create` | Directory tree visualization & creation |
| **Code Execution** | `code_execute` | Python/Shell in sandboxed environment with auto-retry |
| **System Monitor** | `system_info`, `system_processes`, `system_packages` | CPU, RAM, disk, processes, pip packages |
| **PDF Tools** | `pdf_create`, `pdf_read`, `pdf_merge`, `pdf_info` | Create, read, merge, inspect PDFs |
| **Browser** | `browser_open`, `browser_search` | Open URL, Google search (via Selenium) |
| **Screen** | `screenshot` | Desktop screen capture |

### How It Works

```
User: "list all Python files in the current directory"
  │
  ▼
LLM Classification (qwen3:8b)
  │ tool: "file_list", params: {path: ".", pattern: "*.py"}
  ▼
FileSystemExecutor.list_files(path=".", pattern="*.py")
  │
  ▼
DesktopResult(success=true, tool="file_list", result="Found 6 items")
```

---

## MiroFish ABM Simulation (Grandmaster Edition)

The **MiroFish Agent-Based Model** runs Monte Carlo sentiment simulations with 50–80 autonomous agents. It models how market sentiment evolves across dual-channel platforms (Microblog + Forum) with a **5-phase Grandmaster architecture**.

### 5-Phase Architecture

| # | Phase | Key Capabilities |
|---|-------|-------------------|
| 01 | **Graph Construction** | Reality seed extraction from pipeline data, GraphRAG engine, entity-relationship graph, individual + collective memory injection |
| 02 | **Environment Setup** | LLM config agent auto-tunes params (agents, ticks, budget) from data richness, catalyst auto-detection, prediction target extraction |
| 03 | **Simulation** | Dual-platform parallel execution, 15-slot importance-weighted memory with decay, 20% cross-platform discovery, dynamic temporal memory |
| 04 | **Report Generation** | Ensemble trend prediction (T+5/T+10/T+30), linear regression + momentum + mean-reversion, risk factor identification |
| 05 | **Deep Interaction** | Chat with any individual agent (in-character), converse with ReportAgent, D3.js graph visualization |

### Grandmaster Features

| Feature | Description |
|---------|-------------|
| **GraphRAG Engine** | Natural language query → subgraph retrieval → agent prompt context injection |
| **Graph Visualization** | D3.js JSON, Mermaid diagrams, adjacency matrix, temporal evolution snapshots |
| **Reality Seed Extraction** | Rule-based + LLM entity extraction from pipeline data (tickers, people, metrics) |
| **Config Agent** | Auto-scales simulation params from pipeline data richness scoring (0–1) |
| **Dynamic Temporal Memory** | 15-slot importance-weighted memory with eviction scoring (importance × 0.6 + recency × 0.4) |
| **Cross-Platform Discovery** | 20% chance agents discover hot posts from the other channel each tick |
| **Trend Prediction** | Ensemble forecasts at T+5/T+10/T+30 with confidence intervals and risk factors |
| **Agent Chat** | Chat with any agent in-character using their persona, memory, sentiment, and coalition |
| **ReportAgent Chat** | Ask follow-up questions with full simulation context (coalitions, contagion, predictions) |
| **Coalition Stability** | Herfindahl index + polarization penalty for cluster stability measurement |

### ABM API Response Fields

When MiroFish runs, the `/prompt/async` response includes:

| Field | Type | Description |
|-------|------|-------------|
| `abm_contagion_events` | `List[Dict]` | Viral cascade moments with tick, magnitude, direction |
| `abm_phase_transitions` | `List[Dict]` | Critical tipping points with trigger analysis |
| `abm_coalitions` | `List[Dict]` | Agent clusters with mean sentiment and size |
| `abm_saddle_point` | `Dict` | Dominant trajectory from MC analysis |
| `abm_path_variance` | `float` | Branch divergence across MC paths |
| `abm_confidence_interval` | `Dict` | Statistical CI on final sentiments |
| `abm_channel_trajectories` | `Dict` | Per-channel (microblog/forum) tick-by-tick breakdown |
| `abm_catalyst_shocks` | `List[Dict]` | Auto-detected event shocks from pipeline data |
| `abm_graph_data` | `Dict` | D3.js-compatible graph nodes + links + metadata |
| `abm_trend_predictions` | `Dict` | Multi-horizon forecasts with confidence intervals |

---

## 25+ MCP Connectors

All connectors extend `BaseConnector` with async execution, timeout handling, and health metrics.

| Connector | Name | Description |
|-----------|------|-------------|
| `TavilySearchConnector` | `tavily_web_search` | Web search via Tavily API |
| `DuckDuckGoSearchConnector` | `duckduckgo_web_search` | Web search (no API key needed) |
| `WikipediaConnector` | `wikipedia_search` | Wikipedia article search |
| `ArXivConnector` | `arxiv_search` | Academic paper search |
| `SemanticScholarConnector` | `semantic_scholar` | Research paper search |
| `PubMedConnector` | `pubmed_search` | Medical/biotech research |
| `YahooFinanceConnector` | `yahoo_finance` | Stock data & financials |
| `GitHubSearchConnector` | `github_search` | GitHub repo/code search |
| `RedditSearchConnector` | `reddit_search` | Reddit discussion search |
| `YouTubeSearchConnector` | `youtube_search` | YouTube video search |
| `NewsConnector` | `news_headlines` | News headlines API |
| `WeatherConnector` | `weather` | Weather data |
| `MathConnector` | `math_solver` | Mathematical computation |
| `SymPySolverConnector` | `sympy_solver` | Algebraic equation solver |
| `SafeWebScraperConnector` | `safe_web_scraper` | URL content extraction |
| `BrightDataRedditSearchConnector` | `brightdata_reddit` | Reddit via Bright Data |
| **`PlaywrightBrowserConnector`** | `playwright_browser` | Browser automation (navigate, screenshot, extract, click, fill) |
| **`BrowserUseConnector`** | `browser_use_agent` | AI browser agent with Playwright fallback |
| **`MemoryKnowledgeGraphConnector`** | `memory_knowledge_graph` | Persistent knowledge graph (store/query entities & facts) |
| **`DesktopAgentConnector`** | `desktop_agent` | Bridge to Desktop Agent module (21 tools) |
| **`N8nWebhookConnector`** | `n8n_webhook` | Trigger n8n workflow automation via webhooks |

> **Bold** = New Tier 1+2 connectors

---

## API Endpoints

### Deep Research

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v6/research` | v6.0 Deep Research — any topic, iterative multi-agent pipeline |

```json
POST /v6/research
{
  "query": "What are the implications of the US-China trade war on semiconductors?",
  "max_iterations": 4,
  "dry_run": false
}
```

### Desktop Agent

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/desktop/execute` | Execute natural language desktop tasks |
| `GET` | `/desktop/capabilities` | List all 21 tools + descriptions |
| `GET` | `/desktop/history` | Execution audit trail |

```json
POST /desktop/execute
{"task": "show system info"}
{"task": "list all Python files in the current directory"}
{"task": "execute python: print(2**100)"}
{"task": "create a PDF with title 'Report' and content 'Hello World'"}
{"task": "show the directory tree"}
{"task": "show top 10 running processes"}
```

### Financial Pipelines (v6.0)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/prompt` | Prompt-driven pipeline (12 workflows) |
| `POST` | `/prompt/classify` | Classify intent without executing |
| `POST` | `/prompt/async` | Async pipeline (returns job_id + ABM Grandmaster analytics) |
| `GET` | `/workflow/status/{job_id}` | Poll async job result |

### MiroFish ABM Grandmaster

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/abm/chat/agent` | Chat with any simulated agent (in-character response) |
| `POST` | `/abm/chat/report` | Converse with ReportAgent about simulation results |
| `GET` | `/abm/graph/{job_id}` | D3.js graph visualization data (nodes + links + coalitions) |
| `GET` | `/abm/predictions/{job_id}` | Trend predictions with confidence intervals (T+5/T+10/T+30) |
| `GET` | `/api/pipeline/runs` | List all persisted pipeline runs |
| `GET` | `/api/pipeline/{run_id}/state` | Load full state for a specific run |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Ollama + model status |
| `GET` | `/agents` | Agent → LLM assignments |
| `GET` | `/workflows` | Available workflow list |

---

## Project Structure

```
Multi_Agentic_Testing_V1/
│
├── app.py                     # FastAPI server (all endpoints + 4 ABM Grandmaster)
├── orchestrator_v6.py         # v6.0 Deep Research Engine (2000+ lines)
├── orchestrator.py            # v6.0 Financial pipelines
├── llm_router.py              # Agent → Ollama model registry
├── prompt_router.py           # 12-workflow intent classifier
├── workflow_config.yaml       # 12 workflow pipeline definitions
├── requirements.txt           # Python dependencies
├── .env                       # API keys (TAVILY, OLLAMA, etc.)
├── docker-compose.n8n.yml     # n8n self-hosted deployment
│
├── desktop_agent/             # Desktop Automation Module
│   ├── __init__.py            # Module exports
│   └── agent.py               # LLM tool router (21 tools)
│
├── src/                       # Core modules
│   ├── mcp.py                 # 25+ MCP connectors (MCPManager)
│   ├── graph_knowledge_manager.py  # GraphRAG + reality seeds + memory injection
│   ├── document_generator.py  # Report generation utilities
│   ├── evaluation.py          # Pipeline quality evaluation
│   ├── abm/                   # MiroFish ABM Simulation (Grandmaster)
│   │   ├── simulation.py      # Mesa model (Monte Carlo engine)
│   │   ├── agents.py          # Agent types with dynamic temporal memory
│   │   ├── analysis.py        # Saddle-point, coalitions, contagion, phase transitions
│   │   ├── report_agent.py    # LLM-powered narrative report (4000 tokens)
│   │   ├── contracts.py       # Pydantic/dataclass data contracts
│   │   ├── environment.py     # Dual-channel platform model
│   │   ├── graphrag.py        # [NEW] GraphRAG engine (NL query → subgraph)
│   │   ├── graph_visualizer.py# [NEW] D3.js, Mermaid, adjacency, temporal
│   │   ├── config_agent.py    # [NEW] LLM auto-config from data richness
│   │   ├── trend_predictor.py # [NEW] Ensemble forecast T+5/T+10/T+30
│   │   ├── chat_interface.py  # [NEW] Agent chat + ReportAgent chat
│   │   └── llm_utils.py       # Ollama LLM utilities
│   ├── epistemic_agent/       # Epistemic safety agent
│   │   └── desktop_tools/     # 9 tool classes (file, code, system, PDF, browser, etc.)
│   └── reasoning_agent/       # ReAct reasoning agent
│
├── Crewai_module/             # Financial Analysis Module (CrewAI)
│   ├── agents/                # Research, Financial, Investment agents
│   ├── tasks/                 # Task definitions per agent
│   ├── tools/                 # SEC, web search, sentiment tools
│   ├── evaluation/            # Judge validator + adversarial validation
│   └── rag/                   # ChromaDB RAG for filing analysis
│
├── Marketing/                 # Sales Operations Module
│   ├── agents/                # Secretary, Analyst agents
│   ├── services/              # LLM, Gmail, Sheets, RAG services
│   └── models/                # Pydantic data models
│
├── reports/                   # Generated research reports
│   ├── v6_research_*/         # v6.0 deep research outputs
│   └── due_diligence_*/       # Financial pipeline reports
│
├── .github/workflows/         # CI/CD pipeline (lint + test + Docker)
├── Dockerfile                 # Multi-stage production build
├── .gitignore                 # Git exclusions
└── tests/                     # Test suite
```

---

## Quick Start

### Prerequisites

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start Ollama with required models
ollama serve
ollama pull llama3.2:latest
ollama pull qwen3:8b
ollama pull qwen3:4b
ollama pull nomic-embed-text:latest

# 3. (Optional) Desktop Agent extras
pip install playwright browser-use fpdf2 pypdf
playwright install chromium

# 4. (Optional) n8n workflow automation
docker-compose -f docker-compose.n8n.yml up -d
```

### Run the Server

```bash
python -m uvicorn app:app --port 8080
```

### Test Desktop Agent

```bash
# List tools
curl http://localhost:8080/desktop/capabilities

# Execute a task
curl -X POST http://localhost:8080/desktop/execute \
  -H "Content-Type: application/json" \
  -d '{"task": "show system info"}'

# Check audit trail
curl http://localhost:8080/desktop/history
```

### Test Deep Research

```bash
curl -X POST http://localhost:8080/v6/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Is nuclear energy the best solution for climate change?",
    "max_iterations": 3,
    "dry_run": false
  }'
```

---

## Configuration

### `.env`

```env
OLLAMA_BASE_URL=http://localhost:11434

# LLM Models
RESEARCH_LLM_MODEL=llama3.2:latest
FINANCIAL_LLM_MODEL=qwen3:4b
ADVISOR_LLM_MODEL=qwen3:8b
EMBEDDING_MODEL=nomic-embed-text:latest
DESKTOP_LLM_MODEL=qwen3:8b

# API Keys
TAVILY_API_KEY=tvly-xxxxx
GITHUB_TOKEN=ghp_xxxxx          # Optional
YOUTUBE_API_KEY=AIza-xxxxx      # Optional
OPENWEATHERMAP_API_KEY=xxxxx    # Optional
NEWS_API_KEY=xxxxx              # Optional

# n8n (Optional)
N8N_BASE_URL=http://localhost:5678
N8N_RESEARCH_WEBHOOK=research-complete

# Google Services
# Requires OAuth2 credentials for Gmail + Sheets
```

---

## Production Hardening

| Feature | Implementation |
|---------|---------------|
| **Context Window Protection** | SemanticChunker (nomic-embed-text) reduces 160K→30K tokens/iteration |
| **Rate-Limit Safety** | Per-connector asyncio.Semaphore (Tavily=3, Reddit=1, etc.) |
| **Concurrency Cap** | Global /v6/research semaphore: max 3 simultaneous requests, 503 on overflow |
| **Citation Integrity** | Immutable fact_id→source_id JSON + orphan citation removal |
| **Fact Verification** | Verifier Agent (qwen3:4b) cross-checks facts from Extractor |
| **Quality Gate** | 3-tier fallback scorer (JSON → regex → metric-based) with configurable threshold |
| **Graceful Degradation** | All new connectors handle missing deps gracefully (Playwright, browser-use, n8n) |
| **Audit Trail** | Desktop Agent logs every execution with tool, params, duration, result |

## Report Output

Every deep research run saves 3 files to `reports/`:

| File | Contents |
|------|----------|
| `research_report.md` | Full publication-quality report with TOC, executive summary, analysis, methodology, sources |
| `meta.json` | Query metadata, quality score, agent timings, tools used |
| `facts.json` | All extracted facts with locked source citations and verification status |

Additionally, results are exported to **Google Sheets** and an email summary is sent via **Gmail**.

---

## n8n Workflow Automation (Optional)

n8n is a self-hosted workflow automation platform. After research completes, the orchestrator fires a webhook to n8n with research metadata.

```bash
# Deploy n8n
docker-compose -f docker-compose.n8n.yml up -d

# Access UI: http://localhost:5678
# Default login: admin / changeme
```

**Webhook payload sent after each research:**
```json
{
  "event": "research_complete",
  "query": "...",
  "domain": "financial",
  "quality_score": 8.5,
  "facts_count": 22,
  "report_path": "reports/v6_research_...",
  "duration_sec": 245.3
}
```

Use n8n to build visual workflows: Slack alerts, CRM updates, team notifications, dashboard refreshes.

---

## Docker Deployment

```bash
# Build
docker build -t multi-agent-orchestrator .

# Run (Ollama must be accessible)
docker run -p 8000:8000 --env-file .env \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  multi-agent-orchestrator
```

---

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push/PR to `main`:

| Job | What It Does |
|-----|--------------|
| **Lint** | `py_compile` all `.py` files for syntax errors |
| **Test** | `pytest tests/` with dependency install |
| **Docker Build** | Validates `Dockerfile` builds successfully |

---

## Changelog

### v6.0.2 (2026-03-21)

**MiroFish ABM Grandmaster Architecture (5 new modules):**
- `graphrag.py` — GraphRAG engine: NL query → subgraph context for agent prompts
- `graph_visualizer.py` — D3.js JSON, Mermaid diagrams, adjacency matrix, temporal snapshots
- `config_agent.py` — Auto-configure simulation params from pipeline data richness
- `trend_predictor.py` — Ensemble forecasts at T+5/T+10/T+30 with confidence intervals
- `chat_interface.py` — Chat with any agent (in-character) + ReportAgent conversations

**Enhanced Modules:**
- `graph_knowledge_manager.py` — Reality seed extraction, entity-triple extraction, individual/collective memory injection
- `agents.py` — 15-slot importance-weighted memory (up from 5), cross-platform discovery (20%), eviction/recall methods
- `analysis.py` — Phase transition false positive filter (delta ≥ 0.01 required)
- `report_agent.py` — max_tokens 2000→4000, timeout 120→180s
- `orchestrator.py` — Adversarial validation threshold 0.7→0.5, evidence window 1000→3000 chars, v5.0→v6.0 version strings

**4 New API Endpoints:**
- `POST /abm/chat/agent` — Chat with any simulated agent in-character
- `POST /abm/chat/report` — Converse with ReportAgent about simulation results
- `GET /abm/graph/{job_id}` — Graph visualization data (D3.js JSON)
- `GET /abm/predictions/{job_id}` — Trend predictions with confidence intervals

### v6.0.1 (2026-03-21)

**MiroFish ABM Pro Features:**
- Dual-channel platform support (Microblog + Forum)
- Pipeline-seeded agent personas from real financial data
- Contagion cascade detection (≥15% sentiment shift/tick)
- Phase transition detection (critical tipping points)
- Coalition formation via k-means clustering
- Saddle-point Monte Carlo analysis with path variance
- Agent memory buffers with exponential decay
- 8 new Pro analytics fields in API response
- Report preview increased to 2000 characters

**Bug Fixes (17 resolved):**
- Fixed SQL injection in ABM scenario injection
- Fixed memory leaks in `app.py` and `Crewai_module/api.py` (job cap + eviction)
- Fixed thread-safety issues with `settings.verbose`
- Replaced bare `except:` with `except Exception:` across codebase
- Added LLM availability guards in `graph_knowledge_manager.py`
- Fixed broken imports in `evaluation.py`
- Deprecated FMP tools with module-level warning
- Synced version strings across modules
- Environment variables for email addresses

**DevOps:**
- Added `.gitignore` with comprehensive exclusions
- Added multi-stage `Dockerfile` for production deployment
- Added GitHub Actions CI/CD pipeline (lint + test + Docker)
- Updated `requirements.txt` with `scikit-learn`, Google API deps
