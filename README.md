# agentic-backend

> Multi-module agentic AI backend. One FastAPI front door, one prompt
> router, one LLM router, and a fleet of specialised modules — CrewAI
> marketing crew, financial due-diligence, market-pulse, risk-scan,
> social-media research, agent-based simulation, and a desktop receipt
> processor.

`agentic-backend` is the platform that ties several independent agentic
modules into a single runnable backend. Every incoming request goes
through intent classification, then through model selection, and finally
to the right module's orchestrator. State for long-running research runs
is persisted under `src/reports/output/<run_id>/state.json` so a crashed
or paused run can be resumed.

---

## High-level request flow

```
HTTP request (FastAPI: app.py)
        │
        ▼
prompt_router.py        ← intent classification (LLM-driven)
        │
        ▼
llm_router.py           ← picks model per task type / provider
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                    Module dispatch                         │
│                                                            │
│  due-diligence ─►  orchestrator_v6.py                     │
│                   ├─ research                              │
│                   ├─ financial analysis                    │
│                   ├─ risk scan                             │
│                   └─ synthesis  →  reports/output/<id>/    │
│                                                            │
│  market-pulse  ─►  orchestrator_v6.py (market-pulse mode) │
│                                                            │
│  risk-scan     ─►  orchestrator_v6.py (risk-scan mode)    │
│                                                            │
│  marketing     ─►  Crewai_module/crew.py  (CrewAI)        │
│                                                            │
│  campaign      ─►  Marketing/  (Gmail OAuth)              │
│                                                            │
│  receipts      ─►  desktop_agent/receipt_processor.py     │
│                                                            │
│  abm sim       ─►  src/simulation_lite + orchestrator     │
└───────────────────────────────────────────────────────────┘
        │
        ▼
output:
  reports/v6_research_*/{facts.json, meta.json, research_report.md}
  src/reports/output/<id>/state.json
  Gmail / Sheets delivery (where applicable)
```

---

## The v6 orchestrator

`orchestrator_v6.py` is the workhorse for research-style runs (due-diligence,
market-pulse, risk-scan). It runs a multi-step graph:

1. **Plan** — decompose the user request into research questions.
2. **Gather** — Tavily / Exa search per question, with cache hits served
   from `diskcache` to cut repeated calls.
3. **Read** — fetch sources, extract structured facts, store with citations.
4. **Reconcile** — dedupe facts, mark contradictions for downstream
   resolution, attach confidence scores.
5. **Analyse** — module-specific analysis (financial ratios, market
   indicators, risk vectors).
6. **Synthesise** — assemble a Markdown report (`research_report.md`) with
   numbered citations and a structured `facts.json` companion.
7. **Persist** — write `state.json` after every checkpoint so the run is
   resumable.

The orchestrator is idempotent on `state.json`: if a step has already been
checkpointed, re-running picks up from there.

---

## Modules in detail

### Crewai_module — Marketing crew

A focused CrewAI deployment with its own evaluation harness:

```
Crewai_module/
├── api.py                   FastAPI endpoints scoped to the crew
├── config/settings.py       Crew-level config + provider routing
├── crew.py                  Crew definition (agents, tasks, tools)
└── evaluation/
    └── evaluation_service.py  Behavioural eval of crew outputs
```

The crew handles campaign-brief → research → draft → review loops.

### Marketing — Campaign pipeline

Independent Gmail-OAuth-integrated campaign runner. Uses the shared LLM
router to draft and personalise outbound messages, with a guardrail on
delivery rate.

### desktop_agent — Receipt processor

`desktop_agent/receipt_processor.py` runs locally over a folder of receipts,
extracts structured line-items, and posts them to a downstream accounting
endpoint.

### src/reports — Persisted output

Every research-style run lands here as a folder per run:

- `due_diligence_<entity>_<timestamp>/`
- `market_pulse_<entity>_<timestamp>/`
- `risk_scan_<entity>_<timestamp>/`
- `v6_research_<topic>_<timestamp>/`

Each folder contains `state.json` (resumable run state),
`research_report.md` (final report), `facts.json` (structured citations),
and `meta.json` (run metadata).

### tests/ — ABM, ledger, scenario

The simulation side ships its own test pack:

- `test_abm_population.py`, `test_abm_sandbox.py`,
  `test_kol_agent.py`, `test_kol_followers.py`, `test_orchestrator_abm.py`,
  `test_follow_graph.py`, `test_no_homogenization.py`,
  `test_bounded_assimilation.py`, `test_scenario_injection.py`,
  `test_monte_carlo.py`, `test_mc_agreement.py`,
  `test_simulation_lite.py`.
- Ledger & persistence:
  `test_ledger_crud.py`, `test_ledger_wal.py`, `test_state_persist.py`.
- Agents: `test_llm_agent.py`, `test_rule_agent.py`,
  `test_report_agent.py`, `test_agent_step.py`, `test_adversarial.py`.

---

## Routing

Two routers sit at the front:

- `prompt_router.py` — classifies user intent into one of the supported
  modules / orchestrator modes. Falls back to a default if the
  classification confidence is below threshold.
- `llm_router.py` — picks an actual model per task. Heavy reasoning →
  Claude / GPT; cheap classification → Ollama; embeddings → local. Honours
  provider-level rate limits.

---

## Quickstart

```bash
git clone https://github.com/krishddd/agentic-backend.git
cd agentic-backend
pip install -r requirements.txt
cp .env.example .env  # provider keys + TAVILY_API_KEY + Gmail OAuth, etc.

# Bring up the API
uvicorn app:app --reload --port 8000

# Or run a v6 orchestrator job directly
python orchestrator_v6.py --mode due_diligence --entity "Tesla Inc"
```

Inspect outputs:

```bash
ls src/reports/output/
cat src/reports/output/due_diligence_Tesla_Inc_*/research_report.md
```

Docker:

```bash
docker compose up --build
```

n8n integration: `docker-compose.n8n.yml` brings up an n8n companion that
can trigger or consume the backend through webhooks.

---

## Project structure

```
app.py                       FastAPI app factory
orchestrator.py              v5 orchestrator (legacy)
orchestrator_v6.py           v6 orchestrator (current — multi-step research)
llm_router.py                Model selection per task / provider
prompt_router.py             Intent classification
Crewai_module/               Marketing-focused CrewAI crew + evaluation
Marketing/                   Campaign pipeline + Gmail OAuth
desktop_agent/
└── receipt_processor.py     Local receipt extractor
src/
└── reports/                 Generated due-diligence / market-pulse / risk-scan
reports/                     v6_research_* run outputs
tests/                       ABM, agent, ledger, scenario tests (20+ files)
docker-compose.yml
docker-compose.n8n.yml       n8n companion
Dockerfile
workflow_config.yaml         End-to-end workflow definition
walkthrough.md               Long-form usage walkthrough
.github/workflows/ci.yml     Syntax check + pytest + Docker build
```

---

## Persisted run state

`state.json` shape (abbreviated):

```jsonc
{
  "run_id": "due_diligence_Tesla_Inc_20260326_115802",
  "mode": "due_diligence",
  "entity": "Tesla Inc",
  "steps": [
    { "name": "plan",   "status": "done",  "output_ref": "..." },
    { "name": "gather", "status": "done",  "output_ref": "..." },
    { "name": "read",   "status": "done",  "output_ref": "..." },
    { "name": "analyse", "status": "running" }
  ],
  "facts_path": "facts.json",
  "report_path": "research_report.md",
  "checkpoints": [ "...iso8601..." ]
}
```

Restarting `orchestrator_v6.py --resume <run_id>` picks up at `analyse`.

---

## CI

GitHub Actions runs:
- Python syntax compile-check on every `.py` file.
- pytest on the remaining 20 test files (the 3 importing non-existent
  modules were removed during the strict-CI pass).
- Docker build validation.
- `diskcache` was added to `requirements.txt` to fix a downstream import
  error in `cache.py`.

---

## Status

Personal portfolio. Designed as a launchpad to layer additional agentic
modules behind a single routing layer.

## License

MIT
