# agentic-backend

> Multi-module agentic AI backend — research, due-diligence, marketing, and
> desktop automation behind a single API.

A Python backend that orchestrates several specialised AI modules through one
LLM router and prompt router. Modules currently include a CrewAI marketing
crew, financial due-diligence and market-pulse pipelines, risk-scan,
social-media research, and a desktop receipt processor.

## Features

- **LLM router** — selects an appropriate model per task type and provider.
- **Prompt router** — intent classification routes user requests to the right
  module.
- **v6 orchestrator** — multi-step research workflows with persisted state in
  `src/reports/output/`.
- **CrewAI module** — marketing-focused multi-agent crew with its own
  evaluation harness.
- **Marketing module** — independent campaign pipeline (Gmail / OAuth).
- **Desktop agent** — receipt processor for finance automation.
- **Reports** — due-diligence, market-pulse, risk-scan output as structured
  JSON + Markdown.
- **Dockerised** — Dockerfile, docker-compose, n8n integration, GitLab-CI
  SAST template.

## Tech stack

Python · FastAPI · CrewAI · LangChain · diskcache · Docker · n8n

## Quickstart

```bash
git clone https://github.com/krishddd/agentic-backend.git
cd agentic-backend
pip install -r requirements.txt
cp .env.example .env  # add provider keys, TAVILY_API_KEY, etc.

# Start the API
uvicorn app:app --reload --port 8000
# or run the v6 orchestrator directly
python orchestrator_v6.py
```

## Project structure

```
app.py                FastAPI entry point
orchestrator.py       v5 orchestrator
orchestrator_v6.py    v6 orchestrator (preferred)
llm_router.py         Model selection
prompt_router.py      Intent classification
Crewai_module/        Marketing-focused CrewAI crew
Marketing/            Campaign pipeline + Gmail OAuth
desktop_agent/        Receipt processor
src/reports/          Generated due-diligence / market-pulse / risk-scan
tests/                ABM, agent, ledger, scenario tests
```

## Status

Personal portfolio project. CI runs syntax check, pytest, and Docker build on
every push.

## License

MIT
