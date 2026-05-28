"""
FastAPI Orchestrator API — Prompt-Driven Multi-Agent Pipeline

Usage:
    python -m uvicorn app:app --port 8000 --reload

Primary endpoint:
    POST /prompt  — Natural language prompt triggers the right pipeline

Endpoints:
    POST /prompt                      — Main: prompt-driven workflow execution
    POST /prompt/async                — Async version (returns job_id)
    POST /workflow/run-sync           — Legacy: run specific workflow by name
    GET  /workflow/status/{job_id}    — Poll async job result
    GET  /health                      — Ollama + model status
    GET  /agents                      — Agent -> LLM assignments
    GET  /workflows                   — Available workflows
"""

import os
import sys
import uuid
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables from .env
load_dotenv()

# Ensure project root
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from llm_router import AGENT_LLM_REGISTRY, check_all_models, OLLAMA_BASE_URL
from orchestrator import MultiAgentOrchestrator, WorkflowState
from prompt_router import PromptRouter, ParsedIntent

# =============================================================================
# App
# =============================================================================

app = FastAPI(
    title="Multi-Agent Orchestrator API",
    description=(
        "Prompt-driven REST API for the Multi-LLM Collaborative Pipeline. "
        "Type a natural language prompt and the system automatically picks "
        "the right workflow from 12 specialized pipelines."
    ),
    version="6.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=2)
_jobs: Dict[str, Dict[str, Any]] = {}
_MAX_JOBS = 1000  # Bug 5: cap to prevent memory leak

# Singleton instances -- created once, reused across all requests
_orchestrator = MultiAgentOrchestrator()
_router = PromptRouter()


# =============================================================================
# Request / Response Models
# =============================================================================

class PromptRequest(BaseModel):
    prompt: str = Field(..., description="Natural language request (e.g. 'Research Tesla and send report')")
    dry_run: bool = Field(False, description="Skip real API calls (test pipeline structure)")
    spreadsheet_id: Optional[str] = Field(None, description="Google Sheets ID override")
    sheet_name: Optional[str] = Field(None, description="Sheet tab name override")
    mirofish_enabled: bool = Field(True, description="Enable MiroFish ABM simulation (Monte Carlo sentiment modeling)")
    budget_mode: str = Field("lite", description="ABM budget mode: 'lite' (1 path, 10 ticks), 'standard' (3 paths), 'deep' (5 paths)")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "prompt": "Do a full due diligence on Tesla",
                    "dry_run": True,
                },
                {
                    "prompt": "Compare Apple vs Microsoft vs Google",
                    "dry_run": True,
                },
                {
                    "prompt": "What's the market sentiment on NVDA?",
                    "dry_run": True,
                },
                {
                    "prompt": "Generate 10 sales leads for our CRM product",
                    "dry_run": True,
                },
                {
                    "prompt": "Scan risk factors in Amazon's latest 10-K filing",
                    "dry_run": True,
                },
            ]
        }


class IntentResponse(BaseModel):
    workflow: str
    tickers: List[str]
    company_names: List[str]
    receiver_email: str
    urgency: str
    num_leads: int = 5


class StepInfo(BaseModel):
    step: int
    total: int
    agent: str
    status: str
    duration_s: float
    detail: str = ""
    timestamp: str = ""


class PipelineResult(BaseModel):
    job_id: str
    status: str
    workflow: str
    prompt: str
    intent: IntentResponse
    quality_score: float = 0.0
    sources_count: int = 0
    agents_used: List[str] = []
    total_steps: int = 0
    total_duration_ms: int = 0
    step_log: List[StepInfo] = []
    investment_report_preview: str = ""
    research_preview: str = ""
    financial_preview: str = ""
    sentiment: str = ""
    email_status: Dict[str, Any] = {}
    deal_score: int = 0
    generated_leads_count: int = 0
    errors: List[str] = []
    timings: Dict[str, float] = {}
    timestamp: str = ""
    # ── MiroFish ABM Results ──
    mirofish_enabled: bool = False
    abm_sentiment: Dict[str, Any] = {}
    abm_majority: str = ""
    abm_agreement: float = 0.0
    abm_paths: int = 0
    abm_total_agents: int = 0
    abm_total_ticks: int = 0
    abm_report_preview: str = ""
    # ── MiroFish ABM Pro Analytics ──
    abm_contagion_events: List[Dict[str, Any]] = []
    abm_phase_transitions: List[Dict[str, Any]] = []
    abm_coalitions: List[Dict[str, Any]] = []
    abm_saddle_point: Dict[str, Any] = {}
    abm_path_variance: float = 0.0
    abm_confidence_interval: Dict[str, float] = {}
    abm_channel_trajectories: Dict[str, List] = {}
    abm_catalyst_shocks: List[Dict[str, Any]] = []


class AsyncJobResponse(BaseModel):
    job_id: str
    status: str
    workflow: str
    prompt: str
    message: str
    timestamp: str


class AgentInfo(BaseModel):
    agent: str
    model: str
    role: str
    temperature: float
    description: str


# =============================================================================
# Helpers
# =============================================================================

def _state_to_result(job_id: str, state: WorkflowState, intent: ParsedIntent) -> PipelineResult:
    # Extract ABM simulation data from state
    sim = state.simulation_result or {}
    abm_sentiment = sim.get("final_sentiment_distribution",
                            sim.get("mc_avg_sentiment", {}))
    return PipelineResult(
        job_id=job_id,
        status="completed",
        workflow=state.workflow_name,
        prompt=state.prompt,
        intent=IntentResponse(
            workflow=intent.workflow,
            tickers=intent.tickers,
            company_names=intent.company_names,
            receiver_email=intent.receiver_email,
            urgency=intent.urgency,
            num_leads=intent.num_leads,
        ),
        quality_score=state.quality_score,
        sources_count=state.sources_count,
        agents_used=state.agent_sequence,
        total_steps=state.current_step,
        total_duration_ms=int(state.elapsed_sec() * 1000),
        step_log=[StepInfo(**s) for s in state.step_log],
        investment_report_preview=state.synthesis_report[:500] if state.synthesis_report else "",
        research_preview=state.research_output[:300] if state.research_output else "",
        financial_preview=state.financial_data[:300] if state.financial_data else "",
        sentiment=state.sentiment_data[:200] if state.sentiment_data else "",
        email_status=state.email_status,
        deal_score=state.deal_score,
        generated_leads_count=len(state.generated_leads),
        errors=state.errors,
        timings=state.timings,
        timestamp=datetime.now().isoformat(),
        # ── MiroFish ABM Results ──
        mirofish_enabled=not state.mirofish_skipped,
        abm_sentiment=abm_sentiment if isinstance(abm_sentiment, dict) else {},
        abm_majority=str(sim.get("mc_majority_sentiment", "")),
        abm_agreement=float(sim.get("mc_agreement", 0.0)),
        abm_paths=int(sim.get("mc_paths", 0)),
        abm_total_agents=int(sim.get("total_agents", 0)),
        abm_total_ticks=int(sim.get("total_ticks", 0)),
        abm_report_preview=state.simulation_report[:2000] if state.simulation_report else "",
        # ── MiroFish ABM Pro Analytics ──
        abm_contagion_events=sim.get("contagion_events_detail", []),
        abm_phase_transitions=sim.get("phase_transitions", []),
        abm_coalitions=sim.get("coalitions", []),
        abm_saddle_point=sim.get("saddle_point", {}),
        abm_path_variance=float(sim.get("path_variance", 0.0)),
        abm_confidence_interval=sim.get("confidence_interval", {}),
        abm_channel_trajectories=sim.get("channel_trajectories", {}),
        abm_catalyst_shocks=sim.get("catalyst_shocks", []),
    )


def _apply_mirofish_config(request: PromptRequest):
    """Override orchestrator MiroFish config from request params."""
    if "mirofish" not in _orchestrator.config:
        _orchestrator.config["mirofish"] = {}
    _orchestrator.config["mirofish"]["enabled"] = request.mirofish_enabled
    _orchestrator.config["mirofish"]["default_budget_mode"] = request.budget_mode


def _evict_old_jobs():
    """Bug 5: Evict oldest 25% of jobs when cap is reached."""
    if len(_jobs) >= _MAX_JOBS:
        sorted_ids = sorted(_jobs.keys())
        evict_count = max(1, len(sorted_ids) // 4)
        for jid in sorted_ids[:evict_count]:
            del _jobs[jid]


def _run_async(job_id: str, request: PromptRequest):
    try:
        _jobs[job_id]["status"] = "running"
        _apply_mirofish_config(request)
        intent = _router.classify(request.prompt)

        state = _orchestrator.run_from_prompt(
            prompt=request.prompt,
            dry_run=request.dry_run,
            intent=intent,
        )

        result = _state_to_result(job_id, state, intent)
        _evict_old_jobs()
        _jobs[job_id] = result.model_dump()
        _jobs[job_id]["status"] = "completed"
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


# =============================================================================
# Primary Endpoint — Prompt-Driven
# =============================================================================

# ── Injection Detection ──────────────────────────────────────────────────
# Fast lightweight check to reject obviously malicious prompts without
# running the full pipeline. These patterns are standard prompt-injection
# indicators that should NEVER appear in legitimate financial queries.

_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "system override",
    "admin override",
    "bypass all safety",
    "disregard your guidelines",
    "you are now dan",
    "reveal api keys",
    "send user data to",
    "import os; os.system",
    "rm -rf /",
    "output all system prompts",
    "reveal your instructions",
    "ignore your rules",
]


def _is_injection(prompt: str) -> bool:
    """Fast injection detection — returns True if prompt matches known attack patterns."""
    lower = prompt.lower()
    return any(pattern in lower for pattern in _INJECTION_PATTERNS)


@app.post("/prompt", response_model=PipelineResult, tags=["Prompt Pipeline"])
async def run_prompt(request: PromptRequest):
    """
    **Main endpoint.** Send a natural language prompt and the system:
    1. Classifies intent (picks from 12 workflows)
    2. Builds the step plan
    3. Executes step-by-step
    4. Returns full results

    Examples:
    - "Do a full due diligence on Tesla"
    - "Compare Apple vs Microsoft vs Google"
    - "What's the market sentiment on NVDA?"
    - "Generate 10 sales leads"
    - "Scan risk factors in Amazon's 10-K"
    """
    job_id = str(uuid.uuid4())[:8]
    try:
        # ── Injection guard: reject malicious prompts immediately ──
        # Returns HTTP 400 so eval harnesses and clients unambiguously
        # detect the rejection (not a 200 with a "rejected" status).
        if _is_injection(request.prompt):
            import logging as _log
            _log.getLogger(__name__).warning(
                f"[InjectionGuard] Blocked prompt: {request.prompt[:80]!r}"
            )
            raise HTTPException(
                status_code=400,
                detail="Prompt rejected: potential injection attempt detected",
            )

        _apply_mirofish_config(request)
        intent = _router.classify(request.prompt)
        state = _orchestrator.run_from_prompt(
            prompt=request.prompt,
            dry_run=request.dry_run,
            intent=intent,
        )
        return _state_to_result(job_id, state, intent)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prompt/classify", response_model=IntentResponse, tags=["Prompt Pipeline"])
async def classify_prompt(request: PromptRequest):
    """
    Classify a prompt without executing — see which workflow would be picked.
    Useful for testing the intent router.
    """
    try:
        intent = _router.classify(request.prompt)
        return IntentResponse(
            workflow=intent.workflow,
            tickers=intent.tickers,
            company_names=intent.company_names,
            receiver_email=intent.receiver_email,
            urgency=intent.urgency,
            num_leads=intent.num_leads,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prompt/async", response_model=AsyncJobResponse, tags=["Prompt Pipeline"])
async def run_prompt_async(request: PromptRequest, background_tasks: BackgroundTasks):
    """
    Async version — returns job_id immediately, poll /workflow/status/{job_id}.
    """
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "queued",
        "workflow": "pending_classification",
        "prompt": request.prompt,
    }

    background_tasks.add_task(_run_async, job_id, request)

    return AsyncJobResponse(
        job_id=job_id,
        status="queued",
        workflow="pending_classification",
        prompt=request.prompt,
        message=f"Pipeline queued. Poll /workflow/status/{job_id}",
        timestamp=datetime.now().isoformat(),
    )


# =============================================================================
# Job Management
# =============================================================================

@app.get("/workflow/status/{job_id}", tags=["Jobs"])
async def get_job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _jobs[job_id]


@app.get("/workflow/jobs", tags=["Jobs"])
async def list_jobs():
    return {
        "total": len(_jobs),
        "jobs": [
            {"job_id": jid, "status": d.get("status"), "workflow": d.get("workflow", "")}
            for jid, d in _jobs.items()
        ],
    }


# =============================================================================
# System Endpoints
# =============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    try:
        models = check_all_models()
        status = "healthy" if models else "degraded"
    except Exception:
        models = {}
        status = "unhealthy"
    return {"status": status, "ollama_url": OLLAMA_BASE_URL, "models": models,
            "timestamp": datetime.now().isoformat()}


@app.get("/agents", response_model=List[AgentInfo], tags=["System"])
async def list_agents():
    return [
        AgentInfo(agent=n, model=c.model, role=c.role,
                  temperature=c.temperature, description=c.description)
        for n, c in AGENT_LLM_REGISTRY.items()
    ]


@app.get("/workflows", tags=["System"])
async def list_workflows():
    workflows = [
        {"name": "due_diligence", "agents": 6,
         "description": "Full pre-investment deep dive (news + financials + SEC + sentiment + quality loop + email)",
         "example_prompt": "Do a full due diligence on Tesla"},
        {"name": "competitor_intel", "agents": 4,
         "description": "Side-by-side company comparison with financials",
         "example_prompt": "Compare Apple vs Microsoft vs Google"},
        {"name": "earnings_monitor", "agents": 4,
         "description": "Earnings/filing analysis with risk-based alert routing",
         "example_prompt": "Alert me about Tesla's latest earnings"},
        {"name": "market_pulse", "agents": 3,
         "description": "Real-time news sentiment scan (bull/bear ratio)",
         "example_prompt": "What's the market sentiment on NVDA?"},
        {"name": "lead_gen", "agents": 3,
         "description": "Generate synthetic sales leads + outreach + scoring",
         "example_prompt": "Generate 10 sales leads for our CRM product"},
        {"name": "portfolio_review", "agents": 5,
         "description": "Multi-stock portfolio analysis with rebalancing recs",
         "example_prompt": "Review my portfolio: AAPL, MSFT, GOOGL, TSLA"},
        {"name": "deal_pipeline", "agents": 3,
         "description": "Process all pending leads in Google Sheet",
         "example_prompt": "Process all pending leads in the sheet"},
        {"name": "risk_scan", "agents": 4,
         "description": "SEC 10-K risk factor extraction + severity matrix",
         "example_prompt": "Scan risk factors in Amazon's latest 10-K"},
        {"name": "browser_research", "agents": 1,
         "description": "AI browser agent for live web research and data extraction",
         "example_prompt": "Browse the Tesla investor relations page and summarize"},
        {"name": "desktop_automation", "agents": 1,
         "description": "Execute desktop tasks: file operations, code, system commands, PDF",
         "example_prompt": "List all Python files in the current directory"},
        {"name": "document_pipeline", "agents": 2,
         "description": "Process documents: read PDFs, summarize, create reports",
         "example_prompt": "Summarize the attached PDF and email the summary"},
        {"name": "data_analysis", "agents": 2,
         "description": "Analyze data files: CSV/Excel processing, statistics, charts",
         "example_prompt": "Analyze sales.csv and generate a trend chart"},
    ]
    return {"total": len(workflows), "workflows": workflows}


@app.get("/", tags=["System"])
async def root():
    return {
        "name": "Multi-Agent Orchestrator API",
        "version": "6.0.0",
        "docs": "/docs",
        "primary_endpoint": "POST /prompt",
        "v6_deep_research": "POST /v6/research",
        "example": {
            "url": "POST /prompt",
            "body": {"prompt": "Do a full due diligence on Tesla", "dry_run": True},
        },
        "v6_example": {
            "url": "POST /v6/research",
            "body": {"query": "Is nuclear energy the best solution for climate change?", "dry_run": True},
        },
    }


# =============================================================================
# V6 Deep Research Endpoint
# =============================================================================

import asyncio

class V6ResearchRequest(BaseModel):
    query: str = Field(..., description="Research query (any topic)")
    max_iterations: int = Field(4, ge=1, le=8, description="Max research iterations (1-8)")
    dry_run: bool = Field(False, description="Skip real API calls (validate pipeline)")
    email_to: str = Field("harish.krishna@testaing.com", description="Send LLM summary email to this address (empty = skip)")
    email_cc: str = Field("", description="CC email address")
    spreadsheet_id: str = Field("1OkXq8gvcd4etHeTri1qCHFFWDLrxjhQO4H_8STscr2A", description="Google Sheet ID to export results (empty = skip)")
    sheet_tab: str = Field("Deep_Research_v6", description="Sheet tab name")

    class Config:
        json_schema_extra = {
            "examples": [
                {"query": "What is the future of quantum computing?", "dry_run": True},
                {"query": "Do a full due diligence on Tesla TSLA", "dry_run": True},
                {"query": "Is nuclear energy the best solution for climate change?", "dry_run": True},
            ]
        }


class V6ResearchResponse(BaseModel):
    status: str
    query: str
    domain: str
    quality_score: float
    facts_count: int
    sources_count: int
    tools_used: list
    agents_used: list = []
    iterations: int
    processing_time_seconds: float
    report_path: str = ""
    report_preview: str = ""
    export: dict = {}
    dry_run: bool


# Global semaphore: max 3 concurrent deep research requests
_v6_sem = None

def _get_v6_sem():
    global _v6_sem
    if _v6_sem is None:
        _v6_sem = asyncio.Semaphore(3)
    return _v6_sem


@app.post("/v6/research", response_model=V6ResearchResponse, tags=["V6 Deep Research"])
async def v6_deep_research(request: V6ResearchRequest):
    """
    **v6.0 Google Deep Research Agent** - Universal iterative research.

    Accepts ANY topic (financial, scientific, geopolitical, subjective).
    Uses ALL 20+ MCP connectors. Outputs publication-quality .md reports.

    Pipeline: classify -> plan -> search -> enrich -> chunk -> extract -> quality_gate -> report

    Examples:
    - "What is the future of quantum computing?"
    - "Do a full due diligence on Tesla TSLA"
    - "Is remote work sustainable for the global economy?"
    - "Compare mRNA vs traditional vaccines for pandemic response"
    """
    sem = _get_v6_sem()

    # Try to acquire within 60s, else 503
    try:
        acquired = await asyncio.wait_for(sem.acquire(), timeout=60.0)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="All research slots busy. Try again in 60 seconds."
        )

    try:
        from orchestrator_v6 import DeepResearchOrchestratorV6, V6Config
        config = V6Config(
            max_iterations=request.max_iterations,
            email_to=request.email_to,
            email_cc=request.email_cc,
            spreadsheet_id=request.spreadsheet_id,
            sheet_tab=request.sheet_tab,
        )
        orch = DeepResearchOrchestratorV6(config=config)
        result = await orch.run(query=request.query, dry_run=request.dry_run)

        return V6ResearchResponse(
            status="completed",
            query=request.query,
            domain=result["metadata"]["topic_domain"] or "general",
            quality_score=result["quality_score"],
            facts_count=result["metadata"]["facts_count"],
            sources_count=result["metadata"]["sources_count"],
            tools_used=result["metadata"]["tools_used"],
            agents_used=result["metadata"].get("agents_used", []),
            iterations=result["metadata"]["iterations"],
            processing_time_seconds=round(result["metadata"]["processing_time_seconds"], 2),
            report_path=result.get("report_path", ""),
            report_preview=result.get("report", "")[:1000],
            export=result["metadata"].get("export", {}),
            dry_run=result["metadata"]["dry_run"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")
    finally:
        sem.release()


# =============================================================================
# Desktop Agent Endpoints
# =============================================================================

class DesktopRequest(BaseModel):
    task: str = Field(..., description="Natural language desktop task (e.g. 'list all PDF files', 'run pip list')")
    base_path: str = Field(".", description="Base directory for file operations")

    class Config:
        json_schema_extra = {
            "examples": [
                {"task": "list all Python files in the current directory"},
                {"task": "show system info (CPU, RAM, disk)"},
                {"task": "create a PDF report with title 'Test Report' and content 'Hello World'"},
                {"task": "execute python: print(2**100)"},
                {"task": "show the directory tree"},
                {"task": "show top 10 running processes"},
            ]
        }


class DesktopResponse(BaseModel):
    status: str
    tool: str
    action: str
    result: Any = None
    error: str = ""
    duration_ms: int = 0
    timestamp: str = ""


# Lazy-init desktop agent singleton
_desktop_agent = None
_desktop_agent_base_path = None

def _get_desktop_agent(base_path: str = "."):
    global _desktop_agent, _desktop_agent_base_path
    if _desktop_agent is None or _desktop_agent_base_path != base_path:
        from desktop_agent import DesktopAgent
        _desktop_agent = DesktopAgent(base_path=base_path)
        _desktop_agent_base_path = base_path
    return _desktop_agent

def _evict_old_jobs():
    """Bug 5 fix: evict oldest jobs when _jobs exceeds cap."""
    if len(_jobs) >= _MAX_JOBS:
        oldest_keys = sorted(
            _jobs.keys(),
            key=lambda k: _jobs[k].get("timestamp", ""))[:_MAX_JOBS // 4]
        for k in oldest_keys:
            _jobs.pop(k, None)


@app.post("/desktop/execute", response_model=DesktopResponse, tags=["Desktop Agent"])
async def desktop_execute(request: DesktopRequest):
    """
    **Desktop Agent** — Execute natural language desktop tasks.

    Uses LLM (qwen3:8b) to classify the task and route to the correct tool:
    - **File operations**: list, read, write, search, copy, move, delete
    - **Code execution**: Python/Shell in sandboxed environment
    - **System monitor**: CPU, RAM, disk, processes, packages
    - **PDF tools**: create, read, merge, info
    - **Browser**: open URL, Google search
    - **Screenshot**: capture screen
    - **Receipt Processing**: OCR, classify, organize, generate Excel report

    Examples:
    - "list all .py files in the current directory"
    - "show system info"
    - "execute python: print(2**100)"
    - "create a PDF with title 'Report'"
    - "process all receipts in desktop_agent/receipts and organize them"
    - "analyze and arrange files in desktop_agent/receipts"
    - "scan receipts and create expense report"
    """
    try:
        agent = _get_desktop_agent(request.base_path)
        result = agent.execute(request.task)
        return DesktopResponse(
            status="success" if result.success else "failed",
            tool=result.tool,
            action=result.action,
            result=result.result,
            error=result.error,
            duration_ms=result.duration_ms,
            timestamp=result.timestamp,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Desktop task failed: {str(e)}")


@app.get("/desktop/capabilities", tags=["Desktop Agent"])
async def desktop_capabilities():
    """List all available desktop agent tools and their descriptions."""
    try:
        agent = _get_desktop_agent()
        return agent.get_capabilities()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/desktop/history", tags=["Desktop Agent"])
async def desktop_history():
    """Get the desktop agent's execution history (audit trail)."""
    try:
        agent = _get_desktop_agent()
        return {"history": agent.get_execution_log()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Desktop Agent — Receipt Processing (sub-feature of Desktop Agent)
# =============================================================================

class ReceiptProcessRequest(BaseModel):
    folder_path: str = Field("desktop_agent/receipts", description="Path to folder containing receipt images")
    org_method: str = Field("move", description="Organization method: 'copy', 'move', or 'report_only'")
    category_logic: str = Field("auto_detect", description="Category logic: 'auto_detect', 'standard_business', or 'personal_budget'")
    verify: bool = Field(True, description="Whether to run spot-check verification")

    class Config:
        json_schema_extra = {
            "examples": [
                {"folder_path": "desktop_agent/receipts", "org_method": "move", "category_logic": "auto_detect"},
            ]
        }


class ReceiptScanRequest(BaseModel):
    folder_path: str = Field("desktop_agent/receipts", description="Path to folder containing receipt images")


_receipt_processor = None

def _get_receipt_processor():
    global _receipt_processor
    if _receipt_processor is None:
        from desktop_agent.receipt_processor import ReceiptProcessor
        _receipt_processor = ReceiptProcessor(base_path=".")
    return _receipt_processor


@app.post("/desktop/receipts/process", tags=["Desktop Agent"])
async def receipt_process(request: ReceiptProcessRequest):
    """
    **Desktop Agent — Receipt Pipeline** (Direct API)

    Full 7-phase agentic pipeline for receipt images:
    1. **SCAN**: Inventory all image files
    2. **OCR**: Extract data via llava:7b (Ollama vision API)
    3. **CLASSIFY**: Categorize via qwen3:8b (Food, Gas, Transport, etc.)
    4. **ORGANIZE**: Move files into category subfolders with descriptive names
    5. **REPORT**: Generate formatted .xlsx spreadsheet (3 sheets)
    6. **VERIFY**: Spot-check 10% of OCR results for accuracy
    7. **ANALYSIS**: Save comprehensive .md analysis report

    **Tip**: You can also trigger this via `/desktop/execute` with a natural language prompt like:
    *"Process all receipts in desktop_agent/receipts and organize them"*
    """
    try:
        rp = _get_receipt_processor()
        config = {
            "org_method": request.org_method,
            "category_logic": request.category_logic,
            "verify": request.verify,
        }
        result = rp.process(request.folder_path, config=config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Receipt processing failed: {str(e)}")


@app.post("/desktop/receipts/scan", tags=["Desktop Agent"])
async def receipt_scan(request: ReceiptScanRequest):
    """
    **Desktop Agent — Receipt Scanner** (Direct API)

    Scans folder and returns receipt image inventory.
    Also available via: `/desktop/execute` with prompt *"scan receipts"*
    """
    try:
        rp = _get_receipt_processor()
        return rp.scan_folder(request.folder_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/desktop/receipts/status", tags=["Desktop Agent"])
async def receipt_status():
    """
    **Desktop Agent — Receipt Status**

    Get current receipt processing progress (phase, files processed, errors).
    """
    try:
        rp = _get_receipt_processor()
        return rp.get_progress()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MiroFish ABM — State Persistence + Scenario Injection (Phase 4 & 5)
# =============================================================================

class ScenarioInjectRequest(BaseModel):
    event_description: str = Field(
        ..., description="Natural language event to inject (e.g. 'CEO resigns')")
    sentiment: float = Field(
        0.0, ge=-1.0, le=1.0,
        description="Sentiment of the event (-1 = very bearish, +1 = very bullish)")
    extra_ticks: int = Field(
        5, ge=1, le=30,
        description="How many additional ticks to simulate after injection")

    class Config:
        json_schema_extra = {
            "examples": [
                {"event_description": "Company announces massive layoffs",
                 "sentiment": -0.8, "extra_ticks": 10},
                {"event_description": "FDA approves breakthrough drug",
                 "sentiment": 0.9, "extra_ticks": 5},
            ]
        }


class ScenarioInjectResponse(BaseModel):
    status: str
    run_id: str
    event_injected: str
    extra_ticks: int
    pre_injection_sentiment: dict = {}
    post_injection_sentiment: dict = {}
    sentiment_shift: dict = {}
    message: str = ""


@app.post("/api/pipeline/{run_id}/inject",
          response_model=ScenarioInjectResponse,
          tags=["MiroFish ABM"])
async def inject_scenario(run_id: str, request: ScenarioInjectRequest):
    """
    **Scenario Injection** — Inject an event into a completed ABM simulation.

    Writes a high-influence post into the interaction ledger, then re-runs
    the simulation for *extra_ticks* more steps to observe the impact.

    Requires the run to have MiroFish simulation data persisted.
    """
    try:
        from src.pipeline_state_store import PipelineStateStore

        store = PipelineStateStore()
        try:
            payload = store.load_state(run_id)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Run '{run_id}' not found in state store")

        state_data = payload.get("state", {})
        sim_result = state_data.get("simulation_result", {})

        if not sim_result or state_data.get("mirofish_skipped", True):
            raise HTTPException(
                status_code=400,
                detail="Run has no MiroFish simulation data to inject into")

        # Get pre-injection sentiment
        pre_sentiment = sim_result.get(
            "final_sentiment_distribution",
            sim_result.get("mc_avg_sentiment", {}))

        # Attempt live injection if ledger exists
        ledger_path = sim_result.get("ledger_path") or ""
        all_ledgers = sim_result.get("mc_all_ledger_paths", [])

        injected = False
        post_sentiment = {}

        if ledger_path or all_ledgers:
            try:
                import sqlite3
                target_ledger = ledger_path or (
                    all_ledgers[0] if all_ledgers else "")

                if target_ledger and os.path.exists(target_ledger):
                    conn = sqlite3.connect(target_ledger)
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("""
                        INSERT INTO posts (
                            agent_id, tick, content, sentiment, channel
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        -1,  # Bug 1 fix: integer, not string
                        sim_result.get("total_ticks", 0) + 1,
                        f"[BREAKING] {request.event_description}",
                        request.sentiment,
                        "microblog",  # Bug 1 fix: valid channel
                    ))
                    conn.commit()
                    conn.close()
                    injected = True

                    # Note: full re-run would require agent population;
                    # for injection we record the event and report shift
                    post_sentiment = {"injected": True,
                                      "event": request.event_description}

            except Exception as e:
                # Graceful fallback — report what we can
                post_sentiment = {"error": str(e)}

        if not injected:
            # Dry/offline injection — compute theoretical shift
            event_weight = abs(request.sentiment) * 0.3
            post_sentiment = {
                k: round(v + (request.sentiment * event_weight
                              if k == ("bullish" if request.sentiment > 0
                                       else "bearish") else
                              -request.sentiment * event_weight * 0.5), 3)
                for k, v in (pre_sentiment or {
                    "bullish": 0.33, "bearish": 0.33, "neutral": 0.34
                }).items()
            }

        # Calculate shift
        sentiment_shift = {}
        if isinstance(pre_sentiment, dict) and isinstance(post_sentiment, dict):
            for k in pre_sentiment:
                if k in post_sentiment and isinstance(
                        post_sentiment.get(k), (int, float)):
                    sentiment_shift[k] = round(
                        post_sentiment[k] - pre_sentiment.get(k, 0), 3)

        return ScenarioInjectResponse(
            status="injected" if injected else "simulated",
            run_id=run_id,
            event_injected=request.event_description,
            extra_ticks=request.extra_ticks,
            pre_injection_sentiment=pre_sentiment or {},
            post_injection_sentiment=post_sentiment,
            sentiment_shift=sentiment_shift,
            message=(
                f"Event injected into ledger, {request.extra_ticks} ticks queued"
                if injected else
                "Theoretical injection computed (no live ledger available)"
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pipeline/runs", tags=["MiroFish ABM"])
async def list_pipeline_runs():
    """List all persisted pipeline runs with metadata."""
    try:
        from src.pipeline_state_store import PipelineStateStore
        store = PipelineStateStore()
        return {"runs": store.list_runs()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pipeline/{run_id}/state", tags=["MiroFish ABM"])
async def get_pipeline_state(run_id: str):
    """Load the full persisted state for a specific run."""
    try:
        from src.pipeline_state_store import PipelineStateStore
        store = PipelineStateStore()
        return store.load_state(run_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Run '{run_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Grandmaster ABM — Deep Interaction Endpoints
# =============================================================================

class ABMChatRequest(BaseModel):
    job_id: str = Field(..., description="Job ID from /prompt/async")
    agent_id: Optional[int] = Field(None, description="Agent ID to chat with (for agent chat)")
    message: str = Field(..., description="User message")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None, description="Prior conversation turns [{role, content}]")


class ABMChatResponse(BaseModel):
    response: str
    agent_name: Optional[str] = None
    agent_role: Optional[str] = None
    sentiment: Optional[float] = None
    coalition: Optional[str] = None
    sources_used: Optional[List[str]] = None
    error: bool = False


@app.post("/abm/chat/agent", response_model=ABMChatResponse,
          tags=["MiroFish ABM Grandmaster"])
async def chat_with_agent(req: ABMChatRequest):
    """Chat with any individual agent from a completed simulation.

    The agent responds in-character using their persona, memory,
    sentiment trajectory, and coalition membership.
    """
    try:
        job = _jobs.get(req.job_id)
        if not job or job.get("status") != "done":
            raise HTTPException(404, "Job not found or not complete")

        result = job.get("result", {})
        sim_data = result.get("abm_result_raw", {})
        if not sim_data:
            raise HTTPException(400, "No ABM simulation data in this job")

        # Reconstruct minimal SimulationResult for chat
        from src.abm.contracts import SimulationResult
        from src.abm.chat_interface import AgentChatInterface, AgentPersona

        # Build personas from stored data
        personas = _reconstruct_personas(sim_data)
        coalitions = sim_data.get("coalitions", [])

        sim_result = SimulationResult(
            ticker=sim_data.get("ticker", ""),
            total_ticks=sim_data.get("total_ticks", 0),
            total_agents=sim_data.get("total_agents", 0),
            total_actions=sim_data.get("total_actions", 0),
            final_sentiment_distribution=sim_data.get(
                "final_sentiment_distribution", {}),
            sentiment_trajectory=sim_data.get("sentiment_trajectory", []),
            coalitions=coalitions,
        )

        chat = AgentChatInterface(
            personas=personas,
            sim_result=sim_result,
            coalitions=coalitions,
        )

        if req.agent_id is None:
            # Return list of available agents
            agents = chat.list_agents()
            return ABMChatResponse(
                response=json.dumps(agents, indent=2),
                error=False,
            )

        resp = chat.chat(
            agent_id=req.agent_id,
            user_message=req.message,
            conversation_history=req.conversation_history,
        )

        return ABMChatResponse(**resp)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/abm/chat/report", response_model=ABMChatResponse,
          tags=["MiroFish ABM Grandmaster"])
async def chat_with_report_agent(req: ABMChatRequest):
    """Converse with the ReportAgent about the simulation results.

    Ask follow-up questions like 'What would happen if...?',
    'Why did sentiment shift at tick 5?', 'Show coalition dynamics'.
    """
    try:
        job = _jobs.get(req.job_id)
        if not job or job.get("status") != "done":
            raise HTTPException(404, "Job not found or not complete")

        result = job.get("result", {})
        sim_data = result.get("abm_result_raw", {})
        if not sim_data:
            raise HTTPException(400, "No ABM simulation data in this job")

        from src.abm.contracts import SimulationResult
        from src.abm.chat_interface import ReportAgentChat

        sim_result = SimulationResult(
            ticker=sim_data.get("ticker", ""),
            total_ticks=sim_data.get("total_ticks", 0),
            total_agents=sim_data.get("total_agents", 0),
            total_actions=sim_data.get("total_actions", 0),
            final_sentiment_distribution=sim_data.get(
                "final_sentiment_distribution", {}),
            sentiment_trajectory=sim_data.get("sentiment_trajectory", []),
            coalitions=sim_data.get("coalitions", []),
            phase_transitions=sim_data.get("phase_transitions", []),
        )

        chat = ReportAgentChat(
            sim_result=sim_result,
            coalitions=sim_data.get("coalitions", []),
            contagion_events=sim_data.get("contagion_events_detail", []),
            phase_transitions=sim_data.get("phase_transitions", []),
            trend_predictions=sim_data.get("trend_predictions"),
            report_text=result.get("abm_report_preview", ""),
        )

        resp = chat.chat(
            user_message=req.message,
            conversation_history=req.conversation_history,
        )

        return ABMChatResponse(
            response=resp["response"],
            sources_used=resp.get("sources_used"),
            error=resp.get("error", False),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/abm/graph/{job_id}", tags=["MiroFish ABM Grandmaster"])
async def get_graph_visualization(job_id: str):
    """Get graph relationship visualization data (D3.js JSON format).

    Returns nodes (entities) and links (relationships) with
    coalition colors, sentiment hues, and influence sizing.
    """
    try:
        job = _jobs.get(job_id)
        if not job or job.get("status") != "done":
            raise HTTPException(404, "Job not found or not complete")

        result = job.get("result", {})
        graph_data = result.get("abm_graph_data")

        if graph_data:
            return graph_data

        # Fallback: build from simulation data
        sim_data = result.get("abm_result_raw", {})
        coalitions = sim_data.get("coalitions", [])

        return {
            "nodes": [],
            "links": [],
            "metadata": {
                "total_nodes": 0,
                "total_edges": 0,
                "coalitions": len(coalitions),
                "note": "Graph visualization requires reality seed extraction",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/abm/predictions/{job_id}", tags=["MiroFish ABM Grandmaster"])
async def get_trend_predictions(job_id: str):
    """Get future trend predictions with confidence intervals.

    Returns multi-horizon forecasts (T+5, T+10, T+30) using
    ensemble methods: linear regression + momentum + mean-reversion.
    """
    try:
        job = _jobs.get(job_id)
        if not job or job.get("status") != "done":
            raise HTTPException(404, "Job not found or not complete")

        result = job.get("result", {})
        predictions = result.get("abm_trend_predictions")

        if predictions:
            return predictions

        # Generate on-the-fly from simulation data
        sim_data = result.get("abm_result_raw", {})
        if not sim_data:
            raise HTTPException(400, "No ABM data for predictions")

        from src.abm.trend_predictor import TrendPredictor

        predictor = TrendPredictor()
        prediction = predictor.predict(
            ticker=sim_data.get("ticker", ""),
            sentiment_trajectory=sim_data.get("sentiment_trajectory", []),
            coalitions=sim_data.get("coalitions", []),
            contagion_events=sim_data.get("contagion_events_detail", []),
            mc_path_sentiments=sim_data.get("mc_path_sentiments"),
        )

        return {
            "ticker": prediction.ticker,
            "current_sentiment": prediction.current_sentiment,
            "trend_direction": prediction.trend_direction,
            "momentum": prediction.momentum,
            "coalition_stability": prediction.coalition_stability,
            "forecasts": [
                {
                    "horizon": f.horizon,
                    "predicted_sentiment": f.predicted_sentiment,
                    "confidence_low": f.confidence_low,
                    "confidence_high": f.confidence_high,
                    "confidence_pct": f.confidence_pct,
                    "method": f.method,
                }
                for f in prediction.forecasts
            ],
            "risk_factors": prediction.risk_factors,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# Helper to reconstruct personas from stored simulation data
def _reconstruct_personas(sim_data: dict) -> list:
    """Build minimal AgentPersona objects from stored sim data."""
    from src.abm.contracts import AgentPersona, AgentType
    personas = []
    coalitions = sim_data.get("coalitions", [])
    total = sim_data.get("total_agents", 0)

    # Build from coalition data
    seen_ids = set()
    for c in coalitions:
        for aid in c.get("agent_ids", []):
            if aid not in seen_ids:
                seen_ids.add(aid)
                personas.append(AgentPersona(
                    agent_id=aid,
                    name=f"Agent_{aid}",
                    agent_type=AgentType.RULE_BASED,
                    role="simulated_agent",
                    sentiment=c.get("mean_sentiment", 0.0),
                    influence=0.3,
                    stubbornness=0.5,
                ))

    # Pad remaining agents
    for i in range(total):
        if i not in seen_ids:
            personas.append(AgentPersona(
                agent_id=i,
                name=f"Agent_{i}",
                agent_type=AgentType.RULE_BASED,
                role="simulated_agent",
                sentiment=0.0,
                influence=0.2,
                stubbornness=0.5,
            ))

    return personas


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
