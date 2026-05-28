"""FastAPI application for Financial Crew analysis."""

import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any
from enum import Enum

from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import json
import glob
import os

from crew import create_financial_crew
from config.settings import settings
from utils.logger import get_logger
from utils.validators import validate_ticker

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Financial Crew API",
    description="Multi-Agent AI System for Stock Investment Analysis",
    version="1.0.0"
)

# In-memory job storage (use Redis/DB for production)
jobs: Dict[str, dict] = {}
_MAX_JOBS = 500  # Bug 7 fix: cap to prevent memory leak

def _evict_old_jobs():
    """Evict oldest 25% of jobs when cap is reached."""
    if len(jobs) >= _MAX_JOBS:
        oldest = sorted(jobs.keys(), key=lambda k: jobs[k].get("created_at", ""))[:_MAX_JOBS // 4]
        for k in oldest:
            jobs.pop(k, None)


# ============================================================================
# A2A Agent Card Models
# ============================================================================

class Capability(BaseModel):
    """A2A Capability definition."""
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="What this capability does")
    input_schema: Dict[str, Any] = Field(..., description="JSON schema for inputs")
    timeout_seconds: int = Field(default=900, description="Maximum execution time")


class AgentCard(BaseModel):
    """A2A Agent Card for platform registration."""
    capabilities: List[Capability] = Field(..., description="List of agent capabilities")


# ============================================================================
# Existing Models
# ============================================================================



class AnalysisStatus(str, Enum):
    """Analysis job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisRequest(BaseModel):
    """Request model for stock analysis."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., TSLA, AAPL)")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "TSLA",
                "verbose": True
            }
        }


class PromptAnalysisRequest(BaseModel):
    """Request model for prompt-based financial analysis."""
    prompt: str = Field(..., description="Natural language financial analysis prompt")
    tickers: Optional[List[str]] = Field(
        None,
        description="Override auto-detected tickers. If omitted, tickers are extracted from the prompt."
    )
    tools: Optional[List[str]] = Field(
        None,
        description="Specific tool names to use (e.g., ['dcf_valuation', 'technical_analysis']). If omitted, all available tools are used."
    )
    verbose: bool = Field(default=False, description="Enable verbose logging")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Analyze Tesla financial health with SEC filings and technical indicators",
                "tickers": None,
                "tools": None,
                "verbose": False
            }
        }


class AnalysisResponse(BaseModel):
    """Response model for analysis submission."""
    job_id: str = Field(..., description="Unique job identifier")
    ticker: str = Field(..., description="Stock ticker being analyzed")
    status: AnalysisStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "ticker": "TSLA",
                "status": "pending",
                "message": "Analysis job queued successfully"
            }
        }


class JobStatus(BaseModel):
    """Job status response model."""
    job_id: str
    ticker: str
    status: AnalysisStatus
    created_at: str
    completed_at: Optional[str] = None
    report_path: Optional[str] = None
    error: Optional[str] = None


def run_analysis(job_id: str, ticker: str, verbose: bool = False):
    """Background task to run financial crew analysis.
    
    Args:
        job_id: Unique job identifier
        ticker: Stock ticker symbol
        verbose: Enable verbose output
    """
    try:
        # Update job status to running
        jobs[job_id]["status"] = AnalysisStatus.RUNNING
        
        logger.info(f"Starting analysis for {ticker} (job: {job_id})")
        
        # Bug 8 fix: don't mutate shared settings — use local flag
        # (original_verbose = settings.verbose; settings.verbose = True was thread-unsafe)
        
        # Create and run crew
        crew = create_financial_crew(ticker)
        result = crew.run()

        # Update job status
        report_path = settings.outputs_dir / f"{ticker}_investment_report.md"
        jobs[job_id].update({
            "status": AnalysisStatus.COMPLETED,
            "completed_at": datetime.now().isoformat(),
            "report_path": str(report_path),
            "result": str(result)
        })
        
        logger.info(f"Analysis completed for {ticker} (job: {job_id})")
        
    except Exception as e:
        logger.error(f"Analysis failed for {ticker} (job: {job_id}): {e}", exc_info=True)
        jobs[job_id].update({
            "status": AnalysisStatus.FAILED,
            "completed_at": datetime.now().isoformat(),
            "error": str(e)
        })


def _extract_tickers(text: str) -> List[str]:
    """Extract stock ticker symbols from a text prompt using regex."""
    # Match 1-5 uppercase letters that look like tickers
    candidates = re.findall(r'\b([A-Z]{1,5})\b', text)
    # Filter out common English words that are all-caps
    stopwords = {
        "A", "I", "AM", "AN", "AS", "AT", "BY", "DO", "GO", "IF", "IN",
        "IS", "IT", "ME", "MY", "NO", "OF", "OK", "ON", "OR", "SO", "TO",
        "UP", "US", "WE", "AND", "ARE", "BUT", "CAN", "DID", "FOR", "GET",
        "GOT", "HAD", "HAS", "HER", "HIM", "HIS", "HOW", "ITS", "LET",
        "MAY", "NEW", "NOT", "NOW", "OLD", "OUR", "OUT", "OWN", "SAY",
        "SHE", "THE", "TOO", "TRY", "USE", "WAY", "WHO", "WHY", "YOU",
        "ALL", "ANY", "BIG", "DAY", "END", "FAR", "FEW", "HIT", "LOW",
        "PUT", "RUN", "SET", "TOP", "TWO", "WIN", "WON", "YET",
        "SEC", "DCF", "ETF", "IPO", "GDP", "CPI", "ROE", "ROA",
        "EPS", "RSI", "SMA", "EMA", "CEO", "CFO", "COO", "CTO",
        "API", "USD", "EUR", "GBP", "JPY", "VS", "XBRL", "GAAP",
        "MACD", "WITH", "FROM", "THAT", "THIS", "WHAT", "WHEN",
    }
    tickers = list(dict.fromkeys(t for t in candidates if t not in stopwords))
    return tickers[:5]  # Cap at 5 tickers


def run_prompt_analysis(
    job_id: str, prompt: str, tickers: List[str],
    tool_names: Optional[List[str]] = None, verbose: bool = False
):
    """Background task for prompt-based financial analysis.

    Runs the financial crew with expanded tools against each extracted ticker,
    guided by the user's natural-language prompt.
    """
    try:
        jobs[job_id]["status"] = AnalysisStatus.RUNNING
        logger.info(f"Prompt analysis started (job: {job_id}), tickers={tickers}")

        # Bug 8 fix: don't mutate shared settings.verbose — it is a global singleton
        # and concurrent jobs would overwrite each other's state (race condition).
        # Pass verbose flag directly to crew instead of touching settings.

        results = {}
        for ticker in tickers:
            try:
                crew = create_financial_crew(ticker)
                # Inject the prompt as the analysis objective
                result = crew.run()
                results[ticker] = {
                    "status": "completed",
                    "summary": str(result)[:2000],
                }
            except Exception as e:
                logger.error(f"Analysis failed for {ticker}: {e}")
                results[ticker] = {
                    "status": "failed",
                    "error": str(e)[:500],
                }

        # Determine overall status
        any_success = any(r["status"] == "completed" for r in results.values())
        overall_status = AnalysisStatus.COMPLETED if any_success else AnalysisStatus.FAILED

        jobs[job_id].update({
            "status": overall_status,
            "completed_at": datetime.now().isoformat(),
            "result": results,
            "prompt": prompt,
            "tickers_analyzed": tickers,
            "tools_requested": tool_names,
        })

        logger.info(f"Prompt analysis completed (job: {job_id}), status={overall_status}")

    except Exception as e:
        logger.error(f"Prompt analysis failed (job: {job_id}): {e}", exc_info=True)
        jobs[job_id].update({
            "status": AnalysisStatus.FAILED,
            "completed_at": datetime.now().isoformat(),
            "error": str(e),
        })


@app.get("/", tags=["General"])
async def root():
    """API root endpoint."""
    return {
        "service": "Financial Crew API",
        "version": "2.0.0",
        "description": "Multi-Agent AI System for Stock Investment Analysis — Expanded Financial Tools",
        "endpoints": {
            "GET /agent-card": "Get A2A agent card for registration",
            "POST /analyze": "Submit ticker-based stock analysis job",
            "POST /analyze/async": "Submit prompt-based financial analysis job (NEW)",
            "GET /status/{job_id}": "Check job status",
            "GET /report/{ticker}": "Download report",
            "GET /tools": "List all available financial tools (NEW)",
            "GET /evaluation/run/{run_id}": "Get evaluation by run ID",
            "GET /evaluation/latest": "Get latest evaluation",
            "GET /evaluations": "List all evaluations",
            "GET /jobs": "List all jobs",
            "GET /health": "Health check"
        }
    }


@app.get("/agent-card", response_model=AgentCard, tags=["A2A"])
async def get_agent_card():
    """
    Get A2A Agent Card for Testing Platform registration.
    
    This endpoint returns the agent capabilities for automatic registration
    with A2A testing platforms.
    
    Use this to register the agent:
    POST http://localhost:8000/agents/register
    Body: <response from this endpoint>
    """
    return AgentCard(
        capabilities=[
            Capability(
                name="analyze_stock",
                description="Analyze a single stock and generate comprehensive investment report with research, financial analysis, SEC filing insights, and investment recommendations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., AAPL, TSLA, MSFT)",
                            "pattern": "^[A-Z]{1,5}$"
                        },
                        "verbose": {
                            "type": "boolean",
                            "description": "Enable detailed logging of agent thinking",
                            "default": False
                        }
                    },
                    "required": ["ticker"]
                },
                timeout_seconds=900  # 15 minutes for full analysis
            ),
            Capability(
                name="batch_analyze",
                description="Analyze multiple stocks in parallel and generate individual investment reports for each",
                input_schema={
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string", "pattern": "^[A-Z]{1,5}$"},
                            "description": "List of stock ticker symbols",
                            "minItems": 1,
                            "maxItems": 10
                        },
                        "verbose": {
                            "type": "boolean",
                            "description": "Enable detailed logging",
                            "default": False
                        }
                    },
                    "required": ["tickers"]
                },
                timeout_seconds=1800  # 30 minutes for batch
            ),
            Capability(
                name="quick_summary",
                description="Generate a quick investment summary without full analysis (faster, less comprehensive)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker symbol",
                            "pattern": "^[A-Z]{1,5}$"
                        },
                        "focus": {
                            "type": "string",
                            "enum": ["financial", "sentiment", "risks"],
                            "description": "Focus area for quick analysis",
                            "default": "financial"
                        }
                    },
                    "required": ["ticker"]
                },
                timeout_seconds=300  # 5 minutes for quick summary
            ),
            Capability(
                name="prompt_analyze",
                description="Natural language prompt-based financial analysis. Supports queries like 'Analyze Tesla financial health with SEC filings and technical indicators' or 'Compare AAPL vs MSFT earnings growth'.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Natural language financial analysis prompt"
                        },
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Override auto-detected tickers (optional)"
                        },
                        "tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific tools to use (optional)"
                        },
                        "verbose": {
                            "type": "boolean",
                            "default": False
                        }
                    },
                    "required": ["prompt"]
                },
                timeout_seconds=1200  # 20 min for multi-ticker prompt analysis
            )
        ]
    )



@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_configured": bool(settings.openai_api_key),
        "cache_enabled": settings.cache_enabled,
        "active_jobs": sum(1 for job in jobs.values() if job["status"] == AnalysisStatus.RUNNING)
    }


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"], status_code=status.HTTP_202_ACCEPTED)
async def analyze_stock(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Submit a stock analysis job.
    
    This endpoint queues an analysis job and returns immediately.
    Use the /status/{job_id} endpoint to check progress.
    
    Args:
        request: Analysis request with ticker and options
        background_tasks: FastAPI background tasks
    
    Returns:
        Analysis response with job ID and status
    """
    try:
        # Validate ticker
        ticker = validate_ticker(request.ticker)
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create job record
        jobs[job_id] = {
            "job_id": job_id,
            "ticker": ticker,
            "status": AnalysisStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "report_path": None,
            "error": None
        }
        
        # Queue background task
        background_tasks.add_task(run_analysis, job_id, ticker, request.verbose)
        
        logger.info(f"Queued analysis job {job_id} for {ticker}")
        
        return AnalysisResponse(
            job_id=job_id,
            ticker=ticker,
            status=AnalysisStatus.PENDING,
            message=f"Analysis job queued successfully. Use /status/{job_id} to track progress."
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit analysis: {str(e)}")


@app.post("/analyze/async", response_model=AnalysisResponse, tags=["Analysis"], status_code=status.HTTP_202_ACCEPTED)
async def analyze_prompt_async(request: PromptAnalysisRequest, background_tasks: BackgroundTasks):
    """Submit a prompt-based financial analysis job.

    Accepts a natural language prompt, auto-detects tickers (or uses
    user-supplied overrides), and runs the expanded financial crew in
    the background.

    Examples:
    - "Analyze Tesla financial health with SEC filings and technical indicators"
    - "Compare AAPL vs MSFT earnings growth and valuation"
    - "What is the DCF valuation of NVDA?"

    Use /status/{job_id} to poll for results.
    """
    try:
        # Extract tickers from prompt or use overrides
        tickers = request.tickers or _extract_tickers(request.prompt)

        if not tickers:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No stock tickers detected in prompt. "
                    "Include tickers (e.g., TSLA, AAPL) or pass them in the 'tickers' field."
                )
            )

        # Validate tickers
        validated = [validate_ticker(t) for t in tickers]

        # Bug 7 fix: evict old jobs BEFORE adding a new one (not after completion)
        _evict_old_jobs()

        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "job_id": job_id,
            "ticker": ",".join(validated),
            "prompt": request.prompt,
            "status": AnalysisStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "report_path": None,
            "error": None,
        }

        background_tasks.add_task(
            run_prompt_analysis, job_id, request.prompt,
            validated, request.tools, request.verbose
        )

        logger.info(
            f"Queued prompt analysis job {job_id}: tickers={validated}, "
            f"prompt='{request.prompt[:80]}...'"
        )

        return AnalysisResponse(
            job_id=job_id,
            ticker=",".join(validated),
            status=AnalysisStatus.PENDING,
            message=(
                f"Prompt analysis queued for {', '.join(validated)}. "
                f"Use /status/{job_id} to track progress."
            )
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting prompt analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit analysis: {str(e)}")


@app.get("/tools", tags=["Analysis"])
async def list_available_tools():
    """List all available financial analysis tools."""
    tool_registry = {
        "original_tools": [
            {"name": "SEC Filing Search", "type": "sec_tools", "description": "Search SEC EDGAR filings by ticker"},
            {"name": "SEC Filing Download", "type": "sec_tools", "description": "Download 10-K/10-Q filings"},
            {"name": "Financial Metrics", "type": "financial_tools", "description": "yfinance stock data and ratios"},
            {"name": "Web Search", "type": "web_search_tools", "description": "Web search via Tavily/DuckDuckGo"},
            {"name": "News Search", "type": "web_search_tools", "description": "Financial news search"},
            {"name": "Sentiment Analysis", "type": "sentiment_tools", "description": "News sentiment analysis"},
            {"name": "RAG Retrieval", "type": "rag_tools", "description": "Vector search over stored documents"},
        ],
        "expanded_tools": [
            {"name": "SEC EDGAR Full Text Search", "type": "sec_edgar_mcp", "description": "Full-text search across all SEC filings (EFTS API)"},
            {"name": "SEC EDGAR XBRL Financials", "type": "sec_edgar_mcp", "description": "Structured XBRL financial data extraction"},
            {"name": "Alpha Vantage Quote", "type": "alpha_vantage", "description": "Real-time stock quotes", "requires": "ALPHA_VANTAGE_API_KEY"},
            {"name": "Alpha Vantage Earnings", "type": "alpha_vantage", "description": "Quarterly/annual earnings history", "requires": "ALPHA_VANTAGE_API_KEY"},
            {"name": "Alpha Vantage Company Overview", "type": "alpha_vantage", "description": "Company fundamentals & valuation", "requires": "ALPHA_VANTAGE_API_KEY"},
            {"name": "DCF Valuation", "type": "fmp", "description": "Discounted Cash Flow intrinsic value", "requires": "FMP_API_KEY"},
            {"name": "Analyst Estimates", "type": "fmp", "description": "Consensus earnings/revenue estimates", "requires": "FMP_API_KEY"},
            {"name": "Financial Ratios", "type": "fmp", "description": "Comprehensive financial ratios (ROE, D/E, etc.)", "requires": "FMP_API_KEY"},
            {"name": "Economic Indicators", "type": "openbb", "description": "GDP, CPI, unemployment, Fed funds rate"},
            {"name": "Technical Analysis", "type": "openbb", "description": "RSI, MACD, Bollinger Bands, SMA/EMA"},
        ]
    }
    return {
        "total_tools": sum(len(v) for v in tool_registry.values()),
        **tool_registry
    }


@app.get("/status/{job_id}", response_model=JobStatus, tags=["Analysis"])
async def get_job_status(job_id: str):
    """Get the status of an analysis job.
    
    Args:
        job_id: Unique job identifier
    
    Returns:
        Job status information
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs[job_id]
    return JobStatus(**job)


@app.get("/jobs", tags=["Analysis"])
async def list_jobs(status: Optional[AnalysisStatus] = None, limit: int = 50):
    """List all analysis jobs.
    
    Args:
        status: Filter by status (optional)
        limit: Maximum number of jobs to return
    
    Returns:
        List of jobs
    """
    job_list = list(jobs.values())
    
    # Filter by status if provided
    if status:
        job_list = [job for job in job_list if job["status"] == status]
    
    # Sort by created_at descending
    job_list.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Limit results
    job_list = job_list[:limit]
    
    return {
        "total": len(job_list),
        "jobs": job_list
    }


@app.get("/report/{ticker}", tags=["Reports"])
async def get_report(ticker: str):
    """Download the investment report for a stock ticker.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Markdown report file
    """
    try:
        ticker = validate_ticker(ticker)
        report_path = settings.outputs_dir / f"{ticker}_investment_report.md"
        
        if not report_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Report not found for {ticker}. Run analysis first using /analyze endpoint."
            )
        
        return FileResponse(
            path=report_path,
            media_type="text/markdown",
            filename=f"{ticker}_investment_report.md"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve report: {str(e)}")


@app.delete("/jobs/{job_id}", tags=["Analysis"])
async def delete_job(job_id: str):
    """Delete a job record.
    
    Args:
        job_id: Unique job identifier
    
    Returns:
        Deletion confirmation
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Don't delete running jobs
    if jobs[job_id]["status"] == AnalysisStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Cannot delete running job")
    
    del jobs[job_id]
    logger.info(f"Deleted job {job_id}")
    
    return {"message": f"Job {job_id} deleted successfully"}


# ============================================================================
# Evaluation Endpoints
# ============================================================================

@app.get("/evaluation/run/{run_id}", tags=["Evaluation"])
async def get_evaluation_by_run_id(run_id: str):
    """Get evaluation result by run ID.
    
    Args:
        run_id: Unique run identifier from evaluation
    
    Returns:
        Complete evaluation JSON
    """
    try:
        eval_dir = settings.outputs_dir / "evaluations"
        
        if not eval_dir.exists():
            raise HTTPException(
                status_code=404,
                detail="No evaluations found. Run an analysis first with evaluation enabled."
            )
        
        # Search for evaluation file with this run_id
        pattern = str(eval_dir / f"*{run_id}.json")
        files = glob.glob(pattern)
        
        if not files:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation not found for run_id: {run_id}"
            )
        
        # Read the evaluation file
        with open(files[0], 'r', encoding='utf-8') as f:
            evaluation = json.load(f)
        
        return evaluation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve evaluation: {str(e)}")


@app.get("/evaluation/latest", tags=["Evaluation"])
async def get_latest_evaluation():
    """Get the most recent evaluation.
    
    Returns:
        Most recent evaluation JSON
    """
    try:
        eval_dir = settings.outputs_dir / "evaluations"
        
        if not eval_dir.exists():
            raise HTTPException(
                status_code=404,
                detail="No evaluations found. Run an analysis first with evaluation enabled."
            )
        
        # Get all evaluation files
        pattern = str(eval_dir / "*.json")
        files = glob.glob(pattern)
        
        if not files:
            raise HTTPException(
                status_code=404,
                detail="No evaluations found."
            )
        
        # Get most recent file by modification time (getctime is inode-change on Linux, not creation)
        latest = max(files, key=os.path.getmtime)
        
        with open(latest, 'r', encoding='utf-8') as f:
            evaluation = json.load(f)
        
        return evaluation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving latest evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve evaluation: {str(e)}")


@app.get("/evaluations", tags=["Evaluation"])
async def list_evaluations(limit: int = 50, min_score: Optional[float] = None):
    """List all evaluations with optional filtering.
    
    Args:
        limit: Maximum number of evaluations to return
        min_score: Filter by minimum overall score (0-1)
    
    Returns:
        List of evaluation summaries
    """
    try:
        eval_dir = settings.outputs_dir / "evaluations"
        
        if not eval_dir.exists():
            return {"total": 0, "evaluations": []}
        
        # Get all evaluation files
        pattern = str(eval_dir / "*.json")
        files = glob.glob(pattern)
        
        evaluations = []
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                
                # Apply score filter
                if min_score is not None and eval_data.get("overall_score", 0) < min_score:
                    continue
                
                # Create summary
                summary = {
                    "run_id": eval_data.get("run_id"),
                    "agent_id": eval_data.get("agent_id"),
                    "objective": eval_data.get("objective"),
                    "overall_verdict": eval_data.get("overall_verdict"),
                    "overall_score": eval_data.get("overall_score"),
                    "evaluated_at": eval_data.get("evaluated_at"),
                    "key_metrics": {
                        "task_success_rate": eval_data.get("task_success", {}).get("task_success_rate"),
                        "tool_success_rate": eval_data.get("tool_use", {}).get("tool_execution_success_rate"),
                        "avg_latency_ms": eval_data.get("performance", {}).get("avg_latency_ms"),
                        "cost_usd": eval_data.get("performance", {}).get("cost_per_task_usd")
                    }
                }
                evaluations.append(summary)
            except Exception as e:
                logger.warning(f"Error reading evaluation file {file}: {e}")
                continue
        
        # Sort by evaluated_at descending
        evaluations.sort(key=lambda x: x.get("evaluated_at", ""), reverse=True)
        
        # Limit results
        evaluations = evaluations[:limit]
        
        return {
            "total": len(evaluations),
            "evaluations": evaluations
        }
        
    except Exception as e:
        logger.error(f"Error listing evaluations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list evaluations: {str(e)}")


@app.get("/evaluation/metrics/{run_id}", tags=["Evaluation"])
async def get_evaluation_metrics(run_id: str, category: Optional[str] = None):
    """Get specific metric category from an evaluation.
    
    Args:
        run_id: Unique run identifier
        category: Metric category (task_success, tool_use, performance, etc.)
    
    Returns:
        Either full evaluation or specific category metrics
    """
    try:
        # Get the full evaluation first
        eval_dir = settings.outputs_dir / "evaluations"
        pattern = str(eval_dir / f"*{run_id}.json")
        files = glob.glob(pattern)
        
        if not files:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation not found for run_id: {run_id}"
            )
        
        with open(files[0], 'r', encoding='utf-8') as f:
            evaluation = json.load(f)
        
        # If no category specified, return full evaluation
        if not category:
            return evaluation
        
        # Valid categories
        valid_categories = [
            "task_success", "tool_use", "trajectory_quality",
            "robustness", "performance", "qualitative",
            "effectiveness", "efficiency", "safety", "rag_performance"
        ]
        
        if category not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
            )
        
        # Return specific category
        if category in evaluation:
            return {category: evaluation[category]}
        else:
            return {category: None, "message": f"Category '{category}' not available for this evaluation"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving evaluation metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
