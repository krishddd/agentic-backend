"""
FastAPI Wrapper for Testing Platform Integration

This wrapper exposes the Sales Operations AI Agent for the Low-Code Agentic Testing Platform.
Run on Port 9000 to avoid conflicts with the Testing Platform (Port 8000).

Usage:
    python fastapi_wrapper.py
    # Or with uvicorn:
    uvicorn fastapi_wrapper:app --host 0.0.0.0 --port 9000 --reload
"""
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from services.sheets_service import SheetsService
from services.llm_service import get_llm_service
from services.data_generator_service import get_data_generator
from services.retry_handler import get_retry_handler, CircuitBreakerOpen
from services.metrics_service import get_metrics_collector
from services.audit_logger import get_audit_logger
from services.validator import WorkflowRequest, validate_workflow_request, WorkflowStatus
from agents.secretary import AgentSecretary
from agents.analyst import AgentAnalyst
from utils.logger import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# =============================================================================
# FastAPI App - Testing Platform Compatible
# =============================================================================

app = FastAPI(
    title="Sales Ops Agent API",
    description="""
    Sales Operations AI Agent API for Testing Platform Integration.
    
    ## Agents
    - **Secretary Agent**: Processes pending rows, generates emails via LLM+RAG, sends via Gmail
    - **Analyst Agent**: Analyzes sent rows, scores deals, detects risks
    
    ## Testing Platform Endpoints
    - `/secretary` - Trigger Secretary agent
    - `/analyst` - Trigger Analyst agent
    - `/generate-data` - Generate synthetic lead data in Google Sheets
    - `/workflow` - Run complete PEER workflow
    - `/agent-card` - Get A2A Agent Card for registration
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - Allow Testing Platform access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Request/Response Models (A2A Compatible)
# =============================================================================

class SheetRequest(BaseModel):
    """Request model for sheet operations"""
    spreadsheet_id: str = Field(
        default=settings.SPREADSHEET_ID or "",
        description="Google Spreadsheet ID"
    )
    sheet_name: str = Field(
        default=settings.SHEET_NAME,
        description="Sheet name within the spreadsheet"
    )

class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    RUNNING = "running"

class AgentResult(BaseModel):
    """Standardized agent result for Testing Platform"""
    status: ExecutionStatus
    agent: str
    message: str
    stats: Dict[str, Any] = {}
    trajectory: List[Dict[str, Any]] = []  # For PEER review
    timestamp: str

class WorkflowResult(BaseModel):
    """Result for complete workflow execution"""
    status: ExecutionStatus
    workflow: str = "PEER"
    agents: List[AgentResult]
    total_duration_ms: int
    timestamp: str

class GenerateDataRequest(BaseModel):
    """Request model for data generation"""
    spreadsheet_id: str = Field(
        default=settings.SPREADSHEET_ID or "",
        description="Google Spreadsheet ID"
    )
    num_rows: int = Field(default=5, ge=1, le=20, description="Number of rows to generate")
    sheet_name: str = Field(default=settings.SHEET_NAME, description="Target sheet name")

class Capability(BaseModel):
    """Capability definition for Testing Platform"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    timeout_seconds: int = 120

class AgentCard(BaseModel):
    """Agent Card for registration with Testing Platform"""
    name: str = "Sales Ops Agent"
    description: str = "Dual-agent system for sales communication and intelligence"
    endpoint: str = "http://127.0.0.1:9000"
    sse_endpoint: Optional[str] = None
    capabilities: List[Capability]
    auth_required: bool = False
    auth_type: Optional[str] = None
    tags: List[str] = ["sales", "marketing", "gmail", "sheets"]
    metadata: Dict[str, Any] = {"version": "2.0.0"}

# =============================================================================
# Helper Functions
# =============================================================================

def create_trajectory_entry(action: str, result: str, metadata: Dict = None) -> Dict:
    """Create a trajectory entry for PEER evaluation"""
    return {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "result": result,
        "metadata": metadata or {}
    }

# =============================================================================
# Testing Platform Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Sales Ops Agent API",
        "version": "2.0.0",
        "platform": "Testing Platform Compatible",
        "docs": "/docs",
        "endpoints": {
            "secretary": "POST /secretary",
            "analyst": "POST /analyst",
            "workflow": "POST /workflow",
            "agent_card": "GET /agent-card"
        }
    }


@app.get("/health")
async def health():
    """Health check for Testing Platform monitoring"""
    try:
        llm = get_llm_service()
        llm_ok = llm.is_available()
    except Exception:
        llm_ok = False
    
    return {
        "status": "healthy" if llm_ok else "degraded",
        "llm_available": llm_ok,
        "spreadsheet_configured": bool(settings.SPREADSHEET_ID),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/agent-card", response_model=AgentCard)
async def get_agent_card():
    """
    Get A2A Agent Card for Testing Platform registration.
    
    Use this to register the agent:
    POST http://localhost:8000/agents/register
    Body: <response from this endpoint>
    """
    return AgentCard(
        capabilities=[
            Capability(
                name="secretary",
                description="Process pending rows, draft/send emails, and update sheet",
                input_schema={
                    "type": "object",
                    "properties": {
                        "spreadsheet_id": {"type": "string"},
                        "sheet_name": {"type": "string", "default": settings.SHEET_NAME}
                    }
                },
                timeout_seconds=120
            ),
            Capability(
                name="analyst",
                description="Analyze sent emails, score deals, and check for risks",
                input_schema={
                    "type": "object",
                    "properties": {
                        "spreadsheet_id": {"type": "string"},
                        "sheet_name": {"type": "string", "default": settings.SHEET_NAME}
                    }
                },
                timeout_seconds=120
            ),
            Capability(
                name="generator",
                description="Generate synthetic sales lead data using LLM and append to Google Sheets",
                input_schema={
                    "type": "object",
                    "properties": {
                        "spreadsheet_id": {"type": "string"},
                        "num_rows": {"type": "integer", "default": 5}
                    }
                },
                timeout_seconds=120
            )
        ]
    )


@app.post("/secretary", response_model=AgentResult)
async def run_secretary(request: SheetRequest = SheetRequest()):
    """
    Trigger the Secretary Agent to process pending rows.
    
    **Actions:**
    1. Reads rows with Mail_Status = 'pending'
    2. Uses LLM + RAG to generate personalized emails
    3. Sends emails via Gmail API
    4. Updates Mail_Status to 'sent' or 'failed'
    
    **For Testing Platform:**
    - Returns trajectory for PEER evaluation
    - Includes detailed stats for benchmarking
    """
    trajectory = []
    start_time = datetime.now()
    
    try:
        spreadsheet_id = request.spreadsheet_id or settings.SPREADSHEET_ID
        if not spreadsheet_id:
            raise HTTPException(status_code=400, detail="No spreadsheet_id provided and SPREADSHEET_ID not configured")
        
        trajectory.append(create_trajectory_entry(
            "initialize",
            "success",
            {"spreadsheet_id": spreadsheet_id, "sheet_name": request.sheet_name}
        ))
        
        # Initialize services
        sheets = SheetsService(spreadsheet_id, request.sheet_name)
        agent = AgentSecretary(sheets=sheets)
        
        trajectory.append(create_trajectory_entry(
            "agent_created",
            "success",
            {"agent_type": "AgentSecretary"}
        ))
        
        # Run agent
        logger.info(f"[Wrapper] Running Secretary Agent on: {spreadsheet_id}")
        stats = agent.run()
        
        trajectory.append(create_trajectory_entry(
            "agent_run",
            "success",
            {"stats": stats}
        ))
        
        status = ExecutionStatus.SUCCESS
        if stats.get('failed', 0) > 0:
            status = ExecutionStatus.PARTIAL if stats.get('sent', 0) > 0 else ExecutionStatus.FAILED
        
        return AgentResult(
            status=status,
            agent="secretary",
            message=f"Processed: {stats.get('processed', 0)}, Sent: {stats.get('sent', 0)}, Failed: {stats.get('failed', 0)}",
            stats=stats,
            trajectory=trajectory,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"[Wrapper] Secretary Agent error: {e}")
        trajectory.append(create_trajectory_entry(
            "error",
            "failed",
            {"error": str(e)}
        ))
        
        return AgentResult(
            status=ExecutionStatus.FAILED,
            agent="secretary",
            message=f"Error: {str(e)}",
            stats={},
            trajectory=trajectory,
            timestamp=datetime.now().isoformat()
        )


@app.post("/analyst", response_model=AgentResult)
async def run_analyst(request: SheetRequest = SheetRequest()):
    """
    Trigger the Analyst Agent to analyze sent rows.
    
    **Actions:**
    1. Reads rows with Mail_Status = 'sent'
    2. Uses LLM to score deals and detect risks
    3. Updates Deal_Score and Risk_Flags columns
    4. Sends alert emails for high-score leads
    
    **For Testing Platform:**
    - Returns trajectory for PEER evaluation
    - Includes detailed stats for benchmarking
    """
    trajectory = []
    
    try:
        spreadsheet_id = request.spreadsheet_id or settings.SPREADSHEET_ID
        if not spreadsheet_id:
            raise HTTPException(status_code=400, detail="No spreadsheet_id provided and SPREADSHEET_ID not configured")
        
        trajectory.append(create_trajectory_entry(
            "initialize",
            "success",
            {"spreadsheet_id": spreadsheet_id, "sheet_name": request.sheet_name}
        ))
        
        # Initialize services
        sheets = SheetsService(spreadsheet_id, request.sheet_name)
        agent = AgentAnalyst(sheets=sheets)
        
        trajectory.append(create_trajectory_entry(
            "agent_created",
            "success",
            {"agent_type": "AgentAnalyst"}
        ))
        
        # Run agent
        logger.info(f"[Wrapper] Running Analyst Agent on: {spreadsheet_id}")
        stats = agent.run()
        
        trajectory.append(create_trajectory_entry(
            "agent_run",
            "success",
            {"stats": stats}
        ))
        
        return AgentResult(
            status=ExecutionStatus.SUCCESS,
            agent="analyst",
            message=f"Analyzed: {stats.get('analyzed', 0)}, High Score: {stats.get('high_score', 0)}",
            stats=stats,
            trajectory=trajectory,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"[Wrapper] Analyst Agent error: {e}")
        trajectory.append(create_trajectory_entry(
            "error",
            "failed",
            {"error": str(e)}
        ))
        
        return AgentResult(
            status=ExecutionStatus.FAILED,
            agent="analyst",
            message=f"Error: {str(e)}",
            stats={},
            trajectory=trajectory,
            timestamp=datetime.now().isoformat()
        )


@app.post("/generate-data", response_model=AgentResult)
async def generate_data(request: GenerateDataRequest = GenerateDataRequest()):
    """
    Generate synthetic lead data and append to Google Sheets.
    
    **Actions:**
    1. Calls LLM to generate `num_rows` of realistic sales call data
    2. Prepends/Appends unique Row_IDs based on current sheet state
    3. Adds generation timestamps
    4. Writes data to the configured Google Sheet
    """
    trajectory = []
    
    try:
        spreadsheet_id = request.spreadsheet_id or settings.SPREADSHEET_ID
        if not spreadsheet_id:
            raise HTTPException(status_code=400, detail="No spreadsheet_id provided")
            
        trajectory.append(create_trajectory_entry(
            "initialize_generation",
            "success",
            {"num_rows": request.num_rows, "spreadsheet_id": spreadsheet_id}
        ))
        
        # Initialize services
        sheets = SheetsService(spreadsheet_id, request.sheet_name)
        generator = get_data_generator()
        
        # 1. Get current max ID to avoid collisions
        max_id = sheets.get_max_row_id()
        trajectory.append(create_trajectory_entry("query_sheet", "success", {"current_max_id": max_id}))
        
        # 2. Generate raw data from LLM
        raw_rows = generator.generate_rows(request.num_rows)
        trajectory.append(create_trajectory_entry("llm_generate", "success", {"rows_count": len(raw_rows)}))
        
        # 3. Prepare for sheets (add IDs, status, timestamps)
        final_rows = generator.prepare_for_sheets(raw_rows, max_id)
        
        # 4. Write to sheets
        sheets.append_rows(final_rows)
        trajectory.append(create_trajectory_entry("sheet_append", "success", {"appended_count": len(final_rows)}))
        
        return AgentResult(
            status=ExecutionStatus.SUCCESS,
            agent="generator",
            message=f"Successfully generated and appended {len(final_rows)} rows to sheet.",
            stats={"generated": len(final_rows), "start_id": max_id + 1, "end_id": max_id + len(final_rows)},
            trajectory=trajectory,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"[Wrapper] Generation error: {e}")
        trajectory.append(create_trajectory_entry("error", "failed", {"error": str(e)}))
        
        return AgentResult(
            status=ExecutionStatus.FAILED,
            agent="generator",
            message=f"Error generating data: {str(e)}",
            stats={},
            trajectory=trajectory,
            timestamp=datetime.now().isoformat()
        )


@app.post("/workflow", response_model=WorkflowResult)
async def run_workflow(request: SheetRequest = SheetRequest()):
    """
    Run complete PEER workflow: Secretary → Analyst
    
    This is the "Happy Path" workflow for Testing Platform:
    1. Node 1: Secretary Agent (process pending, send emails)
    2. Node 2: Analyst Agent (analyze sent, score deals)
    
    Returns combined trajectory for PEER evaluation.
    """
    import time
    start_time = time.time()
    agents_results = []
    
    try:
        spreadsheet_id = request.spreadsheet_id or settings.SPREADSHEET_ID
        if not spreadsheet_id:
            raise HTTPException(status_code=400, detail="No spreadsheet_id provided")
        
        sheets = SheetsService(spreadsheet_id, request.sheet_name)
        
        # Step 1: Secretary Agent
        logger.info("[Wrapper] Workflow Step 1: Secretary Agent")
        secretary = AgentSecretary(sheets=sheets)
        sec_stats = secretary.run()
        
        agents_results.append(AgentResult(
            status=ExecutionStatus.SUCCESS if sec_stats.get('failed', 0) == 0 else ExecutionStatus.PARTIAL,
            agent="secretary",
            message=f"Sent: {sec_stats.get('sent', 0)}, Failed: {sec_stats.get('failed', 0)}",
            stats=sec_stats,
            trajectory=[create_trajectory_entry("secretary_run", "complete", sec_stats)],
            timestamp=datetime.now().isoformat()
        ))
        
        # Step 2: Analyst Agent
        logger.info("[Wrapper] Workflow Step 2: Analyst Agent")
        analyst = AgentAnalyst(sheets=sheets)
        ana_stats = analyst.run()
        
        agents_results.append(AgentResult(
            status=ExecutionStatus.SUCCESS,
            agent="analyst",
            message=f"Analyzed: {ana_stats.get('analyzed', 0)}, High Score: {ana_stats.get('high_score', 0)}",
            stats=ana_stats,
            trajectory=[create_trajectory_entry("analyst_run", "complete", ana_stats)],
            timestamp=datetime.now().isoformat()
        ))
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Determine overall status
        all_success = all(r.status == ExecutionStatus.SUCCESS for r in agents_results)
        any_failed = any(r.status == ExecutionStatus.FAILED for r in agents_results)
        
        return WorkflowResult(
            status=ExecutionStatus.SUCCESS if all_success else (ExecutionStatus.FAILED if any_failed else ExecutionStatus.PARTIAL),
            workflow="PEER",
            agents=agents_results,
            total_duration_ms=duration_ms,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"[Wrapper] Workflow error: {e}")
        duration_ms = int((time.time() - start_time) * 1000)
        
        return WorkflowResult(
            status=ExecutionStatus.FAILED,
            workflow="PEER",
            agents=agents_results,
            total_duration_ms=duration_ms,
            timestamp=datetime.now().isoformat()
        )


@app.post("/workflow/async")
async def run_workflow_async_enhanced(
    request: SheetRequest,
    background_tasks: BackgroundTasks,
    priority: int = Query(default=5, ge=1, le=10, description="Execution priority (1-10)")
):
    """
    Run PEER workflow in background with production-grade capabilities.
    
    **Features:**
    - ✅ Automatic retry with exponential backoff
    - ✅ Real-time metrics collection
    - ✅ Complete audit trail (PII-masked)
    - ✅ Input validation & sanitization
    - ✅ Circuit breaker protection
    - ✅ Status tracking via thread_id
    
    **Use this for:**
    - Long-running workflows
    - Bulk processing
    - Production deployments
    """
    # 1. VALIDATE INPUT
    try:
        spreadsheet_id = request.spreadsheet_id or settings.SPREADSHEET_ID
        if not spreadsheet_id:
            raise HTTPException(status_code=400, detail="No spreadsheet_id provided")
        
        validated_request = WorkflowRequest(
            spreadsheet_id=spreadsheet_id,
            sheet_name=request.sheet_name,
            priority=priority,
            user_id="api_user"
        )
    except ValidationError as e:
        logger.error(f"[Async Workflow] Validation failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    
    # 2. GENERATE THREAD ID for tracking
    thread_id = str(uuid.uuid4())
    
    # 3. INITIALIZE SERVICES
    audit = get_audit_logger()
    metrics = get_metrics_collector()
    
    # 4. LOG WORKFLOW START
    audit.log_workflow_start(
        thread_id=thread_id,
        spreadsheet_id=validated_request.spreadsheet_id,
        sheet_name=validated_request.sheet_name,
        user_id=validated_request.user_id
    )
    
    # 5. START METRICS TRACKING
    metrics.start_workflow(thread_id)
    
    # 6. DEFINE BACKGROUND TASK with all capabilities
    def execute_workflow_with_all_capabilities():
        """Production-grade workflow execution"""
        retry_handler = get_retry_handler()
        workflow_start_time = time.time()
        
        try:
            # STEP 1: Initialize Sheets Service with retry
            metrics.update_current_step(thread_id, "initializing_sheets")
            
            sheets = retry_handler.execute_with_retry(
                lambda: SheetsService(
                    validated_request.spreadsheet_id,
                    validated_request.sheet_name
                ),
                operation_name="sheets_initialization"
            )
            
            audit.log_data_access(
                thread_id=thread_id,
                operation="initialize",
                resource=f"{validated_request.spreadsheet_id}:{validated_request.sheet_name}",
                row_count=0,
                success=True
            )
            
            # STEP 2: SECRETARY AGENT with retry + metrics
            metrics.update_current_step(thread_id, "running_secretary")
            logger.info(f"[Async Workflow {thread_id}] Starting Secretary Agent")
            
            sec_start = time.time()
            secretary = AgentSecretary(sheets=sheets)
            
            sec_stats = retry_handler.execute_with_retry(
                lambda: secretary.run(),
                operation_name="secretary_agent"
            )
            
            sec_duration = (time.time() - sec_start) * 1000
            metrics.record_latency(thread_id, "secretary_agent", sec_duration)
            metrics.record_agent_completion(thread_id, "secretary")
            metrics.record_success(thread_id, "secretary")
            
            audit.log_agent_action(
                thread_id=thread_id,
                agent_name="secretary",
                action="execute",
                stats=sec_stats,
                success=True
            )
            
            logger.info(f"[Async Workflow {thread_id}] Secretary completed: {sec_stats}")
            
            # STEP 3: ANALYST AGENT with retry + metrics
            metrics.update_current_step(thread_id, "running_analyst")
            logger.info(f"[Async Workflow {thread_id}] Starting Analyst Agent")
            
            ana_start = time.time()
            analyst = AgentAnalyst(sheets=sheets)
            
            ana_stats = retry_handler.execute_with_retry(
                lambda: analyst.run(),
                operation_name="analyst_agent"
            )
            
            ana_duration = (time.time() - ana_start) * 1000
            metrics.record_latency(thread_id, "analyst_agent", ana_duration)
            metrics.record_agent_completion(thread_id, "analyst")
            metrics.record_success(thread_id, "analyst")
            
            audit.log_agent_action(
                thread_id=thread_id,
                agent_name="analyst",
                action="execute",
                stats=ana_stats,
                success=True
            )
            
            logger.info(f"[Async Workflow {thread_id}] Analyst completed: {ana_stats}")
            
            # STEP 4: COMPLETE WORKFLOW
            total_duration = int((time.time() - workflow_start_time) * 1000)
            metrics.complete_workflow(thread_id, status="completed")
            
            audit.log_workflow_complete(
                thread_id=thread_id,
                success=True,
                duration_ms=total_duration,
                stats={
                    "secretary": sec_stats,
                    "analyst": ana_stats
                }
            )
            
            logger.info(f"[Async Workflow {thread_id}] Completed successfully in {total_duration}ms")
            
        except CircuitBreakerOpen as e:
            # Circuit breaker triggered - critical error
            logger.error(f"[Async Workflow {thread_id}] Circuit breaker open: {e}")
            
            metrics.record_failure(thread_id, "workflow", "circuit_breaker_open")
            metrics.complete_workflow(thread_id, status="failed")
            
            audit.log_security_event(
                event_type="circuit_breaker_triggered",
                severity="high",
                description=f"Workflow {thread_id} failed due to circuit breaker",
                metadata={"thread_id": thread_id}
            )
            
            total_duration = int((time.time() - workflow_start_time) * 1000)
            audit.log_workflow_complete(
                thread_id=thread_id,
                success=False,
                duration_ms=total_duration,
                error="Circuit breaker open"
            )
            
        except Exception as e:
            # General error handling
            logger.error(f"[Async Workflow {thread_id}] Error: {e}", exc_info=True)
            
            metrics.record_failure(thread_id, "workflow", str(e))
            metrics.complete_workflow(thread_id, status="failed")
            
            total_duration = int((time.time() - workflow_start_time) * 1000)
            audit.log_workflow_complete(
                thread_id=thread_id,
                success=False,
                duration_ms=total_duration,
                error=str(e)
            )
    
    # 7. ADD TO BACKGROUND TASKS
    background_tasks.add_task(execute_workflow_with_all_capabilities)
    
    # 8. RETURN IMMEDIATELY with tracking info
    logger.info(f"[Async Workflow] Started workflow {thread_id}")
    
    return {
        "status": "running",
        "thread_id": thread_id,
        "message": "Workflow started in background with production-grade capabilities",
        "tracking_url": f"/workflow/status/{thread_id}",
        "priority": validated_request.priority,
        "timestamp": datetime.now().isoformat(),
        "features": [
            "Auto-retry with exponential backoff",
            "Circuit breaker protection",
            "Real-time metrics collection",
            "Audit trail (PII-masked)",
            "Input validation"
        ]
    }


@app.get("/workflow/status/{thread_id}", response_model=WorkflowStatus)
async def get_workflow_status(thread_id: str = Path(..., description="Workflow thread ID")):
    """
    Get real-time status of async workflow.
    
    **Returns:**
    - Current execution status
    - Progress percentage
    - Performance metrics
    - Error information (if any)
    """
    metrics = get_metrics_collector()
    workflow_metrics = metrics.get_workflow_metrics(thread_id)
    
    if not workflow_metrics:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow {thread_id} not found. It may have expired or never existed."
        )
    
    return WorkflowStatus(
        thread_id=thread_id,
        status=workflow_metrics.status,
        progress_percent=workflow_metrics.progress_percent(),
        current_step=workflow_metrics.current_step,
        duration_ms=workflow_metrics.duration_ms(),
        agents_completed=workflow_metrics.agents_completed,
        error_count=len(workflow_metrics.errors),
        last_updated=datetime.now().isoformat()
    )


@app.get("/metrics")
async def get_metrics():
    """
    Get global metrics for monitoring.
    
    **Use for:**
    - Performance dashboards
    - Alerting systems
    - Capacity planning
    """
    metrics = get_metrics_collector()
    return metrics.get_global_stats()


@app.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """
    Export metrics in Prometheus format.
    
    **Integration:**
    Configure Prometheus to scrape this endpoint.
    """
    metrics = get_metrics_collector()
    return metrics.export_prometheus()


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("SALES OPS AGENT API - Testing Platform Compatible")
    print("=" * 60)
    print(f"  Port: 9000")
    print(f"  Docs: http://localhost:9000/docs")
    print(f"  Agent Card: http://localhost:9000/agent-card")
    print(f"  Sheet ID: {settings.SPREADSHEET_ID or 'Not configured'}")
    print("=" * 60)
    print("\nRegistration with Testing Platform:")
    print("   1. GET http://localhost:9000/agent-card")
    print("   2. POST to http://localhost:8000/agents/register")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=9000)
