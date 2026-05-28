"""
FastAPI Application - Sales Operations AI Agent API
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

from config import settings
from services.sheets_service import SheetsService
from services.llm_service import get_llm_service
from services.rag_engine import get_rag_engine
from agents.secretary import AgentSecretary
from agents.analyst import AgentAnalyst
from utils.logger import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# FastAPI app
app = FastAPI(
    title="Sales Operations AI Agent",
    description="API for automating sales communication and intelligence",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Request/Response Models
# =============================================================================

class SheetRequest(BaseModel):
    spreadsheet_id: str
    sheet_name: Optional[str] = settings.SHEET_NAME

class AgentResponse(BaseModel):
    success: bool
    message: str
    stats: Optional[Dict[str, int]] = None
    timestamp: str

class RowData(BaseModel):
    row_id: str
    client_name: str
    company_name: str
    consultant_email: str
    manager_email: str
    call_type: str
    raw_notes: str
    requirements: str
    scheduling: str
    mail_status: str
    deal_score: Optional[str] = ""
    risk_flags: Optional[str] = ""
    sync_status: Optional[str] = ""

class SheetDataResponse(BaseModel):
    success: bool
    total_rows: int
    pending_count: int
    sent_count: int
    failed_count: int
    rows: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_model: str
    llm_available: bool
    timestamp: str

# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "name": "Sales Operations AI Agent",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    llm = get_llm_service()
    return HealthResponse(
        status="healthy",
        llm_provider=llm.provider,
        llm_model=llm.model,
        llm_available=llm.is_available(),
        timestamp=datetime.now().isoformat()
    )


@app.post("/sheet/data", response_model=SheetDataResponse)
async def get_sheet_data(request: SheetRequest):
    """Get all data from the Google Sheet"""
    try:
        sheets = SheetsService(request.spreadsheet_id)
        all_rows = sheets.get_all_rows()
        
        pending = sum(1 for r in all_rows if r.get('Mail_Status', '').lower() == 'pending')
        sent = sum(1 for r in all_rows if r.get('Mail_Status', '').lower() == 'sent')
        failed = sum(1 for r in all_rows if r.get('Mail_Status', '').lower() == 'failed')
        
        return SheetDataResponse(
            success=True,
            total_rows=len(all_rows),
            pending_count=pending,
            sent_count=sent,
            failed_count=failed,
            rows=all_rows
        )
    except Exception as e:
        logger.error(f"Error fetching sheet data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/secretary", response_model=AgentResponse)
async def run_secretary_agent(request: SheetRequest):
    """
    Run the Secretary Agent (Communication)
    
    - Reads pending rows from sheet
    - Generates emails using LLM + RAG
    - Sends emails via Gmail
    - Updates sheet status
    """
    try:
        logger.info(f"Running Secretary Agent on sheet: {request.spreadsheet_id}")
        
        # Create services with the provided sheet ID
        sheets = SheetsService(request.spreadsheet_id)
        
        # Run agent
        agent = AgentSecretary(sheets=sheets)
        stats = agent.run()
        
        return AgentResponse(
            success=True,
            message=f"Secretary agent completed. Processed: {stats.get('processed', 0)}, Sent: {stats.get('sent', 0)}, Failed: {stats.get('failed', 0)}",
            stats=stats,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Secretary agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/analyst", response_model=AgentResponse)
async def run_analyst_agent(request: SheetRequest):
    """
    Run the Analyst Agent (Intelligence)
    
    - Reads sent rows from sheet
    - Scores deals and detects risks
    - Sends alerts for high-score leads
    - Updates sheet with analysis
    """
    try:
        logger.info(f"Running Analyst Agent on sheet: {request.spreadsheet_id}")
        
        sheets = SheetsService(request.spreadsheet_id)
        
        agent = AgentAnalyst(sheets=sheets)
        stats = agent.run()
        
        return AgentResponse(
            success=True,
            message=f"Analyst agent completed. Analyzed: {stats.get('analyzed', 0)}, High Score: {stats.get('high_score', 0)}",
            stats=stats,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Analyst agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/run-all", response_model=AgentResponse)
async def run_all_agents(request: SheetRequest):
    """
    Run both Secretary and Analyst agents in sequence
    """
    try:
        logger.info(f"Running all agents on sheet: {request.spreadsheet_id}")
        
        sheets = SheetsService(request.spreadsheet_id)
        
        # Run Secretary first
        secretary = AgentSecretary(sheets=sheets)
        secretary_stats = secretary.run()
        
        # Then run Analyst
        analyst = AgentAnalyst(sheets=sheets)
        analyst_stats = analyst.run()
        
        combined_stats = {
            "secretary": secretary_stats,
            "analyst": analyst_stats
        }
        
        return AgentResponse(
            success=True,
            message=f"All agents completed. Emails sent: {secretary_stats.get('sent', 0)}, Deals analyzed: {analyst_stats.get('analyzed', 0)}",
            stats=combined_stats,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Run all agents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/run-all-async", response_model=AgentResponse)
async def run_all_agents_async(request: SheetRequest, background_tasks: BackgroundTasks):
    """
    Run both agents in background (non-blocking)
    """
    def run_agents_task(spreadsheet_id: str):
        try:
            sheets = SheetsService(spreadsheet_id)
            secretary = AgentSecretary(sheets=sheets)
            secretary.run()
            analyst = AgentAnalyst(sheets=sheets)
            analyst.run()
            logger.info(f"Background agents completed for sheet: {spreadsheet_id}")
        except Exception as e:
            logger.error(f"Background agents error: {e}")
    
    background_tasks.add_task(run_agents_task, request.spreadsheet_id)
    
    return AgentResponse(
        success=True,
        message="Agents started in background. Check sheet for updates.",
        stats=None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/rag/search")
async def search_knowledge_base(query: str):
    """Search the product knowledge base"""
    try:
        rag = get_rag_engine()
        context = rag.retrieve(query)
        questions = rag.extract_questions(query)
        
        return {
            "success": True,
            "query": query,
            "extracted_questions": questions,
            "context": context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/test")
async def test_llm(prompt: str = "Say hello"):
    """Test the LLM connection"""
    try:
        llm = get_llm_service()
        response = llm.chat("Be brief.", prompt)
        
        return {
            "success": True,
            "provider": llm.provider,
            "model": llm.model,
            "prompt": prompt,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Run with: uvicorn api.main:app --reload
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
