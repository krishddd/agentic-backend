"""Financial Analyst agent for financial statement analysis and SEC filings."""

import sys
import os

# Add project root (parent of Crewai_module) to path for llm_router import
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from crewai import Agent
from tools import (
    SECFilingSearchTool,
    SECFilingDownloadTool,
    RAGRetrievalTool,
    FinancialMetricsTool
)
from config.settings import settings
from utils.logger import get_logger
from llm_router import get_crewai_llm

logger = get_logger(__name__)


def create_financial_analyst() -> Agent:
    """Create and configure the Financial Analyst agent.
    
    Uses qwen3:4b via Ollama — fast and precise for structured
    financial data parsing and ratio calculations.
    
    Returns:
        Configured Financial Analyst agent
    """
    logger.info("Creating Financial Analyst agent")
    
    # Get per-agent LLM from router
    llm = get_crewai_llm("financial_analyst")
    
    # Initialize tools
    tools = [
        SECFilingSearchTool(),
        SECFilingDownloadTool(),
        RAGRetrievalTool(),
        FinancialMetricsTool(),
    ]
    
    agent = Agent(
        role="Expert Financial Analyst",
        goal="Analyze financial statements, SEC filings, and key metrics for {stock_symbol} to assess financial health",
        backstory="""You are an expert financial analyst with deep expertise in financial statement analysis,
        ratio calculations, and SEC filing interpretation. With a CFA charter and 12+ years of experience,
        you specialize in dissecting 10-Q and 10-K forms to uncover insights about revenue trends,
        profitability, cash flows, and financial risks. You use advanced RAG techniques to extract
        relevant information from lengthy regulatory filings and can quickly identify red flags or
        opportunities in financial data.""",
        tools=tools,
        llm=llm,
        verbose=settings.verbose,
        allow_delegation=settings.enable_delegation,
        max_iter=settings.max_iter,
    )
    
    logger.info(f"Financial Analyst agent created (LLM: {llm.model})")
    return agent
