"""Research Analyst agent for market research and competitive analysis."""

import sys
import os

# Add project root (parent of Crewai_module) to path for llm_router import
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from crewai import Agent
from tools import WebSearchTool, NewsSearchTool, SentimentAnalysisTool
from config.settings import settings
from utils.logger import get_logger
from llm_router import get_crewai_llm

logger = get_logger(__name__)


def create_research_analyst() -> Agent:
    """Create and configure the Research Analyst agent.
    
    Uses llama3.2:latest via Ollama — strong at general NLP,
    research summarization, and competitive analysis.
    
    Returns:
        Configured Research Analyst agent
    """
    logger.info("Creating Research Analyst agent")
    
    # Get per-agent LLM from router
    llm = get_crewai_llm("research_analyst")
    
    # Initialize tools
    tools = [
        WebSearchTool(),
        NewsSearchTool(),
        SentimentAnalysisTool(),
    ]
    
    agent = Agent(
        role="Senior Research Analyst",
        goal="Conduct comprehensive market research and competitive analysis for {stock_symbol}",
        backstory="""You are a seasoned research analyst with 15+ years of experience in equity research.
        Your expertise lies in gathering and synthesizing market intelligence, tracking industry trends,
        analyzing competitor positioning, and identifying key market drivers. You have a keen eye for
        detail and excel at connecting disparate pieces of information to form a coherent picture of
        the market landscape. Your research forms the foundation for investment decisions.""",
        tools=tools,
        llm=llm,
        verbose=settings.verbose,
        allow_delegation=settings.enable_delegation,
        max_iter=settings.max_iter,
    )
    
    logger.info(f"Research Analyst agent created (LLM: {llm.model})")
    return agent

