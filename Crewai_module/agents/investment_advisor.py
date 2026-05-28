"""Investment Advisor agent for synthesizing analysis into recommendations."""

import sys
import os

# Add project root (parent of Crewai_module) to path for llm_router import
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from crewai import Agent
from config.settings import settings
from utils.logger import get_logger
from llm_router import get_crewai_llm

logger = get_logger(__name__)


def create_investment_advisor() -> Agent:
    """Create and configure the Investment Advisor agent.
    
    Uses qwen3:8b via Ollama — strongest reasoning model for
    synthesizing research into actionable investment recommendations.
    
    Returns:
        Configured Investment Advisor agent
    """
    logger.info("Creating Investment Advisor agent")
    
    # Get per-agent LLM from router
    llm = get_crewai_llm("investment_advisor")
    
    # Investment advisor doesn't need tools - it synthesizes work from other agents
    agent = Agent(
        role="Senior Investment Advisor",
        goal="Synthesize research and analysis to provide clear, actionable investment recommendations for {stock_symbol}",
        backstory="""You are a senior investment advisor with 20+ years of experience managing portfolios for
        high-net-worth individuals and institutional clients. Your strength is synthesizing complex
        research and financial analysis into clear, actionable investment recommendations. You consider
        multiple perspectives, weigh risks against opportunities, and provide balanced recommendations
        that account for both fundamental analysis and market dynamics. Your reports are known for
        their clarity, depth, and practical value.""",
        tools=[],  # No tools needed - synthesizes from other agents
        llm=llm,
        verbose=settings.verbose,
        allow_delegation=settings.enable_delegation,
        max_iter=settings.max_iter,
    )
    
    logger.info(f"Investment Advisor agent created (LLM: {llm.model})")
    return agent
