"""Research task for market sentiment and competitive analysis."""

from crewai import Task, Agent
from utils.logger import get_logger

logger = get_logger(__name__)


def create_research_task(agent: Agent, stock_symbol: str) -> Task:
    """Create research task for market analysis.
    
    Args:
        agent: Research Analyst agent
        stock_symbol: Stock ticker symbol  
    
    Returns:
        Configured research task
    """
    logger.info(f"Creating research task for {stock_symbol}")
    
    description = f"""Conduct comprehensive market research on {stock_symbol}:
    
    1. Market Sentiment Analysis:
       - Analyze recent news articles and social media sentiment
       - Identify positive and negative sentiment drivers
       - Gauge overall market perception
    
    2. Regulatory Changes:
       - Monitor recent regulatory developments affecting the company
       - Track policy changes in relevant jurisdictions
       - Identify compliance risks or opportunities
    
    3. Insider Trading Activity:
       - Research recent insider buying or selling activity
       - Analyze patterns in executive stock transactions
       - Interpret significance of insider moves
    
    4. Upcoming Events:
       - Identify upcoming earnings calls, investor days, product launches
       - Track important dates (ex-dividend, analyst meetings, etc.)
       - Note any scheduled announcements
    
    5. Competitive Landscape:
       - Compare {stock_symbol} with key competitors
       - Analyze market share trends
       - Identify competitive advantages or threats
    
    Provide a comprehensive research summary with sources and key findings."""
    
    expected_output = f"""A detailed research report including:
    - Market sentiment summary (bullish/bearish/neutral) with supporting evidence
    - List of recent regulatory changes and their impact
    - Insider trading activity analysis with transaction details
    - Calendar of upcoming events with significance ratings
    - Competitive analysis comparing {stock_symbol} with 3-5 key competitors
    - Key research findings and implications"""
    
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
    )
    
    logger.info(f"Research task created for {stock_symbol}")
    return task
