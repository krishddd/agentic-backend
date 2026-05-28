"""Financial analysis task for analyzing financial statements and metrics."""

from crewai import Task, Agent
from utils.logger import get_logger

logger = get_logger(__name__)


def create_financial_analysis_task(agent: Agent, stock_symbol: str) -> Task:
    """Create financial analysis task.
    
    Args:
        agent: Financial Analyst agent
        stock_symbol: Stock ticker symbol
    
    Returns:
        Configured financial analysis task
    """
    logger.info(f"Creating financial analysis task for {stock_symbol}")
    
    description = f"""Perform deep financial analysis on {stock_symbol}:
    
    1. Revenue Analysis:
       - Analyze revenue trends over the past 4 quarters
       - Break down revenue by segment and geography if available
       - Calculate revenue growth rates (YoY, QoQ)
       - Identify revenue drivers and risks
    
    2. Profitability Metrics:
       - Calculate gross margin, operating margin, net margin
       - Analyze profitability trends
       - Compare margins with industry averages
       - Identify margin expansion or compression drivers
    
    3. Financial Ratios:
       - Calculate key liquidity ratios (current, quick)
       - Compute leverage ratios (debt-to-equity, interest coverage)
       - Calculate efficiency ratios (ROE, ROA, ROIC)
    
    4. Cash Flow Analysis:
       - Analyze operating, investing, and financing cash flows
       - Calculate free cash flow
       - Assess cash generation quality
    
    5. Balance Sheet Health:
       - Analyze asset composition and quality
       - Evaluate debt levels and maturity profile
       - Assess liquidity position
    
    Provide quantitative analysis with specific numbers and trends."""
    
    expected_output = f"""A comprehensive financial analysis report containing:
    - Revenue breakdown by segment/geography with growth rates
    - Profitability metrics table with trend analysis
    - Complete financial ratios scorecard
    - Cash flow statement analysis with free cash flow calculation
    - Balance sheet health assessment
    - Key financial strengths and weaknesses
    - Comparison with industry benchmarks where available"""
    
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
    )
    
    logger.info(f"Financial analysis task created for {stock_symbol}")
    return task
