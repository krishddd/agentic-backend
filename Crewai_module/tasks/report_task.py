"""Report generation task for synthesizing all analysis into investment recommendations."""

from crewai import Task, Agent
from utils.logger import get_logger

logger = get_logger(__name__)


def create_report_task(
    agent: Agent,
    stock_symbol: str,
    context_tasks: list = None
) -> Task:
    """Create report generation task.
    
    Args:
        agent: Investment Advisor agent
        stock_symbol: Stock ticker symbol
        context_tasks: List of tasks to use as context
    
    Returns:
        Configured report task
    """
    logger.info(f"Creating report generation task for {stock_symbol}")
    
    description = f"""Create a comprehensive investment report for {stock_symbol} by synthesizing all research and analysis:
    
    1. Executive Summary:
       - Investment thesis in 2-3 paragraphs
       - Clear recommendation (Strong Buy, Buy, Hold, Sell, Strong Sell)
       - Target price and timeframe
    
    2. Financial Health:
       - Synthesize financial analysis findings
       - Highlight revenue trends, profitability, and cash flow strength
       - Include key financial metrics in an easy-to-read format
    
    3. Market Sentiment:
       - Summarize market sentiment analysis
       - Include key news and sentiment drivers
       - Note analyst consensus if available
    
    4. Regulatory & Risk Analysis:
       - Highlight regulatory changes and their impact
       - Summarize key risks from 10-Q filing
       - Assess overall risk level (Low, Medium, High)
    
    5. Insider Trading:
       - Report insider activity and interpretation
       - Note any significant patterns
    
    6. Upcoming Events:
       - List key upcoming events and their significance
       - Include earnings date, ex-dividend date, etc.
    
    7. Competitive Position:
       - Summarize competitive landscape
       - Highlight competitive advantages or disadvantages
    
    8. Investment Recommendation:
       - Clear buy/hold/sell recommendation with rationale
       - Suggested entry price range
       - Price target with 12-month horizon
       - Key catalysts to watch
       - Risk factors to monitor
    
    The report should be professional, well-structured, and actionable."""
    
    expected_output = f"""A comprehensive investment report (1500-2500 words) with the following structure:
    
# Investment Report: {stock_symbol}
## Executive Summary
## Financial Health
## Market Sentiment Analysis
## Regulatory Changes & Risk Factors
## Insider Trading Activity
## Upcoming Events Calendar
## Competitive Positioning
## Investment Recommendation

The report should be clear, data-driven, and include specific numbers, dates, and actionable insights."""
    
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        context=context_tasks or [],
        output_file=f"outputs/{stock_symbol}_investment_report.md"
    )
    
    logger.info(f"Report generation task created for {stock_symbol}")
    return task
