"""Filing analysis task for analyzing SEC 10-Q filings using RAG."""

from crewai import Task, Agent
from utils.logger import get_logger

logger = get_logger(__name__)


def create_filing_analysis_task(agent: Agent, stock_symbol: str) -> Task:
    """Create filing analysis task using RAG.
    
    Args:
        agent: Financial Analyst agent
        stock_symbol: Stock ticker symbol
    
    Returns:
        Configured filing analysis task
    """
    logger.info(f"Creating filing analysis task for {stock_symbol}")
    
    description = f"""Analyze SEC 10-Q filings for {stock_symbol} using RAG retrieval:
    
    1. Risk Factors Analysis:
       - Use RAG to extract risk factors mentioned in the filing
       - Identify new or escalating risks compared to previous filings
       - Categorize risks (operational, financial, regulatory, market)
    
    2. Management Discussion & Analysis (MD&A):
       - Extract key management insights about business performance
       - Identify forward-looking statements
       - Note management's commentary on trends and outlook
    
    3. Business Segment Performance:
       - Extract performance metrics for each business segment
       - Identify segment-specific trends and challenges
       - Note any restructuring or strategic changes
    
    4. Contingent Liabilities:
       - Identify pending litigation or regulatory investigations
       - Extract details about contingent liabilities
       - Assess potential financial impact
    
    5. Notable Disclosures:
       - Flag any unusual or concerning disclosures
       - Identify changes in accounting policies
       - Note any material events or transactions
    
    Use RAG tools to retrieve relevant information efficiently from the filings."""
    
    expected_output = f"""A detailed 10-Q analysis report including:
    - Summary of key risk factors with categorization
    - Management's discussion highlights with direct quotes
    - Segment performance breakdown with metrics
    - List of contingent liabilities and their potential impact
    - Notable changes from previous filings
    - Key insights extracted via RAG retrieval
    - Flags for any concerning disclosures"""
    
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
    )
    
    logger.info(f"Filing analysis task created for {stock_symbol}")
    return task
