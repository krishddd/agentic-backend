"""Tools package for Financial Crew agents."""

from .sec_tools import SECFilingSearchTool, SECFilingDownloadTool
from .rag_tools import RAGRetrievalTool
from .web_search_tools import WebSearchTool, NewsSearchTool
from .financial_tools import FinancialMetricsTool
from .sentiment_tools import SentimentAnalysisTool

# --- Phase 1 Expansion: Financial Domain Tools ---
from .sec_edgar_mcp_tool import SECEdgarFullTextSearchTool, SECEdgarXBRLTool
from .alpha_vantage_tools import AlphaVantageQuoteTool, AlphaVantageEarningsTool, AlphaVantageOverviewTool
from .fmp_tools import FMPDCFTool, FMPAnalystEstimatesTool, FMPRatiosTool
from .openbb_tools import OpenBBEconomicTool, OpenBBTechnicalTool

__all__ = [
    # Original tools
    "SECFilingSearchTool",
    "SECFilingDownloadTool",
    "RAGRetrievalTool",
    "WebSearchTool",
    "NewsSearchTool",
    "FinancialMetricsTool",
    "SentimentAnalysisTool",
    # SEC EDGAR MCP
    "SECEdgarFullTextSearchTool",
    "SECEdgarXBRLTool",
    # Alpha Vantage
    "AlphaVantageQuoteTool",
    "AlphaVantageEarningsTool",
    "AlphaVantageOverviewTool",
    # FinancialModelingPrep
    "FMPDCFTool",
    "FMPAnalystEstimatesTool",
    "FMPRatiosTool",
    # OpenBB / Technical
    "OpenBBEconomicTool",
    "OpenBBTechnicalTool",
]
