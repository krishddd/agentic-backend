"""Financial analysis tools for calculating metrics and ratios."""

from typing import Dict, Any, Optional
import yfinance as yf
from datetime import datetime
from crewai.tools import BaseTool
from utils.logger import get_logger
from utils.cache import cache
from utils.validators import validate_ticker

logger = get_logger(__name__)


class FinancialMetricsTool(BaseTool):
    """Tool for retrieving and analyzing financial metrics."""
    
    name: str = "Financial Metrics"
    description: str = (
        "Get financial metrics and ratios for a stock ticker. "
        "Input should be a stock ticker symbol (e.g., 'TSLA'). "
        "Returns key financial metrics including revenue, profit margins, "
        "cash flow, debt ratios, and more."
    )
    
    def _run(self, ticker: str) -> str:
        """Get financial metrics for a stock.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Formatted financial metrics
        """
        try:
            ticker = validate_ticker(ticker)
            logger.info(f"Fetching financial metrics for {ticker}")
            
            # Check cache
            cached_result = cache.get("financial_metrics", ticker=ticker)
            if cached_result:
                return cached_result
            
            # Get stock data
            stock = yf.Ticker(ticker)
            
            # Get financial info
            info = stock.info
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Build output
            output = f"Financial Metrics for {ticker} ({info.get('longName', 'N/A')}):\n\n"
            
            # Basic info
            output += "=== Company Overview ===\n"
            output += f"Sector: {info.get('sector', 'N/A')}\n"
            output += f"Industry: {info.get('industry', 'N/A')}\n"
            output += f"Market Cap: ${info.get('marketCap', 0):,.0f}\n"
            output += f"Current Price: ${info.get('currentPrice', 'N/A')}\n\n"
            
            # Revenue metrics
            output += "=== Revenue & Profitability ===\n"
            output += f"Total Revenue: ${info.get('totalRevenue', 0):,.0f}\n"
            output += f"Revenue Growth (YoY): {info.get('revenueGrowth', 0) * 100:.2f}%\n"
            output += f"Gross Margin: {info.get('grossMargins', 0) * 100:.2f}%\n"
            output += f"Operating Margin: {info.get('operatingMargins', 0) * 100:.2f}%\n"
            output += f"Profit Margin: {info.get('profitMargins', 0) * 100:.2f}%\n"
            output += f"EBITDA: ${info.get('ebitda', 0):,.0f}\n\n"
            
            # Profitability ratios
            output += "=== Profitability Ratios ===\n"
            output += f"ROE (Return on Equity): {info.get('returnOnEquity', 0) * 100:.2f}%\n"
            output += f"ROA (Return on Assets): {info.get('returnOnAssets', 0) * 100:.2f}%\n"
            output += f"ROIC (Return on Capital): {info.get('returnOnCapital', 0) * 100:.2f}%\n\n"
            
            # Cash flow
            output += "=== Cash Flow ===\n"
            output += f"Operating Cash Flow: ${info.get('operatingCashflow', 0):,.0f}\n"
            output += f"Free Cash Flow: ${info.get('freeCashflow', 0):,.0f}\n\n"
            
            # Debt & liquidity
            output += "=== Debt & Liquidity ===\n"
            output += f"Total Cash: ${info.get('totalCash', 0):,.0f}\n"
            output += f"Total Debt: ${info.get('totalDebt', 0):,.0f}\n"
            output += f"Debt to Equity: {info.get('debtToEquity', 0) / 100:.2f}\n"
            output += f"Current Ratio: {info.get('currentRatio', 0):.2f}\n"
            output += f"Quick Ratio: {info.get('quickRatio', 0):.2f}\n\n"
            
            # Valuation
            output += "=== Valuation Metrics ===\n"
            output += f"P/E Ratio: {info.get('trailingPE', 'N/A')}\n"
            output += f"Forward P/E: {info.get('forwardPE', 'N/A')}\n"
            output += f"PEG Ratio: {info.get('pegRatio', 'N/A')}\n"
            output += f"Price to Book: {info.get('priceToBook', 'N/A')}\n"
            output += f"Price to Sales: {info.get('priceToSalesTrailing12Months', 'N/A')}\n\n"
            
            # Growth metrics
            output += "=== Growth Metrics ===\n"
            output += f"Revenue Growth: {info.get('revenueGrowth', 0) * 100:.2f}%\n"
            output += f"Earnings Growth: {info.get('earningsGrowth', 0) * 100:.2f}%\n"
            output += f"EPS (TTM): ${info.get('trailingEps', 'N/A')}\n"
            output += f"Forward EPS: ${info.get('forwardEps', 'N/A')}\n\n"
            
            # Analyst recommendations
            output += "=== Analyst Recommendations ===\n"
            output += f"Target Price: ${info.get('targetMeanPrice', 'N/A')}\n"
            output += f"Recommendation: {info.get('recommendationKey', 'N/A').upper()}\n"
            
            # Cache result
            cache.set("financial_metrics", output, expire_hours=12, ticker=ticker)
            
            logger.info(f"Successfully retrieved metrics for {ticker}")
            return output
            
        except Exception as e:
            error_msg = f"Error fetching financial metrics for {ticker}: {str(e)}"
            logger.error(error_msg)
            return error_msg
