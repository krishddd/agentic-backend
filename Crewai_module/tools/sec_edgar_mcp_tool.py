"""SEC EDGAR MCP Tools — Enhanced EDGAR connector for the Financial Crew.

Provides:
  - SECEdgarFullTextSearchTool: Full-text search across ALL SEC filings (EFTS API)
  - SECEdgarXBRLTool: Structured financial data extraction from XBRL companyfacts

Both use free SEC EDGAR APIs (no API key required, just a User-Agent header).
"""

from typing import Optional, Dict, Any, List
import requests
from datetime import datetime
from crewai.tools import BaseTool
from config.settings import settings
from utils.logger import get_logger
from utils.cache import cache
from utils.validators import validate_ticker

logger = get_logger(__name__)

# Shared SEC headers
_SEC_HEADERS = {
    "User-Agent": "financial-crew github.com/user",
    "Accept-Encoding": "gzip, deflate",
}


class SECEdgarFullTextSearchTool(BaseTool):
    """Search across ALL SEC filings using the EDGAR Full-Text Search (EFTS) API."""

    name: str = "SEC EDGAR Full Text Search"
    description: str = (
        "Search across all SEC filings using full-text search. "
        "Input should be a search query (e.g., 'Tesla revenue growth risk factors'). "
        "Returns matching filing excerpts with filing type, company, date, and links."
    )

    def _run(self, query: str) -> str:
        """Search SEC EDGAR full-text index.

        Args:
            query: Free-text search query

        Returns:
            Formatted search results from SEC filings
        """
        try:
            logger.info(f"EDGAR full-text search: {query}")

            cached_result = cache.get("sec_efts", query=query)
            if cached_result:
                return cached_result

            url = "https://efts.sec.gov/LATEST/search-index"
            # Note: Official endpoint. If this returns errors, try /search-index
            params = {
                "q": query,
                "dateRange": "custom",
                "startdt": "2022-01-01",
                "enddt": datetime.now().strftime("%Y-%m-%d"),
                "forms": "10-K,10-Q,8-K,S-1,DEF 14A",
            }

            response = requests.get(url, params=params, headers=_SEC_HEADERS, timeout=30)
            response.raise_for_status()
            data = response.json()

            hits = data.get("hits", {}).get("hits", [])
            total = data.get("hits", {}).get("total", {}).get("value", 0)

            if not hits:
                return f"No SEC filings found for: {query}"

            output = f"SEC EDGAR Full-Text Search Results ({total} total matches):\n\n"
            for i, hit in enumerate(hits[:8], 1):
                src = hit.get("_source", {})
                form = src.get("form_type", "N/A")
                display_names = src.get("display_names") or src.get("entity_name", "")
                company = display_names[0] if isinstance(display_names, list) and display_names else str(display_names) or "Unknown"
                date_filed = src.get("file_date", "N/A")
                file_num = src.get("file_num", "")
                accession = hit.get("_id", "")

                # Build filing URL
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{src.get('entity_id', '')}/{accession.replace('-', '')}"

                output += f"[{i}] {form} — {company}\n"
                output += f"    Filed: {date_filed}\n"
                if file_num:
                    output += f"    File Number: {file_num}\n"
                output += f"    URL: {filing_url}\n\n"

            cache.set("sec_efts", output, expire_hours=12, query=query)
            logger.info(f"Found {len(hits)} EDGAR results for: {query}")
            return output

        except Exception as e:
            error_msg = f"Error searching SEC EDGAR: {str(e)}"
            logger.error(error_msg)
            return error_msg


class SECEdgarXBRLTool(BaseTool):
    """Extract structured financial data from SEC XBRL companyfacts API."""

    name: str = "SEC EDGAR XBRL Financials"
    description: str = (
        "Extract structured financial data (revenue, net income, EPS, assets, "
        "liabilities) from SEC XBRL filings for a stock ticker. "
        "Input should be a stock ticker symbol (e.g., 'TSLA'). "
        "Returns key financial metrics from the most recent filings."
    )

    def _run(self, ticker: str) -> str:
        """Extract XBRL financial data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Formatted financial data from XBRL filings
        """
        try:
            ticker = validate_ticker(ticker)
            logger.info(f"Fetching XBRL financials for {ticker}")

            cached_result = cache.get("sec_xbrl", ticker=ticker)
            if cached_result:
                return cached_result

            # Step 1: Get CIK for ticker
            cik = self._get_cik(ticker)
            if not cik:
                return f"Could not find CIK for ticker {ticker}"

            # Step 2: Fetch company facts
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            response = requests.get(url, headers=_SEC_HEADERS, timeout=30)
            response.raise_for_status()
            data = response.json()

            company_name = data.get("entityName", ticker)
            facts = data.get("facts", {})
            us_gaap = facts.get("us-gaap", {})

            if not us_gaap:
                return f"No XBRL financial data found for {ticker} ({company_name})"

            # Key financial metrics to extract
            metrics = {
                "Revenues": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
                "NetIncomeLoss": ["NetIncomeLoss"],
                "EarningsPerShareBasic": ["EarningsPerShareBasic"],
                "EarningsPerShareDiluted": ["EarningsPerShareDiluted"],
                "Assets": ["Assets"],
                "Liabilities": ["Liabilities"],
                "StockholdersEquity": ["StockholdersEquity"],
                "OperatingIncomeLoss": ["OperatingIncomeLoss"],
                "CashAndCashEquivalents": ["CashAndCashEquivalentsAtCarryingValue"],
                "LongTermDebt": ["LongTermDebt", "LongTermDebtNoncurrent"],
            }

            output = f"XBRL Financial Data for {company_name} ({ticker}):\n\n"

            for label, possible_keys in metrics.items():
                for key in possible_keys:
                    if key in us_gaap:
                        units_data = us_gaap[key].get("units", {})
                        # Try USD first, then USD/shares
                        values = units_data.get("USD", units_data.get("USD/shares", []))
                        if values:
                            # Get the 4 most recent 10-K or 10-Q values
                            annual = [v for v in values if v.get("form") in ("10-K", "10-Q")]
                            annual.sort(key=lambda x: x.get("end", ""), reverse=True)
                            recent = annual[:4]

                            if recent:
                                output += f"=== {label} ===\n"
                                for val in recent:
                                    period = val.get("end", "N/A")
                                    form = val.get("form", "?")
                                    amount = val.get("val", 0)
                                    # Format large numbers
                                    if isinstance(amount, (int, float)) and abs(amount) >= 1_000_000:
                                        formatted = f"${amount / 1_000_000:,.1f}M"
                                    elif isinstance(amount, (int, float)) and abs(amount) >= 1_000:
                                        formatted = f"${amount:,.0f}"
                                    else:
                                        formatted = f"{amount}"
                                    output += f"  {period} ({form}): {formatted}\n"
                                output += "\n"
                            break  # Found this metric, move to next

            cache.set("sec_xbrl", output, expire_hours=12, ticker=ticker)
            logger.info(f"Successfully extracted XBRL data for {ticker}")
            return output

        except Exception as e:
            error_msg = f"Error fetching XBRL data for {ticker}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for ticker from SEC company tickers JSON."""
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(url, headers=_SEC_HEADERS, timeout=30)
            response.raise_for_status()
            data = response.json()

            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry["cik_str"]).zfill(10)

            logger.warning(f"CIK not found for ticker {ticker}")
            return None

        except Exception as e:
            logger.error(f"Error getting CIK: {e}")
            return None
