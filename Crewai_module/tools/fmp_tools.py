"""FinancialModelingPrep Tools — DCF models, analyst estimates, and financial ratios.

⚠️ DEPRECATED: FMP was removed from trending_api.py (see conversation e86cd57d).
   This module is retained for Crewai-only usage but may be removed in a future version.
   Consider migrating to yfinance-based alternatives.

Free tier: 250 API calls per day.
Requires FMP_API_KEY in .env.
API docs: https://site.financialmodelingprep.com/developer/docs
"""

from typing import Optional
import requests
from crewai.tools import BaseTool
from config.settings import settings
from utils.logger import get_logger
from utils.cache import cache
from utils.validators import validate_ticker
import os

logger = get_logger(__name__)

_BASE_URL = "https://financialmodelingprep.com/stable"


def _get_api_key() -> Optional[str]:
    """Get FMP API key from env."""
    return os.getenv("FMP_API_KEY", getattr(settings, "fmp_api_key", None))


def _fmp_request(endpoint: str, **extra_params) -> list:
    """Make a FinancialModelingPrep API request."""
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("FMP_API_KEY not configured. Set it in .env (free at financialmodelingprep.com)")

    params = {"apikey": api_key, **extra_params}
    url = f"{_BASE_URL}/{endpoint}"
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict) and "Error Message" in data:
        raise ValueError(f"FMP API error: {data['Error Message']}")

    return data if isinstance(data, list) else [data] if data else []


class FMPDCFTool(BaseTool):
    """Get Discounted Cash Flow (DCF) valuation for a stock."""

    name: str = "DCF Valuation"
    description: str = (
        "Get Discounted Cash Flow (DCF) intrinsic value for a stock ticker. "
        "Input should be a stock ticker symbol (e.g., 'TSLA'). "
        "Returns DCF value, current price, and whether stock is under/overvalued."
    )

    def _run(self, ticker: str) -> str:
        try:
            ticker = validate_ticker(ticker)
            logger.info(f"FMP DCF valuation for {ticker}")

            cached = cache.get("fmp_dcf", ticker=ticker)
            if cached:
                return cached

            data = _fmp_request("discounted-cash-flow", symbol=ticker)

            if not data:
                return f"No DCF data available for {ticker}"

            dcf = data[0] if isinstance(data, list) else data
            dcf_value = dcf.get("dcf", "N/A")
            stock_price = dcf.get("Stock Price", "N/A")
            date = dcf.get("date", "N/A")

            output = f"DCF Valuation for {ticker}:\n\n"
            output += f"Date: {date}\n"
            output += f"DCF Intrinsic Value: ${dcf_value}\n"
            output += f"Current Stock Price: ${stock_price}\n"

            try:
                dcf_val = float(dcf_value)
                price_val = float(stock_price)
                diff_pct = ((dcf_val - price_val) / price_val) * 100
                if diff_pct > 10:
                    output += f"Assessment: UNDERVALUED by {diff_pct:.1f}%\n"
                elif diff_pct < -10:
                    output += f"Assessment: OVERVALUED by {abs(diff_pct):.1f}%\n"
                else:
                    output += f"Assessment: FAIRLY VALUED (within 10%)\n"
            except (ValueError, TypeError, ZeroDivisionError):
                pass

            # Also get historical DCF
            hist = _fmp_request("historical-discounted-cash-flow-statement", symbol=ticker, limit=4)
            if hist:
                output += "\n=== Historical DCF (Last 4 Periods) ===\n"
                for h in hist[:4]:
                    output += f"  {h.get('date', 'N/A')}: DCF ${h.get('dcf', 'N/A')}, "
                    output += f"Price ${h.get('Stock Price', 'N/A')}\n"

            cache.set("fmp_dcf", output, expire_hours=12, ticker=ticker)
            return output

        except Exception as e:
            error_msg = f"Error fetching DCF for {ticker}: {str(e)}"
            logger.error(error_msg)
            return error_msg


class FMPAnalystEstimatesTool(BaseTool):
    """Get analyst consensus estimates for earnings and revenue."""

    name: str = "Analyst Estimates"
    description: str = (
        "Get analyst consensus estimates for a stock: revenue estimates, "
        "EPS estimates, number of analysts, and estimate revisions. "
        "Input should be a stock ticker symbol (e.g., 'TSLA')."
    )

    def _run(self, ticker: str) -> str:
        try:
            ticker = validate_ticker(ticker)
            logger.info(f"FMP analyst estimates for {ticker}")

            cached = cache.get("fmp_estimates", ticker=ticker)
            if cached:
                return cached

            data = _fmp_request("analyst-estimates", symbol=ticker, limit=8)

            if not data:
                return f"No analyst estimates available for {ticker}"

            output = f"Analyst Estimates for {ticker}:\n\n"
            for est in data[:8]:
                date = est.get("date", "N/A")
                output += f"=== Period: {date} ===\n"

                # Revenue estimates
                rev_avg = est.get("estimatedRevenueAvg", 0)
                rev_high = est.get("estimatedRevenueHigh", 0)
                rev_low = est.get("estimatedRevenueLow", 0)
                num_analysts = est.get("numberAnalystEstimatedRevenue", "N/A")

                if rev_avg:
                    output += f"  Revenue Est (Avg): ${rev_avg / 1e9:.2f}B\n"
                    output += f"  Revenue Range: ${rev_low / 1e9:.2f}B - ${rev_high / 1e9:.2f}B\n"
                    output += f"  Analysts: {num_analysts}\n"

                # EPS estimates
                eps_avg = est.get("estimatedEpsAvg", "N/A")
                eps_high = est.get("estimatedEpsHigh", "N/A")
                eps_low = est.get("estimatedEpsLow", "N/A")

                output += f"  EPS Est (Avg): ${eps_avg}\n"
                output += f"  EPS Range: ${eps_low} - ${eps_high}\n"

                # Net income
                ni_avg = est.get("estimatedNetIncomeAvg", 0)
                if ni_avg:
                    output += f"  Net Income Est: ${ni_avg / 1e9:.2f}B\n"

                output += "\n"

            cache.set("fmp_estimates", output, expire_hours=12, ticker=ticker)
            return output

        except Exception as e:
            error_msg = f"Error fetching estimates for {ticker}: {str(e)}"
            logger.error(error_msg)
            return error_msg


class FMPRatiosTool(BaseTool):
    """Get comprehensive financial ratios for a stock."""

    name: str = "Financial Ratios"
    description: str = (
        "Get comprehensive financial ratios: profitability (ROE, ROA, margins), "
        "liquidity (current, quick), leverage (D/E), efficiency (asset turnover). "
        "Input should be a stock ticker symbol (e.g., 'TSLA')."
    )

    def _run(self, ticker: str) -> str:
        try:
            ticker = validate_ticker(ticker)
            logger.info(f"FMP financial ratios for {ticker}")

            cached = cache.get("fmp_ratios", ticker=ticker)
            if cached:
                return cached

            data = _fmp_request("ratios", symbol=ticker, limit=4)

            if not data:
                return f"No financial ratios available for {ticker}"

            output = f"Financial Ratios for {ticker}:\n\n"
            for period in data[:4]:
                date = period.get("date", "N/A")
                output += f"=== {date} ===\n"

                output += "  Profitability:\n"
                output += f"    Gross Margin: {self._fmt_pct(period.get('grossProfitMargin'))}\n"
                output += f"    Operating Margin: {self._fmt_pct(period.get('operatingProfitMargin'))}\n"
                output += f"    Net Margin: {self._fmt_pct(period.get('netProfitMargin'))}\n"
                output += f"    ROE: {self._fmt_pct(period.get('returnOnEquity'))}\n"
                output += f"    ROA: {self._fmt_pct(period.get('returnOnAssets'))}\n"
                output += f"    ROIC: {self._fmt_pct(period.get('returnOnCapitalEmployed'))}\n"

                output += "  Liquidity:\n"
                output += f"    Current Ratio: {self._fmt_num(period.get('currentRatio'))}\n"
                output += f"    Quick Ratio: {self._fmt_num(period.get('quickRatio'))}\n"
                output += f"    Cash Ratio: {self._fmt_num(period.get('cashRatio'))}\n"

                output += "  Leverage:\n"
                output += f"    Debt/Equity: {self._fmt_num(period.get('debtEquityRatio'))}\n"
                output += f"    Debt/Assets: {self._fmt_num(period.get('debtRatio'))}\n"
                output += f"    Interest Coverage: {self._fmt_num(period.get('interestCoverage'))}\n"

                output += "  Efficiency:\n"
                output += f"    Asset Turnover: {self._fmt_num(period.get('assetTurnover'))}\n"
                output += f"    Inventory Turnover: {self._fmt_num(period.get('inventoryTurnover'))}\n"
                output += f"    Receivables Turnover: {self._fmt_num(period.get('receivablesTurnover'))}\n"

                output += "  Valuation:\n"
                output += f"    P/E: {self._fmt_num(period.get('priceEarningsRatio'))}\n"
                output += f"    P/B: {self._fmt_num(period.get('priceToBookRatio'))}\n"
                output += f"    P/S: {self._fmt_num(period.get('priceToSalesRatio'))}\n"
                output += f"    EV/EBITDA: {self._fmt_num(period.get('enterpriseValueMultiple'))}\n"
                output += "\n"

            cache.set("fmp_ratios", output, expire_hours=12, ticker=ticker)
            return output

        except Exception as e:
            error_msg = f"Error fetching ratios for {ticker}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    @staticmethod
    def _fmt_pct(value) -> str:
        """Format a decimal as percentage."""
        if value is None:
            return "N/A"
        try:
            return f"{float(value) * 100:.2f}%"
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def _fmt_num(value) -> str:
        """Format a number nicely."""
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.2f}"
        except (ValueError, TypeError):
            return str(value)
