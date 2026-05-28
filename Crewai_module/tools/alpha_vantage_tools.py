"""Alpha Vantage Tools — Real-time quotes, earnings, and company fundamentals.

Free tier: 25 API calls per day.
Requires ALPHA_VANTAGE_API_KEY in .env.
"""

from typing import Optional
import requests
from crewai.tools import BaseTool
from config.settings import settings
from utils.logger import get_logger
from utils.cache import cache
from utils.validators import validate_ticker

logger = get_logger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"


def _get_api_key() -> Optional[str]:
    """Get Alpha Vantage API key from settings or env."""
    return getattr(settings, "alpha_vantage_api_key", None) or None


def _av_request(function: str, **extra_params) -> dict:
    """Make an Alpha Vantage API request."""
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY not configured. Set it in .env")

    params = {"function": function, "apikey": api_key, **extra_params}
    response = requests.get(_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Check for API limit error
    if "Note" in data or "Information" in data:
        msg = data.get("Note", data.get("Information", ""))
        raise ValueError(f"Alpha Vantage API limit: {msg}")

    return data


class AlphaVantageQuoteTool(BaseTool):
    """Get real-time and daily stock price data."""

    name: str = "Alpha Vantage Quote"
    description: str = (
        "Get real-time stock quote and recent price history from Alpha Vantage. "
        "Input should be a stock ticker symbol (e.g., 'TSLA'). "
        "Returns current price, change, volume, and 5-day price history."
    )

    def _run(self, ticker: str) -> str:
        try:
            ticker = validate_ticker(ticker)
            logger.info(f"Alpha Vantage quote for {ticker}")

            cached = cache.get("av_quote", ticker=ticker)
            if cached:
                return cached

            data = _av_request("GLOBAL_QUOTE", symbol=ticker)
            quote = data.get("Global Quote", {})

            if not quote:
                return f"No quote data from Alpha Vantage for {ticker}"

            price = quote.get("05. price", "N/A")
            change = quote.get("09. change", "N/A")
            change_pct = quote.get("10. change percent", "N/A")
            volume = quote.get("06. volume", "N/A")
            high = quote.get("03. high", "N/A")
            low = quote.get("04. low", "N/A")
            prev_close = quote.get("08. previous close", "N/A")
            latest_day = quote.get("07. latest trading day", "N/A")

            output = f"Alpha Vantage Quote for {ticker}:\n\n"
            output += f"=== Real-Time Quote ({latest_day}) ===\n"
            output += f"Price: ${price}\n"
            output += f"Change: {change} ({change_pct})\n"
            output += f"Day High: ${high} | Day Low: ${low}\n"
            output += f"Previous Close: ${prev_close}\n"
            try:
                output += f"Volume: {int(float(volume)):,}\n"
            except (ValueError, TypeError):
                output += f"Volume: {volume}\n"

            cache.set("av_quote", output, expire_hours=1, ticker=ticker)
            return output

        except Exception as e:
            error_msg = f"Error fetching Alpha Vantage quote for {ticker}: {str(e)}"
            logger.error(error_msg)
            return error_msg


class AlphaVantageEarningsTool(BaseTool):
    """Get quarterly and annual earnings history with surprises."""

    name: str = "Alpha Vantage Earnings"
    description: str = (
        "Get quarterly and annual earnings history for a stock ticker. "
        "Input should be a stock ticker symbol (e.g., 'TSLA'). "
        "Returns EPS actual vs estimate, surprise percentage, revenue data."
    )

    def _run(self, ticker: str) -> str:
        try:
            ticker = validate_ticker(ticker)
            logger.info(f"Alpha Vantage earnings for {ticker}")

            cached = cache.get("av_earnings", ticker=ticker)
            if cached:
                return cached

            data = _av_request("EARNINGS", symbol=ticker)

            output = f"Earnings History for {ticker}:\n\n"

            # Quarterly earnings
            quarterly = data.get("quarterlyEarnings", [])[:8]
            if quarterly:
                output += "=== Quarterly Earnings (Last 8 Quarters) ===\n"
                for q in quarterly:
                    date = q.get("fiscalDateEnding", "N/A")
                    reported = q.get("reportedEPS", "N/A")
                    estimated = q.get("estimatedEPS", "N/A")
                    surprise = q.get("surprise", "N/A")
                    surprise_pct = q.get("surprisePercentage", "N/A")

                    beat = ""
                    if surprise != "N/A" and surprise != "None":
                        try:
                            beat = " BEAT" if float(surprise) > 0 else " MISS"
                        except (ValueError, TypeError):
                            pass

                    output += f"  {date}: EPS ${reported} vs Est ${estimated}"
                    output += f" (Surprise: {surprise_pct}%{beat})\n"

            # Annual earnings
            annual = data.get("annualEarnings", [])[:5]
            if annual:
                output += "\n=== Annual Earnings (Last 5 Years) ===\n"
                for a in annual:
                    date = a.get("fiscalDateEnding", "N/A")
                    reported = a.get("reportedEPS", "N/A")
                    output += f"  {date}: EPS ${reported}\n"

            cache.set("av_earnings", output, expire_hours=12, ticker=ticker)
            return output

        except Exception as e:
            error_msg = f"Error fetching earnings for {ticker}: {str(e)}"
            logger.error(error_msg)
            return error_msg


class AlphaVantageOverviewTool(BaseTool):
    """Get comprehensive company fundamentals and overview."""

    name: str = "Alpha Vantage Company Overview"
    description: str = (
        "Get comprehensive company fundamentals: sector, market cap, P/E, "
        "52-week range, dividend, beta, and more. "
        "Input should be a stock ticker symbol (e.g., 'TSLA')."
    )

    def _run(self, ticker: str) -> str:
        try:
            ticker = validate_ticker(ticker)
            logger.info(f"Alpha Vantage overview for {ticker}")

            cached = cache.get("av_overview", ticker=ticker)
            if cached:
                return cached

            data = _av_request("OVERVIEW", symbol=ticker)

            if not data or "Symbol" not in data:
                return f"No overview data from Alpha Vantage for {ticker}"

            output = f"Company Overview for {data.get('Name', ticker)} ({ticker}):\n\n"

            output += "=== Company Info ===\n"
            output += f"Sector: {data.get('Sector', 'N/A')}\n"
            output += f"Industry: {data.get('Industry', 'N/A')}\n"
            output += f"Description: {data.get('Description', 'N/A')[:300]}...\n\n"

            output += "=== Valuation ===\n"
            try:
                mkt_cap = int(data.get('MarketCapitalization', 0) or 0)
                output += f"Market Cap: ${mkt_cap:,}\n"
            except (ValueError, TypeError):
                output += f"Market Cap: {data.get('MarketCapitalization', 'N/A')}\n"
            output += f"P/E Ratio (TTM): {data.get('TrailingPE', 'N/A')}\n"
            output += f"Forward P/E: {data.get('ForwardPE', 'N/A')}\n"
            output += f"PEG Ratio: {data.get('PEGRatio', 'N/A')}\n"
            output += f"Price/Book: {data.get('PriceToBookRatio', 'N/A')}\n"
            output += f"Price/Sales: {data.get('PriceToSalesRatioTTM', 'N/A')}\n"
            output += f"EV/Revenue: {data.get('EVToRevenue', 'N/A')}\n"
            output += f"EV/EBITDA: {data.get('EVToEBITDA', 'N/A')}\n\n"

            output += "=== Price Range ===\n"
            output += f"52-Week High: ${data.get('52WeekHigh', 'N/A')}\n"
            output += f"52-Week Low: ${data.get('52WeekLow', 'N/A')}\n"
            output += f"50-Day MA: ${data.get('50DayMovingAverage', 'N/A')}\n"
            output += f"200-Day MA: ${data.get('200DayMovingAverage', 'N/A')}\n"
            output += f"Beta: {data.get('Beta', 'N/A')}\n\n"

            output += "=== Dividend ===\n"
            output += f"Dividend Yield: {data.get('DividendYield', 'N/A')}\n"
            output += f"Dividend Per Share: ${data.get('DividendPerShare', 'N/A')}\n"
            output += f"Ex-Dividend Date: {data.get('ExDividendDate', 'N/A')}\n\n"

            output += "=== Profitability ===\n"
            output += f"Profit Margin: {data.get('ProfitMargin', 'N/A')}\n"
            output += f"Operating Margin: {data.get('OperatingMarginTTM', 'N/A')}\n"
            output += f"ROE: {data.get('ReturnOnEquityTTM', 'N/A')}\n"
            output += f"ROA: {data.get('ReturnOnAssetsTTM', 'N/A')}\n"
            try:
                rev_ttm = int(data.get('RevenueTTM', 0) or 0)
                output += f"Revenue (TTM): ${rev_ttm:,}\n"
            except (ValueError, TypeError):
                output += f"Revenue (TTM): {data.get('RevenueTTM', 'N/A')}\n"
            output += f"EPS (TTM): ${data.get('EPS', 'N/A')}\n"

            cache.set("av_overview", output, expire_hours=12, ticker=ticker)
            return output

        except Exception as e:
            error_msg = f"Error fetching overview for {ticker}: {str(e)}"
            logger.error(error_msg)
            return error_msg
