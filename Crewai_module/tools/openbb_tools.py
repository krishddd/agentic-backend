"""OpenBB Platform Tools — Economic indicators and technical analysis.

OpenBB SDK is optional — if not installed, falls back to yfinance + manual
calculation for technical indicators.

Install: pip install openbb>=4.0.0 (optional)
"""

from typing import Optional
import os
import yfinance as yf
from crewai.tools import BaseTool
from config.settings import settings
from utils.logger import get_logger
from utils.cache import cache
from utils.validators import validate_ticker

logger = get_logger(__name__)

# Try to import OpenBB — fall back gracefully if not available
try:
    from openbb import obb as _obb
    _HAS_OPENBB = True
    logger.info("OpenBB SDK available")
except ImportError:
    _obb = None
    _HAS_OPENBB = False
    logger.info("OpenBB SDK not installed, using yfinance fallback")


class OpenBBEconomicTool(BaseTool):
    """Get macroeconomic indicators: GDP, CPI, unemployment, Fed funds rate."""

    name: str = "Economic Indicators"
    description: str = (
        "Get macroeconomic data: GDP growth, CPI inflation, unemployment rate, "
        "Fed funds rate, and more. "
        "Input should be an indicator name or a general query "
        "(e.g., 'GDP', 'CPI', 'unemployment US', 'federal funds rate')."
    )

    def _run(self, indicator: str) -> str:
        try:
            indicator_lower = indicator.strip().lower()
            logger.info(f"Economic indicator request: {indicator}")

            cached = cache.get("econ_indicator", indicator=indicator_lower)
            if cached:
                return cached

            if _HAS_OPENBB:
                result = self._openbb_fetch(indicator_lower)
            else:
                result = self._fallback_fetch(indicator_lower)

            cache.set("econ_indicator", result, expire_hours=24, indicator=indicator_lower)
            return result

        except Exception as e:
            error_msg = f"Error fetching economic data for '{indicator}': {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _openbb_fetch(self, indicator: str) -> str:
        """Fetch via OpenBB SDK."""
        try:
            # Map user input to OpenBB functions
            if any(k in indicator for k in ["gdp", "gross domestic"]):
                data = _obb.economy.gdp.nominal(provider="oecd", country="united_states")
                return f"US GDP Data (OpenBB):\n{data.to_df().tail(8).to_string()}"
            elif any(k in indicator for k in ["cpi", "inflation", "consumer price"]):
                data = _obb.economy.cpi(country="united_states", provider="fred")
                return f"US CPI/Inflation Data (OpenBB):\n{data.to_df().tail(12).to_string()}"
            elif any(k in indicator for k in ["unemployment", "jobless"]):
                data = _obb.economy.unemployment(country="united_states", provider="oecd")
                return f"US Unemployment Data (OpenBB):\n{data.to_df().tail(12).to_string()}"
            elif any(k in indicator for k in ["fed", "interest rate", "funds rate"]):
                data = _obb.economy.fred_series(symbol="FEDFUNDS")
                return f"Federal Funds Rate (OpenBB):\n{data.to_df().tail(12).to_string()}"
            else:
                return self._fallback_fetch(indicator)
        except Exception as e:
            logger.warning(f"OpenBB fetch failed: {e}, using fallback")
            return self._fallback_fetch(indicator)

    def _fallback_fetch(self, indicator: str) -> str:
        """Fallback using FRED REST API (no key needed for basic data)."""
        import requests

        fred_series = {
            "gdp": ("GDP", "US Gross Domestic Product"),
            "gross domestic": ("GDP", "US Gross Domestic Product"),
            "cpi": ("CPIAUCSL", "Consumer Price Index (All Urban)"),
            "inflation": ("CPIAUCSL", "Consumer Price Index (All Urban)"),
            "unemployment": ("UNRATE", "US Unemployment Rate"),
            "jobless": ("UNRATE", "US Unemployment Rate"),
            "fed": ("FEDFUNDS", "Federal Funds Effective Rate"),
            "interest rate": ("FEDFUNDS", "Federal Funds Effective Rate"),
            "funds rate": ("FEDFUNDS", "Federal Funds Effective Rate"),
        }

        series_id = None
        series_name = indicator
        for key, (sid, sname) in fred_series.items():
            if key in indicator:
                series_id = sid
                series_name = sname
                break

        if not series_id:
            return (
                f"Unknown economic indicator: '{indicator}'. "
                f"Try one of: GDP, CPI, unemployment, Fed funds rate."
            )

        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            fred_key = os.getenv("FRED_API_KEY", "")
            if not fred_key:
                return f"FRED_API_KEY not configured. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            params = {
                "series_id": series_id,
                "api_key": fred_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 12,
            }
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                return f"FRED API unavailable for {series_name}. Install openbb for full access."

            data = resp.json()
            observations = data.get("observations", [])

            if not observations:
                return f"No data available for {series_name}"

            output = f"{series_name} (FRED Series: {series_id}):\n\n"
            for obs in observations[:12]:
                output += f"  {obs.get('date', 'N/A')}: {obs.get('value', 'N/A')}\n"

            return output

        except Exception as e:
            return f"Could not fetch {series_name}: {str(e)}. Install openbb for better results."


class OpenBBTechnicalTool(BaseTool):
    """Calculate technical indicators: RSI, MACD, Bollinger Bands, moving averages."""

    name: str = "Technical Analysis"
    description: str = (
        "Calculate technical indicators for a stock: RSI, MACD, Bollinger Bands, "
        "SMA/EMA moving averages. "
        "Input should be 'TICKER INDICATOR [PERIOD]' "
        "(e.g., 'TSLA RSI 14', 'AAPL MACD', 'MSFT SMA 50')."
    )

    def _run(self, input_str: str) -> str:
        try:
            parts = input_str.strip().upper().split()
            if len(parts) < 2:
                return "Input format: TICKER INDICATOR [PERIOD]. E.g., 'TSLA RSI 14'"

            ticker = validate_ticker(parts[0])
            indicator = parts[1]
            period = int(parts[2]) if len(parts) > 2 else None

            logger.info(f"Technical analysis: {ticker} {indicator} {period}")

            cached = cache.get("tech_analysis", ticker=ticker, indicator=indicator, period=str(period))
            if cached:
                return cached

            # Fetch price data via yfinance
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")

            if hist.empty:
                return f"No price data available for {ticker}"

            close = hist["Close"]

            if indicator == "RSI":
                result = self._calc_rsi(close, period or 14, ticker)
            elif indicator == "MACD":
                result = self._calc_macd(close, ticker)
            elif indicator in ("BB", "BOLLINGER"):
                result = self._calc_bollinger(close, period or 20, ticker)
            elif indicator == "SMA":
                result = self._calc_sma(close, period or 50, ticker)
            elif indicator == "EMA":
                result = self._calc_ema(close, period or 20, ticker)
            else:
                result = (
                    f"Unknown indicator: {indicator}. "
                    f"Available: RSI, MACD, BB (Bollinger), SMA, EMA"
                )

            cache.set("tech_analysis", result, expire_hours=1,
                       ticker=ticker, indicator=indicator, period=str(period))
            return result

        except Exception as e:
            error_msg = f"Error calculating technical indicator: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _calc_rsi(self, close, period: int, ticker: str) -> str:
        """Calculate Relative Strength Index."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]
        signal = "OVERBOUGHT" if current_rsi > 70 else "OVERSOLD" if current_rsi < 30 else "NEUTRAL"

        output = f"RSI({period}) for {ticker}:\n\n"
        output += f"Current RSI: {current_rsi:.2f}\n"
        output += f"Signal: {signal}\n\n"
        output += "Recent RSI values:\n"
        for date, val in rsi.tail(5).items():
            output += f"  {date.strftime('%Y-%m-%d')}: {val:.2f}\n"
        return output

    def _calc_macd(self, close, ticker: str) -> str:
        """Calculate MACD (12, 26, 9)."""
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_hist = histogram.iloc[-1]
        trend = "BULLISH" if current_macd > current_signal else "BEARISH"

        output = f"MACD (12, 26, 9) for {ticker}:\n\n"
        output += f"MACD Line: {current_macd:.4f}\n"
        output += f"Signal Line: {current_signal:.4f}\n"
        output += f"Histogram: {current_hist:.4f}\n"
        output += f"Signal: {trend}\n\n"
        output += "Recent MACD values:\n"
        for i in range(-5, 0):
            date = close.index[i].strftime('%Y-%m-%d')
            output += f"  {date}: MACD={macd_line.iloc[i]:.4f}, Signal={signal_line.iloc[i]:.4f}\n"
        return output

    def _calc_bollinger(self, close, period: int, ticker: str) -> str:
        """Calculate Bollinger Bands."""
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)

        current_price = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        current_sma = sma.iloc[-1]
        bandwidth = ((current_upper - current_lower) / current_sma) * 100

        position = "ABOVE UPPER (Overbought)" if current_price > current_upper \
            else "BELOW LOWER (Oversold)" if current_price < current_lower \
            else "WITHIN BANDS"

        output = f"Bollinger Bands ({period}) for {ticker}:\n\n"
        output += f"Current Price: ${current_price:.2f}\n"
        output += f"Upper Band: ${current_upper:.2f}\n"
        output += f"Middle (SMA): ${current_sma:.2f}\n"
        output += f"Lower Band: ${current_lower:.2f}\n"
        output += f"Bandwidth: {bandwidth:.2f}%\n"
        output += f"Position: {position}\n"
        return output

    def _calc_sma(self, close, period: int, ticker: str) -> str:
        """Calculate Simple Moving Average."""
        sma = close.rolling(window=period).mean()
        current_price = close.iloc[-1]
        current_sma = sma.iloc[-1]
        signal = "ABOVE SMA (Bullish)" if current_price > current_sma else "BELOW SMA (Bearish)"

        output = f"SMA({period}) for {ticker}:\n\n"
        output += f"Current Price: ${current_price:.2f}\n"
        output += f"SMA({period}): ${current_sma:.2f}\n"
        output += f"Signal: {signal}\n"
        return output

    def _calc_ema(self, close, period: int, ticker: str) -> str:
        """Calculate Exponential Moving Average."""
        ema = close.ewm(span=period, adjust=False).mean()
        current_price = close.iloc[-1]
        current_ema = ema.iloc[-1]
        signal = "ABOVE EMA (Bullish)" if current_price > current_ema else "BELOW EMA (Bearish)"

        output = f"EMA({period}) for {ticker}:\n\n"
        output += f"Current Price: ${current_price:.2f}\n"
        output += f"EMA({period}): ${current_ema:.2f}\n"
        output += f"Signal: {signal}\n"
        return output
