"""
Trending Stocks & News Dashboard API
=====================================
Standalone FastAPI that serves trending stock data with Google Finance-like
metadata + financial news in structured JSON for premium UI dashboards.

Run:  python -m uvicorn trending_api:app --port 8001
Docs: http://localhost:8001/docs

No LLM required -- purely aggregates free API data.
"""

import os
import time
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Trending Stocks & News Dashboard",
    description="Real-time trending stocks with Google Finance-like metadata and financial news",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Pydantic Response Models
# ===========================================================================

class StockCard(BaseModel):
    """Compact card view — for small icon / tile display."""
    ticker: str
    name: str
    price: float
    change: float
    change_percent: float
    exchange: str
    direction: str = Field(description="'up', 'down', or 'flat'")
    currency: str = "USD"


class StockDetail(BaseModel):
    """Full detail popup — mirrors Google Finance metadata."""
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    prev_close: Optional[float] = None
    market_cap: Optional[str] = None
    market_cap_raw: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    beta: Optional[float] = None
    dividend_yield: Optional[str] = None
    qtrly_dividend: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    volume: Optional[int] = None
    avg_volume: Optional[int] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None
    employees: Optional[int] = None
    earnings_growth: Optional[str] = None
    revenue_growth: Optional[str] = None
    profit_margin: Optional[str] = None
    operating_margin: Optional[str] = None
    roe: Optional[str] = None
    roa: Optional[str] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    free_cash_flow: Optional[str] = None


class TechnicalIndicators(BaseModel):
    """Technical analysis snapshot."""
    rsi_14: Optional[float] = None
    rsi_signal: Optional[str] = None
    macd_value: Optional[float] = None
    macd_signal_line: Optional[float] = None
    macd_direction: Optional[str] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    technical_summary: Optional[str] = None


class QuarterlyData(BaseModel):
    """Latest quarterly financials."""
    revenue: Optional[str] = None
    revenue_growth: Optional[str] = None
    earnings: Optional[str] = None
    earnings_growth: Optional[str] = None
    eps: Optional[float] = None
    eps_estimate: Optional[float] = None
    eps_surprise: Optional[str] = None


class TrendingStock(BaseModel):
    """Complete stock object for UI — card + popup detail."""
    card: StockCard
    detail: Optional[StockDetail] = None
    technicals: Optional[TechnicalIndicators] = None
    quarterly: Optional[QuarterlyData] = None


class NewsArticle(BaseModel):
    """Single news article."""
    title: str
    source: Optional[str] = None
    url: Optional[str] = None
    published: Optional[str] = None
    snippet: Optional[str] = None
    image_url: Optional[str] = None


class TrendingStocksResponse(BaseModel):
    timestamp: str
    count: int
    fetch_time_ms: int
    stocks: List[TrendingStock]


class TrendingNewsResponse(BaseModel):
    timestamp: str
    count: int
    news: List[NewsArticle]


class DashboardResponse(BaseModel):
    timestamp: str
    stocks: TrendingStocksResponse
    news: TrendingNewsResponse


# ===========================================================================
# Helper: Format large numbers
# ===========================================================================

def _fmt_number(val, prefix="$") -> Optional[str]:
    """Format large number to human readable: $2.78T, $142.5B, etc."""
    if val is None or val == 0:
        return None
    try:
        v = float(val)
    except (ValueError, TypeError):
        return None
    abs_v = abs(v)
    if abs_v >= 1e12:
        return f"{prefix}{v / 1e12:.2f}T"
    elif abs_v >= 1e9:
        return f"{prefix}{v / 1e9:.2f}B"
    elif abs_v >= 1e6:
        return f"{prefix}{v / 1e6:.2f}M"
    elif abs_v >= 1e3:
        return f"{prefix}{v / 1e3:.1f}K"
    return f"{prefix}{v:,.2f}"


def _pct_str(val) -> Optional[str]:
    """Format float to percentage string."""
    if val is None:
        return None
    try:
        return f"{float(val) * 100:.1f}%"
    except (ValueError, TypeError):
        return None


# ===========================================================================
# Helper: Fetch trending tickers using yfinance (free, no API key)
# ===========================================================================

# Pool of popular/high-volume tickers to scan for trending
_TICKER_POOL = [
    # Mega-cap tech
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
    # Semiconductors
    "AMD", "INTC", "AVGO", "QCOM", "MU",
    # Fintech / Finance
    "SOFI", "COIN", "JPM", "GS", "V", "MA",
    # Growth / Popular retail
    "PLTR", "NFLX", "DIS", "UBER", "SHOP", "XYZ", "SNOW",
    # Energy / EV
    "XOM", "CVX", "RIVN", "LCID", "NIO",
    # Healthcare / Biotech
    "JNJ", "PFE", "MRNA", "LLY",
    # Indices / ETFs for market context
    "SPY", "QQQ", "IWM", "DIA",
    # Crypto-adjacent
    "MSTR", "MARA",
]


def _fetch_trending_tickers(limit: int = 15) -> List[dict]:
    """Get trending stocks by scanning popular tickers via yfinance.

    Fetches real-time data for a pool of ~40 popular tickers and ranks
    them by absolute % change (most movement = most trending).
    Returns list of dicts with symbol, name, price, change, etc.
    """
    try:
        # Batch download 1-day data for the whole pool
        tickers_str = " ".join(_TICKER_POOL)
        data = yf.download(tickers_str, period="2d", group_by="ticker", progress=False, threads=True)

        results = []
        for ticker in _TICKER_POOL:
            try:
                if len(_TICKER_POOL) == 1:
                    close_col = data["Close"]
                else:
                    close_col = data[ticker]["Close"]

                if close_col.empty or len(close_col.dropna()) < 2:
                    continue

                prices = close_col.dropna()
                current_price = float(prices.iloc[-1])
                prev_price = float(prices.iloc[-2])
                change = round(current_price - prev_price, 2)
                change_pct = round((change / prev_price) * 100, 2) if prev_price != 0 else 0

                # Get company name from yfinance fast_info or info
                try:
                    t = yf.Ticker(ticker)
                    name = t.info.get("shortName", ticker) if hasattr(t, "info") else ticker
                    exchange = t.info.get("exchange", "") if hasattr(t, "info") else ""
                except Exception:
                    name = ticker
                    exchange = ""

                results.append({
                    "symbol": ticker,
                    "name": name,
                    "price": current_price,
                    "change": change,
                    "changesPercentage": change_pct,
                    "exchange": exchange,
                    "abs_change_pct": abs(change_pct),  # for sorting
                })
            except Exception:
                continue

        # Sort by absolute % change (most volatile / trending first)
        results.sort(key=lambda x: x.get("abs_change_pct", 0), reverse=True)
        return results[:limit]

    except Exception as e:
        logger.error(f"yfinance trending fetch failed: {e}")
        return []


# Fallback popular tickers if yfinance batch fails
FALLBACK_TICKERS = [
    "NVDA", "AAPL", "TSLA", "AMZN", "GOOGL", "MSFT", "META",
    "PLTR", "AMD", "INTC", "SOFI", "COIN", "NFLX", "JPM", "V",
]


# ===========================================================================
# Helper: yfinance metadata for a single ticker
# ===========================================================================

def _fetch_stock_metadata(ticker: str) -> dict:
    """Fetch full Google Finance-like metadata from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or not info.get("shortName"):
            return {}
        return info
    except Exception as e:
        logger.warning(f"yfinance metadata failed for {ticker}: {e}")
        return {}


def _build_stock_detail(info: dict) -> StockDetail:
    """Build StockDetail from yfinance info dict."""
    mcap = info.get("marketCap")
    fcf = info.get("freeCashflow")
    return StockDetail(
        open=info.get("open") or info.get("regularMarketOpen"),
        high=info.get("dayHigh") or info.get("regularMarketDayHigh"),
        low=info.get("dayLow") or info.get("regularMarketDayLow"),
        prev_close=info.get("previousClose") or info.get("regularMarketPreviousClose"),
        market_cap=_fmt_number(mcap),
        market_cap_raw=float(mcap) if mcap else None,
        pe_ratio=info.get("trailingPE"),
        forward_pe=info.get("forwardPE"),
        peg_ratio=info.get("pegRatio"),
        beta=info.get("beta"),
        dividend_yield=_pct_str(info.get("dividendYield")),
        qtrly_dividend=info.get("lastDividendValue"),
        week_52_high=info.get("fiftyTwoWeekHigh"),
        week_52_low=info.get("fiftyTwoWeekLow"),
        volume=info.get("volume") or info.get("regularMarketVolume"),
        avg_volume=info.get("averageVolume"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        website=info.get("website"),
        description=info.get("longBusinessSummary", "")[:300] if info.get("longBusinessSummary") else None,
        employees=info.get("fullTimeEmployees"),
        earnings_growth=_pct_str(info.get("earningsGrowth")),
        revenue_growth=_pct_str(info.get("revenueGrowth")),
        profit_margin=_pct_str(info.get("profitMargins")),
        operating_margin=_pct_str(info.get("operatingMargins")),
        roe=_pct_str(info.get("returnOnEquity")),
        roa=_pct_str(info.get("returnOnAssets")),
        debt_to_equity=round(info.get("debtToEquity", 0) / 100, 2) if info.get("debtToEquity") else None,
        current_ratio=round(info.get("currentRatio", 0), 2) if info.get("currentRatio") else None,
        free_cash_flow=_fmt_number(fcf),
    )


# ===========================================================================
# Helper: Technical indicators
# ===========================================================================

def _compute_technicals(ticker: str) -> Optional[TechnicalIndicators]:
    """Compute RSI, MACD, SMA, Bollinger from yfinance price data."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty or len(hist) < 26:
            return None

        close = hist["Close"]

        # RSI (14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_val = round(float(rsi.iloc[-1]), 1)
        rsi_sig = "OVERBOUGHT" if rsi_val > 70 else ("OVERSOLD" if rsi_val < 30 else "NEUTRAL")

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_val = round(float(macd_line.iloc[-1]), 2)
        sig_val = round(float(signal_line.iloc[-1]), 2)
        macd_dir = "BULLISH" if macd_val > sig_val else "BEARISH"

        # SMAs
        sma20 = round(float(close.rolling(20).mean().iloc[-1]), 2)
        sma50 = round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else None
        sma200 = round(float(close.rolling(200).mean().iloc[-1]), 2) if len(close) >= 200 else None

        # Bollinger
        bb_std = float(close.rolling(20).std().iloc[-1])
        bb_upper = round(sma20 + 2 * bb_std, 2)
        bb_lower = round(sma20 - 2 * bb_std, 2)

        # Summary
        signals = []
        if rsi_val > 70:
            signals.append("overbought")
        elif rsi_val < 30:
            signals.append("oversold")
        signals.append(f"MACD {macd_dir.lower()}")
        current = float(close.iloc[-1])
        if sma50 and current > sma50:
            signals.append("above SMA50")
        elif sma50:
            signals.append("below SMA50")

        return TechnicalIndicators(
            rsi_14=rsi_val,
            rsi_signal=rsi_sig,
            macd_value=macd_val,
            macd_signal_line=sig_val,
            macd_direction=macd_dir,
            sma_20=sma20,
            sma_50=sma50,
            sma_200=sma200,
            bollinger_upper=bb_upper,
            bollinger_lower=bb_lower,
            technical_summary=", ".join(signals),
        )
    except Exception as e:
        logger.warning(f"Technicals failed for {ticker}: {e}")
        return None


# ===========================================================================
# Helper: Quarterly financials
# ===========================================================================

def _fetch_quarterly(ticker: str, info: dict) -> Optional[QuarterlyData]:
    """Extract latest quarterly data from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        qf = stock.quarterly_financials
        if qf is None or qf.empty:
            return None

        latest = qf.iloc[:, 0]  # most recent quarter
        prev = qf.iloc[:, 1] if qf.shape[1] > 1 else None

        revenue = latest.get("Total Revenue")
        earnings = latest.get("Net Income")

        rev_growth = None
        earn_growth = None
        if prev is not None:
            prev_rev = prev.get("Total Revenue")
            prev_earn = prev.get("Net Income")
            if prev_rev and revenue and prev_rev != 0:
                rev_growth = f"{((revenue - prev_rev) / abs(prev_rev)) * 100:+.1f}%"
            if prev_earn and earnings and prev_earn != 0:
                earn_growth = f"{((earnings - prev_earn) / abs(prev_earn)) * 100:+.1f}%"

        return QuarterlyData(
            revenue=_fmt_number(revenue),
            revenue_growth=rev_growth,
            earnings=_fmt_number(earnings),
            earnings_growth=earn_growth,
            eps=info.get("trailingEps"),
            eps_estimate=info.get("forwardEps"),
            eps_surprise=None,
        )
    except Exception as e:
        logger.warning(f"Quarterly data failed for {ticker}: {e}")
        return None


# ===========================================================================
# Helper: Build complete TrendingStock from yfinance data
# ===========================================================================

def _build_trending_stock(stock_item: dict, include_detail: bool = True) -> TrendingStock:
    """Build a complete TrendingStock object."""
    ticker = stock_item.get("symbol", "???")
    price = stock_item.get("price", 0)
    change = stock_item.get("change", 0)
    change_pct = stock_item.get("changesPercentage", 0)

    direction = "up" if change > 0 else ("down" if change < 0 else "flat")

    card = StockCard(
        ticker=ticker,
        name=stock_item.get("name", ticker),
        price=round(price, 2),
        change=round(change, 2),
        change_percent=round(change_pct, 2),
        exchange=stock_item.get("exchange", ""),
        direction=direction,
    )

    detail = None
    technicals = None
    quarterly = None

    if include_detail:
        info = _fetch_stock_metadata(ticker)
        if info:
            detail = _build_stock_detail(info)
            quarterly = _fetch_quarterly(ticker, info)
        technicals = _compute_technicals(ticker)

    return TrendingStock(
        card=card,
        detail=detail,
        technicals=technicals,
        quarterly=quarterly,
    )


# ===========================================================================
# Helper: Fetch financial news from Tavily
# ===========================================================================

def _fetch_financial_news(count: int = 20) -> List[NewsArticle]:
    """Fetch trending financial news from Tavily."""
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set, no news")
        return []
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": "stock market financial news today trending",
                "search_depth": "basic",
                "max_results": count,
                "include_domains": [
                    "reuters.com", "bloomberg.com", "cnbc.com", "wsj.com",
                    "seekingalpha.com", "yahoo.com", "marketwatch.com",
                    "fool.com", "investopedia.com", "barrons.com",
                    "finance.yahoo.com", "businessinsider.com",
                ],
            },
            timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        articles = []
        for item in results:
            url = item.get("url", "")
            source = url.split("/")[2] if "/" in url else ""
            articles.append(NewsArticle(
                title=item.get("title", ""),
                source=source,
                url=url,
                published=item.get("published_date"),
                snippet=item.get("content", "")[:250],
                image_url=item.get("image"),
            ))
        return articles
    except Exception as e:
        logger.error(f"News fetch failed: {e}")
        return []


# ===========================================================================
# API Endpoints
# ===========================================================================

@app.get("/", tags=["Health"])
async def root():
    """Health check."""
    return {
        "service": "Trending Stocks & News Dashboard",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": [
            "/trending/stocks",
            "/trending/news",
            "/trending/dashboard",
            "/trending/stock/{ticker}",
        ],
    }


@app.get("/trending/stocks", response_model=TrendingStocksResponse, tags=["Trending"])
async def get_trending_stocks(
    limit: int = Query(default=15, ge=1, le=30, description="Number of stocks to return"),
    detail: bool = Query(default=True, description="Include full metadata (slower)"),
):
    """
    Get top trending / most-active stocks with Google Finance-like metadata.

    - **Card view**: ticker, name, price, change, direction (always returned)
    - **Detail view**: market cap, P/E, 52wk range, open/high/low, sector, margins (when detail=true)
    - **Technicals**: RSI, MACD, SMA, Bollinger Bands (when detail=true)
    - **Quarterly**: Latest revenue, earnings, EPS (when detail=true)
    """
    t0 = time.time()

    trending_items = _fetch_trending_tickers(limit)
    if not trending_items:
        # Fallback: build from popular tickers using yfinance
        trending_items = [
            {"symbol": t, "name": t, "price": 0, "change": 0, "changesPercentage": 0, "exchange": ""}
            for t in FALLBACK_TICKERS[:limit]
        ]

    stocks = []
    for item in trending_items:
        try:
            stock = _build_trending_stock(item, include_detail=detail)
            stocks.append(stock)
        except Exception as e:
            logger.warning(f"Failed to build stock {item.get('symbol')}: {e}")

    elapsed = int((time.time() - t0) * 1000)
    return TrendingStocksResponse(
        timestamp=datetime.utcnow().isoformat() + "Z",
        count=len(stocks),
        fetch_time_ms=elapsed,
        stocks=stocks,
    )


@app.get("/trending/news", response_model=TrendingNewsResponse, tags=["Trending"])
async def get_trending_news(
    count: int = Query(default=20, ge=1, le=50, description="Number of news articles"),
):
    """
    Get latest trending financial news from major sources.

    Returns title, source, URL, published date, snippet for each article.
    """
    articles = _fetch_financial_news(count)
    return TrendingNewsResponse(
        timestamp=datetime.utcnow().isoformat() + "Z",
        count=len(articles),
        news=articles,
    )


@app.get("/trending/dashboard", response_model=DashboardResponse, tags=["Trending"])
async def get_dashboard(
    stock_limit: int = Query(default=15, ge=1, le=30, description="Number of stocks"),
    news_count: int = Query(default=20, ge=1, le=50, description="Number of news articles"),
    detail: bool = Query(default=True, description="Include full stock metadata"),
):
    """
    Combined dashboard: trending stocks + financial news in one call.

    Use this for initial page load to get everything in a single request.
    """
    t0 = time.time()

    # Fetch trending tickers
    trending_items = _fetch_trending_tickers(stock_limit)
    if not trending_items:
        trending_items = [
            {"symbol": t, "name": t, "price": 0, "change": 0, "changesPercentage": 0, "exchange": ""}
            for t in FALLBACK_TICKERS[:stock_limit]
        ]

    stocks = []
    for item in trending_items:
        try:
            stock = _build_trending_stock(item, include_detail=detail)
            stocks.append(stock)
        except Exception as e:
            logger.warning(f"Failed to build stock {item.get('symbol')}: {e}")

    elapsed = int((time.time() - t0) * 1000)
    stocks_response = TrendingStocksResponse(
        timestamp=datetime.utcnow().isoformat() + "Z",
        count=len(stocks),
        fetch_time_ms=elapsed,
        stocks=stocks,
    )

    articles = _fetch_financial_news(news_count)
    news_response = TrendingNewsResponse(
        timestamp=datetime.utcnow().isoformat() + "Z",
        count=len(articles),
        news=articles,
    )

    return DashboardResponse(
        timestamp=datetime.utcnow().isoformat() + "Z",
        stocks=stocks_response,
        news=news_response,
    )


@app.get("/trending/stock/{ticker}", response_model=TrendingStock, tags=["Trending"])
async def get_stock_detail(ticker: str):
    """
    Get full detail for a single stock ticker (popup view).

    Returns card + detail + technicals + quarterly data.
    """
    ticker = ticker.upper().strip()
    if not ticker or len(ticker) > 10:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    info = _fetch_stock_metadata(ticker)
    if not info:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

    price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose", 0)
    change = round(price - prev_close, 2) if price and prev_close else 0
    change_pct = round((change / prev_close) * 100, 2) if prev_close and prev_close != 0 else 0
    direction = "up" if change > 0 else ("down" if change < 0 else "flat")

    card = StockCard(
        ticker=ticker,
        name=info.get("shortName") or info.get("longName", ticker),
        price=round(price, 2) if price else 0,
        change=change,
        change_percent=change_pct,
        exchange=info.get("exchange", ""),
        direction=direction,
    )

    detail = _build_stock_detail(info)
    technicals = _compute_technicals(ticker)
    quarterly = _fetch_quarterly(ticker, info)

    return TrendingStock(
        card=card,
        detail=detail,
        technicals=technicals,
        quarterly=quarterly,
    )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("trending_api:app", host="0.0.0.0", port=8001, reload=True, log_level="info")
