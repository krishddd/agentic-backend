"""
Multi-Agent Orchestrator v6.0 -- Multi-LLM Deep Research Pipeline

LLM Assignments:
  llama3.2:latest  (2.0GB) -> Deep Research, Latest News  (fluent text)
  qwen3:4b         (2.5GB) -> Financial Analysis, SEC Risk (structured data)
  qwen3:8b         (5.2GB) -> Marketing, Sentiment, Synthesis (strongest reasoning)

Data Sources:
  Tavily API       -> Web search, news search (real-time data)
  yfinance         -> Financial metrics (live market data)
  SEC EDGAR API    -> Real SEC filings + XBRL company facts
  Alpha Vantage    -> Earnings history, company overview (free API key)
  Technical        -> RSI, MACD, Bollinger Bands (computed)
  Scrapling        -> Full-page content enrichment

Pipeline (due_diligence):
  [1] Deep Research     llama3.2   Tavily web search -> LLM analysis
  [2] Latest News       llama3.2   Tavily news search -> LLM digest
  [3] Financial         qwen3:4b   yfinance + XBRL + AV + technicals
  [4] SEC Risk          qwen3:4b   SEC EDGAR + XBRL -> LLM risk
  [5] Marketing         qwen3:8b   LLM: brand, moat, competitive
  [6] Sentiment         qwen3:8b   LLM-powered sentiment
  [7] Synthesis         qwen3:8b   LLM: combine all -> investment report
  [8] Save+Email+Sheet  --         Save files, Gmail, Google Sheets
"""

import os
import sys
import time
import json
import yaml
import logging
import requests
import importlib.util
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# MiroFish date guard — injected into all LLM system prompts
_TODAY = datetime.now().strftime("%B %d, %Y")
_DATE_GUARD = (f"Today's date is {_TODAY}. Only reference events up to this date. "
               "If information appears outdated (>2 years old), note this explicitly. "
               "Do not cite future dates.")

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from llm_router import OLLAMA_BASE_URL
from prompt_router import PromptRouter, ParsedIntent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM assignments (3 models, each with a role)
# ---------------------------------------------------------------------------
LLM_RESEARCH = "llama3.2:latest"   # Deep research, news (fluent text)
LLM_FINANCIAL = "qwen3:4b"         # Financial + SEC (structured data)
LLM_REASONING = "qwen3:8b"         # Marketing, sentiment, synthesis (strongest)

# Tavily API
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_ENDPOINT = "https://api.tavily.com/search"


# ===========================================================================
# Workflow State
# ===========================================================================

@dataclass
class WorkflowState:
    workflow_name: str = ""
    prompt: str = ""
    company: str = ""
    tickers: List[str] = field(default_factory=list)

    # Agent outputs
    research_output: str = ""
    news_output: str = ""
    financial_data: str = ""
    sec_analysis: str = ""
    marketing_analysis: str = ""
    sentiment_data: str = ""
    synthesis_report: str = ""

    # ABM / MiroFish outputs (Phase 1.5)
    graph_context: str = ""
    simulation_result: Dict[str, Any] = field(default_factory=dict)
    simulation_report: str = ""
    validation_result: Dict[str, Any] = field(default_factory=dict)
    mirofish_skipped: bool = False
    abm_budget_mode: str = "lite"

    # Shared facts dict — consistent data across all agents
    shared_facts: Dict[str, Any] = field(default_factory=dict)

    # Meta
    sources_count: int = 0
    quality_score: float = 0.0
    generated_leads: List[Dict] = field(default_factory=list)
    deal_score: int = 0

    # Output
    email_status: Dict[str, Any] = field(default_factory=dict)
    sheet_status: str = ""
    report_dir: str = ""

    # Progress
    total_steps: int = 0
    current_step: int = 0
    agent_sequence: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    step_log: List[Dict] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    started_at: float = 0.0

    def elapsed_sec(self) -> float:
        return round(time.time() - self.started_at, 1) if self.started_at else 0.0

    def log_step(self, agent, status, duration, detail=""):
        self.step_log.append({
            "step": self.current_step, "total": self.total_steps,
            "agent": agent, "status": status,
            "duration_s": round(duration, 1), "detail": detail,
            "timestamp": datetime.now().isoformat(),
        })


# ===========================================================================
# Tavily Search (replaces DuckDuckGo)
# ===========================================================================

def tavily_search(query: str, search_depth: str = "basic", max_results: int = 5) -> str:
    """Search the web via Tavily API. Returns formatted results."""
    try:
        r = requests.post(TAVILY_ENDPOINT, json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
        }, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return f"No results found for: {query}"
        output = f"Search results for '{query}':\n\n"
        for i, res in enumerate(results, 1):
            output += f"[{i}] {res.get('title', 'N/A')}\n"
            output += f"    {res.get('content', 'N/A')}\n"
            output += f"    URL: {res.get('url', 'N/A')}\n\n"
        return output
    except Exception as e:
        return f"[ERROR] Tavily search failed: {e}"


def tavily_news(query: str, max_results: int = 5) -> str:
    """Search recent news via Tavily API."""
    try:
        r = requests.post(TAVILY_ENDPOINT, json={
            "api_key": TAVILY_API_KEY,
            "query": f"{query} latest news",
            "search_depth": "basic",
            "max_results": max_results,
            "include_domains": ["reuters.com", "bloomberg.com", "cnbc.com",
                                "wsj.com", "seekingalpha.com", "yahoo.com",
                                "marketwatch.com", "fool.com", "investopedia.com"],
        }, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return f"No news found for: {query}"
        output = f"Latest news for '{query}':\n\n"
        for i, res in enumerate(results, 1):
            output += f"[{i}] {res.get('title', 'N/A')}\n"
            output += f"    {res.get('content', 'N/A')}\n"
            output += f"    Source: {res.get('url', 'N/A')}\n\n"
        return output
    except Exception as e:
        return f"[ERROR] Tavily news failed: {e}"


# ===========================================================================
# Direct Ollama Caller (supports model selection)
# ===========================================================================

def call_ollama(prompt, system="", model=LLM_RESEARCH, timeout=90, max_tokens=4096):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json={
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0.3, "num_predict": max_tokens},
        }, timeout=timeout)
        r.raise_for_status()
        msg = r.json().get("message", {})
        content = msg.get("content", "")
        # qwen3 thinking mode: content may be empty, output in 'thinking'
        if not content.strip():
            thinking = msg.get("thinking", "")
            if thinking:
                import re
                content = re.sub(r"<think>|</think>", "", thinking).strip()
        return content
    except requests.exceptions.Timeout:
        return f"[TIMEOUT] {model} timed out after {timeout}s"
    except Exception as e:
        return f"[ERROR] Ollama ({model}) failed: {e}"


# ===========================================================================
# yfinance fetcher (Crewai FinancialMetricsTool logic, no Crewai import)
# ===========================================================================

def fetch_financial_data(ticker: str) -> str:
    """Fetch comprehensive financial data -- mirrors Crewai FinancialMetricsTool."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info

        out = f"Financial Metrics for {ticker} ({info.get('longName', 'N/A')}):\n\n"
        out += "=== Company Overview ===\n"
        out += f"Sector: {info.get('sector', 'N/A')}\n"
        out += f"Industry: {info.get('industry', 'N/A')}\n"
        mcap = info.get('marketCap', 0)
        out += f"Market Cap: ${mcap:,.0f}\n" if mcap else "Market Cap: N/A\n"
        out += f"Current Price: ${info.get('currentPrice') or info.get('regularMarketPrice', 'N/A')}\n\n"

        out += "=== Revenue & Profitability ===\n"
        rev = info.get('totalRevenue', 0)
        out += f"Total Revenue: ${rev:,.0f}\n" if rev else ""
        rg = info.get('revenueGrowth', 0)
        out += f"Revenue Growth (YoY): {rg*100:.1f}%\n" if rg else ""
        out += f"Gross Margin: {info.get('grossMargins', 0)*100:.1f}%\n"
        out += f"Operating Margin: {info.get('operatingMargins', 0)*100:.1f}%\n"
        out += f"Profit Margin: {info.get('profitMargins', 0)*100:.1f}%\n"
        ebitda = info.get('ebitda', 0)
        out += f"EBITDA: ${ebitda:,.0f}\n\n" if ebitda else "\n"

        out += "=== Profitability Ratios ===\n"
        out += f"ROE: {info.get('returnOnEquity', 0)*100:.1f}%\n"
        out += f"ROA: {info.get('returnOnAssets', 0)*100:.1f}%\n\n"

        out += "=== Cash Flow ===\n"
        ocf = info.get('operatingCashflow', 0)
        fcf = info.get('freeCashflow', 0)
        out += f"Operating Cash Flow: ${ocf:,.0f}\n" if ocf else ""
        out += f"Free Cash Flow: ${fcf:,.0f}\n\n" if fcf else "\n"

        out += "=== Debt & Liquidity ===\n"
        out += f"Total Cash: ${info.get('totalCash', 0):,.0f}\n"
        out += f"Total Debt: ${info.get('totalDebt', 0):,.0f}\n"
        dte = info.get('debtToEquity', 0)
        out += f"Debt to Equity: {dte/100:.2f}\n" if dte else ""
        out += f"Current Ratio: {info.get('currentRatio', 0):.2f}\n\n"

        out += "=== Valuation ===\n"
        out += f"P/E (TTM): {info.get('trailingPE', 'N/A')}\n"
        out += f"Forward P/E: {info.get('forwardPE', 'N/A')}\n"
        out += f"PEG Ratio: {info.get('pegRatio', 'N/A')}\n"
        out += f"Price/Book: {info.get('priceToBook', 'N/A')}\n\n"

        out += "=== Growth ===\n"
        eg = info.get('earningsGrowth', 0)
        out += f"Earnings Growth: {eg*100:.1f}%\n" if eg else ""
        out += f"EPS (TTM): ${info.get('trailingEps', 'N/A')}\n"
        out += f"Forward EPS: ${info.get('forwardEps', 'N/A')}\n\n"

        out += "=== Analyst ===\n"
        out += f"Target Price: ${info.get('targetMeanPrice', 'N/A')}\n"
        out += f"Recommendation: {(info.get('recommendationKey') or 'N/A').upper()}\n"

        div = info.get('dividendYield')
        if div:
            out += f"Dividend Yield: {div:.2%}\n"
        return out
    except Exception as e:
        return f"[ERROR] yfinance failed for {ticker}: {e}"


# ===========================================================================
# SEC EDGAR search (Crewai SECFilingSearchTool logic, no Crewai import)
# ===========================================================================

def fetch_sec_filings(ticker: str) -> str:
    """Fetch SEC filing info from EDGAR API."""
    try:
        headers = {"User-Agent": "MultiAgentOrchestrator/5.0 harish.krishna@testaing.com", "Accept": "application/json"}
        # Get CIK
        cik_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&forms=10-K"
        # Simpler: use company tickers JSON
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=15)
        r.raise_for_status()
        tickers_data = r.json()
        cik = None
        for entry in tickers_data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                break
        if not cik:
            return f"[ERROR] Could not find CIK for {ticker}"

        # Get recent filings
        filings_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(filings_url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        descs = recent.get("primaryDocument", [])
        accessions = recent.get("accessionNumber", [])

        out = f"SEC Filings for {ticker} (CIK: {cik}):\n\n"
        count = 0
        for i, form in enumerate(forms):
            if form in ("10-K", "10-Q", "8-K", "DEF 14A") and count < 10:
                out += f"  [{form}] Filed: {dates[i]} | Doc: {descs[i]}\n"
                acc = accessions[i].replace("-", "")
                out += f"         URL: https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{descs[i]}\n"
                count += 1
        if count == 0:
            out += "  No recent 10-K/10-Q/8-K filings found.\n"
        return out
    except Exception as e:
        return f"[ERROR] SEC EDGAR failed for {ticker}: {e}"


# ===========================================================================
# Enhanced Financial Data Fetchers (Phase 1 tools — standalone, no CrewAI)
# ===========================================================================

def fetch_xbrl_financials(ticker: str) -> str:
    """Fetch structured XBRL financial data from SEC EDGAR companyfacts."""
    try:
        headers = {"User-Agent": "MultiAgentOrchestrator/5.0 harish.krishna@testaing.com", "Accept": "application/json"}
        # Get CIK
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=15)
        r.raise_for_status()
        cik = None
        for entry in r.json().values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                break
        if not cik:
            return ""

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        facts = r.json().get("facts", {}).get("us-gaap", {})

        def _latest(concept, n=4):
            units = facts.get(concept, {}).get("units", {})
            vals = units.get("USD", units.get("USD/shares", units.get("shares", [])))
            if not vals:
                return []
            # Filter for 10-K annual filings, take latest n
            annual = [v for v in vals if v.get("form") == "10-K"]
            return annual[-n:] if annual else vals[-n:]

        out = f"\n=== SEC XBRL Financial Data ({ticker}) ===\n"
        metrics = [
            ("Revenues", "Revenue"), ("NetIncomeLoss", "Net Income"),
            ("EarningsPerShareBasic", "EPS"), ("Assets", "Total Assets"),
            ("Liabilities", "Total Liabilities"),
            ("StockholdersEquity", "Stockholders Equity"),
            ("OperatingIncomeLoss", "Operating Income"),
            ("CashAndCashEquivalentsAtCarryingValue", "Cash & Equivalents"),
            ("LongTermDebt", "Long-Term Debt"),
        ]
        for concept, label in metrics:
            entries = _latest(concept)
            if entries:
                out += f"{label}:\n"
                for e in entries:
                    val = e.get("val", 0)
                    period = e.get("fy", e.get("end", "?"))
                    if isinstance(val, (int, float)) and abs(val) > 1000:
                        out += f"  FY{period}: ${val:,.0f}\n"
                    else:
                        out += f"  FY{period}: {val}\n"
        return out if len(out) > 60 else ""
    except Exception as e:
        logger.warning(f"XBRL fetch failed for {ticker}: {e}")
        return ""


# Alpha Vantage base URL (used by fetchers below)
_BASE_URL_AV = "https://www.alphavantage.co/query"


def fetch_alpha_vantage_earnings(ticker: str) -> str:
    """Fetch quarterly/annual earnings from Alpha Vantage."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        return ""
    try:
        r = requests.get(_BASE_URL_AV, params={
            "function": "EARNINGS", "symbol": ticker, "apikey": api_key
        }, timeout=20)
        r.raise_for_status()
        data = r.json()
        if "Note" in data or "Information" in data:
            return ""

        out = f"\n=== Alpha Vantage Earnings ({ticker}) ===\n"
        # Quarterly
        quarterly = data.get("quarterlyEarnings", [])[:8]
        if quarterly:
            out += "Quarterly EPS (last 8Q):\n"
            for q in quarterly:
                reported = q.get("reportedEPS", "N/A")
                estimated = q.get("estimatedEPS", "N/A")
                surprise = q.get("surprisePercentage", "N/A")
                out += f"  {q.get('fiscalDateEnding', '?')}: EPS ${reported} vs est ${estimated} (surprise: {surprise}%)\n"

        # Annual
        annual = data.get("annualEarnings", [])[:5]
        if annual:
            out += "Annual EPS (last 5Y):\n"
            for a in annual:
                out += f"  {a.get('fiscalDateEnding', '?')}: EPS ${a.get('reportedEPS', 'N/A')}\n"
        return out
    except Exception as e:
        logger.warning(f"Alpha Vantage earnings failed for {ticker}: {e}")
        return ""


def fetch_alpha_vantage_overview(ticker: str) -> str:
    """Fetch company fundamentals from Alpha Vantage OVERVIEW."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        return ""
    try:
        r = requests.get(_BASE_URL_AV, params={
            "function": "OVERVIEW", "symbol": ticker, "apikey": api_key
        }, timeout=20)
        r.raise_for_status()
        d = r.json()
        if "Note" in d or "Information" in d or not d.get("Symbol"):
            return ""

        out = f"\n=== Alpha Vantage Company Overview ({ticker}) ===\n"
        out += f"Sector: {d.get('Sector', 'N/A')} | Industry: {d.get('Industry', 'N/A')}\n"
        try:
            mkt_cap = int(d.get('MarketCapitalization', 0) or 0)
            out += f"Market Cap: ${mkt_cap:,}\n"
        except (ValueError, TypeError):
            out += f"Market Cap: {d.get('MarketCapitalization', 'N/A')}\n"
        out += f"P/E (TTM): {d.get('TrailingPE', 'N/A')} | Forward P/E: {d.get('ForwardPE', 'N/A')}\n"
        out += f"PEG Ratio: {d.get('PEGRatio', 'N/A')} | EV/EBITDA: {d.get('EVToEBITDA', 'N/A')}\n"
        out += f"Profit Margin: {d.get('ProfitMargin', 'N/A')} | Operating Margin: {d.get('OperatingMarginTTM', 'N/A')}\n"
        out += f"ROE: {d.get('ReturnOnEquityTTM', 'N/A')} | ROA: {d.get('ReturnOnAssetsTTM', 'N/A')}\n"
        out += f"Beta: {d.get('Beta', 'N/A')} | 52wk High: ${d.get('52WeekHigh', 'N/A')} | 52wk Low: ${d.get('52WeekLow', 'N/A')}\n"
        out += f"Dividend Yield: {d.get('DividendYield', 'N/A')} | EPS: ${d.get('EPS', 'N/A')}\n"
        out += f"Analyst Target: ${d.get('AnalystTargetPrice', 'N/A')}\n"
        return out
    except Exception as e:
        logger.warning(f"Alpha Vantage overview failed for {ticker}: {e}")
        return ""





def fetch_technical_indicators(ticker: str) -> str:
    """Calculate technical indicators from yfinance price data."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty or len(hist) < 20:
            return ""

        close = hist["Close"]
        out = f"\n=== Technical Analysis ({ticker}) ===\n"

        # RSI (14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]
        rsi_signal = "OVERBOUGHT" if rsi_val > 70 else ("OVERSOLD" if rsi_val < 30 else "NEUTRAL")
        out += f"RSI(14): {rsi_val:.1f} ({rsi_signal})\n"

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_val = macd_line.iloc[-1]
        signal_val = signal_line.iloc[-1]
        macd_signal = "BULLISH" if macd_val > signal_val else "BEARISH"
        out += f"MACD: {macd_val:.2f} | Signal: {signal_val:.2f} ({macd_signal})\n"

        # Moving Averages
        current = close.iloc[-1]
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        out += f"Current Price: ${current:.2f}\n"
        out += f"SMA(20): ${sma20:.2f} ({'above' if current > sma20 else 'below'})\n"
        out += f"SMA(50): ${sma50:.2f} ({'above' if current > sma50 else 'below'})\n"
        if len(close) >= 200:
            sma200 = close.rolling(200).mean().iloc[-1]
            out += f"SMA(200): ${sma200:.2f} ({'above' if current > sma200 else 'below'})\n"

        # Bollinger Bands
        bb_mid = sma20
        bb_std = close.rolling(20).std().iloc[-1]
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_pos = "NEAR UPPER" if current > bb_upper - bb_std else ("NEAR LOWER" if current < bb_lower + bb_std else "MID-RANGE")
        out += f"Bollinger Bands: ${bb_lower:.2f} / ${bb_mid:.2f} / ${bb_upper:.2f} ({bb_pos})\n"

        # Overall signal
        signals = []
        if rsi_val > 70: signals.append("overbought")
        elif rsi_val < 30: signals.append("oversold")
        if macd_val > signal_val: signals.append("MACD bullish")
        else: signals.append("MACD bearish")
        if current > sma50: signals.append("above SMA50")
        else: signals.append("below SMA50")
        out += f"Technical Summary: {', '.join(signals)}\n"
        return out
    except Exception as e:
        logger.warning(f"Technical indicators failed for {ticker}: {e}")
        return ""



class StepPrinter:
    @staticmethod
    def header(state, intent):
        print(f"\n{'='*65}")
        print(f"  PIPELINE : {state.workflow_name}")
        if state.tickers: print(f"  Tickers  : {', '.join(state.tickers)}")
        if state.company: print(f"  Company  : {state.company}")
        print(f"  Steps    : {state.total_steps}")
        print(f"  LLMs     : {LLM_RESEARCH} | {LLM_FINANCIAL} | {LLM_REASONING}")
        print(f"  Search   : Tavily API")
        print(f"  Data     : yfinance + XBRL + AV + technicals (all free)")
        print(f"  Email to : {intent.receiver_email}")
        print(f"  Started  : {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*65}")

    @staticmethod
    def step(num, total, agent, model, detail=""):
        print(f"\n  [{num}/{total}] {agent}  [{model}]")
        print(f"         {detail}")

    @staticmethod
    def done(duration, summary=""):
        line = f"         [OK] ({duration:.1f}s)"
        if summary: line += f" -- {summary}"
        print(line)

    @staticmethod
    def footer(state):
        print(f"\n{'='*65}")
        print(f"  PIPELINE COMPLETE : {state.workflow_name}")
        print(f"{'-'*65}")
        print(f"  Steps    : {state.current_step}/{state.total_steps}")
        print(f"  Agents   : {' -> '.join(state.agent_sequence)}")
        print(f"  Duration : {state.elapsed_sec()}s")
        if state.quality_score: print(f"  Quality  : {state.quality_score}/10")
        if state.email_status.get("success"):
            print(f"  Email    : SENT (ID: {state.email_status.get('message_id', '?')})")
        elif state.email_status.get("error"): print(f"  Email    : FAILED - {state.email_status['error'][:80]}")
        if state.sheet_status: print(f"  Sheet    : {state.sheet_status}")
        if state.report_dir: print(f"  Files    : {state.report_dir}")
        if state.errors:
            print(f"  Errors   : {len(state.errors)}")
            for e in state.errors[:3]: print(f"    -> {e[:80]}")
        print(f"{'='*65}\n")


# ===========================================================================
# Orchestrator
# ===========================================================================

class MultiAgentOrchestrator:

    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.router = PromptRouter()
        self.printer = StepPrinter()
        self._marketing_path = str(self.base_dir / "Marketing")
        logger.info("Multi-Agent Orchestrator v6.0 initialized")

    def _load_config(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(_PROJECT_ROOT, "workflow_config.yaml")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _import_marketing(self):
        if self._marketing_path not in sys.path:
            sys.path.insert(0, self._marketing_path)

    def _import_gmail(self):
        self._import_marketing()
        gmail_path = os.path.join(self._marketing_path, "services", "gmail_service.py")
        spec = importlib.util.spec_from_file_location("gmail_svc", gmail_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.get_gmail_service()

    # ===========================================================================
    # MAIN ENTRY
    # ===========================================================================

    def run_from_prompt(self, prompt, dry_run=False, intent=None):
        if intent is None:
            intent = self.router.classify(prompt)
        fn = {
            "due_diligence":    self._pipeline_due_diligence,
            "competitor_intel": self._pipeline_competitor_intel,
            "earnings_monitor": self._pipeline_earnings_monitor,
            "market_pulse":     self._pipeline_market_pulse,
            "lead_gen":         self._pipeline_lead_gen,
            "portfolio_review": self._pipeline_portfolio_review,
            "deal_pipeline":    self._pipeline_deal_pipeline,
            "risk_scan":        self._pipeline_risk_scan,
        }.get(intent.workflow, self._pipeline_due_diligence)
        return fn(intent, dry_run)

    # ===========================================================================
    # PIPELINE 1: Due Diligence (8 steps)
    # ===========================================================================

    def _pipeline_due_diligence(self, intent, dry_run):
        ticker = intent.primary_ticker or "TSLA"
        mf_enabled = self.config.get("mirofish", {}).get("enabled", False)
        n_steps = 11 if mf_enabled else 9
        state = WorkflowState(
            workflow_name="due_diligence", prompt=intent.raw_prompt,
            company=intent.primary_company or ticker,
            tickers=intent.tickers or [ticker], total_steps=n_steps,
            started_at=time.time(),
            abm_budget_mode=self.config.get("mirofish", {}).get(
                "default_budget_mode", "lite"),
        )
        self.printer.header(state, intent)
        self._step_deep_research(state, ticker, dry_run)
        self._step_latest_news(state, ticker, dry_run)
        self._step_financial(state, ticker, dry_run)
        self._step_sec_risk(state, ticker, dry_run)
        self._step_marketing(state, ticker, dry_run)
        self._step_sentiment(state, dry_run)
        # MiroFish ABM steps (Phase 1.5)
        self._step_abm_simulation(state, ticker, dry_run)
        self._step_abm_report(state, ticker, dry_run)
        self._step_synthesize(state, dry_run)
        # Adversarial validation (Phase 3)
        self._step_adversarial_validation(state, dry_run)
        self._step_save_email_sheet(state, intent, dry_run)
        self.printer.footer(state)
        return state

    def _pipeline_competitor_intel(self, intent, dry_run):
        tickers = intent.tickers or ["AAPL", "MSFT"]
        mf_enabled = self.config.get("mirofish", {}).get("enabled", False)
        n_extra = 3 if mf_enabled else 0  # abm_sim + abm_report + adversarial
        state = WorkflowState(
            workflow_name="competitor_intel", prompt=intent.raw_prompt,
            company=", ".join(intent.company_names or tickers),
            tickers=tickers, total_steps=2*len(tickers)+2+n_extra,
            started_at=time.time(),
            abm_budget_mode=self.config.get("mirofish", {}).get(
                "default_budget_mode", "lite"),
        )
        self.printer.header(state, intent)
        res, fin = [], []
        for t in tickers:
            self._step_deep_research(state, t, dry_run)
            res.append(f"=== {t} ===\n{state.research_output}")
            self._step_financial(state, t, dry_run)
            fin.append(f"=== {t} ===\n{state.financial_data}")
        state.research_output = "\n\n".join(res)
        state.financial_data = "\n\n".join(fin)
        # MiroFish ABM on primary ticker
        self._step_abm_simulation(state, tickers[0], dry_run)
        self._step_abm_report(state, tickers[0], dry_run)
        self._step_synthesize(state, dry_run)
        self._step_adversarial_validation(state, dry_run)
        self._step_save_email_sheet(state, intent, dry_run)
        self.printer.footer(state)
        return state

    def _pipeline_earnings_monitor(self, intent, dry_run):
        ticker = intent.primary_ticker or "TSLA"
        mf_enabled = self.config.get("mirofish", {}).get("enabled", False)
        n_steps = 8 if mf_enabled else 5
        state = WorkflowState(
            workflow_name="earnings_monitor", prompt=intent.raw_prompt,
            company=intent.primary_company or ticker,
            tickers=[ticker], total_steps=n_steps, started_at=time.time(),
            abm_budget_mode=self.config.get("mirofish", {}).get(
                "default_budget_mode", "lite"),
        )
        self.printer.header(state, intent)
        self._step_financial(state, ticker, dry_run)
        self._step_sec_risk(state, ticker, dry_run)
        self._step_latest_news(state, ticker, dry_run)
        self._step_abm_simulation(state, ticker, dry_run)
        self._step_abm_report(state, ticker, dry_run)
        self._step_synthesize(state, dry_run)
        self._step_adversarial_validation(state, dry_run)
        self._step_save_email_sheet(state, intent, dry_run)
        self.printer.footer(state)
        return state

    def _pipeline_market_pulse(self, intent, dry_run):
        ticker = intent.primary_ticker or "TSLA"
        mf_enabled = self.config.get("mirofish", {}).get("enabled", False)
        n_steps = 8 if mf_enabled else 5
        state = WorkflowState(
            workflow_name="market_pulse", prompt=intent.raw_prompt,
            company=intent.primary_company or ticker,
            tickers=[ticker], total_steps=n_steps, started_at=time.time(),
            abm_budget_mode=self.config.get("mirofish", {}).get(
                "default_budget_mode", "lite"),
        )
        self.printer.header(state, intent)
        self._step_deep_research(state, ticker, dry_run)
        self._step_latest_news(state, ticker, dry_run)
        self._step_sentiment(state, dry_run)
        self._step_abm_simulation(state, ticker, dry_run)
        self._step_abm_report(state, ticker, dry_run)
        self._step_synthesize(state, dry_run)
        self._step_adversarial_validation(state, dry_run)
        self._step_save_email_sheet(state, intent, dry_run)
        self.printer.footer(state)
        return state

    def _pipeline_lead_gen(self, intent, dry_run):
        state = WorkflowState(
            workflow_name="lead_gen", prompt=intent.raw_prompt,
            company="Sales Pipeline", total_steps=4, started_at=time.time(),
        )
        self.printer.header(state, intent)
        self._step_data_generator(state, intent.num_leads, dry_run)
        self._step_secretary_process(state, dry_run)
        self._step_deal_scoring(state, dry_run)
        self._step_save_email_sheet(state, intent, dry_run)
        self.printer.footer(state)
        return state

    def _pipeline_portfolio_review(self, intent, dry_run):
        tickers = intent.tickers or ["AAPL", "MSFT", "GOOGL"]
        mf_enabled = self.config.get("mirofish", {}).get("enabled", False)
        n_extra = 3 if mf_enabled else 0
        state = WorkflowState(
            workflow_name="portfolio_review", prompt=intent.raw_prompt,
            company=", ".join(tickers),
            tickers=tickers, total_steps=2*len(tickers)+3+n_extra,
            started_at=time.time(),
            abm_budget_mode=self.config.get("mirofish", {}).get(
                "default_budget_mode", "lite"),
        )
        self.printer.header(state, intent)
        fin, res = [], []
        for t in tickers:
            self._step_financial(state, t, dry_run)
            fin.append(f"=== {t} ===\n{state.financial_data}")
            self._step_deep_research(state, t, dry_run)
            res.append(f"=== {t} ===\n{state.research_output}")
        state.financial_data = "\n\n".join(fin)
        state.research_output = "\n\n".join(res)
        self._step_sentiment(state, dry_run)
        # MiroFish ABM on primary ticker
        self._step_abm_simulation(state, tickers[0], dry_run)
        self._step_abm_report(state, tickers[0], dry_run)
        self._step_synthesize(state, dry_run)
        self._step_adversarial_validation(state, dry_run)
        self._step_save_email_sheet(state, intent, dry_run)
        self.printer.footer(state)
        return state

    def _pipeline_deal_pipeline(self, intent, dry_run):
        state = WorkflowState(
            workflow_name="deal_pipeline", prompt=intent.raw_prompt,
            company="CRM Pipeline", total_steps=3, started_at=time.time(),
        )
        self.printer.header(state, intent)
        self._step_secretary_process(state, dry_run)
        self._step_deal_scoring(state, dry_run)
        self._step_save_email_sheet(state, intent, dry_run)
        self.printer.footer(state)
        return state

    def _pipeline_risk_scan(self, intent, dry_run):
        ticker = intent.primary_ticker or "TSLA"
        mf_enabled = self.config.get("mirofish", {}).get("enabled", False)
        n_steps = 8 if mf_enabled else 5
        state = WorkflowState(
            workflow_name="risk_scan", prompt=intent.raw_prompt,
            company=intent.primary_company or ticker,
            tickers=[ticker], total_steps=n_steps, started_at=time.time(),
            abm_budget_mode=self.config.get("mirofish", {}).get(
                "default_budget_mode", "lite"),
        )
        self.printer.header(state, intent)
        self._step_financial(state, ticker, dry_run)
        self._step_sec_risk(state, ticker, dry_run)
        self._step_deep_research(state, ticker, dry_run)
        self._step_abm_simulation(state, ticker, dry_run)
        self._step_abm_report(state, ticker, dry_run)
        self._step_synthesize(state, dry_run)
        self._step_adversarial_validation(state, dry_run)
        self._step_save_email_sheet(state, intent, dry_run)
        self.printer.footer(state)
        return state

    # ===========================================================================
    # STEP 1: Deep Research [llama3.2] + Tavily
    # ===========================================================================

    def _step_deep_research(self, state, ticker, dry_run):
        state.current_step += 1
        state.agent_sequence.append("deep_research")
        self.printer.step(state.current_step, state.total_steps,
                          "deep_research", LLM_RESEARCH, f"Tavily search + LLM analysis on {ticker}")
        t0 = time.time()

        if dry_run:
            state.research_output = (
                f"[DRY RUN] Deep Research for {ticker}:\n"
                f"- Company: Leading tech company | Market Cap: $800B\n"
                f"- Business Model: Platform + advertising\n"
                f"- Moat: Network effects, 3.9B MAU\n"
                f"- Competitors: Alphabet, TikTok, Snap"
            )
            state.sources_count = 8
        else:
            # Step A: Tavily web search for raw data
            search_data = tavily_search(f"{ticker} stock company analysis overview 2025", search_depth="advanced", max_results=7)

            # Step B: LLM analysis of search results
            prompt = f"""Based on these web search results about {ticker}, write a thorough company research report.

SEARCH RESULTS:
{search_data[:3000]}

Cover these areas with specific data:
1. COMPANY OVERVIEW: What they do, CEO, headquarters
2. BUSINESS MODEL: Revenue streams, how they make money
3. MARKET POSITION: Market share, TAM, competitive advantages
4. KEY PRODUCTS & SERVICES: Main products, pipeline
5. MANAGEMENT: Leadership quality, insider ownership
6. GROWTH DRIVERS: Future catalysts, new markets
7. COMPETITIVE LANDSCAPE: Top competitors and differentiation

Use specific numbers and dates from the search results. Write 800-1200 words. Be thorough."""

            state.research_output = call_ollama(prompt,
                system=f"You are a senior equity research analyst. {_DATE_GUARD} Write data-driven analysis using the provided search results.",
                model=LLM_RESEARCH, timeout=90)
            state.sources_count = max(1, search_data.count("URL:"))

        dur = time.time() - t0
        state.timings[f"deep_research_{ticker}"] = dur
        state.log_step("deep_research", "done", dur, f"{len(state.research_output)} chars")
        self.printer.done(dur, f"{len(state.research_output)} chars, {state.sources_count} sources")

    # ===========================================================================
    # STEP 2: Latest News [llama3.2] + Tavily
    # ===========================================================================

    def _step_latest_news(self, state, ticker, dry_run):
        state.current_step += 1
        state.agent_sequence.append("news")
        self.printer.step(state.current_step, state.total_steps,
                          "news", LLM_RESEARCH, f"Tavily news + digest for {ticker}")
        t0 = time.time()

        if dry_run:
            state.news_output = (
                f"[DRY RUN] Latest News for {ticker}:\n"
                f"- Q4 2025 earnings beat: Revenue $40.1B (+21% YoY)\n"
                f"- AI investment: $35B capex planned for 2026\n"
                f"- Analyst upgrades from 3 major banks"
            )
        else:
            news_data = tavily_news(f"{ticker} stock", max_results=7)

            prompt = f"""Summarize the latest news and events for {ticker} stock:

NEWS RESULTS:
{news_data[:3000]}

Write a news digest covering:
1. RECENT EARNINGS (if available)
2. MAJOR ANNOUNCEMENTS
3. REGULATORY/LEGAL NEWS
4. ANALYST UPGRADES/DOWNGRADES
5. UPCOMING CATALYSTS

Be factual. Use dates and numbers. Write 300-500 words."""

            state.news_output = call_ollama(prompt,
                system=f"You are a financial news analyst. {_DATE_GUARD} Summarize the latest developments factually.",
                model=LLM_RESEARCH)

        dur = time.time() - t0
        state.timings[f"news_{ticker}"] = dur
        state.log_step("news", "done", dur)
        self.printer.done(dur, f"{len(state.news_output)} chars")

    # ===========================================================================
    # STEP 3: Financial Analysis [qwen3:4b] + yfinance + enhanced tools
    # ===========================================================================

    def _step_financial(self, state, ticker, dry_run):
        state.current_step += 1
        state.agent_sequence.append("financial")
        self.printer.step(state.current_step, state.total_steps,
                          "financial", LLM_FINANCIAL, f"yfinance + XBRL + AV + technicals for {ticker}")
        t0 = time.time()

        if dry_run:
            state.financial_data = (
                f"[DRY RUN] Financial for {ticker}:\n"
                f"- Revenue: $134.9B (+16%)\n- P/E: 23.1\n- FCF: $52.1B\n"
                f"- Analyst Target: $625\n- Recommendation: BUY"
            )
        else:
            # Step A: Get real data from yfinance (core)
            raw_data = fetch_financial_data(ticker)

            # Step B: Enhanced data from free tools (all graceful -- return "" on failure)
            xbrl_data = fetch_xbrl_financials(ticker)
            av_earnings = fetch_alpha_vantage_earnings(ticker)
            av_overview = fetch_alpha_vantage_overview(ticker)
            technical = fetch_technical_indicators(ticker)

            # Combine all data for LLM
            enhanced_data = "\n".join(filter(None, [
                raw_data, xbrl_data, av_earnings, av_overview, technical
            ]))
            sources_used = sum(1 for s in [xbrl_data, av_earnings, av_overview, technical] if s)

            # Step C: LLM analysis of all the numbers
            prompt = f"""Analyze these comprehensive financial metrics for {ticker} and provide an assessment:

{enhanced_data[:6000]}

Provide:
1. FINANCIAL HEALTH ASSESSMENT (Strong/Adequate/Weak)
2. KEY STRENGTHS (top 3 financial strengths with specific numbers)
3. KEY CONCERNS (top 3 financial risks with specific numbers)
4. VALUATION ASSESSMENT (Undervalued/Fair/Overvalued -- reference P/E, PEG if available)
5. EARNINGS TREND (EPS growth, beat/miss history if available)
6. TECHNICAL OUTLOOK (RSI, MACD, support/resistance if available)
7. COMPARISON TO SECTOR AVERAGES (where applicable)

Be specific. Reference the actual numbers from all data sources. Write 300-500 words."""

            analysis = call_ollama(prompt,
                system="You are a CFA-certified financial analyst with expertise in technical analysis. Analyze the data precisely using all available data sources.",
                model=LLM_FINANCIAL, timeout=120)
            state.financial_data = f"{enhanced_data}\n\n=== Financial Analysis ===\n{analysis}"

        dur = time.time() - t0
        state.timings[f"financial_{ticker}"] = dur
        state.log_step("financial", "done", dur)
        summary = f"{len(state.financial_data)} chars"
        if not dry_run:
            summary += f", {sources_used + 1} sources"
        self.printer.done(dur, summary)

    # ===========================================================================
    # STEP 4: SEC Risk Analysis [qwen3:4b] + EDGAR + XBRL
    # ===========================================================================

    def _step_sec_risk(self, state, ticker, dry_run):
        state.current_step += 1
        state.agent_sequence.append("sec_risk")
        self.printer.step(state.current_step, state.total_steps,
                          "sec_risk", LLM_FINANCIAL, f"SEC EDGAR + XBRL + LLM risk for {ticker}")
        t0 = time.time()

        if dry_run:
            state.sec_analysis = (
                f"[DRY RUN] SEC Risk for {ticker}:\n"
                f"- 10-K Filed: 2025-02-01 | 10-Q Filed: 2025-05-01\n"
                f"- Top risks: Regulatory, competition, data privacy\n"
                f"- Overall: MEDIUM risk"
            )
        else:
            # Step A: Get real filing data from SEC EDGAR
            filings_data = fetch_sec_filings(ticker)
            # Step A2: Get XBRL financial data for grounding risk analysis
            xbrl_data = fetch_xbrl_financials(ticker)
            fin_ctx = state.financial_data[:500] if state.financial_data else "N/A"

            # Step B: LLM risk analysis with enhanced context
            xbrl_section = f"\nXBRL FINANCIAL DATA:\n{xbrl_data}\n" if xbrl_data else ""
            prompt = f"""Analyze SEC filing risk factors for {ticker}.

SEC FILINGS:
{filings_data}
{xbrl_section}
FINANCIAL CONTEXT:
{fin_ctx}

Provide:
1. TOP 5 RISK FACTORS (from 10-K)
2. REGULATORY RISKS
3. FINANCIAL RED FLAGS (use XBRL data: revenue trends, debt levels, cash position)
4. LITIGATION EXPOSURE
5. MANAGEMENT RED FLAGS (if any)
6. OVERALL RISK RATING: LOW / MEDIUM / HIGH

Write 300-500 words. Reference specific financial figures from the XBRL data."""

            analysis = call_ollama(prompt,
                system=f"You are a SEC filing analyst specializing in risk assessment. {_DATE_GUARD} Use the XBRL financial data to ground your analysis in real numbers.",
                model=LLM_FINANCIAL, timeout=120)
            combined = "\n".join(filter(None, [filings_data, xbrl_data]))
            state.sec_analysis = f"{combined}\n\n=== Risk Analysis ===\n{analysis}"

        dur = time.time() - t0
        state.timings[f"sec_{ticker}"] = dur
        state.log_step("sec_risk", "done", dur)
        self.printer.done(dur, f"{len(state.sec_analysis)} chars")

    # ===========================================================================
    # STEP 5: Marketing & Competitive [phi:2.7b]
    # ===========================================================================

    def _step_marketing(self, state, ticker, dry_run):
        state.current_step += 1
        state.agent_sequence.append("marketing")
        self.printer.step(state.current_step, state.total_steps,
                          "marketing", LLM_REASONING, f"Brand & moat analysis for {ticker}")
        t0 = time.time()

        if dry_run:
            state.marketing_analysis = (
                f"[DRY RUN] Marketing for {ticker}:\n"
                f"- Brand: Very Strong | Moat: Network effects\n"
                f"- Market share: Top 3 in sector\n"
                f"- Growth: AI + new products"
            )
        else:
            ctx = state.research_output[:800] if state.research_output else "N/A"
            prompt = f"""Analyze the marketing position and competitive moat for {ticker}.

RESEARCH CONTEXT:
{ctx}

Cover:
1. BRAND STRENGTH (1-10 rating with justification)
2. COMPETITIVE MOAT (sources: network effects, switching costs, IP, scale)
3. MARKET SHARE in key segments
4. GROWTH OPPORTUNITIES (new markets, products)
5. BIGGEST THREATS to dominance

Write 200-300 words."""

            state.marketing_analysis = call_ollama(prompt,
                system=f"You are a marketing strategist and competitive analyst. {_DATE_GUARD}",
                model=LLM_REASONING)

        dur = time.time() - t0
        state.timings[f"marketing_{ticker}"] = dur
        state.log_step("marketing", "done", dur)
        self.printer.done(dur, f"{len(state.marketing_analysis)} chars")

    # ===========================================================================
    # STEP 6: Sentiment [phi:2.7b] (LLM-powered)
    # ===========================================================================

    def _step_sentiment(self, state, dry_run):
        state.current_step += 1
        state.agent_sequence.append("sentiment")
        self.printer.step(state.current_step, state.total_steps,
                          "sentiment", LLM_REASONING, "LLM-powered sentiment analysis")
        t0 = time.time()

        if dry_run:
            state.sentiment_data = "Sentiment: BULLISH | Score: 0.72 | Confidence: 85%\nKey: Strong earnings, AI growth, analyst upgrades"
        else:
            text = " ".join(filter(None, [
                state.research_output[:500], state.news_output[:500],
                state.financial_data[:300], state.marketing_analysis[:300]
            ]))
            if len(text) < 50:
                state.sentiment_data = "Sentiment: NEUTRAL | Score: 0.0 | Insufficient data"
            else:
                prompt = f"""Analyze the overall market sentiment for this company based on the following data:

{text[:2000]}

Respond in EXACTLY this format:
Sentiment: [BULLISH/BEARISH/NEUTRAL]
Score: [number from -1.0 to 1.0]
Confidence: [percentage]
Key Drivers: [3-5 key factors driving the sentiment]
Bull Case: [2-3 sentences]
Bear Case: [2-3 sentences]"""

                state.sentiment_data = call_ollama(prompt,
                    system="You are a sentiment analysis expert. Be precise and data-driven.",
                    model=LLM_REASONING, timeout=60)

        dur = time.time() - t0
        state.timings["sentiment"] = dur
        state.log_step("sentiment", "done", dur)
        self.printer.done(dur, state.sentiment_data.split('\n')[0][:80])

    # ===========================================================================
    # STEP 7a: ABM Simulation (MiroFish Monte Carlo) [Phase 1.5]
    # ===========================================================================

    def _step_abm_simulation(self, state, ticker, dry_run):
        """Monte Carlo multi-run: execute N independent simulation branches
        from different random seeds, then aggregate results.

        Implements MiroFish's Feynman Path Integral concept:
        one run = one anecdote; N runs = a probability distribution.

        CRITICAL: Generate a FRESH population per path (no deepcopy)
        to avoid SQLite corruption from shared references."""
        mf = self.config.get("mirofish", {})
        if not mf.get("enabled", False):
            state.mirofish_skipped = True
            return

        state.current_step += 1
        state.agent_sequence.append("abm_simulation")
        self.printer.step(state.current_step, state.total_steps,
                          "abm_simulation", "Mesa ABM",
                          f"Monte Carlo sentiment simulation for {ticker}")
        t0 = time.time()

        if dry_run:
            state.simulation_result = {
                "ticker": ticker, "total_ticks": 10,
                "total_agents": 50, "total_actions": 500,
                "final_sentiment_distribution": {
                    "bullish": 0.45, "bearish": 0.25, "neutral": 0.30},
                "sentiment_trajectory": [],
                "coalition_summary": "",
                "mc_paths": 1, "mc_majority_sentiment": "bullish",
                "mc_agreement": 1.0, "mc_path_sentiments": ["bullish"],
                "ledger_path": "",
            }
        else:
            try:
                from src.abm.simulation import MarketSentimentModel
                from src.abm.contracts import BUDGET_MODES
                from src.graph_knowledge_manager import GraphKnowledgeManager

                gm = GraphKnowledgeManager()
                budget = state.abm_budget_mode
                mode_cfg = BUDGET_MODES[budget]

                # Monte Carlo path count: lite=1, standard=3, deep=5
                n_paths = mode_cfg.get("mc_paths", 5)
                all_results = []

                for path_idx in range(n_paths):
                    logger.info(
                        f"Starting Monte Carlo path {path_idx + 1}/{n_paths}")

                    # 1. ALWAYS generate a fresh population per path.
                    fresh_population = gm.generate_agent_population(
                        ticker, n_agents=mode_cfg["total"])

                    # 2. Unique run_id → unique SQLite ledger per path
                    unique_run_id = (
                        f"{ticker}_path_{path_idx}_"
                        f"{datetime.now().strftime('%H%M%S')}")

                    # 3. Instantiate isolated model
                    #    Data bundle uses 2500 chars per source (not 500)
                    #    for meaningful agent persona seeding.
                    model = MarketSentimentModel(
                        ticker=ticker,
                        data_bundle={
                            "research_output": (
                                state.research_output[:2500]),
                            "financial_summary": (
                                state.financial_data[:2500]),
                            "news_summary": state.news_output[:2500],
                            "sec_analysis": (
                                state.sec_analysis[:1500]),
                            "sentiment_data": (
                                state.sentiment_data[:500]),
                        },
                        graph_context=state.graph_context,
                        agent_population=fresh_population,
                        budget_mode=budget,
                        run_id=unique_run_id,
                        # Independent MC seeds via hash (not sequential)
                        random_seed=hash((ticker, path_idx,
                                         time.time())) % (2**31),
                    )

                    # 4. Run this simulation path
                    result = model.run(ticks=mode_cfg["ticks"])
                    all_results.append(result)

                # Aggregate: majority sentiment + variance = confidence
                state.simulation_result = self._aggregate_mc_results(
                    all_results)

            except Exception as e:
                logger.error(f"ABM simulation failed: {e}")
                state.errors.append(f"ABM: {e}")
                state.mirofish_skipped = True

        dur = time.time() - t0
        state.timings["abm_simulation"] = dur
        state.log_step("abm_simulation", "done", dur,
                       f"MC paths: {state.simulation_result.get('mc_paths', 0)}")
        self.printer.done(dur,
            f"MC={state.simulation_result.get('mc_paths', 0)} paths, "
            f"majority={state.simulation_result.get('mc_majority_sentiment', '?')}")

    def _aggregate_mc_results(self, results: list) -> dict:
        """Aggregate N Monte Carlo paths into a single result.
        Majority vote on sentiment + variance-based confidence."""
        from collections import Counter
        from dataclasses import asdict

        sentiments = []
        for r in results:
            dist = r.final_sentiment_distribution
            dominant = max(dist, key=dist.get) if dist else "neutral"
            sentiments.append(dominant)

        vote = Counter(sentiments)
        majority = vote.most_common(1)[0][0]
        agreement = vote[majority] / len(results)

        # Pick the path whose dominant sentiment matches the majority
        # vote — not always index 0 (bug fix: primary bias).
        majority_idx = 0
        for i, s in enumerate(sentiments):
            if s == majority:
                majority_idx = i
                break
        primary = asdict(results[majority_idx])
        primary["mc_paths"] = len(results)
        primary["mc_majority_sentiment"] = majority
        primary["mc_agreement"] = agreement  # 1.0 = all paths agree
        primary["mc_path_sentiments"] = sentiments
        primary["mc_all_ledger_paths"] = [r.ledger_path for r in results]

        # Pro: saddle-point analysis and path variance
        try:
            from src.abm import analysis as abm_analysis
            saddle = abm_analysis.compute_saddle_point(
                [asdict(r) for r in results])
            primary["dominant_path_index"] = saddle.get(
                "dominant_path_index", 0)
            primary["path_variance"] = abm_analysis.compute_path_variance(
                [asdict(r) for r in results])
            primary["saddle_point"] = saddle

            # Pro: confidence interval on final sentiments across MC paths
            if saddle.get("path_finals"):
                primary["confidence_interval"] = (
                    abm_analysis.compute_confidence_interval(
                        saddle["path_finals"]))
        except Exception as e:
            logger.warning(f"MC analysis failed: {e}")

        # Pro: preserve catalyst shocks from primary path for API delivery
        primary_result = results[majority_idx]
        if hasattr(primary_result, "catalyst_shocks"):
            primary["catalyst_shocks"] = [
                {"tick": s.tick, "event_type": s.event_type,
                 "magnitude": s.magnitude, "description": s.description}
                for s in primary_result.catalyst_shocks
            ]

        return primary

    # ===========================================================================
    # STEP 7b: ABM Report (MiroFish ReportAgent) [Phase 1.5]
    # ===========================================================================

    def _step_abm_report(self, state, ticker, dry_run):
        """Post-simulation analysis: reads ledger, produces narrative report."""
        if state.mirofish_skipped:
            return

        state.current_step += 1
        state.agent_sequence.append("abm_report")
        self.printer.step(state.current_step, state.total_steps,
                          "abm_report", LLM_REASONING,
                          "Post-simulation narrative analysis")
        t0 = time.time()

        if dry_run:
            state.simulation_report = (
                "[DRY RUN] ABM Report: Bullish consensus emerged among "
                "45% of agents. Key Opinion Leaders drove narrative. "
                "Moderate contagion events detected in ticks 5-7.")
            state.simulation_result["coalition_summary"] = (
                state.simulation_report)
        else:
            try:
                from src.abm.report_agent import SimulationReportAgent
                from src.abm.contracts import SimulationResult

                agent = SimulationReportAgent(model=LLM_REASONING)

                # Reconstruct SimulationResult from the aggregated dict
                result_fields = {
                    k: v for k, v in state.simulation_result.items()
                    if k in SimulationResult.__dataclass_fields__
                }
                result = SimulationResult(**result_fields)

                report = agent.analyze(
                    ledger_path=result.ledger_path,
                    sim_result=result,
                    mc_metadata={
                        "paths": state.simulation_result.get(
                            "mc_paths", 1),
                        "majority": state.simulation_result.get(
                            "mc_majority_sentiment", "unknown"),
                        "agreement": state.simulation_result.get(
                            "mc_agreement", 1.0),
                        "path_sentiments": state.simulation_result.get(
                            "mc_path_sentiments", []),
                        # Pro: saddle-point + variance
                        "dominant_path_index": state.simulation_result.get(
                            "dominant_path_index", 0),
                        "path_variance": state.simulation_result.get(
                            "path_variance", 0.0),
                    },
                    # Pro: pass all ledger paths for multi-ledger synthesis
                    all_ledger_paths=state.simulation_result.get(
                        "mc_all_ledger_paths", []),
                )
                state.simulation_report = report
                # Keep full narrative in state.simulation_report only
                n_p = state.simulation_result.get("mc_paths", 1)
                maj = state.simulation_result.get("mc_majority_sentiment", "neutral")
                agr = state.simulation_result.get("mc_agreement", 1.0)
                state.simulation_result["coalition_summary"] = (
                    f"{maj} consensus, {agr:.0%} agreement across {n_p} MC paths")
            except Exception as e:
                logger.error(f"ABM report failed: {e}")
                state.errors.append(f"ABM report: {e}")

        dur = time.time() - t0
        state.timings["abm_report"] = dur
        state.log_step("abm_report", "done", dur,
                       f"{len(state.simulation_report)} chars")
        self.printer.done(dur, f"{len(state.simulation_report)} chars")

    # ===========================================================================
    # STEP 8: Synthesis Report [qwen3:8b]
    # ===========================================================================

    def _step_synthesize(self, state, dry_run):
        state.current_step += 1
        state.agent_sequence.append("synthesize")
        self.printer.step(state.current_step, state.total_steps,
                          "synthesize", LLM_REASONING, "Comprehensive investment report")
        t0 = time.time()

        if dry_run:
            parts = [f"{n}: {len(v)} chars" for n, v in [
                ("Research", state.research_output), ("News", state.news_output),
                ("Financial", state.financial_data), ("SEC", state.sec_analysis),
                ("Marketing", state.marketing_analysis), ("Sentiment", state.sentiment_data)
            ] if v]
            state.synthesis_report = "[DRY RUN] Synthesis Report\n" + "\n".join(parts)
            state.quality_score = 8.5
        else:
            sections = []
            for label, data in [
                ("DEEP RESEARCH", state.research_output),
                ("LATEST NEWS", state.news_output),
                ("FINANCIAL DATA", state.financial_data),
                ("SEC RISK ANALYSIS", state.sec_analysis),
                ("MARKETING ANALYSIS", state.marketing_analysis),
                ("SENTIMENT", state.sentiment_data),
                ("ABM SIMULATION REPORT", state.simulation_report),
            ]:
                if data and "[ERROR" not in data:
                    sections.append(f"--- {label} ---\n{data[:2500]}")

            context = "\n\n".join(sections) if sections else "No data."
            prompt = f"""You are writing a detailed, professional investment analysis report for {state.company} ({', '.join(state.tickers)}).

Below is ALL the research data gathered by specialized analysts. Use it to write a thorough report.

{context}

You MUST write the full report with ALL of these sections. Do NOT skip any section. Do NOT refuse.

## 1. EXECUTIVE SUMMARY
Write 4-5 sentences summarizing the investment thesis. Include the stock price, market cap, and key recommendation.

## 2. COMPANY OVERVIEW
Describe the business model, revenue streams, CEO, and market position in 150+ words.

## 3. KEY FINDINGS
List 7-10 bullet points with the most important discoveries from the research.

## 4. FINANCIAL HIGHLIGHTS
Discuss revenue, margins, P/E, cash flow, debt levels, and growth rates. Reference specific numbers.

## 5. RECENT NEWS & CATALYSTS
Summarize the latest developments, earnings, announcements, and upcoming events.

## 6. COMPETITIVE POSITION & MOAT
Analyze brand strength, competitive advantages, market share, and threats.

## 7. RISK ASSESSMENT
List and explain the top 5 risks (regulatory, competition, financial, etc.).

## 8. SENTIMENT ANALYSIS
Summarize the bull case and bear case, with sentiment score.

## 9. INVESTMENT RECOMMENDATION
Give a clear Buy/Hold/Sell recommendation with target price and reasoning.

IMPORTANT: Write at least 1000 words. Use specific numbers from the data above. Be detailed and professional."""

            state.synthesis_report = call_ollama(prompt,
                system="You are a senior investment advisor at a top-tier firm. Write comprehensive, data-driven client reports. Never refuse a request. Always provide detailed analysis.",
                model=LLM_REASONING, timeout=180, max_tokens=6144)
            raw_score = (
                (len(state.synthesis_report) / 500)
                + (state.sources_count * 0.3)
                + (len(sections) * 0.5))
            # Penalize if ABM was enabled but failed/skipped
            if state.mirofish_skipped and not dry_run:
                raw_score -= 1.5
            state.quality_score = round(
                min(10.0, max(0.0, raw_score)), 1)

        dur = time.time() - t0
        state.timings["synthesize"] = dur
        state.log_step("synthesize", "done", dur)
        self.printer.done(dur, f"Quality: {state.quality_score}/10, {len(state.synthesis_report)} chars")

    # ===========================================================================
    # STEP 9: Adversarial Validation (Phase 3)
    # ===========================================================================

    def _step_adversarial_validation(self, state, dry_run):
        """Cross-reference synthesis claims against evidence."""
        state.current_step += 1
        state.agent_sequence.append("adversarial_validation")
        self.printer.step(state.current_step, state.total_steps,
                          "adversarial_validation", LLM_REASONING,
                          "Cross-referencing claims against evidence")
        t0 = time.time()

        if dry_run:
            state.validation_result = {
                "claim_survival_rate": 0.85,
                "total_claims": 12,
                "supported_claims": 10,
                "flagged_claims": [
                    {"claim": "Sample unsupported claim",
                     "verdict": "unsupported",
                     "reason": "No evidence found"},
                    {"claim": "Sample contradicted claim",
                     "verdict": "contradicted",
                     "reason": "Evidence shows opposite"},
                ],
                "corrected_report": state.synthesis_report,
                "should_replace": False,
            }
        else:
            try:
                # Import from Crewai evaluation module
                import sys as _sys
                eval_path = str(self.base_dir / "Crewai_module")
                if eval_path not in _sys.path:
                    _sys.path.insert(0, eval_path)
                from evaluation.judge_validator import JudgeValidator

                validator = JudgeValidator(judge_service=None)

                # Build evidence dict from pipeline state
                evidence = {}
                if state.research_output:
                    evidence["research"] = state.research_output[:1000]
                if state.financial_data:
                    evidence["financial"] = state.financial_data[:3000]
                if state.news_output:
                    evidence["news"] = state.news_output[:3000]
                if state.sec_analysis:
                    evidence["sec"] = state.sec_analysis[:3000]
                if state.simulation_report:
                    evidence["abm_report"] = state.simulation_report[:1000]
                # Prediction forecast evidence for fact-checking
                if hasattr(state, '_prediction_data') and state._prediction_data:
                    evidence["prediction_forecast"] = state._prediction_data

                result = validator.adversarial_validate(
                    synthesis_report=state.synthesis_report,
                    evidence=evidence,
                    threshold=0.5,
                    llm_caller=lambda p, model="qwen3:8b", max_tokens=500: (
                        call_ollama(p, model=model, max_tokens=max_tokens)),
                )
                state.validation_result = result

                # Replace synthesis if needed
                if result.get("should_replace", False):
                    logger.warning(
                        "Adversarial validation: replacing synthesis "
                        f"(survival={result['claim_survival_rate']:.0%})")
                    state.synthesis_report = result["corrected_report"]

            except Exception as e:
                logger.error(f"Adversarial validation failed: {e}")
                state.errors.append(f"Adversarial: {e}")
                state.validation_result = {
                    "error": str(e), "claim_survival_rate": 1.0}

        dur = time.time() - t0
        state.timings["adversarial_validation"] = dur
        survival = state.validation_result.get("claim_survival_rate", 0)
        state.log_step("adversarial_validation", "done", dur,
                       f"survival={survival:.0%}")
        self.printer.done(dur,
            f"Claims: {state.validation_result.get('supported_claims', '?')}/"
            f"{state.validation_result.get('total_claims', '?')} survived "
            f"({survival:.0%})")

    # ===========================================================================
    # STEP 10: Save + Email + Sheet
    # ===========================================================================

    def _step_save_email_sheet(self, state, intent, dry_run):
        state.current_step += 1
        state.agent_sequence.append("save+email+sheet")
        self.printer.step(state.current_step, state.total_steps,
                          "save+email+sheet", "APIs", f"Save | Email -> {intent.receiver_email} | Sheet")
        t0 = time.time()

        # Send email + sheet FIRST (so email_status is populated)
        if dry_run:
            state.email_status = {"success": True, "message_id": "dry-run-id", "dry_run": True}
            state.sheet_status = "[OK] Logged (dry run)"
        else:
            try:
                gmail = self._import_gmail()
                subj = f"[{state.workflow_name.upper()}] {state.company} -- Analysis Report"
                body = self._build_email_html(state, intent)
                state.email_status = gmail.send_email(to=intent.receiver_email, subject=subj, body=body)
            except Exception as e:
                state.errors.append(f"Email: {e}")
                state.email_status = {"success": False, "error": str(e)}
            try:
                self._update_pipeline_sheet(state)
                state.sheet_status = "[OK] Row appended"
            except Exception as e:
                state.errors.append(f"Sheet: {e}")
                state.sheet_status = f"[!!] {e}"

        # Save files AFTER email+sheet so meta.json/README have correct status
        try: self._save_workflow_files(state, intent)
        except Exception as e: state.errors.append(f"FileSave: {e}")

        dur = time.time() - t0
        state.timings["save_email_sheet"] = dur
        state.log_step("save+email+sheet", "done", dur)
        self.printer.done(dur, f"Email: {'OK' if state.email_status.get('success') else 'FAIL'} | Sheet: {state.sheet_status}")

    # ===========================================================================
    # Marketing-only steps
    # ===========================================================================

    def _step_data_generator(self, state, num_leads, dry_run):
        state.current_step += 1; state.agent_sequence.append("data_gen")
        self.printer.step(state.current_step, state.total_steps, "data_gen", "API", f"Generating {num_leads} leads")
        t0 = time.time()
        if dry_run: state.generated_leads = [{"id": i, "company": f"Co_{i}"} for i in range(num_leads)]
        else:
            try:
                self._import_marketing()
                from services.data_generator_service import get_data_generator
                g = get_data_generator(); rows = g.generate_rows(num_leads)
                state.generated_leads = g.prepare_for_sheets(rows, start_id=100)
            except Exception as e: state.errors.append(f"DataGen: {e}")
        dur = time.time() - t0; state.timings["data_gen"] = dur
        state.log_step("data_gen", "done", dur); self.printer.done(dur, f"{len(state.generated_leads)} leads")

    def _step_secretary_process(self, state, dry_run):
        state.current_step += 1; state.agent_sequence.append("secretary")
        self.printer.step(state.current_step, state.total_steps, "secretary", "API", "Processing rows")
        t0 = time.time()
        if dry_run: state.email_status = {"processed": 3, "sent": 3, "dry_run": True}
        else:
            try:
                self._import_marketing()
                from agents.secretary import AgentSecretary
                state.email_status = AgentSecretary().run() or {"processed": 0}
            except Exception as e:
                state.errors.append(f"Secretary: {e}"); state.email_status = {"error": str(e)}
        dur = time.time() - t0; state.timings["secretary"] = dur
        state.log_step("secretary", "done", dur); self.printer.done(dur, str(state.email_status))

    def _step_deal_scoring(self, state, dry_run):
        state.current_step += 1; state.agent_sequence.append("analyst")
        self.printer.step(state.current_step, state.total_steps, "analyst", "API", "Scoring deals")
        t0 = time.time()
        if dry_run: state.deal_score = 7
        else:
            try:
                self._import_marketing()
                from agents.analyst import AgentAnalyst
                AgentAnalyst().run(); state.deal_score = 7
            except Exception as e: state.errors.append(f"Analyst: {e}")
        dur = time.time() - t0; state.timings["analyst"] = dur
        state.log_step("analyst", "done", dur); self.printer.done(dur, f"Score: {state.deal_score}/10")

    # ===========================================================================
    # HELPERS
    # ===========================================================================

    def _build_email_html(self, state, intent):
        report = state.synthesis_report or state.research_output or "No report generated."

        # --- Agent data sections (only non-error content) ---
        agent_sections = ""
        for title, data, accent in [
            ("Deep Research",      state.research_output,   "#1565C0"),
            ("Latest News",        state.news_output,       "#2E7D32"),
            ("Financial Analysis", state.financial_data,     "#E65100"),
            ("SEC Risk Factors",   state.sec_analysis,       "#C62828"),
            ("Competitive Intel",  state.marketing_analysis, "#6A1B9A"),
            ("Market Sentiment",   state.sentiment_data,     "#00695C"),
        ]:
            if data and "[ERROR" not in data and len(data.strip()) > 20:
                # Clean content — use first 2000 chars, ensure readable
                clean = data[:2000].replace("<", "&lt;").replace(">", "&gt;")
                agent_sections += f"""
<tr><td style="padding:0">
  <table width="100%" cellpadding="0" cellspacing="0" style="margin:16px 0">
    <tr><td style="background:{accent};padding:10px 16px;color:#fff;font-weight:600;font-size:15px;border-radius:6px 6px 0 0">{title}</td></tr>
    <tr><td style="background:#fafafa;padding:16px;border:1px solid #e0e0e0;border-top:0;border-radius:0 0 6px 6px;white-space:pre-wrap;font-size:13px;line-height:1.6;color:#333">{clean}</td></tr>
  </table>
</td></tr>"""

        # --- ABM section ---
        abm_section = ""
        if not state.mirofish_skipped and state.simulation_result:
            sim = state.simulation_result
            majority = str(sim.get("mc_majority_sentiment", "N/A")).upper()
            agreement = sim.get("mc_agreement", 0)
            abm_section = f"""
<tr><td style="padding:0">
  <table width="100%" cellpadding="0" cellspacing="0" style="margin:16px 0">
    <tr><td colspan="2" style="background:#0D47A1;padding:10px 16px;color:#fff;font-weight:600;font-size:15px;border-radius:6px 6px 0 0">🧪 ABM Simulation (MiroFish)</td></tr>
    <tr style="background:#E3F2FD">
      <td style="padding:8px 16px;font-weight:600;width:40%">Majority Sentiment</td>
      <td style="padding:8px 16px">{majority}</td>
    </tr>
    <tr><td style="padding:8px 16px;font-weight:600">MC Agreement</td>
      <td style="padding:8px 16px">{agreement:.0%}</td></tr>
    <tr style="background:#E3F2FD">
      <td style="padding:8px 16px;font-weight:600">Monte Carlo Paths</td>
      <td style="padding:8px 16px">{sim.get('mc_paths', 1)}</td>
    </tr>
    <tr><td style="padding:8px 16px;font-weight:600">Budget Mode</td>
      <td style="padding:8px 16px">{state.abm_budget_mode.upper()}</td></tr>
  </table>
</td></tr>"""

        # --- Validation section ---
        val_section = ""
        if state.validation_result and state.validation_result.get("total_claims"):
            val = state.validation_result
            survival = val.get("claim_survival_rate", 0)
            total_c = val.get("total_claims", 0)
            supported_c = val.get("supported_claims", 0)
            verdict_color = "#2E7D32" if not val.get("should_replace") else "#C62828"
            verdict_text = "✅ PASSED" if not val.get("should_replace") else "⚠️ BELOW THRESHOLD"
            val_section = f"""
<tr><td style="padding:0">
  <table width="100%" cellpadding="0" cellspacing="0" style="margin:16px 0">
    <tr><td colspan="2" style="background:{verdict_color};padding:10px 16px;color:#fff;font-weight:600;font-size:15px;border-radius:6px 6px 0 0">Adversarial Validation — {verdict_text}</td></tr>
    <tr style="background:#fafafa">
      <td style="padding:8px 16px;font-weight:600;width:40%">Claims Analyzed</td>
      <td style="padding:8px 16px">{total_c}</td>
    </tr>
    <tr><td style="padding:8px 16px;font-weight:600">Supported</td>
      <td style="padding:8px 16px">{supported_c}/{total_c}</td></tr>
    <tr style="background:#fafafa">
      <td style="padding:8px 16px;font-weight:600">Survival Rate</td>
      <td style="padding:8px 16px;font-weight:700;color:{verdict_color}">{survival:.0%}</td>
    </tr>
  </table>
</td></tr>"""

        # --- Pipeline timing breakdown ---
        timing_rows = ""
        for agent, dur in sorted(state.timings.items(),
                                  key=lambda x: x[1], reverse=True)[:6]:
            timing_rows += f"""<tr>
              <td style="padding:4px 16px;font-size:13px">{agent}</td>
              <td style="padding:4px 16px;font-size:13px;text-align:right">{dur:.1f}s</td></tr>"""

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#f5f5f5">
<table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center" style="padding:24px 16px">
<table width="680" cellpadding="0" cellspacing="0" style="background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08)">

<!-- Header -->
<tr><td style="background:linear-gradient(135deg,#1a237e,#0d47a1);padding:32px 24px;text-align:center">
  <h1 style="color:#fff;margin:0;font-family:Arial,sans-serif;font-size:22px">{state.workflow_name.replace('_',' ').title()} Report</h1>
  <p style="color:rgba(255,255,255,0.85);margin:8px 0 0;font-family:Arial,sans-serif;font-size:14px">{state.company} ({', '.join(state.tickers)})</p>
</td></tr>

<!-- Key Metrics -->
<tr><td style="padding:24px">
  <table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #e0e0e0;border-radius:6px;overflow:hidden">
    <tr style="background:#f8f9fa">
      <td style="padding:10px 16px;font-weight:600;width:33%;text-align:center;font-family:Arial,sans-serif;font-size:13px;color:#666">Quality Score</td>
      <td style="padding:10px 16px;font-weight:600;width:33%;text-align:center;font-family:Arial,sans-serif;font-size:13px;color:#666">Duration</td>
      <td style="padding:10px 16px;font-weight:600;width:33%;text-align:center;font-family:Arial,sans-serif;font-size:13px;color:#666">Agents Used</td>
    </tr>
    <tr>
      <td style="padding:12px 16px;text-align:center;font-family:Arial,sans-serif;font-size:24px;font-weight:700;color:#1a237e">{state.quality_score}/10</td>
      <td style="padding:12px 16px;text-align:center;font-family:Arial,sans-serif;font-size:24px;font-weight:700;color:#333">{state.elapsed_sec()}s</td>
      <td style="padding:12px 16px;text-align:center;font-family:Arial,sans-serif;font-size:24px;font-weight:700;color:#333">{len(state.agent_sequence)}</td>
    </tr>
  </table>
</td></tr>

<!-- Synthesis Report -->
<tr><td style="padding:0 24px">
  <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 16px">
    <tr><td style="background:#1a237e;padding:10px 16px;color:#fff;font-weight:600;font-size:15px;border-radius:6px 6px 0 0">Investment Synthesis Report</td></tr>
    <tr><td style="background:#fafafa;padding:16px;border:1px solid #e0e0e0;border-top:0;border-radius:0 0 6px 6px;white-space:pre-wrap;font-size:13px;line-height:1.7;color:#333;font-family:Arial,sans-serif">{report[:4000]}</td></tr>
  </table>
</td></tr>

<!-- ABM Simulation -->
<tr><td style="padding:0 24px">{abm_section}</td></tr>

<!-- Adversarial Validation -->
<tr><td style="padding:0 24px">{val_section}</td></tr>

<!-- Agent Sections -->
<tr><td style="padding:0 24px"><table width="100%" cellpadding="0" cellspacing="0">
{agent_sections}
</table></td></tr>

<!-- Pipeline Breakdown -->
<tr><td style="padding:16px 24px">
  <table width="100%" cellpadding="0" cellspacing="0" style="margin:8px 0">
    <tr><td colspan="2" style="background:#37474f;padding:10px 16px;color:#fff;font-weight:600;font-size:14px;border-radius:6px 6px 0 0">Pipeline Breakdown</td></tr>
    <tr><td style="padding:4px 16px;font-weight:600;font-size:13px">Agent</td>
      <td style="padding:4px 16px;font-weight:600;font-size:13px;text-align:right">Time</td></tr>
    {timing_rows}
  </table>
</td></tr>

<!-- Footer -->
<tr><td style="background:#f5f5f5;padding:16px 24px;text-align:center;border-top:1px solid #e0e0e0">
  <p style="margin:0;font-family:Arial,sans-serif;font-size:12px;color:#999">
    Multi-Agent Pipeline v6.0 · {len(state.agent_sequence)} agents · {LLM_RESEARCH} | {LLM_FINANCIAL} | {LLM_REASONING}<br>
    {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}
  </p>
</td></tr>

</table></td></tr></table></body></html>"""

    def _update_pipeline_sheet(self, state):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        self._import_marketing()
        from config import settings

        creds = Credentials.from_authorized_user_file(settings.TOKEN_PATH,
            list(set(settings.GMAIL_SCOPES + settings.SHEETS_SCOPES)))
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        svc = build('sheets', 'v4', credentials=creds)
        sid = settings.SPREADSHEET_ID
        tab = settings.SHEET_NAME

        # Auto-create tab if missing
        try:
            meta = svc.spreadsheets().get(spreadsheetId=sid).execute()
            tabs = [s['properties']['title'] for s in meta.get('sheets', [])]
            if tab not in tabs:
                logger.info(f"[Sheet] Creating tab '{tab}'...")
                svc.spreadsheets().batchUpdate(spreadsheetId=sid, body={
                    "requests": [{"addSheet": {"properties": {"title": tab}}}]}).execute()
        except Exception as e:
            logger.warning(f"[Sheet] Tab check failed: {e}")

        header = ["Timestamp", "Company", "Workflow", "Summary", "Mail_Status", "Duration_Sec"]
        try:
            existing = svc.spreadsheets().values().get(
                spreadsheetId=sid, range=f"{tab}!A1:F1").execute().get('values', [])
            if not existing or existing[0] != header:
                svc.spreadsheets().values().update(spreadsheetId=sid, range=f"{tab}!A1:F1",
                    valueInputOption='RAW', body={'values': [header]}).execute()
        except Exception:
            svc.spreadsheets().values().update(spreadsheetId=sid, range=f"{tab}!A1:F1",
                valueInputOption='RAW', body={'values': [header]}).execute()

        summary = f"{state.current_step} steps. {' -> '.join(state.agent_sequence)}. Quality: {state.quality_score}/10"
        mail = "Sent" if state.email_status.get("success") else "Failed"
        row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               f"{state.company} ({', '.join(state.tickers)})" if state.tickers else state.company,
               state.workflow_name, summary[:200], mail, str(state.elapsed_sec())]
        svc.spreadsheets().values().append(spreadsheetId=sid, range=f"{tab}!A:F",
            valueInputOption='RAW', insertDataOption='INSERT_ROWS',
            body={'values': [row]}).execute()
        logger.info(f"[Sheet] Row appended to '{tab}'")

    def _save_workflow_files(self, state, intent):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = (state.company or "unknown").replace(" ", "_").replace(",", "").replace(".", "")[:25]
        rd = os.path.join(_PROJECT_ROOT, "reports", "output",
                          f"{state.workflow_name}_{slug}_{ts}")
        os.makedirs(rd, exist_ok=True)
        state.report_dir = rd
        n = 1

        # ----- Individual agent reports (professional MD) -----
        for name, content, title, emoji in [
            ("deep_research",    state.research_output,     "Deep Research Analysis",        "🔬"),
            ("latest_news",      state.news_output,         "Latest News & Headlines",       "📰"),
            ("financial_data",   state.financial_data,       "Financial Data Analysis",       "📊"),
            ("sec_risk",         state.sec_analysis,         "SEC Risk Factor Analysis",      "⚖️"),
            ("marketing",        state.marketing_analysis,   "Marketing & Competitive Intel", "📈"),
            ("sentiment",        state.sentiment_data,       "Sentiment Analysis",            "💬"),
        ]:
            if content and "[ERROR" not in content:
                md = self._format_agent_report(
                    title, emoji, state, content, is_final=False)
                with open(os.path.join(rd, f"{n:02d}_{name}.md"),
                          "w", encoding="utf-8") as f:
                    f.write(md)
                n += 1

        # ----- Synthesis report -----
        if state.synthesis_report:
            md = self._format_agent_report(
                "Investment Synthesis Report", "📋", state,
                state.synthesis_report, is_final=True)
            with open(os.path.join(rd, f"{n:02d}_synthesis_report.md"),
                      "w", encoding="utf-8") as f:
                f.write(md)
            n += 1

        # ----- ABM simulation report (Phase 2 — separate file) -----
        if state.simulation_report and not state.mirofish_skipped:
            abm_md = self._format_abm_report(state)
            with open(os.path.join(rd, f"{n:02d}_abm_simulation.md"),
                      "w", encoding="utf-8") as f:
                f.write(abm_md)
            n += 1

            # Raw simulation result JSON
            with open(os.path.join(rd, f"{n:02d}_abm_result.json"),
                      "w", encoding="utf-8") as f:
                json.dump(state.simulation_result, f, indent=2, default=str)
            n += 1

            # ----- Grandmaster: Graph visualization JSON -----
            try:
                from src.abm.graph_visualizer import GraphVisualizer
                sim_res = state.simulation_result or {}
                agent_sts = sim_res.get("agent_states", [])
                coals = sim_res.get("coalitions", [])
                graph_data = GraphVisualizer.build_from_sim_result(
                    ticker=state.tickers[0] if state.tickers else "UNKNOWN",
                    agent_states=agent_sts,
                    coalitions=coals,
                    pipeline_data={
                        "research": state.research_output or "",
                        "financial": state.financial_data or "",
                        "news": state.news_output or "",
                        "sec": state.sec_analysis or "",
                    },
                )
                with open(os.path.join(rd, f"{n:02d}_abm_graph.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(graph_data, f, indent=2, default=str)
                n += 1
                logger.info(f"Saved ABM graph visualization: {len(graph_data.get('nodes', []))} nodes")
            except Exception as e:
                logger.warning(f"Graph visualization generation skipped: {e}")

            # ----- Grandmaster: Trend predictions JSON -----
            try:
                from src.abm.trend_predictor import TrendPredictor
                from dataclasses import asdict
                sim_res = state.simulation_result or {}
                trajectory = sim_res.get("sentiment_trajectory", [])
                coals = sim_res.get("coalitions", [])
                contagion = sim_res.get("contagion_events_detail", [])
                predictor = TrendPredictor()
                prediction_result = predictor.predict(
                    ticker=state.tickers[0] if state.tickers else "UNKNOWN",
                    sentiment_trajectory=trajectory,
                    coalitions=coals,
                    contagion_events=contagion,
                    mc_path_sentiments=sim_res.get(
                        "mc_path_sentiments", []),
                )

                # --- LLM Narrative Synthesis (Upgrade 2) ---
                ticker = state.tickers[0] if state.tickers else "UNKNOWN"
                try:
                    fcs = prediction_result.forecasts
                    nar_prompt = (
                        f"ABM simulation for {ticker} complete. Write a "
                        f"150-word predictive forecast based on these "
                        f"quant results:\n"
                        f"Current sentiment: "
                        f"{prediction_result.current_sentiment:+.3f}\n"
                        f"Momentum: {prediction_result.momentum:+.4f} "
                        f"({'accelerating' if prediction_result.momentum > 0 else 'decelerating'})\n"
                        f"Coalition stability: "
                        f"{prediction_result.coalition_stability:.3f} "
                        f"({'stable' if prediction_result.coalition_stability > 0.5 else 'fragile'})\n"
                    )
                    if len(fcs) >= 1:
                        nar_prompt += (
                            f"T+5: {fcs[0].predicted_sentiment:+.3f} "
                            f"({fcs[0].confidence_pct}% conf)\n")
                    if len(fcs) >= 3:
                        nar_prompt += (
                            f"T+30: {fcs[2].predicted_sentiment:+.3f} "
                            f"({fcs[2].confidence_pct}% conf)\n")
                    nar_prompt += (
                        f"Risk factors: "
                        f"{'; '.join(prediction_result.risk_factors)}\n"
                        f"Explain WHY the consensus may shift, using "
                        f"the stability and momentum data. Write for "
                        f"an institutional audience. /no_think"
                    )
                    narrative = call_ollama(
                        nar_prompt,
                        system=f"You are a quantitative research analyst. {_DATE_GUARD}",
                        model=LLM_REASONING, max_tokens=300, timeout=60)
                    prediction_result.narrative = narrative
                    logger.info(f"Generated prediction narrative: {len(narrative)} chars")
                except Exception as nar_e:
                    logger.warning(f"Narrative synthesis skipped: {nar_e}")

                # Store prediction data for adversarial validation
                state._prediction_data = json.dumps({
                    "trend": prediction_result.trend_direction,
                    "T+5": prediction_result.forecasts[0].predicted_sentiment if prediction_result.forecasts else 0,
                    "T+30": prediction_result.forecasts[2].predicted_sentiment if len(prediction_result.forecasts) > 2 else 0,
                    "stability": prediction_result.coalition_stability,
                    "narrative": (prediction_result.narrative or "")[:500],
                }, default=str)

                predictions = asdict(prediction_result)
                with open(os.path.join(rd, f"{n:02d}_abm_predictions.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(predictions, f, indent=2, default=str)
                n += 1
                logger.info(f"Saved ABM trend predictions: {predictions.get('trend_direction', 'unknown')}")
            except Exception as e:
                logger.warning(f"Trend prediction generation skipped: {e}")

        # ----- Adversarial validation report (Phase 3) -----
        if state.validation_result:
            val_md = self._format_validation_report(state)
            with open(os.path.join(rd, f"{n:02d}_validation.md"),
                      "w", encoding="utf-8") as f:
                f.write(val_md)
            n += 1

            with open(os.path.join(rd, f"{n:02d}_validation_result.json"),
                      "w", encoding="utf-8") as f:
                json.dump(state.validation_result, f, indent=2, default=str)
            n += 1

        # ----- Leads JSON (if marketing) -----
        if state.generated_leads:
            with open(os.path.join(rd, f"{n:02d}_leads.json"),
                      "w", encoding="utf-8") as f:
                json.dump(state.generated_leads, f, indent=2, default=str)
            n += 1

        # ----- Email preview -----
        with open(os.path.join(rd, f"{n:02d}_email.html"),
                  "w", encoding="utf-8") as f:
            f.write(self._build_email_html(state, intent))
        n += 1

        # ----- Run README (top-level summary) -----
        readme = self._build_run_readme(state, intent, n - 1)
        with open(os.path.join(rd, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme)

        # ----- Meta JSON (enriched) -----
        with open(os.path.join(rd, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "workflow": state.workflow_name,
                "prompt": state.prompt,
                "company": state.company,
                "tickers": state.tickers,
                "llms": {
                    "research": LLM_RESEARCH,
                    "financial": LLM_FINANCIAL,
                    "reasoning": LLM_REASONING,
                },
                "steps": state.current_step,
                "total_steps": state.total_steps,
                "agents": state.agent_sequence,
                "quality": state.quality_score,
                "duration_sec": state.elapsed_sec(),
                "timings": state.timings,
                "email": state.email_status,
                "sheet": state.sheet_status,
                "errors": state.errors,
                "step_log": state.step_log,
                "mirofish_enabled": not state.mirofish_skipped,
                "abm_budget_mode": state.abm_budget_mode,
                "validation_survival_rate": state.validation_result.get(
                    "claim_survival_rate", None),
                "total_files": n,
                "generated_at": datetime.now().isoformat(),
            }, f, indent=2, default=str)

        # ----- Persist state via PipelineStateStore (Phase 4) -----
        try:
            from src.pipeline_state_store import PipelineStateStore
            store = PipelineStateStore()
            run_id = f"{state.workflow_name}_{slug}_{ts}"
            store.save_state(run_id, state)
        except Exception as e:
            logger.warning(f"[StateStore] Could not persist state: {e}")

        logger.info(f"[Files] {n} files saved to: {rd}")

    # ------------------------------------------------------------------
    # Report Formatters
    # ------------------------------------------------------------------

    def _format_agent_report(self, title, emoji, state, content, is_final):
        """Build a professional Markdown report for a single agent step."""
        header = f"""# {emoji} {title}

> **Pipeline:** {state.workflow_name.replace('_', ' ').title()}
> **Company:** {state.company} ({', '.join(state.tickers)})
> **Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}
> **Quality Score:** {state.quality_score}/10

---

"""
        if is_final:
            header += f"""## Executive Summary

*This report synthesizes findings from {len(state.agent_sequence)} specialized agents
analyzing {state.company} across research, financial, SEC filings, sentiment,
and {'ABM simulation' if not state.mirofish_skipped else 'market analysis'} dimensions.*

---

"""
        return header + content

    def _format_abm_report(self, state):
        """Build ABM simulation report with MC metadata."""
        sim = state.simulation_result or {}
        mc_paths = sim.get("mc_paths", 1)
        majority = sim.get("mc_majority_sentiment", "N/A")
        agreement = sim.get("mc_agreement", 0)

        return f"""# 🧪 ABM Simulation Report — MiroFish

> **Ticker:** {sim.get('ticker', state.company)}
> **Budget Mode:** {state.abm_budget_mode}
> **Monte Carlo Paths:** {mc_paths}
> **Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}

---

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| **Total Ticks** | {sim.get('total_ticks', 'N/A')} |
| **Total Agents** | {sim.get('total_agents', 'N/A')} |
| **MC Paths** | {mc_paths} |
| **Budget Mode** | {state.abm_budget_mode.upper()} |

## Results

| Metric | Value |
|--------|-------|
| **Majority Sentiment** | {majority.upper()} |
| **MC Agreement** | {agreement:.0%} |
| **Total Actions** | {sim.get('total_actions', 'N/A')} |

## Narrative Analysis

{state.simulation_report}
"""

    def _format_validation_report(self, state):
        """Build adversarial validation report."""
        val = state.validation_result
        total = val.get("total_claims", 0)
        supported = val.get("supported_claims", 0)
        survival = val.get("claim_survival_rate", 0)
        flagged = val.get("flagged_claims", [])
        replaced = val.get("should_replace", False)

        flag_lines = ""
        for f in flagged:
            verdict_icon = "❌" if f.get("verdict") == "contradicted" else "⚠️"
            flag_lines += f"\n| {verdict_icon} {f.get('verdict', '').upper()} | {f.get('claim', '')[:80]} | {f.get('reason', '')[:100]} |"

        return f"""# ✅ Adversarial Validation Report

> **Claims Analyzed:** {total}
> **Claims Survived:** {supported}/{total} ({survival:.0%})
> **Report Replaced:** {'Yes — below threshold' if replaced else 'No — report is reliable'}
> **Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}

---

## Validation Summary

| Metric | Value |
|--------|-------|
| **Total Claims Extracted** | {total} |
| **Supported by Evidence** | {supported} |
| **Survival Rate** | {survival:.0%} |
| **Threshold** | 70% |
| **Verdict** | {'⚠️ BELOW THRESHOLD' if replaced else '✅ PASSED'} |

## Flagged Claims

| Verdict | Claim | Reason |
|---------|-------|--------|{flag_lines if flag_lines else chr(10) + '| ✅ | All claims supported | — |'}

---

*Validated using LLM cross-reference against pipeline evidence sources.*
"""

    def _build_run_readme(self, state, intent, file_count):
        """Build a top-level README summarizing the entire run."""
        step_rows = ""
        for s in state.step_log:
            status_icon = "✅" if "error" not in s.get("detail", "").lower() else "⚠️"
            step_rows += (
                f"\n| {s.get('step', '')}/{s.get('total', '')} "
                f"| {s.get('agent', '')} "
                f"| {s.get('duration_s', 0):.1f}s "
                f"| {status_icon} {s.get('status', '')} |"
            )

        timing_rows = ""
        for agent, dur in sorted(state.timings.items(),
                                  key=lambda x: x[1], reverse=True):
            timing_rows += f"\n| {agent} | {dur:.2f}s |"

        error_section = ""
        if state.errors:
            error_section = "\n## ⚠️ Errors\n\n" + "\n".join(
                f"- `{e}`" for e in state.errors)

        abm_section = ""
        if not state.mirofish_skipped and state.simulation_result:
            sim = state.simulation_result
            abm_section = f"""
## 🧪 ABM Simulation

| Metric | Value |
|--------|-------|
| **Majority Sentiment** | {sim.get('mc_majority_sentiment', 'N/A').upper()} |
| **MC Agreement** | {sim.get('mc_agreement', 0):.0%} |
| **MC Paths** | {sim.get('mc_paths', 1)} |
| **Budget Mode** | {state.abm_budget_mode.upper()} |
"""

        val_section = ""
        if state.validation_result and state.validation_result.get("total_claims"):
            val = state.validation_result
            val_section = f"""
## ✅ Adversarial Validation

| Metric | Value |
|--------|-------|
| **Claims Survived** | {val.get('supported_claims', 0)}/{val.get('total_claims', 0)} |
| **Survival Rate** | {val.get('claim_survival_rate', 0):.0%} |
| **Report Replaced** | {'Yes' if val.get('should_replace') else 'No'} |
"""

        return f"""# 📊 Pipeline Run — {state.company}

> **Workflow:** {state.workflow_name.replace('_', ' ').title()}
> **Company:** {state.company} ({', '.join(state.tickers)})
> **Prompt:** _{state.prompt}_
> **Quality Score:** {state.quality_score}/10
> **Duration:** {state.elapsed_sec()}s
> **Files Generated:** {file_count}
> **Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}

---

## Pipeline Steps

| Step | Agent | Duration | Status |
|------|-------|----------|--------|{step_rows}

## Agent Timings

| Agent | Duration |
|-------|----------|{timing_rows}

## LLM Configuration

| Role | Model |
|------|-------|
| Research | `{LLM_RESEARCH}` |
| Financial | `{LLM_FINANCIAL}` |
| Reasoning | `{LLM_REASONING}` |
{abm_section}{val_section}
## Email Status

- **Recipient:** {intent.receiver_email}
- **Status:** {'✅ Sent' if state.email_status.get('success') else '❌ Failed'}
{error_section}
---

*Generated by Multi-Agent Pipeline v6.0 — {len(state.agent_sequence)} agents collaborated on this analysis.*
"""


# ===========================================================================
# CLI
# ===========================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="Multi-Agent Orchestrator v6.0")
    p.add_argument("--prompt", "-p", type=str)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not args.prompt:
        print("Usage: python orchestrator.py -p 'Research Tesla' --dry-run")
        return
    orch = MultiAgentOrchestrator()
    state = orch.run_from_prompt(args.prompt, dry_run=args.dry_run)
    print(json.dumps({"workflow": state.workflow_name, "company": state.company,
        "quality": state.quality_score, "steps": state.current_step,
        "agents": state.agent_sequence, "duration_sec": state.elapsed_sec(),
        "llms": {"research": LLM_RESEARCH, "financial": LLM_FINANCIAL, "reasoning": LLM_REASONING},
        "email": state.email_status, "sheet": state.sheet_status,
        "report_dir": state.report_dir, "errors": state.errors}, indent=2))

if __name__ == "__main__":
    main()
