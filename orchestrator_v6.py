"""
Multi-Agent Orchestrator v6.0 -- Google Deep Research Agent
===========================================================

A universal research agent that:
- Accepts ANY topic (financial, scientific, geopolitical, subjective)
- Utilizes ALL 20+ MCP connectors via MCPManager
- Performs iterative multi-pass research with quality gating
- Outputs publication-quality .md reports

8-node LangGraph pipeline:
  classify_topic -> generate_plan -> execute_searches -> enrich_content
       ^                                                       |
       |                                                semantic_chunk
       |                                                       |
       |                                                extract_facts
       |                                                       |
       +--------------- quality_gate (loop if gaps) <----------+
                              | (pass)
                        compose_report -> END
"""

import os
import sys
import re
import json
import time
import asyncio
import logging
import hashlib
import importlib
import importlib.util
import requests
from typing import (
    TypedDict, List, Dict, Any, Optional, Literal, Annotated, Tuple
)
from operator import add
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_SRC = os.path.join(_PROJECT_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [v6] %(levelname)s %(message)s",
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# LLM model assignments (matches llm_router.py registry)
LLM_CLASSIFIER  = os.getenv("RESEARCH_LLM_MODEL", "llama3.2:latest")
LLM_PLANNER     = os.getenv("ADVISOR_LLM_MODEL", "qwen3:8b")
LLM_EXTRACTOR   = os.getenv("RESEARCH_LLM_MODEL", "llama3.2:latest")
LLM_SCORER      = os.getenv("FINANCIAL_LLM_MODEL", "qwen3:4b")
LLM_COMPOSER    = os.getenv("ADVISOR_LLM_MODEL", "qwen3:8b")

# ===========================================================================
# MULTI-AGENT REGISTRY  (3 LLMs as 6 named domain-expert agents)
# ===========================================================================

AGENT_REGISTRY = {
    "Classifier Agent":  {"model": LLM_CLASSIFIER,  "role": "Topic detection & intent routing",     "llm": "llama3.2"},
    "Strategist Agent":  {"model": LLM_PLANNER,     "role": "Research planning & decomposition",    "llm": "qwen3:8b"},
    "Extractor Agent":   {"model": LLM_EXTRACTOR,   "role": "Fact extraction from raw content",     "llm": "llama3.2"},
    "Verifier Agent":    {"model": LLM_SCORER,      "role": "Cross-checks facts & verifies claims", "llm": "qwen3:4b"},
    "Scorer Agent":      {"model": LLM_SCORER,      "role": "Quality assessment & gap analysis",    "llm": "qwen3:4b"},
    "Composer Agent":    {"model": LLM_COMPOSER,    "role": "Report synthesis & long-form writing",  "llm": "qwen3:8b"},
}

# Node -> Agent mapping for logging
NODE_AGENT_MAP = {
    "classify_topic":   "Classifier Agent",
    "generate_plan":    "Strategist Agent",
    "execute_searches": None,  # Tool execution, no LLM
    "enrich_content":   None,  # Scrapling, no LLM
    "semantic_chunk":   None,  # Embeddings only
    "extract_facts":    "Extractor Agent + Verifier Agent",  # Two-agent collaboration
    "quality_gate":     "Scorer Agent",
    "compose_report":   "Composer Agent",
}

def _log_agent(node: str, detail: str = ""):
    """Log which agent is handling the current node."""
    agent = NODE_AGENT_MAP.get(node)
    if agent:
        reg = AGENT_REGISTRY.get(agent.split(" + ")[0], {})
        model = reg.get("llm", "?")
        logger.info(f"  >> Agent: {agent} ({model}) {detail}")


# ===========================================================================
# LangGraph custom reducers
# ===========================================================================

def _merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts; lists are concatenated, scalars overwritten."""
    if not isinstance(a, dict):
        a = {}
    if not isinstance(b, dict):
        b = {}
    result = dict(a)
    for k, v in b.items():
        if isinstance(v, list) and isinstance(result.get(k), list):
            result[k] = result[k] + v
        else:
            result[k] = v
    return result


def _sum_int(a: int, b: int) -> int:
    return (a or 0) + (b or 0)


# ===========================================================================
# ResearchStateV6  (LangGraph TypedDict)
# ===========================================================================

class SubQuestion(TypedDict):
    question: str
    priority: int          # 1=high, 2=medium, 3=low
    tool: str              # best MCP connector name
    status: str            # pending | done | gap


class LockedFact(TypedDict):
    fact_id: str
    statement: str
    source_id: str
    source_url: str
    source_title: str
    sub_question: str
    relevance: str         # high | medium


class ResearchStateV6(TypedDict):
    # --- Input ---
    query: str
    topic_domain: Optional[str]
    # --- Plan ---
    research_plan: List[SubQuestion]
    # --- Search ---
    search_results: Annotated[List[dict], add]
    enriched_chunks: Annotated[List[dict], add]
    # --- Facts ---
    locked_facts: Annotated[List[LockedFact], add]
    knowledge_gaps: List[str]
    # --- Report ---
    final_report: str
    # --- Quality ---
    quality_score: float
    iteration: int
    max_iterations: int
    # --- Meta ---
    tools_used: Annotated[List[str], add]
    sources_registry: Annotated[List[dict], add]
    timings: Annotated[Dict[str, Any], _merge_dicts]
    error_count: Annotated[int, _sum_int]
    agent_contributions: Annotated[List[dict], add]   # tracks which agent did what
    dry_run: bool


# ===========================================================================
# V6 Research Configuration
# ===========================================================================

@dataclass
class V6Config:
    max_iterations: int = 4
    quality_threshold: float = 7.0
    max_sub_questions: int = 7
    max_pages_to_enrich: int = 5
    semantic_chunk_size: int = 500       # tokens per chunk
    semantic_top_k: int = 3              # chunks to keep per page
    max_facts_per_iteration: int = 10
    report_min_words: int = 800
    # --- Export settings (defaults from Marketing/config/settings.py) ---
    email_to: str = "harish.krishna@testaing.com"
    email_cc: str = ""
    spreadsheet_id: str = "1OkXq8gvcd4etHeTri1qCHFFWDLrxjhQO4H_8STscr2A"
    sheet_tab: str = "Deep_Research_v6"


# ===========================================================================
# Ollama caller (reused pattern from orchestrator.py)
# ===========================================================================

def call_ollama(prompt: str, system: str = "", model: str = LLM_CLASSIFIER,
                timeout: int = 90, max_tokens: int = 4096) -> str:
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
        content = r.json().get("message", {}).get("content", "")
        # Strip <think>...</think> tags (Qwen3)
        return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    except requests.exceptions.Timeout:
        return f"[TIMEOUT] {model} timed out after {timeout}s"
    except Exception as e:
        return f"[ERROR] Ollama ({model}) failed: {e}"


def extract_json(text: str) -> dict:
    """Robustly extract JSON from LLM output."""
    if not isinstance(text, str):
        return {}
    # Strip think tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try code block
    m = re.search(r'```(?:json)?\s*([\{\[].*?[\}\]])\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw braces
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    return {"error": "parse_failed", "raw": text[:300]}


# ===========================================================================
# Per-connector semaphores (Production Hardening)
# ===========================================================================

CONNECTOR_SEMAPHORES: Dict[str, asyncio.Semaphore] = {}

def _get_semaphore(connector_name: str) -> asyncio.Semaphore:
    """Lazy-init semaphores so they're created in the right event loop."""
    if connector_name not in CONNECTOR_SEMAPHORES:
        caps = {
            "tavily_web_search": 3, "duckduckgo_web_search": 2,
            "wikipedia_search": 3, "arxiv_search": 3,
            "semantic_scholar": 2, "pubmed_search": 2,
            "reddit_search": 1, "youtube_search": 2,
            "github_search": 2, "news_headlines": 2,
            "yahoo_finance": 2, "safe_web_scraper": 2,
            "weather": 2, "reddit_search_brightdata": 1,
            "reddit_post_retrieval_brightdata": 1,
        }
        limit = caps.get(connector_name, 2)
        CONNECTOR_SEMAPHORES[connector_name] = asyncio.Semaphore(limit)
    return CONNECTOR_SEMAPHORES[connector_name]


# Global request semaphore (max 3 concurrent deep research requests)
_GLOBAL_RESEARCH_SEM = None

def _get_global_sem() -> asyncio.Semaphore:
    global _GLOBAL_RESEARCH_SEM
    if _GLOBAL_RESEARCH_SEM is None:
        _GLOBAL_RESEARCH_SEM = asyncio.Semaphore(3)
    return _GLOBAL_RESEARCH_SEM


# ===========================================================================
# TOPIC CLASSIFIER  (domain detection + tool mapping)
# ===========================================================================

TOOL_MATRIX = {
    "financial": {
        "primary": ["tavily_web_search", "yahoo_finance", "news_headlines"],
        "secondary": ["reddit_search", "safe_web_scraper", "duckduckgo_web_search"],
    },
    "scientific": {
        "primary": ["arxiv_search", "semantic_scholar", "pubmed_search", "tavily_web_search"],
        "secondary": ["wikipedia_search", "github_search"],
    },
    "technology": {
        "primary": ["github_search", "tavily_web_search", "arxiv_search"],
        "secondary": ["reddit_search", "youtube_search", "news_headlines"],
    },
    "geopolitical": {
        "primary": ["news_headlines", "tavily_web_search", "wikipedia_search"],
        "secondary": ["reddit_search", "youtube_search"],
    },
    "general": {
        "primary": ["tavily_web_search", "duckduckgo_web_search", "wikipedia_search"],
        "secondary": ["news_headlines", "reddit_search", "youtube_search"],
    },
}


def classify_topic(state: ResearchStateV6) -> dict:
    """Node 1: Classify the query domain and select tool mix."""
    t0 = time.time()
    query = state["query"]

    if state.get("dry_run"):
        # Keyword-based fast classification for dry run
        domain = _keyword_classify(query)
        logger.info(f"[1/8 classify_topic] DRY RUN -> domain={domain}")
        _log_agent("classify_topic", f"-> {domain}")
        return {
            "topic_domain": domain,
            "timings": {"classify_topic": time.time() - t0},
            "agent_contributions": [{"node": "classify_topic", "agent": "Classifier Agent", "model": LLM_CLASSIFIER, "output": domain}],
        }

    prompt = f"""Classify this research query into exactly ONE domain.

Query: "{query}"

Domains: financial, scientific, technology, geopolitical, general

Rules:
- financial: stocks, companies, investments, markets, revenue, earnings
- scientific: papers, studies, biology, physics, chemistry, medicine, health
- technology: software, AI, programming, hardware, startups, open source
- geopolitical: politics, countries, policy, war, trade, international relations
- general: everything else (philosophy, lifestyle, comparisons, opinions)

Respond with ONLY a JSON object: {{"domain": "<one of the 5>"}}"""

    resp = call_ollama(prompt, model=LLM_CLASSIFIER, timeout=30, max_tokens=100)
    parsed = extract_json(resp)
    domain = parsed.get("domain", "general")
    if domain not in TOOL_MATRIX:
        domain = _keyword_classify(query)

    logger.info(f"[1/8 classify_topic] domain={domain}")
    _log_agent("classify_topic", f"-> {domain}")
    return {
        "topic_domain": domain,
        "timings": {"classify_topic": time.time() - t0},
        "agent_contributions": [{"node": "classify_topic", "agent": "Classifier Agent", "model": LLM_CLASSIFIER, "output": domain}],
    }


def _keyword_classify(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["stock", "invest", "revenue", "earnings", "ticker",
                             "market cap", "dividend", "sec filing", "due diligence",
                             "portfolio", "financial"]):
        return "financial"
    if any(w in q for w in ["study", "paper", "research", "clinical", "biology",
                             "physics", "chemistry", "medicine", "healthcare",
                             "disease", "genome", "pharmaceutical"]):
        return "scientific"
    if any(w in q for w in ["software", "programming", "ai ", "machine learning",
                             "github", "open source", "algorithm", "cloud",
                             "cybersecurity", "startup", "saas"]):
        return "technology"
    if any(w in q for w in ["country", "government", "policy", "war", "trade",
                             "election", "geopolitic", "sanctions", "nato",
                             "united nations", "diplomacy"]):
        return "geopolitical"
    return "general"


# ===========================================================================
# RESEARCH PLAN GENERATOR  (query decomposition)
# ===========================================================================

def generate_plan(state: ResearchStateV6) -> dict:
    """Node 2: Decompose query into 5-7 sub-questions with tool assignments."""
    t0 = time.time()
    query = state["query"]
    domain = state.get("topic_domain", "general")
    available_tools = TOOL_MATRIX.get(domain, TOOL_MATRIX["general"])
    iteration = state.get("iteration", 0)
    gaps = state.get("knowledge_gaps", [])

    if state.get("dry_run"):
        plan = _dry_run_plan(query, domain)
        logger.info(f"[2/8 generate_plan] DRY RUN -> {len(plan)} sub-questions")
        _log_agent("generate_plan", f"-> {len(plan)} sub-questions")
        return {
            "research_plan": plan,
            "iteration": iteration + 1,
            "timings": {"generate_plan": time.time() - t0},
            "agent_contributions": [{"node": "generate_plan", "agent": "Strategist Agent", "model": LLM_PLANNER, "output": f"{len(plan)} sub-questions"}],
        }

    # Build gap context for subsequent iterations
    gap_ctx = ""
    if gaps:
        gap_ctx = f"\n\nKNOWLEDGE GAPS from previous iteration (PRIORITIZE these):\n"
        gap_ctx += "\n".join(f"- {g}" for g in gaps[:5])

    all_tools = available_tools["primary"] + available_tools["secondary"]
    tools_str = ", ".join(all_tools)

    prompt = f"""You are a senior research strategist. Decompose this query into 5-7 highly specific, targeted sub-questions.

RESEARCH QUERY: "{query}"
DOMAIN: {domain}
AVAILABLE TOOLS: {tools_str}
{gap_ctx}

DECOMPOSITION STRATEGY - Cover these angles:
1. FUNDAMENTALS: What are the core concepts, mechanisms, or entities involved?
2. DATA & METRICS: What specific numbers, statistics, or quantitative evidence exists?
3. REGULATORY/FRAMEWORKS: What policies, regulations, standards, or frameworks apply?
4. EXPERT OPINIONS: What do leading researchers, analysts, or institutions say?
5. CHALLENGES & RISKS: What are the key obstacles, limitations, or controversies?
6. CASE STUDIES: What real-world implementations, pilots, or examples exist?
7. FUTURE OUTLOOK: What are the emerging trends, predictions, and implications?

CRITICAL RULES:
- Questions must be SPECIFIC to the topic, NOT generic templates
- Reference actual entities, technologies, regulations, or frameworks BY NAME
- Each question should be INDEPENDENTLY searchable and produce useful results
- Avoid repeating the full query in each question - extract the specific angle

For each sub-question, assign:
- priority: 1 (critical), 2 (important), or 3 (nice-to-have)
- tool: the single best tool from the available list

Return ONLY JSON:
{{
  "sub_questions": [
    {{"question": "specific sub-question", "priority": 1, "tool": "tool_name"}},
    ...
  ]
}}"""

    resp = call_ollama(prompt, model=LLM_PLANNER, timeout=90, max_tokens=2048,
                       system="You are an expert research planner like Google Gemini Deep Research. "
                              "Generate highly specific sub-questions that decompose the query into "
                              "distinct research angles. Each question should target a specific aspect "
                              "and be directly searchable. Return valid JSON only.")
    parsed = extract_json(resp)
    raw_sqs = parsed.get("sub_questions", [])

    plan: List[SubQuestion] = []
    for sq in raw_sqs[:7]:
        if isinstance(sq, dict) and sq.get("question"):
            tool = sq.get("tool", all_tools[0])
            if tool not in all_tools:
                tool = all_tools[0]
            plan.append({
                "question": str(sq["question"]),
                "priority": min(3, max(1, int(sq.get("priority", 2)))),
                "tool": tool,
                "status": "pending",
            })

    if not plan:
        plan = _dry_run_plan(query, domain)

    logger.info(f"[2/8 generate_plan] iteration={iteration+1}, {len(plan)} sub-questions")
    _log_agent("generate_plan", f"-> {len(plan)} sub-questions")
    return {
        "research_plan": plan,
        "iteration": iteration + 1,
        "timings": {"generate_plan": time.time() - t0},
        "agent_contributions": [{"node": "generate_plan", "agent": "Strategist Agent", "model": LLM_PLANNER, "output": f"{len(plan)} sub-questions"}],
    }


def _dry_run_plan(query: str, domain: str) -> List[SubQuestion]:
    """Generate domain-specific sub-questions using LLM even in dry-run."""
    tools = TOOL_MATRIX.get(domain, TOOL_MATRIX["general"])
    primary = tools["primary"]
    all_tools = primary + tools.get("secondary", [])
    tools_str = ", ".join(all_tools)

    prompt = f"""Decompose this research query into exactly 5 specific, targeted sub-questions.

QUERY: "{query}"
DOMAIN: {domain}
AVAILABLE TOOLS: {tools_str}

RULES:
- Each question must target a SPECIFIC angle of the topic
- Reference real entities, frameworks, technologies, or metrics by name
- DO NOT use generic templates like 'What is the current state of X'
- Make each question independently searchable

Return ONLY JSON:
{{
  "sub_questions": [
    {{"question": "specific question", "priority": 1, "tool": "tool_name"}},
    ...
  ]
}}"""

    try:
        resp = call_ollama(prompt, model=LLM_CLASSIFIER, timeout=30, max_tokens=1024,
                          system="Generate 5 specific research sub-questions. Return valid JSON only.")
        parsed = extract_json(resp)
        raw_sqs = parsed.get("sub_questions", [])
        plan = []
        for sq in raw_sqs[:5]:
            if isinstance(sq, dict) and sq.get("question"):
                tool = sq.get("tool", primary[0])
                if tool not in all_tools:
                    tool = primary[0]
                plan.append({
                    "question": str(sq["question"]),
                    "priority": min(3, max(1, int(sq.get("priority", 2)))),
                    "tool": tool,
                    "status": "pending",
                })
        if plan:
            return plan
    except Exception:
        pass

    # Ultimate fallback if LLM fails
    return [
        {"question": f"Analyze the fundamental concepts and current state of {query[:60]}",
         "priority": 1, "tool": primary[0], "status": "pending"},
        {"question": f"What specific data, statistics, and metrics exist for {query[:60]}?",
         "priority": 1, "tool": primary[min(1, len(primary)-1)], "status": "pending"},
        {"question": f"What do leading researchers and analysts conclude about {query[:60]}?",
         "priority": 2, "tool": primary[0], "status": "pending"},
        {"question": f"What are the key challenges, risks, and limitations of {query[:60]}?",
         "priority": 2, "tool": primary[min(2, len(primary)-1)], "status": "pending"},
        {"question": f"What emerging trends and future developments are expected for {query[:60]}?",
         "priority": 3, "tool": primary[0], "status": "pending"},
    ]


# ===========================================================================
# MULTI-SOURCE SEARCH EXECUTOR  (semaphore-gated parallel search)
# ===========================================================================

def execute_searches(state: ResearchStateV6) -> dict:
    """Node 3: Execute sub-questions across MCP connectors in parallel."""
    t0 = time.time()
    plan = state.get("research_plan", [])
    pending = [sq for sq in plan if sq["status"] == "pending"]

    if state.get("dry_run"):
        results, sources, tools = _dry_run_search(pending)
        for sq in plan:
            sq["status"] = "done"
        logger.info(f"[3/8 execute_searches] DRY RUN -> {len(results)} results")
        return {
            "search_results": results,
            "sources_registry": sources,
            "tools_used": tools,
            "research_plan": plan,
            "timings": {"execute_searches": time.time() - t0},
        }

    # Real search via MCPManager (with graceful fallback)
    try:
        from src.mcp import MCPManager, ConnectorResponse
        mcp = MCPManager(llm=None, tavily_api_key=TAVILY_API_KEY)
        available = mcp.get_available_connectors()
        use_mcp = True
    except Exception as e:
        logger.warning(f"[3/8 execute_searches] MCPManager import failed: {e}")
        logger.warning(f"  -> Falling back to direct Tavily + requests search")
        use_mcp = False
        available = []

    if use_mcp:
        async def _run():
            tasks = []
            for sq in pending:
                tool_name = sq["tool"]
                if tool_name not in available:
                    tool_name = mcp.get_primary_web_search_name()
                tasks.append(_sem_search(mcp, tool_name, sq["question"]))
            return await asyncio.gather(*tasks, return_exceptions=True)

        async def _sem_search(mcp_mgr, tool_name, question):
            sem = _get_semaphore(tool_name)
            async with sem:
                resp = await mcp_mgr.execute_tool(tool_name, {"query": question})
                return (tool_name, question, resp)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                raw_results = loop.run_until_complete(_run())
            else:
                raw_results = asyncio.run(_run())
        except RuntimeError:
            raw_results = asyncio.run(_run())

        results, sources, tools_used = [], [], []
        source_idx = len(state.get("sources_registry", []))

        for i, item in enumerate(raw_results):
            if isinstance(item, Exception):
                logger.warning(f"Search failed: {item}")
                continue
            tool_name, question, resp = item
            tools_used.append(tool_name)

            if not resp.success:
                continue

            data_str = str(resp.data) if resp.data else ""
            urls = _extract_urls_from_response(resp.data, data_str)

            sid = f"[{source_idx + i + 1}]"
            for j, url_info in enumerate(urls[:3]):
                sources.append({
                    "id": f"[{source_idx + i + 1}.{j+1}]",
                    "url": url_info.get("url", f"search_{source_idx+i+1}"),
                    "title": url_info.get("title", "Source"),
                    "tool": tool_name,
                    "query": question,
                })

            results.append({
                "sub_question": question,
                "tool": tool_name,
                "raw_text": data_str[:3000],
                "urls": [u.get("url", "") for u in urls[:3]],
                "source_ids": [f"[{source_idx+i+1}.{j+1}]" for j in range(min(3, len(urls)))],
            })

    else:
        # ---- Fallback: direct Tavily + requests search (no MCP) ----
        results, sources, tools_used = [], [], []
        source_idx = len(state.get("sources_registry", []))

        for i, sq in enumerate(pending):
            question = sq["question"]
            try:
                # Try Tavily search directly
                if TAVILY_API_KEY:
                    tavily_resp = requests.post(
                        "https://api.tavily.com/search",
                        json={"query": question, "search_depth": "basic", "max_results": 3,
                              "api_key": TAVILY_API_KEY},
                        timeout=30,
                    )
                    if tavily_resp.status_code == 200:
                        data = tavily_resp.json()
                        tavily_results = data.get("results", [])
                        tools_used.append("tavily_web_search")

                        for j, r in enumerate(tavily_results[:3]):
                            sources.append({
                                "id": f"[{source_idx + i + 1}.{j+1}]",
                                "url": r.get("url", ""),
                                "title": r.get("title", "Source"),
                                "tool": "tavily_web_search",
                                "query": question,
                            })

                        raw_text = "\n".join(
                            f"- {r.get('title', '')}: {r.get('content', '')[:500]}"
                            for r in tavily_results[:3]
                        )
                        results.append({
                            "sub_question": question,
                            "tool": "tavily_web_search",
                            "raw_text": raw_text[:3000],
                            "urls": [r.get("url", "") for r in tavily_results[:3]],
                            "source_ids": [f"[{source_idx+i+1}.{j+1}]" for j in range(min(3, len(tavily_results)))],
                        })
                    else:
                        logger.warning(f"Tavily search failed: HTTP {tavily_resp.status_code}")
                else:
                    logger.warning(f"No TAVILY_API_KEY set, skipping search for: {question[:40]}")
            except Exception as e:
                logger.warning(f"Fallback search failed for '{question[:40]}': {e}")

    for sq in plan:
        sq["status"] = "done"

    dur = time.time() - t0
    logger.info(f"[3/8 execute_searches] {len(results)} results from {len(set(tools_used))} tools in {dur:.1f}s")
    return {
        "search_results": results,
        "sources_registry": sources,
        "tools_used": tools_used,
        "research_plan": plan,
        "timings": {"execute_searches": dur},
    }


def _extract_urls_from_response(data: Any, data_str: str) -> List[dict]:
    urls = []
    try:
        if isinstance(data, dict):
            for item in data.get("results", data.get("data", []))[:5]:
                if isinstance(item, dict) and "url" in item:
                    urls.append({"url": item["url"], "title": item.get("title", "Source")})
        elif isinstance(data, list):
            for item in data[:5]:
                if isinstance(item, dict) and "url" in item:
                    urls.append({"url": item["url"], "title": item.get("title", "Source")})
    except Exception:
        pass
    if not urls:
        found = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', data_str)
        urls = [{"url": u.rstrip("',)\";\\"), "title": "Source"} for u in found[:3]]
    if not urls:
        urls = [{"url": "internal_search", "title": "Search Result"}]
    return urls


def _dry_run_search(pending: List[SubQuestion]) -> Tuple[list, list, list]:
    results, sources, tools = [], [], []
    for i, sq in enumerate(pending):
        tools.append(sq["tool"])
        sid = f"[{i+1}]"
        sources.append({
            "id": f"[{i+1}.1]",
            "url": f"https://example.com/source-{i+1}",
            "title": f"Source for: {sq['question'][:50]}",
            "tool": sq["tool"], "query": sq["question"],
        })
        results.append({
            "sub_question": sq["question"],
            "tool": sq["tool"],
            "raw_text": f"[DRY RUN] Simulated search result for: {sq['question']}. "
                        f"Key data points: GDP growth 3.2%, unemployment 4.1%, "
                        f"adoption rate 47%, market size $2.1 trillion. "
                        f"According to recent studies, experts project significant changes. "
                        f"Source: World Economic Forum, McKinsey Global Institute.",
            "urls": [f"https://example.com/source-{i+1}"],
            "source_ids": [f"[{i+1}.1]"],
        })
    return results, sources, tools


# ===========================================================================
# CONTENT ENRICHER  (Scrapling + Gemini URL)
# ===========================================================================

SKIP_DOMAINS = {
    "linkedin.com", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "tiktok.com", "youtube.com", "scribd.com",
    "researchgate.net", "api.tavily.com", "localhost",
}


def enrich_content(state: ResearchStateV6) -> dict:
    """Node 4: Scrape top URLs for full page content."""
    t0 = time.time()
    search_results = state.get("search_results", [])

    if state.get("dry_run"):
        enriched = []
        for sr in search_results[-7:]:
            enriched.append({
                "sub_question": sr.get("sub_question", ""),
                "source_ids": sr.get("source_ids", []),
                "content": sr.get("raw_text", "")[:1500],
                "enriched": False,
            })
        logger.info(f"[4/8 enrich_content] DRY RUN -> {len(enriched)} items")
        return {
            "enriched_chunks": enriched,
            "timings": {"enrich_content": time.time() - t0},
        }

    # Collect unique URLs to scrape
    urls_to_scrape = []
    url_to_sr_idx = {}
    for idx, sr in enumerate(search_results[-7:]):
        for url in sr.get("urls", []):
            if url and url.startswith("http") and not any(d in url.lower() for d in SKIP_DOMAINS):
                if url not in url_to_sr_idx:
                    urls_to_scrape.append(url)
                    url_to_sr_idx[url] = idx

    urls_to_scrape = urls_to_scrape[:5]  # Cap at 5

    # --- URL Content Connector (requests + BeautifulSoup) ---
    scraped: Dict[str, str] = {}
    if urls_to_scrape:
        import requests as req_lib
        from bs4 import BeautifulSoup as BS4

        def _fetch_url(url: str) -> dict:
            """Fetch and extract clean text from a URL."""
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                }
                resp = req_lib.get(url, timeout=15, headers=headers, allow_redirects=True)
                resp.raise_for_status()
                soup = BS4(resp.text, "html.parser")
                # Remove non-content elements
                for tag in soup(["script", "style", "nav", "footer", "header",
                                 "aside", "form", "noscript", "iframe", "svg"]):
                    tag.decompose()
                # Extract text
                text = soup.get_text(separator="\n", strip=True)
                lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 20]
                clean = "\n".join(lines)
                if len(clean) < 100:
                    return {"url": url, "content": "", "success": False}
                logger.info(f"  [URL] Fetched {len(clean)} chars from {url[:60]}")
                return {"url": url, "content": clean[:3000], "success": True}
            except Exception as e:
                logger.debug(f"  [URL] Failed: {url[:50]} - {e}")
                return {"url": url, "content": "", "success": False}

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_fetch_url, u): u for u in urls_to_scrape}
            for future in as_completed(futures, timeout=45):
                try:
                    result = future.result(timeout=20)
                    if result.get("success"):
                        scraped[result["url"]] = result["content"]
                except Exception:
                    pass
        logger.info(f"[enrich_content] URL connector fetched {len(scraped)}/{len(urls_to_scrape)} pages")

    # Build enriched chunks
    enriched = []
    for sr in search_results[-7:]:
        content = sr.get("raw_text", "")[:1500]
        is_enriched = False
        for url in sr.get("urls", []):
            if url in scraped:
                content = scraped[url][:2000]
                is_enriched = True
                break
        enriched.append({
            "sub_question": sr.get("sub_question", ""),
            "source_ids": sr.get("source_ids", []),
            "content": content,
            "enriched": is_enriched,
        })

    dur = time.time() - t0
    enriched_count = sum(1 for e in enriched if e["enriched"])
    logger.info(f"[4/8 enrich_content] {enriched_count}/{len(enriched)} enriched in {dur:.1f}s")
    return {
        "enriched_chunks": enriched,
        "timings": {"enrich_content": dur},
    }


# ===========================================================================
# SEMANTIC CHUNKER  (embedding-based relevance filtering)
# ===========================================================================

def _simple_chunk(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into roughly equal chunks by words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
    return chunks or [text[:2000]]


def _get_embedding(text: str, model: str = "nomic-embed-text:latest") -> Optional[List[float]]:
    """Get embedding vector from Ollama."""
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/embed", json={
            "model": model, "input": text,
        }, timeout=15)
        r.raise_for_status()
        data = r.json()
        # Ollama returns {"embeddings": [[...]]} or {"embedding": [...]}
        embs = data.get("embeddings", [])
        if embs and isinstance(embs[0], list):
            return embs[0]
        emb = data.get("embedding", [])
        if emb:
            return emb
        return None
    except Exception as e:
        logger.debug(f"Embedding failed: {e}")
        return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def semantic_chunk(state: ResearchStateV6) -> dict:
    """Node 5: Chunk enriched content and keep only relevant pieces."""
    t0 = time.time()
    enriched = state.get("enriched_chunks", [])
    config = V6Config()

    if state.get("dry_run"):
        # In dry run, just pass through (content is already short)
        logger.info(f"[5/8 semantic_chunk] DRY RUN -> {len(enriched)} chunks passed through")
        return {"timings": {"semantic_chunk": time.time() - t0}}

    filtered_chunks = []
    for item in enriched:
        content = item.get("content", "")
        sub_q = item.get("sub_question", "")

        # Skip short content (already concise enough)
        if len(content.split()) <= config.semantic_chunk_size:
            filtered_chunks.append(item)
            continue

        # Split into chunks
        chunks = _simple_chunk(content, config.semantic_chunk_size)
        if len(chunks) <= config.semantic_top_k:
            filtered_chunks.append(item)
            continue

        # Get question embedding
        q_emb = _get_embedding(sub_q)
        if not q_emb:
            # Fallback: just take first K chunks
            item["content"] = " ".join(chunks[:config.semantic_top_k])
            filtered_chunks.append(item)
            continue

        # Score and rank chunks
        scored = []
        for chunk in chunks:
            c_emb = _get_embedding(chunk)
            if c_emb:
                score = _cosine_similarity(q_emb, c_emb)
                scored.append((score, chunk))
            else:
                scored.append((0.0, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c for _, c in scored[:config.semantic_top_k]]
        item["content"] = " ".join(top_chunks)
        filtered_chunks.append(item)

    # Replace enriched_chunks with filtered versions
    dur = time.time() - t0
    logger.info(f"[5/8 semantic_chunk] Filtered {len(filtered_chunks)} items in {dur:.1f}s")
    return {
        "enriched_chunks": filtered_chunks,
        "timings": {"semantic_chunk": dur},
    }


# ===========================================================================
# FACT SYNTHESIZER  (strict citation-locked JSON extraction)
# ===========================================================================

def extract_facts(state: ResearchStateV6) -> dict:
    """Node 6: Extract facts with strict source locking."""
    t0 = time.time()
    enriched = state.get("enriched_chunks", [])
    sources = state.get("sources_registry", [])
    existing_facts = state.get("locked_facts", [])
    fact_counter = len(existing_facts)

    if state.get("dry_run"):
        facts = []
        for i, item in enumerate(enriched[:5]):
            sids = item.get("source_ids", [f"[{i+1}.1]"])
            src = next((s for s in sources if s.get("id") == sids[0]), {}) if sids else {}
            sub_q = item.get("sub_question", "research topic")
            facts.append({
                "fact_id": f"F-{fact_counter + i + 1:03d}",
                "statement": f"[DRY RUN] Simulated finding for: {sub_q[:80]}. "
                             f"Full content would be extracted from live search results.",
                "source_id": sids[0] if sids else f"[{i+1}.1]",
                "source_url": src.get("url", f"https://example.com/source-{i+1}"),
                "source_title": src.get("title", f"Source {i+1}"),
                "sub_question": sub_q,
                "relevance": "high" if i < 3 else "medium",
                "verified_by": "Verifier Agent (qwen3:4b)",
            })
        logger.info(f"[6/8 extract_facts] DRY RUN -> {len(facts)} facts")
        _log_agent("extract_facts", f"-> {len(facts)} facts (Extractor + Verifier)")
        return {
            "locked_facts": facts,
            "timings": {"extract_facts": time.time() - t0},
            "agent_contributions": [
                {"node": "extract_facts", "agent": "Extractor Agent", "model": LLM_EXTRACTOR, "output": f"{len(facts)} raw facts"},
                {"node": "extract_facts", "agent": "Verifier Agent",  "model": LLM_SCORER, "output": f"{len(facts)} facts verified"},
            ],
        }

    # Build source ID map for the LLM
    source_map = "\n".join(
        f"{s['id']} -> {s.get('title', 'Source')[:60]} ({s.get('url', '')[:60]})"
        for s in sources[-15:]
    )

    # Process each enriched chunk
    all_facts: List[LockedFact] = []
    for item in enriched[-7:]:
        sub_q = item.get("sub_question", "")
        content = item.get("content", "")[:2500]
        sids = item.get("source_ids", [])

        prompt = f"""Extract 2-3 key facts from this search result that answer the question.

QUESTION: {sub_q}
AVAILABLE SOURCE IDs (use ONLY these): {', '.join(sids) if sids else 'none'}

CONTENT:
{content}

Return ONLY JSON:
{{
  "facts": [
    {{
      "statement": "Specific factual claim with numbers or data",
      "source_id": "{sids[0] if sids else '[1.1]'}",
      "relevance": "high"
    }}
  ]
}}"""

        resp = call_ollama(prompt, model=LLM_EXTRACTOR, timeout=45, max_tokens=512,
                          system="Extract ONLY verifiable facts. Each fact must cite the given source ID. Return valid JSON only.")
        parsed = extract_json(resp)
        raw_facts = parsed.get("facts", [])

        for rf in raw_facts[:3]:
            if isinstance(rf, dict) and rf.get("statement"):
                sid = rf.get("source_id", sids[0] if sids else "[1.1]")
                # Lock to actual source
                src = next((s for s in sources if s.get("id") == sid), {})
                if not src and sids:
                    sid = sids[0]
                    src = next((s for s in sources if s.get("id") == sid), {})

                fact_counter += 1
                all_facts.append({
                    "fact_id": f"F-{fact_counter:03d}",
                    "statement": str(rf["statement"]),
                    "source_id": sid,
                    "source_url": src.get("url", "unknown"),
                    "source_title": src.get("title", "Source"),
                    "sub_question": sub_q,
                    "relevance": rf.get("relevance", "medium"),
                })

    # --- Stage 2: Verifier Agent (qwen3:4b) cross-checks facts ---
    verified_facts: List[LockedFact] = []
    if all_facts and not state.get("dry_run"):
        logger.info(f"  >> Verifier Agent (qwen3:4b) cross-checking {len(all_facts)} facts...")
        facts_summary = "\n".join(
            f"{i+1}. {f['statement'][:100]} [source: {f['source_id']}]"
            for i, f in enumerate(all_facts[:10])
        )
        verify_prompt = f"""You are a fact-checking expert. Review these extracted facts for accuracy and relevance.

RESEARCH TOPIC: {state.get('query', '')}

FACTS TO VERIFY:
{facts_summary}

For each fact, respond with a JSON object:
{{
  "verified": [
    {{"index": 1, "status": "confirmed", "confidence": "high"}},
    {{"index": 2, "status": "needs_context", "confidence": "medium"}}
  ]
}}

Status options: confirmed, needs_context, questionable
Confidence: high, medium, low"""

        verify_resp = call_ollama(verify_prompt, model=LLM_SCORER, timeout=45, max_tokens=512,
                                 system="You are a meticulous fact-checker. Verify claims for accuracy. Return valid JSON.")
        verify_parsed = extract_json(verify_resp)
        verifications = {v.get("index", 0): v for v in verify_parsed.get("verified", [])}

        for i, fact in enumerate(all_facts):
            v = verifications.get(i + 1, {"status": "confirmed", "confidence": "medium"})
            fact["verified_by"] = "Verifier Agent (qwen3:4b)"
            fact["verification_status"] = v.get("status", "confirmed")
            fact["verification_confidence"] = v.get("confidence", "medium")
            # Only keep confirmed or needs_context facts
            if v.get("status") != "questionable":
                verified_facts.append(fact)
            else:
                logger.info(f"  >> Verifier rejected: {fact['fact_id']} - {fact['statement'][:50]}")

        logger.info(f"  >> Verifier Agent: {len(verified_facts)}/{len(all_facts)} facts passed verification")
    else:
        verified_facts = all_facts
        for f in verified_facts:
            f["verified_by"] = "Verifier Agent (qwen3:4b)"

    # Identify knowledge gaps
    answered_questions = {f["sub_question"] for f in verified_facts + existing_facts}
    plan = state.get("research_plan", [])
    gaps = [sq["question"] for sq in plan
            if sq["question"] not in answered_questions
            and sum(1 for f in verified_facts + existing_facts if f["sub_question"] == sq["question"]) < 2]

    dur = time.time() - t0
    logger.info(f"[6/8 extract_facts] {len(verified_facts)} verified facts, {len(gaps)} gaps in {dur:.1f}s")
    _log_agent("extract_facts", f"-> {len(verified_facts)} verified facts")
    return {
        "locked_facts": verified_facts,
        "knowledge_gaps": gaps,
        "timings": {"extract_facts": dur},
        "agent_contributions": [
            {"node": "extract_facts", "agent": "Extractor Agent", "model": LLM_EXTRACTOR, "output": f"{len(all_facts)} raw facts"},
            {"node": "extract_facts", "agent": "Verifier Agent",  "model": LLM_SCORER, "output": f"{len(verified_facts)}/{len(all_facts)} verified"},
        ],
    }


# ===========================================================================
# ADAPTIVE QUALITY GATE  (loop controller)
# ===========================================================================

def quality_gate(state: ResearchStateV6) -> Literal["generate_plan", "compose_report"]:
    """Node 7: Scorer Agent (qwen3:4b) decides whether to loop or finalize."""
    iteration = state.get("iteration", 1)
    max_iter = state.get("max_iterations", 4)
    facts = state.get("locked_facts", [])
    gaps = state.get("knowledge_gaps", [])
    plan = state.get("research_plan", [])
    query = state.get("query", "")

    # --- Scorer Agent (qwen3:4b): LLM-based quality assessment ---
    if not state.get("dry_run") and facts:
        logger.info(f"  >> Scorer Agent (qwen3:4b) evaluating research quality...")
        facts_summary = "\n".join(f"- {f['statement'][:80]}" for f in facts[:8])
        gaps_summary = "\n".join(f"- {g[:60]}" for g in gaps[:5]) if gaps else "None"

        score_prompt = f"""You are a strict research quality assessor. Score this research 1-10.

QUERY: {query}
ITERATION: {iteration} of {max_iter}

FACTS COLLECTED ({len(facts)} total):
{facts_summary}

REMAINING GAPS:
{gaps_summary}

SCORING RUBRIC:
- 1-3: Very few facts, major gaps, minimal coverage
- 4-5: Some facts but missing key angles, needs more iterations
- 6-7: Good coverage with minor gaps, acceptable quality
- 8-9: Excellent coverage, diverse sources, specific data points, minimal gaps
- 10: Perfect comprehensive coverage with no gaps

IMPORTANT: If there are {len(facts)} facts and only {len(gaps)} gaps, score SHOULD be higher than 5.
If facts >= 10 and gaps <= 2, score should be 7+.
If facts >= 20 and gaps == 0, score should be 9+.

Return ONLY this JSON (nothing else): {{"score": <number>, "rationale": "reason"}}"""

        score_resp = call_ollama(score_prompt, model=LLM_SCORER, timeout=30, max_tokens=256,
                                system="Score research quality 1-10. Return ONLY JSON with score and rationale. "
                                       "Be proportional: more facts + fewer gaps = higher score.")
        score_parsed = extract_json(score_resp)
        score = None

        # Try JSON-parsed score first
        if isinstance(score_parsed.get("score"), (int, float)):
            score = float(score_parsed["score"])
        elif "error" not in score_parsed:
            try:
                score = float(score_parsed.get("score", 0))
            except (ValueError, TypeError):
                pass

        # Fallback: regex extraction from raw LLM output
        if score is None:
            num_match = re.search(r'(?:score|rating)[^\d]*(\d+(?:\.\d+)?)', score_resp, re.IGNORECASE)
            if not num_match:
                num_match = re.search(r'\b(\d+(?:\.\d+)?)\s*/\s*10', score_resp)
            if num_match:
                score = float(num_match.group(1))
                logger.info(f"  >> Scorer: extracted score {score} via regex fallback")

        # Ultimate fallback: metric-based proportional score
        if score is None:
            coverage = min(1.0, len(facts) / 15)
            gap_penalty = max(0, 1.0 - len(gaps) * 0.15)
            tool_diversity = min(1.0, len(set(state.get('tools_used', []))) / 3)
            score = round((coverage * 4 + gap_penalty * 3 + tool_diversity * 3), 1)
            logger.info(f"  >> Scorer: LLM parse failed, using metric score={score}")

        score = min(10.0, max(0.0, score))
        rationale = score_parsed.get("rationale", "")
        logger.info(f"  >> Scorer Agent verdict: {score}/10 - {rationale[:60]}")
    else:
        # Fallback: metric-based scoring for dry run
        total_sqs = max(len(plan), 1)
        answered = len({f["sub_question"] for f in facts})
        coverage = answered / total_sqs
        fact_density = min(1.0, len(facts) / 10)
        tool_diversity = len(set(state.get("tools_used", []))) / max(len(TOOL_MATRIX.get(state.get("topic_domain", "general"), {}).get("primary", ["x"])), 1)
        tool_diversity = min(1.0, tool_diversity)
        score = round((coverage * 4 + fact_density * 3 + tool_diversity * 3), 1)

    _log_agent("quality_gate", f"score={score}/10")
    logger.info(
        f"[7/8 quality_gate] iter={iteration}/{max_iter} | "
        f"facts={len(facts)} | gaps={len(gaps)} | score={score}/10"
    )

    # Always finalize at max iterations
    if iteration >= max_iter:
        logger.info("  -> FINALIZE (max iterations reached)")
        return "compose_report"

    # Need at least 1 full iteration
    if iteration < 1:
        logger.info("  -> CONTINUE (min iterations)")
        return "generate_plan"

    # Pass if quality is good enough
    if score >= 7.0 and len(facts) >= 6:
        logger.info(f"  -> FINALIZE (score={score} >= 7.0, facts={len(facts)} >= 6)")
        return "compose_report"

    # Continue if there are gaps
    if gaps:
        logger.info(f"  -> CONTINUE ({len(gaps)} gaps remaining)")
        return "generate_plan"

    # Default: finalize if no gaps even with low score
    logger.info("  -> FINALIZE (no gaps)")
    return "compose_report"


# ===========================================================================
# REPORT COMPOSER  (publication-quality .md with citation validator)
# ===========================================================================

def compose_report(state: ResearchStateV6) -> dict:
    """Node 8: Generate publication-quality .md report."""
    t0 = time.time()
    query = state["query"]
    domain = state.get("topic_domain", "general")
    facts = state.get("locked_facts", [])
    sources = state.get("sources_registry", [])
    iteration = state.get("iteration", 1)
    tools_used = list(set(state.get("tools_used", [])))

    if state.get("dry_run"):
        report = _build_dry_run_report(query, domain, facts, sources, tools_used, iteration)
        logger.info(f"[8/8 compose_report] DRY RUN -> {len(report)} chars")
        return {
            "final_report": report,
            "quality_score": 8.5,
            "timings": {"compose_report": time.time() - t0},
        }

    # Build fact pool for the LLM (pre-formatted with locked citations)
    fact_pool = "\n".join(
        f"- {f['fact_id']}: {f['statement']} {f['source_id']}"
        for f in facts
    )

    # Source reference list
    source_list = "\n".join(
        f"{s['id']} {s.get('title', 'Source')}: {s.get('url', '')}"
        for s in sources[:40]
    )

    # Sub-questions as section headings
    plan = state.get("research_plan", [])
    sections_hint = "\n".join(f"- {sq['question']}" for sq in plan[:7])

    prompt = f"""Write a comprehensive, in-depth, publication-quality research report in Markdown.

RESEARCH QUERY: {query}
DOMAIN: {domain}
TOTAL FACTS AVAILABLE: {len(facts)}

FACT POOL (use ONLY these facts, cite using the source IDs shown):
{fact_pool}

SOURCE REFERENCES:
{source_list}

SECTION STRUCTURE (use as subsection headings under Detailed Analysis):
{sections_hint}

CRITICAL RULES:
1. Use ONLY facts from the FACT POOL above
2. Cite using the EXACT source IDs shown (e.g., [1.1], [2.1]) - do NOT invent citations
3. Every claim MUST have a citation from the fact pool
4. Write at MINIMUM 2500 words - this is a DEEP RESEARCH report, not a summary
5. Use ## for main sections, ### for subsections
6. Bold key terms, statistics, and findings
7. Include analysis, context, and implications - not just restating facts
8. Connect findings across subsections where relevant
9. Add a ### Implications subsection discussing real-world impact

REQUIRED SECTIONS:
## Executive Summary (6-10 sentences providing a comprehensive overview)
## Key Findings (detailed bullet points with specific data and citations)
## Detailed Analysis (one ### subsection per research angle - each should be 300+ words with in-depth discussion)
## Implications and Strategic Outlook (forward-looking analysis)
## Methodology (tools used, iterations performed)
## Limitations (gaps, caveats, areas for further research)

IMPORTANT: This is a DEEP RESEARCH report. Each analysis subsection should:
- State the finding with data
- Explain the context and significance
- Discuss implications and connections to other findings
- Reference multiple facts where possible

DO NOT include a Sources section (it will be appended automatically).
Write the complete, detailed report now:"""

    draft = call_ollama(prompt, model=LLM_COMPOSER, timeout=300, max_tokens=10240,
                       system="You are a senior research analyst writing an exhaustive deep-dive report. "
                              "Write detailed, data-driven analysis with multiple paragraphs per section. "
                              "Use ONLY the provided facts and citations. Never invent data. "
                              "Target 2500-4000 words. Be thorough and analytical, not superficial.")

    # Clean up
    draft = draft.replace('\\n', '\n')
    draft = re.sub(r'\n{3,}', '\n\n', draft)

    # Post-generation citation validator
    valid_sids = {s["id"] for s in sources}
    orphan_count = 0
    def _validate_citation(match):
        nonlocal orphan_count
        cit = match.group(0)
        if cit not in valid_sids:
            orphan_count += 1
            return ""  # Remove orphan citation
        return cit
    draft = re.sub(r'\[\d+\.\d+\]', _validate_citation, draft)
    if orphan_count:
        logger.warning(f"[compose_report] Removed {orphan_count} orphan citations")

    # Build full report with metadata and sources
    report = _build_full_report(query, domain, draft, facts, sources, tools_used, iteration, t0)

    # Compute quality
    fact_coverage = min(1.0, len(facts) / 10)
    word_count = len(report.split())
    length_score = min(1.0, word_count / 800)
    quality = round((fact_coverage * 4 + length_score * 3 + min(1.0, len(tools_used)/3) * 3), 1)

    dur = time.time() - t0
    logger.info(f"[8/8 compose_report] {len(report)} chars, {word_count} words, quality={quality}/10 in {dur:.1f}s")
    return {
        "final_report": report,
        "quality_score": quality,
        "timings": {"compose_report": dur},
    }


def _build_full_report(query, domain, draft, facts, sources, tools_used, iteration, t0):
    """Assemble the complete .md report with header and sources."""
    title = query[:80].replace('\n', ' ')
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dur = time.time() - t0

    header = f"""# {title}

> **Research Query:** {query}
> **Domain:** {domain.title()} | **Date:** {ts} | **Sources:** {len(sources)} | **Facts:** {len(facts)}

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Key Findings](#key-findings)
3. [Detailed Analysis](#detailed-analysis)
4. [Methodology](#methodology)
5. [Limitations](#limitations)
6. [Sources & References](#sources--references)

---

"""

    # Build sources section with numbered refs
    sources_md = "\n\n---\n\n## Sources & References\n\n"
    by_tool: Dict[str, list] = {}
    for s in sources:
        tool = s.get("tool", "web_search")
        by_tool.setdefault(tool, []).append(s)

    for tool, tool_sources in by_tool.items():
        sources_md += f"### {tool.replace('_', ' ').title()}\n"
        for s in tool_sources:
            url = s.get("url", "")
            title_s = s.get("title", "Source")
            sid = s.get("id", "")
            if url.startswith("http"):
                sources_md += f"{sid} [{title_s}]({url})\n"
            else:
                sources_md += f"{sid} **{title_s}** (Retrieved via {tool})\n"
        sources_md += "\n"

    # Methodology section with agent collaboration table
    agents_md = "\n".join(
        f"| {name} | {info['llm']} | {info['role']} |"
        for name, info in AGENT_REGISTRY.items()
    )
    methodology = f"""

## Methodology

### Pipeline Parameters

| Parameter | Value |
|-----------|-------|
| Tools Used | {', '.join(tools_used)} |
| Iterations | {iteration} |
| Facts Extracted | {len(facts)} |
| Sources Analyzed | {len(sources)} |
| Domain | {domain.title()} |
| Generated | {ts} |

### Multi-Agent Collaboration

| Agent | LLM Model | Role |
|-------|----------|------|
{agents_md}

*This report was generated by Multi-Agent Orchestrator v6.0 using 3 LLMs as 6 specialized agents.*
"""

    return f"{header}{draft}\n{methodology}\n{sources_md}"


def _build_dry_run_report(query, domain, facts, sources, tools_used, iteration):
    """Generate a complete dry-run report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    facts_md = "\n".join(f"- **{f['fact_id']}:** {f['statement']} {f['source_id']}" for f in facts)
    sources_md = "\n".join(f"{s['id']} [{s.get('title', 'Source')}]({s.get('url', '')})" for s in sources)
    tools_str = ", ".join(tools_used)

    return f"""# {query[:80]}

> **Research Query:** {query}
> **Domain:** {domain.title()} | **Date:** {ts} | **Sources:** {len(sources)} | **Facts:** {len(facts)}

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Key Findings](#key-findings)
3. [Detailed Analysis](#detailed-analysis)
4. [Methodology](#methodology)
5. [Limitations](#limitations)
6. [Sources & References](#sources--references)

---

## Executive Summary

[DRY RUN] This report investigates: **{query}**. The research pipeline classified
this as a **{domain}** domain query and decomposed it into multiple sub-questions.
Using {len(tools_used)} specialized tools ({tools_str}), the system gathered data
across {iteration} iteration(s) and extracted {len(facts)} verified facts from
{len(sources)} sources.

## Key Findings

{facts_md}

## Detailed Analysis

### Current State
[DRY RUN] Simulated analysis of the current state based on gathered research data.
Key indicators show significant trends with measurable metrics.

### Expert Perspectives
[DRY RUN] Expert analysis compiled from multiple authoritative sources confirms
the findings above with additional nuance.

### Challenges and Risks
[DRY RUN] Risk assessment identifies key concerns that stakeholders should monitor.

## Methodology

### Pipeline Parameters

| Parameter | Value |
|-----------|-------|
| Tools Used | {tools_str} |
| Iterations | {iteration} |
| Facts Extracted | {len(facts)} |
| Sources Analyzed | {len(sources)} |
| Domain | {domain.title()} |
| Mode | DRY RUN |
| Generated | {ts} |

### Multi-Agent Collaboration

| Agent | LLM Model | Role |
|-------|----------|------|
| Classifier Agent | llama3.2 | Topic detection & intent routing |
| Strategist Agent | qwen3:8b | Research planning & decomposition |
| Extractor Agent | llama3.2 | Fact extraction from raw content |
| Verifier Agent | qwen3:4b | Cross-checks facts & verifies claims |
| Scorer Agent | qwen3:4b | Quality assessment & gap analysis |
| Composer Agent | qwen3:8b | Report synthesis & long-form writing |

*This report was generated by Multi-Agent Orchestrator v6.0 (dry run mode).*

## Limitations

- [DRY RUN] No real API calls were made
- Data shown is simulated for pipeline validation

---

## Sources & References

{sources_md}
"""


# ===========================================================================
# LANGGRAPH PIPELINE  (8-node iterative graph)
# ===========================================================================

def build_v6_graph():
    """Build the 8-node LangGraph research pipeline."""
    from langgraph.graph import StateGraph, END

    graph = StateGraph(ResearchStateV6)

    # Register all 8 nodes
    graph.add_node("classify_topic", classify_topic)
    graph.add_node("generate_plan", generate_plan)
    graph.add_node("execute_searches", execute_searches)
    graph.add_node("enrich_content", enrich_content)
    graph.add_node("semantic_chunk", semantic_chunk)
    graph.add_node("extract_facts", extract_facts)
    graph.add_node("compose_report", compose_report)

    # Linear edges
    graph.set_entry_point("classify_topic")
    graph.add_edge("classify_topic", "generate_plan")
    graph.add_edge("generate_plan", "execute_searches")
    graph.add_edge("execute_searches", "enrich_content")
    graph.add_edge("enrich_content", "semantic_chunk")
    graph.add_edge("semantic_chunk", "extract_facts")

    # Conditional: quality gate decides loop vs finalize
    graph.add_conditional_edges(
        "extract_facts",
        quality_gate,
        {
            "generate_plan": "generate_plan",
            "compose_report": "compose_report",
        }
    )

    graph.add_edge("compose_report", END)

    return graph.compile()


# ===========================================================================
# DEEP RESEARCH ORCHESTRATOR V6  (top-level class)
# ===========================================================================

class DeepResearchOrchestratorV6:
    """Google Deep Research-style universal research orchestrator."""

    def __init__(self, config: Optional[V6Config] = None):
        self.config = config or V6Config()
        self.base_dir = Path(_PROJECT_ROOT)
        logger.info("DeepResearchOrchestratorV6 initialized")

    async def run(self, query: str, dry_run: bool = False,
                  config_overrides: Optional[dict] = None) -> dict:
        """Run the full deep research pipeline."""
        start_time = time.time()
        logger.info(f"{'='*65}")
        logger.info(f"  DEEP RESEARCH v6.0 | Query: {query[:60]}...")
        logger.info(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        logger.info(f"  Max Iterations: {self.config.max_iterations}")
        logger.info(f"{'='*65}")

        # Build initial state
        initial_state: dict = {
            "query": query,
            "topic_domain": None,
            "research_plan": [],
            "search_results": [],
            "enriched_chunks": [],
            "locked_facts": [],
            "knowledge_gaps": [],
            "final_report": "",
            "quality_score": 0.0,
            "iteration": 0,
            "max_iterations": self.config.max_iterations,
            "tools_used": [],
            "sources_registry": [],
            "timings": {},
            "error_count": 0,
            "agent_contributions": [],
            "dry_run": dry_run,
        }

        # Build and run graph
        graph = build_v6_graph()

        import concurrent.futures
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                final_state = await loop.run_in_executor(
                    pool, lambda: graph.invoke(initial_state)
                )
        except RuntimeError:
            final_state = graph.invoke(initial_state)

        total_time = time.time() - start_time

        # Save report
        report = final_state.get("final_report", "")
        report_path = self._save_report(query, report, final_state)

        logger.info(f"{'='*65}")
        logger.info(f"  RESEARCH COMPLETE")
        logger.info(f"  Duration: {total_time:.1f}s")
        logger.info(f"  Quality:  {final_state.get('quality_score', 0)}/10")
        logger.info(f"  Facts:    {len(final_state.get('locked_facts', []))}")
        logger.info(f"  Sources:  {len(final_state.get('sources_registry', []))}")
        logger.info(f"  Report:   {report_path}")
        logger.info(f"{'='*65}")

        # --- Post-pipeline export workflow ---
        export_results = {}

        # 1. Export to Google Sheets
        if self.config.spreadsheet_id and not dry_run:
            try:
                sheet_status = self._export_to_sheets(query, final_state, total_time, report_path)
                export_results["sheets"] = sheet_status
                logger.info(f"  Sheets:   {sheet_status.get('status', 'unknown')}")
            except Exception as e:
                logger.error(f"  Sheets:   FAILED - {e}")
                export_results["sheets"] = {"status": "failed", "error": str(e)}
        elif self.config.spreadsheet_id and dry_run:
            logger.info(f"  Sheets:   [DRY RUN] Would export to {self.config.spreadsheet_id}")
            export_results["sheets"] = {"status": "dry_run", "spreadsheet_id": self.config.spreadsheet_id}

        # 2. Generate LLM summary + Send email
        if self.config.email_to and not dry_run:
            try:
                summary_html = self._generate_email_summary(query, report, final_state, total_time)
                email_status = self._send_email_summary(query, summary_html, final_state)
                export_results["email"] = email_status
                logger.info(f"  Email:    {email_status.get('status', 'unknown')}")

                # Update Email_Status in Sheet row (from "Pending" to "Sent"/"Failed")
                sheet_info = export_results.get("sheets", {})
                if sheet_info.get("row_number") and hasattr(self, '_sheets_svc') and self._sheets_svc:
                    try:
                        row_num = sheet_info["row_number"]
                        tab = sheet_info["tab"]
                        sid = sheet_info["spreadsheet_id"]
                        email_cell_value = "Sent" if email_status.get("status") == "sent" else f"Failed: {email_status.get('error', 'unknown')[:50]}"
                        self._sheets_svc.spreadsheets().values().update(
                            spreadsheetId=sid,
                            range=f"{tab}!M{row_num}",
                            valueInputOption='RAW',
                            body={'values': [[email_cell_value]]}
                        ).execute()
                        logger.info(f"  Sheets:   Email_Status updated to '{email_cell_value}'")
                    except Exception as ue:
                        logger.warning(f"  Sheets:   Email_Status update failed: {ue}")
            except Exception as e:
                logger.error(f"  Email:    FAILED - {e}")
                export_results["email"] = {"status": "failed", "error": str(e)}
        elif self.config.email_to and dry_run:
            logger.info(f"  Email:    [DRY RUN] Would send summary to {self.config.email_to}")
            export_results["email"] = {"status": "dry_run", "to": self.config.email_to}

        return {
            "report": report,
            "report_path": str(report_path) if report_path else None,
            "quality_score": final_state.get("quality_score", 0),
            "metadata": {
                "topic_domain": final_state.get("topic_domain"),
                "iterations": final_state.get("iteration", 0),
                "facts_count": len(final_state.get("locked_facts", [])),
                "sources_count": len(final_state.get("sources_registry", [])),
                "tools_used": list(set(final_state.get("tools_used", []))),
                "agents_used": list(AGENT_REGISTRY.keys()),
                "timings": final_state.get("timings", {}),
                "processing_time_seconds": total_time,
                "dry_run": dry_run,
                "export": export_results,
            },
            "facts": final_state.get("locked_facts", []),
            "sources": final_state.get("sources_registry", []),
        }

    def _save_report(self, query: str, report: str, state: dict) -> Optional[Path]:
        """Save the .md report to reports/ directory."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = re.sub(r'[^\w\s-]', '', query[:30]).strip().replace(' ', '_')
            rd = self.base_dir / "reports" / f"v6_research_{slug}_{ts}"
            rd.mkdir(parents=True, exist_ok=True)

            # Save main report
            report_path = rd / "research_report.md"
            report_path.write_text(report, encoding="utf-8")

            # Save metadata
            meta = {
                "query": query,
                "domain": state.get("topic_domain"),
                "quality_score": state.get("quality_score", 0),
                "iterations": state.get("iteration", 0),
                "facts_count": len(state.get("locked_facts", [])),
                "sources_count": len(state.get("sources_registry", [])),
                "tools_used": list(set(state.get("tools_used", []))),
                "timings": state.get("timings", {}),
                "generated_at": datetime.now().isoformat(),
                "version": "6.0",
            }
            (rd / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

            # Save facts
            (rd / "facts.json").write_text(
                json.dumps(state.get("locked_facts", []), indent=2, default=str),
                encoding="utf-8"
            )

            logger.info(f"[save] Report saved to: {rd}")
            return report_path
        except Exception as e:
            logger.error(f"[save] Failed: {e}")
            return None

    # ===================================================================
    # EXPORT: Google Sheets  (appends research row)
    # ===================================================================

    def _import_sheets_service(self):
        """Dynamically import SheetsService from Marketing module."""
        marketing_path = str(self.base_dir / "Marketing")
        if marketing_path not in sys.path:
            sys.path.insert(0, marketing_path)
        sheets_path = str(self.base_dir / "Marketing" / "services" / "sheets_service.py")
        spec = importlib.util.spec_from_file_location("sheets_svc", sheets_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.get_sheets_service(spreadsheet_id=self.config.spreadsheet_id)

    def _export_to_sheets(self, query: str, state: dict, duration: float, report_path=None) -> dict:
        """Append a research summary row to Google Sheets."""
        logger.info(f"[export] Updating Google Sheet: {self.config.spreadsheet_id}")

        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        marketing_path = str(self.base_dir / "Marketing")
        if marketing_path not in sys.path:
            sys.path.insert(0, marketing_path)
        from config import settings

        creds = Credentials.from_authorized_user_file(
            settings.TOKEN_PATH,
            list(set(settings.GMAIL_SCOPES + settings.SHEETS_SCOPES))
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())

        svc = build('sheets', 'v4', credentials=creds)
        sid = self.config.spreadsheet_id
        tab = self.config.sheet_tab

        # Auto-create tab if missing
        try:
            meta = svc.spreadsheets().get(spreadsheetId=sid).execute()
            tabs = [s['properties']['title'] for s in meta.get('sheets', [])]
            if tab not in tabs:
                logger.info(f"[export] Creating tab '{tab}'...")
                svc.spreadsheets().batchUpdate(spreadsheetId=sid, body={
                    "requests": [{"addSheet": {"properties": {"title": tab}}}]
                }).execute()
        except Exception as e:
            logger.warning(f"[export] Tab check failed: {e}")

        # Ensure header row exists
        header = [
            "Timestamp", "Query", "Domain", "Quality_Score",
            "Facts", "Sources", "Tools_Used", "Agents_Used",
            "Iterations", "Duration_Sec", "Sub_Questions",
            "Report_Path", "Email_Status"
        ]
        try:
            existing = svc.spreadsheets().values().get(
                spreadsheetId=sid, range=f"{tab}!A1:M1"
            ).execute().get('values', [])
            if not existing or existing[0] != header:
                svc.spreadsheets().values().update(
                    spreadsheetId=sid, range=f"{tab}!A1:M1",
                    valueInputOption='RAW', body={'values': [header]}
                ).execute()
        except Exception:
            svc.spreadsheets().values().update(
                spreadsheetId=sid, range=f"{tab}!A1:M1",
                valueInputOption='RAW', body={'values': [header]}
            ).execute()

        # Append data row
        tools = ', '.join(sorted(set(state.get('tools_used', []))))
        agents = ', '.join(AGENT_REGISTRY.keys())
        plan = state.get('research_plan', [])
        sub_qs = '; '.join(sq.get('question', '')[:80] for sq in plan[:7])
        rp = str(report_path) if report_path else ""
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query[:100],
            state.get('topic_domain', 'general'),
            str(state.get('quality_score', 0)),
            str(len(state.get('locked_facts', []))),
            str(len(state.get('sources_registry', []))),
            tools,
            agents,
            str(state.get('iteration', 0)),
            str(round(duration, 1)),
            sub_qs[:500],
            rp,
            "Pending",
        ]

        svc.spreadsheets().values().append(
            spreadsheetId=sid, range=f"{tab}!A:M",
            valueInputOption='RAW', insertDataOption='INSERT_ROWS',
            body={'values': [row]}
        ).execute()

        # Find the row number we just appended (for later email status update)
        try:
            all_rows = svc.spreadsheets().values().get(
                spreadsheetId=sid, range=f"{tab}!A:A"
            ).execute().get('values', [])
            appended_row = len(all_rows)  # 1-indexed, last row
        except Exception:
            appended_row = None

        logger.info(f"[export] Row {appended_row} appended to '{tab}' in sheet {sid}")
        self._sheets_svc = svc  # Store for email status update
        return {"status": "success", "tab": tab, "spreadsheet_id": sid,
                "row_number": appended_row}

    # ===================================================================
    # EXPORT: Email Summary  (LLM-generated concise summary via Gmail)
    # ===================================================================

    def _import_gmail_service(self):
        """Dynamically import GmailService from Marketing module."""
        marketing_path = str(self.base_dir / "Marketing")
        if marketing_path not in sys.path:
            sys.path.insert(0, marketing_path)
        gmail_path = str(self.base_dir / "Marketing" / "services" / "gmail_service.py")
        spec = importlib.util.spec_from_file_location("gmail_svc", gmail_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.get_gmail_service()

    def _generate_email_summary(self, query: str, report: str,
                                state: dict, duration: float) -> str:
        """Use Composer Agent (qwen3:8b) to generate a concise HTML email summary."""
        logger.info("[email] Composer Agent generating email summary...")

        domain = state.get('topic_domain', 'general')
        facts_count = len(state.get('locked_facts', []))
        sources_count = len(state.get('sources_registry', []))
        quality = state.get('quality_score', 0)
        tools = ', '.join(sorted(set(state.get('tools_used', []))))

        # Extract key findings from report for LLM context
        report_excerpt = report[:3000]

        prompt = f"""Write a concise EMAIL SUMMARY (not the full report) of this research.

RESEARCH QUERY: {query}
DOMAIN: {domain}
QUALITY SCORE: {quality}/10
FACTS: {facts_count} | SOURCES: {sources_count}
DURATION: {duration:.1f}s

FULL REPORT EXCERPT:
{report_excerpt}

WRITE A SUMMARY EMAIL BODY (5-8 sentences max) that includes:
1. What was researched
2. Key findings (top 3-4 bullet points)
3. Quality assessment
4. One-line recommendation or next step

Write in PLAIN TEXT (will be converted to HTML). Be concise and actionable.
Do NOT include the full report. Do NOT include greetings or sign-offs."""

        summary_text = call_ollama(
            prompt, model=LLM_COMPOSER, timeout=60, max_tokens=1024,
            system="You are a senior analyst writing a brief email summary. "
                   "Be concise, data-driven, and actionable. No fluff."
        )

        # Build rich HTML email
        html = f"""
<div style="font-family:Arial,sans-serif;max-width:650px;margin:0 auto;">
  <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;padding:20px 24px;border-radius:12px 12px 0 0;">
    <h1 style="margin:0;font-size:20px;">Deep Research Report Summary</h1>
    <p style="margin:6px 0 0;opacity:0.85;font-size:14px;">{query[:80]}</p>
  </div>

  <div style="background:#f8f9fa;padding:16px 24px;border-bottom:1px solid #e0e0e0;">
    <table style="width:100%;font-size:13px;border-collapse:collapse;">
      <tr>
        <td style="padding:4px 0;"><strong>Domain:</strong> {domain.title()}</td>
        <td style="padding:4px 0;"><strong>Quality:</strong> {quality}/10</td>
      </tr>
      <tr>
        <td style="padding:4px 0;"><strong>Facts:</strong> {facts_count}</td>
        <td style="padding:4px 0;"><strong>Sources:</strong> {sources_count}</td>
      </tr>
      <tr>
        <td style="padding:4px 0;"><strong>Tools:</strong> {tools}</td>
        <td style="padding:4px 0;"><strong>Duration:</strong> {duration:.1f}s</td>
      </tr>
    </table>
  </div>

  <div style="padding:20px 24px;background:white;">
    <h2 style="color:#333;font-size:16px;margin-top:0;">Summary</h2>
    <div style="color:#444;font-size:14px;line-height:1.6;white-space:pre-wrap;">{summary_text}</div>
  </div>

  <div style="background:#f0f0f0;padding:12px 24px;border-radius:0 0 12px 12px;font-size:11px;color:#888;">
    Multi-Agent Orchestrator v6.0 | 3 LLMs x 6 Agents | {datetime.now().strftime('%Y-%m-%d %H:%M')}
  </div>
</div>"""

        logger.info(f"[email] Summary generated: {len(summary_text)} chars")
        return html

    def _send_email_summary(self, query: str, html_body: str, state: dict) -> dict:
        """Send the summary email via Gmail OAuth2."""
        logger.info(f"[email] Sending summary to: {self.config.email_to}")

        gmail = self._import_gmail_service()

        domain = state.get('topic_domain', 'general').title()
        quality = state.get('quality_score', 0)
        subject = f"[Deep Research] {domain}: {query[:50]} (Score: {quality}/10)"

        result = gmail.send_email(
            to=self.config.email_to,
            subject=subject,
            body=html_body,
            cc=self.config.email_cc if self.config.email_cc else None,
        )

        if result.get('success'):
            logger.info(f"[email] Sent successfully (ID: {result.get('message_id')})")
            return {"status": "sent", "message_id": result.get('message_id')}
        else:
            logger.error(f"[email] Send failed: {result.get('error')}")
            return {"status": "failed", "error": result.get('error')}


# ===========================================================================
# CLI
# ===========================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="Deep Research Orchestrator v6.0")
    p.add_argument("--query", "-q", type=str, required=True,
                   help="Research query (any topic)")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip real API calls (validate pipeline)")
    p.add_argument("--max-iterations", "-n", type=int, default=4,
                   help="Max research iterations (default: 4)")
    p.add_argument("--email", "-e", type=str, default="",
                   help="Send LLM summary email to this address")
    p.add_argument("--email-cc", type=str, default="",
                   help="CC email address")
    p.add_argument("--spreadsheet-id", "-s", type=str, default="",
                   help="Google Sheets ID to export results to")
    p.add_argument("--sheet-tab", type=str, default="V6_Research",
                   help="Sheet tab name (default: V6_Research)")
    args = p.parse_args()

    config = V6Config(
        max_iterations=args.max_iterations,
        email_to=args.email,
        email_cc=args.email_cc,
        spreadsheet_id=args.spreadsheet_id,
        sheet_tab=args.sheet_tab,
    )
    orch = DeepResearchOrchestratorV6(config=config)
    result = asyncio.run(orch.run(args.query, dry_run=args.dry_run))

    print(json.dumps({
        "query": args.query,
        "domain": result["metadata"]["topic_domain"],
        "quality_score": result["quality_score"],
        "facts": result["metadata"]["facts_count"],
        "sources": result["metadata"]["sources_count"],
        "tools": result["metadata"]["tools_used"],
        "agents": result["metadata"]["agents_used"],
        "iterations": result["metadata"]["iterations"],
        "duration_sec": round(result["metadata"]["processing_time_seconds"], 1),
        "report_path": result["report_path"],
        "export": result["metadata"].get("export", {}),
        "dry_run": result["metadata"]["dry_run"],
    }, indent=2))


if __name__ == "__main__":
    main()
