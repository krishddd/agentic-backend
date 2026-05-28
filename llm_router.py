"""
Multi-LLM Router — Central registry mapping each agent to its Ollama model.

Provides factory functions for CrewAI agents (via crewai.LLM) and
Marketing agents (via raw Ollama config dicts). All models run locally
through Ollama at http://localhost:11434.

Available models:
    qwen3:8b          → Complex reasoning (Investment Advisor, Marketing Analyst)
    llama3.2:latest   → General NLP (Research Analyst, Secretary)
    qwen3:4b          → Fast structured output (Financial Analyst, Data Generator)
    qwen3:8b          → Document parsing (SEC Filing Parser) [deepseek-ocr:3b not available]
    nomic-embed-text  → Embeddings (RAG for both modules)
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass
class AgentLLMConfig:
    """Configuration for a single agent's LLM assignment."""
    model: str
    temperature: float = 0.7
    role: str = "general"
    max_tokens: int = 4096
    description: str = ""


# =============================================================================
# Agent → Model Registry
# =============================================================================

AGENT_LLM_REGISTRY: Dict[str, AgentLLMConfig] = {
    # ── Crewai_module agents ──────────────────────────────────────────────
    "research_analyst": AgentLLMConfig(
        model=os.getenv("RESEARCH_LLM_MODEL", "llama3.2:latest"),
        temperature=0.7,
        role="research",
        description="Market research, news summarization, competitive analysis"
    ),
    "financial_analyst": AgentLLMConfig(
        model=os.getenv("FINANCIAL_LLM_MODEL", "qwen3:4b"),
        temperature=0.3,
        role="analysis",
        description="Financial metrics parsing, ratio calculations, structured output"
    ),
    "investment_advisor": AgentLLMConfig(
        model=os.getenv("ADVISOR_LLM_MODEL", "qwen3:8b"),
        temperature=0.5,
        role="synthesis",
        description="Investment report synthesis, quality assessment, recommendations"
    ),
    "sec_filing_parser": AgentLLMConfig(
        model=os.getenv("SEC_PARSER_LLM_MODEL", "qwen3:8b"),
        temperature=0.2,
        role="parsing",
        description="SEC 10-Q/10-K filing parsing, document extraction"
    ),

    # ── Marketing agents ──────────────────────────────────────────────────
    "secretary": AgentLLMConfig(
        model=os.getenv("SECRETARY_LLM_MODEL", "llama3.2:latest"),
        temperature=0.7,
        role="drafting",
        description="Email drafting, natural language generation, client communication"
    ),
    "analyst": AgentLLMConfig(
        model=os.getenv("ANALYST_LLM_MODEL", "qwen3:8b"),
        temperature=0.5,
        role="scoring",
        description="Deal scoring, risk detection, strategic analysis"
    ),
    "data_generator": AgentLLMConfig(
        model=os.getenv("GENERATOR_LLM_MODEL", "qwen3:4b"),
        temperature=0.8,
        role="generation",
        description="Synthetic sales lead data generation"
    ),

    # ── Shared ────────────────────────────────────────────────────────────
    "embeddings": AgentLLMConfig(
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest"),
        temperature=0.0,
        role="embeddings",
        description="Vector embeddings for RAG (ChromaDB, knowledge base)"
    ),
}


# =============================================================================
# Factory Functions
# =============================================================================

def get_agent_config(agent_name: str) -> AgentLLMConfig:
    """Get the LLM configuration for a specific agent.
    
    Args:
        agent_name: Name of the agent (e.g., 'research_analyst', 'secretary')
        
    Returns:
        AgentLLMConfig for the agent
        
    Raises:
        KeyError: If agent_name is not registered
    """
    if agent_name not in AGENT_LLM_REGISTRY:
        available = ", ".join(AGENT_LLM_REGISTRY.keys())
        raise KeyError(
            f"Unknown agent '{agent_name}'. Available: {available}"
        )
    return AGENT_LLM_REGISTRY[agent_name]


def get_crewai_llm(agent_name: str):
    """Get a CrewAI-compatible LLM instance for the given agent.
    
    Uses CrewAI's built-in Ollama support via the 'ollama/model-name' format.
    
    Args:
        agent_name: Name of the agent
        
    Returns:
        crewai.LLM instance configured for the agent's Ollama model
    """
    from crewai import LLM
    
    config = get_agent_config(agent_name)
    
    llm = LLM(
        model=f"ollama/{config.model}",
        base_url=OLLAMA_BASE_URL,
        temperature=config.temperature,
    )
    
    logger.info(
        f"[LLM Router] Created CrewAI LLM for '{agent_name}': "
        f"ollama/{config.model} (temp={config.temperature})"
    )
    return llm


def get_ollama_model(agent_name: str) -> str:
    """Get the raw Ollama model name for an agent.
    
    Args:
        agent_name: Name of the agent
        
    Returns:
        Model name string (e.g., 'qwen3:8b')
    """
    return get_agent_config(agent_name).model


def get_ollama_config(agent_name: str) -> Dict[str, Any]:
    """Get the full Ollama configuration dict for an agent.
    
    Useful for Marketing's LLMService which needs model + temperature + base_url.
    
    Args:
        agent_name: Name of the agent
        
    Returns:
        Dict with model, temperature, base_url
    """
    config = get_agent_config(agent_name)
    return {
        "model": config.model,
        "temperature": config.temperature,
        "base_url": OLLAMA_BASE_URL,
        "role": config.role,
    }


# =============================================================================
# Health Check
# =============================================================================

def check_model_available(model_name: str) -> bool:
    """Check if a specific Ollama model is available locally.
    
    Args:
        model_name: Model name (e.g., 'qwen3:8b')
        
    Returns:
        True if model is available
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_names = [m["name"] for m in models]
            return model_name in available_names
        return False
    except requests.exceptions.ConnectionError:
        return False


def check_all_models() -> Dict[str, Dict[str, Any]]:
    """Check availability of all registered models.
    
    Returns:
        Dict mapping agent_name → {model, available, role, description}
    """
    results = {}
    
    # Get available models from Ollama
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_names = {m["name"] for m in models}
            ollama_running = True
        else:
            available_names = set()
            ollama_running = False
    except requests.exceptions.ConnectionError:
        available_names = set()
        ollama_running = False
    
    if not ollama_running:
        print("[FAIL] Ollama is NOT running. Start with: ollama serve")
        return {}
    
    print(f"\n{'='*65}")
    print(f"  MULTI-LLM ROUTER — Model Health Check")
    print(f"  Ollama: {OLLAMA_BASE_URL}")
    print(f"{'='*65}\n")
    
    # Deduplicate models (multiple agents may share the same model)
    unique_models = {}
    for agent_name, config in AGENT_LLM_REGISTRY.items():
        if config.model not in unique_models:
            unique_models[config.model] = []
        unique_models[config.model].append(agent_name)
    
    all_ok = True
    for model_name, agents in unique_models.items():
        available = model_name in available_names
        status = "[OK]" if available else "[MISSING]"
        agents_str = ", ".join(agents)
        print(f"  {status} {model_name:<25} -> {agents_str}")
        
        if not available:
            all_ok = False
            print(f"     -> Fix: ollama pull {model_name}")
        
        for agent_name in agents:
            config = AGENT_LLM_REGISTRY[agent_name]
            results[agent_name] = {
                "model": model_name,
                "available": available,
                "role": config.role,
                "description": config.description,
            }
    
    print(f"\n{'-'*65}")
    if all_ok:
        print(f"  [OK] All {len(unique_models)} models ready!")
    else:
        missing = sum(1 for m in unique_models if m not in available_names)
        print(f"  [WARN] {missing} model(s) missing. Pull them before running.")
    print(f"{'-'*65}\n")
    
    return results


def print_agent_registry():
    """Pretty-print the agent → model mapping."""
    print(f"\n{'='*65}")
    print(f"  AGENT -> LLM REGISTRY")
    print(f"{'='*65}")
    print(f"  {'Agent':<22} {'Model':<22} {'Role':<12} {'Temp'}")
    print(f"  {'-'*22} {'-'*22} {'-'*12} {'-'*5}")
    
    for agent_name, config in AGENT_LLM_REGISTRY.items():
        print(
            f"  {agent_name:<22} {config.model:<22} "
            f"{config.role:<12} {config.temperature}"
        )
    print(f"{'='*65}\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if "--check" in sys.argv:
        check_all_models()
    elif "--registry" in sys.argv:
        print_agent_registry()
    else:
        print("Usage:")
        print("  python llm_router.py --check     Check all model availability")
        print("  python llm_router.py --registry   Print agent → model mapping")
        print()
        print_agent_registry()
        check_all_models()
