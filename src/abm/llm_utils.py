"""
Ollama LLM utility for the ABM sandbox.

Provides ``call_ollama()`` and ``check_model_available()`` used by
LLMCitizen, KOLAgent, and SimulationReportAgent.

Uses /api/chat format to support system prompts (date guards, roles).
Falls back to placeholder on connection error so sims don't crash.
"""

import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

# Default Ollama endpoint
OLLAMA_BASE_URL = "http://localhost:11434"


def call_ollama(prompt: str, model: str = "llama3.2:latest",
                system: str = "", max_tokens: int = 150,
                temperature: float = 0.7, timeout: int = 30) -> str:
    """Call Ollama's chat endpoint and return the response text.

    Uses /api/chat (not /api/generate) so system prompts are supported.
    Falls back to a placeholder on connection error so the simulation
    doesn't crash if Ollama is temporarily unavailable.

    Args:
        prompt: The user message to send.
        model: Ollama model name.
        system: Optional system prompt (date guards, roles).
        max_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
    """
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except requests.ConnectionError:
        logger.warning(f"Ollama not reachable at {OLLAMA_BASE_URL}")
        return f"[LLM unavailable — {model}]"
    except requests.Timeout:
        logger.warning(f"Ollama timeout for model {model}")
        return f"[LLM timeout — {model}]"
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return f"[LLM error — {model}]"


def check_model_available(model_name: str) -> bool:
    """Check if a model is pulled in the local Ollama instance."""
    try:
        resp = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=5,
        )
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return model_name in models or any(
                m.startswith(model_name.split(":")[0]) for m in models
            )
        return False
    except Exception:
        return False
