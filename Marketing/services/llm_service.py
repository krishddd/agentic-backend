"""
LLM Service — Ollama-only, per-agent model support.

Each agent gets its own LLMService instance with a different Ollama model,
configured via the shared llm_router or per-agent settings.
"""
import json
import requests
from typing import Optional, Dict

from config import settings


class LLMService:
    """Ollama LLM interface with per-agent model support."""
    
    def __init__(self, model: str = None, temperature: float = None):
        """Initialize LLM service for a specific Ollama model.
        
        Args:
            model: Ollama model name (e.g., 'qwen3:8b'). Defaults to settings.OLLAMA_MODEL.
            temperature: Override default temperature. Defaults to settings.LLM_TEMPERATURE.
        """
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        self.temperature = temperature or settings.LLM_TEMPERATURE
        self.provider = "ollama"  # backward compat for health endpoints
        
        print(f"[LLM] OK: Using Ollama: {self.model} at {self.base_url}")
    
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        json_mode: bool = False
    ) -> str:
        """
        Send a chat request to the Ollama LLM.
        
        Args:
            system_prompt: System message
            user_prompt: User message
            temperature: Override default temperature
            json_mode: Request JSON output
            
        Returns:
            LLM response text
        """
        temp = temperature or self.temperature
        return self._ollama_chat(system_prompt, user_prompt, temp, json_mode)
    
    def _ollama_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        json_mode: bool
    ) -> str:
        """Ollama API call using /api/generate endpoint."""
        url = f"{self.base_url}/api/generate"
        
        # Combine system and user prompts for generate endpoint
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        if json_mode:
            payload["format"] = "json"
        
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running with: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama error ({self.model}): {e}")
    
    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Simple generation (backward compat for llm_judge_service).
        
        Args:
            prompt: Full prompt text
            temperature: Override temperature
            
        Returns:
            Generated text
        """
        return self.chat("You are a helpful assistant.", prompt, temperature)
    
    def is_available(self) -> bool:
        """Check if this LLM model is available on Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available = {m["name"] for m in models}
                return self.model in available
            return False
        except:
            return False


# =============================================================================
# Per-Agent Factory Functions
# =============================================================================

# Cache of per-agent LLM instances
_agent_llm_cache: Dict[str, LLMService] = {}

# Agent name → model mapping (from settings)
_AGENT_MODEL_MAP = {
    "secretary": settings.SECRETARY_LLM_MODEL,
    "analyst": settings.ANALYST_LLM_MODEL,
    "data_generator": settings.GENERATOR_LLM_MODEL,
}


def get_llm_for_agent(agent_name: str) -> LLMService:
    """Get or create a per-agent LLM service instance.
    
    Each agent gets its own LLMService with the correct Ollama model.
    
    Args:
        agent_name: Name of the agent ('secretary', 'analyst', 'data_generator')
        
    Returns:
        LLMService configured for that agent's model
    """
    if agent_name not in _agent_llm_cache:
        model = _AGENT_MODEL_MAP.get(agent_name, settings.OLLAMA_MODEL)
        _agent_llm_cache[agent_name] = LLMService(model=model)
        print(f"[LLM Router] Agent '{agent_name}' → model: {model}")
    
    return _agent_llm_cache[agent_name]


def get_llm_service() -> LLMService:
    """Legacy singleton — returns default LLM (secretary model).
    
    Kept for backward compatibility with code that doesn't
    specify an agent name.
    """
    return get_llm_for_agent("secretary")
