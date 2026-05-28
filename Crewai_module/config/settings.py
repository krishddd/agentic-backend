"""Centralized configuration management for Financial Crew."""

from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )
    
    # LLM Configuration — Ollama Only
    llm_provider: str = Field(default="ollama", description="LLM provider (ollama)")
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    
    # Per-agent model overrides (defaults match llm_router.py)
    research_llm_model: str = Field(default="llama3.2:latest", description="Research Analyst model")
    financial_llm_model: str = Field(default="qwen3:4b", description="Financial Analyst model")
    advisor_llm_model: str = Field(default="qwen3:8b", description="Investment Advisor model")
    sec_parser_llm_model: str = Field(default="llava:7b", description="SEC Filing Parser model")
    embedding_model_name: str = Field(default="nomic-embed-text:latest", description="Embedding model for RAG")
    
    # OpenAI (optional — only for LLM judge evaluation if enabled)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key (optional, for LLM judge only)")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model (judge only)")
    
    # SEC Configuration
    sec_user_agent: str = Field(
        default="financial-crew github.com/user",
        description="User agent for SEC EDGAR API"
    )
    
    # Search APIs (Optional)
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key")
    serper_api_key: Optional[str] = Field(default=None, description="Serper API key")
    serpapi: Optional[str] = Field(default=None, description="SerpAPI key")
    
    # News API (Optional)
    news_api_key: Optional[str] = Field(default=None, description="News API key")
    
    # Financial Data APIs (Optional)
    alpha_vantage_api_key: Optional[str] = Field(default=None, description="Alpha Vantage API key")
    fmp_api_key: Optional[str] = Field(default=None, description="FinancialModelingPrep API key")
    finnhub_api_key: Optional[str] = Field(default=None, description="Finnhub API key")
    
    # Application Settings
    log_level: str = Field(default="INFO", description="Logging level")
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_expire_hours: int = Field(default=24, description="Cache expiration in hours")
    
    # RAG Settings
    chunk_size: int = Field(default=1000, description="Text chunk size for RAG")
    chunk_overlap: int = Field(default=200, description="Chunk overlap for RAG")
    top_k_results: int = Field(default=5, description="Number of RAG results to retrieve")
    
    # Agent Settings
    max_iter: int = Field(default=25, description="Maximum iterations for agents")
    enable_delegation: bool = Field(default=True, description="Enable agent delegation")
    verbose: bool = Field(default=True, description="Verbose agent output")
    
    # Evaluation Settings
    enable_evaluation: bool = Field(default=True, description="Enable agent evaluation tracking")
    enable_llm_judge: bool = Field(default=False, description="Enable LLM-as-a-Judge evaluation")
    judge_model: str = Field(default="gpt-4o-mini", description="Model for LLM judge")

    # Google OAuth Paths
    CREDENTIALS_PATH: str = Field(
        default="Marketing/credentials.json",
        description="Path to Google OAuth credentials.json"
    )
    TOKEN_PATH: str = Field(
        default="Marketing/token.json",
        description="Path to Google OAuth token.json"
    )

    # Gmail Configuration
    SENDER_EMAIL: str = Field(default="krishnahutrik.n@gmail.com", description="Gmail sender address")
    DEFAULT_RECEIVER_EMAIL: str = Field(default="harish.krishna@testaing.com", description="Default receiver")
    GMAIL_SCOPES: list = Field(
        default=[
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/gmail.readonly",
        ],
        description="Gmail OAuth scopes"
    )

    # Google Sheets Configuration
    SPREADSHEET_ID: str = Field(
        default="1OkXq8gvcd4etHeTri1qCHFFWDLrxjhQO4H_8STscr2A",
        description="Google Sheets spreadsheet ID"
    )
    SHEET_NAME: str = Field(default="Management_demo", description="Default sheet tab name")
    SHEETS_SCOPES: list = Field(
        default=["https://www.googleapis.com/auth/spreadsheets"],
        description="Sheets OAuth scopes"
    )
    
    # Paths
    @property
    def base_dir(self) -> Path:
        """Get base directory of the application."""
        return Path(__file__).parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        path = self.base_dir / "data"
        path.mkdir(exist_ok=True)
        return path
    
    @property
    def filings_dir(self) -> Path:
        """Get SEC filings directory."""
        path = self.data_dir / "filings"
        path.mkdir(exist_ok=True)
        return path
    
    @property
    def chroma_dir(self) -> Path:
        """Get ChromaDB directory."""
        path = self.data_dir / "chroma_db"
        path.mkdir(exist_ok=True)
        return path
    
    @property
    def outputs_dir(self) -> Path:
        """Get outputs directory."""
        path = self.base_dir / "outputs"
        path.mkdir(exist_ok=True)
        return path
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory."""
        path = self.base_dir / "logs"
        path.mkdir(exist_ok=True)
        return path
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of {valid_levels}")
        return v
    
    def has_search_api(self) -> bool:
        """Check if any search API is configured."""
        return any([
            self.tavily_api_key,
            self.serper_api_key,
            self.serpapi
        ])
    
    def get_search_api(self) -> Optional[str]:
        """Get the first available search API."""
        if self.tavily_api_key:
            return "tavily"
        elif self.serper_api_key:
            return "serper"
        elif self.serpapi:
            return "serpapi"
        return None


# Global settings instance
settings = Settings()
