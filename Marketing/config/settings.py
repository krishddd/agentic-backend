"""
Sales Operations AI Agent - Configuration
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# =============================================================================
# API Keys
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# =============================================================================
# Google OAuth Paths
# =============================================================================
CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")

# =============================================================================
# Gmail Configuration
# =============================================================================
SENDER_EMAIL = "krishnahutrik.n@gmail.com"
DEFAULT_RECEIVER_EMAIL = "harish.krishna@testaing.com"
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly"
]

# =============================================================================
# Google Sheets Configuration
# =============================================================================
SPREADSHEET_ID = "1OkXq8gvcd4etHeTri1qCHFFWDLrxjhQO4H_8STscr2A"
SHEET_NAME = "Management_demo"
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# =============================================================================
# Agent Configuration
# =============================================================================
POLLING_INTERVAL_SECONDS = 300  # 5 minutes
PROCESSING_TIMEOUT_MINUTES = 10  # Reset stuck rows after 10 min

# =============================================================================
# Slack Configuration (Optional)
# =============================================================================
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
SLACK_ENABLED = bool(SLACK_WEBHOOK_URL)

# =============================================================================
# Data Paths
# =============================================================================
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, "knowledge_base.txt")
COMPETITORS_PATH = os.path.join(DATA_DIR, "competitors.json")

# =============================================================================
# LLM Configuration — Ollama Only, Per-Agent Models
# =============================================================================
# All agent LLMs run through Ollama. No OpenAI dependency.
LLM_PROVIDER = "ollama"

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Per-agent model assignments
SECRETARY_LLM_MODEL = os.getenv("SECRETARY_LLM_MODEL", "llama3.2:latest")
ANALYST_LLM_MODEL = os.getenv("ANALYST_LLM_MODEL", "qwen3:8b")
GENERATOR_LLM_MODEL = os.getenv("GENERATOR_LLM_MODEL", "qwen3:4b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

# Legacy: kept for backward compat if any eval code references it
OLLAMA_MODEL = SECRETARY_LLM_MODEL  # default fallback

# Common settings
LLM_TEMPERATURE = 0.7

# =============================================================================
# Logging Configuration
# =============================================================================
LOG_FILE = os.path.join(LOGS_DIR, "agent.log")
LOG_LEVEL = "INFO"

