# Sales Operations AI Agent

Enterprise-grade dual-agent system for automating sales communication and intelligence.

> **Orchestrator Integration**: When run via the root `app.py` orchestrator, this module uses **local Ollama LLMs** instead of OpenAI for lead scoring, email generation, and deal analysis. Pipeline data feeds into the **MiroFish ABM Grandmaster** simulation for market sentiment-aware lead prioritization.

## 📁 Project Structure

```
Marketing/
├── config/                 # Configuration
│   ├── __init__.py
│   └── settings.py         # All settings & environment
│
├── services/               # API Wrappers
│   ├── __init__.py
│   ├── gmail_service.py    # Gmail OAuth & sending
│   ├── sheets_service.py   # Google Sheets CRUD
│   ├── rag_engine.py       # Knowledge Base retrieval
│   └── calendar_generator.py # ICS file generation
│
├── agents/                 # AI Agents
│   ├── __init__.py
│   ├── secretary.py        # Communication Agent
│   └── analyst.py          # Intelligence Agent
│
├── utils/                  # Utilities
│   ├── __init__.py
│   └── logger.py           # Centralized logging
│
├── scripts/                # Entry Points
│   ├── run_agents.py       # Main scheduler
│   └── generate_data.py    # Synthetic data generator
│
├── data/                   # Knowledge & Data
│   ├── knowledge_base.txt  # Product FAQ
│   ├── competitors.json    # Battlecard data
│   └── synthetic_sales_data.csv
│
├── logs/                   # Log Files
│   └── agent_YYYYMMDD.log
│
├── credentials.json        # Google OAuth credentials
├── token.json              # OAuth token (auto-generated)
├── .env                    # API keys
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure
Edit `config/settings.py` or use these current values:
```python
SPREADSHEET_ID = "1b-KNdoqKl1gJs3BU4b3eiG_pHHA5jc5FQr7-Lg7faoA"
SHEET_NAME = "Marketing Data"
```

### 3. Generate Test Data
```bash
python scripts/generate_data.py
```

### 4. Run Agents
```bash
# One-shot mode
python scripts/run_agents.py --mode once

# Continuous mode (polls every 5 min)
python scripts/run_agents.py --mode continuous

# Individual agents
python scripts/run_agents.py --secretary-only
python scripts/run_agents.py --analyst-only
```

## 🔧 Configuration

All settings are in `config/settings.py`:

| Setting | Description |
|---------|-------------|
| `SPREADSHEET_ID` | Your Google Sheet ID |
| `SENDER_EMAIL` | Gmail sender address |
| `POLLING_INTERVAL_SECONDS` | Scheduler interval (default: 300) |
| `PROCESSING_TIMEOUT_MINUTES` | Reset stuck rows (default: 10) |
| `LLM_MODEL` | OpenAI model (default: gpt-4o-mini) |

## 📊 Workflow

```
[Sheet: Pending] → Secretary → [Sheet: Sent] → Analyst → [Sheet: Synced]
                        ↓                           ↓
                    [Gmail]                    [Slack/Email]
```

## 📋 Spreadsheet Columns

| Column | Type | Description |
|--------|------|-------------|
| Row_ID | Number | Unique ID |
| Client_Name | Text | Customer name |
| Company_Name | Text | Organization |
| Consultant_Email | Email | To recipient |
| Manager_Email | Email | CC recipient |
| Call_Type | Dropdown | Discovery/Demo/etc |
| Raw_Discussion_Notes | Text | Call notes |
| Client_Requirements | Text | Technical needs |
| Scheduling_Next_Steps | Text | Next meeting |
| Mail_Status | Dropdown | Pending/Processing/Sent/Failed |
| Deal_Potential_Score | Number | 1-10 score |
| Risk_Flags | Text | Concerns |
| Strategy_Notes | Text | Recommendations |
| Sync_Status | Dropdown | Waiting/Synced |

## 📝 Logs

Logs are saved to `logs/agent_YYYYMMDD.log` with format:
```
2024-12-17 08:15:00 | INFO     | agents.secretary | Processing Row #1
```
