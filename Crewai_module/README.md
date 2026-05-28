# Financial Crew - Multi-Agent Investment Analysis System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CrewAI](https://img.shields.io/badge/CrewAI-0.86.0-green.svg)](https://github.com/joaomdmoura/crewAI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready multi-agent AI system for comprehensive stock investment analysis. Built with CrewAI, featuring three specialized AI agents that collaborate to generate professional investment reports.

> **Orchestrator Integration**: When run via the root `app.py` orchestrator, this module uses **local Ollama LLMs** (llama3.2, qwen3) instead of OpenAI, and integrates with the **MiroFish ABM Grandmaster** edition — featuring GraphRAG entity retrieval, ensemble trend prediction (T+5/T+10/T+30), deep interaction chat with simulated agents, and D3.js graph visualization for enhanced market analysis.

## 🌟 Features

### Three Specialized AI Agents

1. **Research Analyst** 
   - Market sentiment analysis from news and social media
   - Competitive landscape analysis
   - Regulatory changes tracking
   - Insider trading activity monitoring
   - Upcoming events calendar

2. **Financial Analyst**
   - SEC 10-Q/10-K filing analysis with RAG retrieval
   - Financial metrics and ratio calculation
   - Revenue, profitability, and cash flow analysis
   - Balance sheet health assessment
   - Industry benchmarking

3. **Investment Advisor**
   - Synthesizes all research and analysis
   - Generates actionable investment recommendations
   - Provides buy/hold/sell ratings with price targets
   - Identifies key catalysts and risks

### Advanced RAG System

- Downloads and processes SEC 10-Q forms automatically
- Semantic search over regulatory filings using ChromaDB
- Extracts relevant information with high accuracy
- Uses Sentence Transformers for embeddings

### Production-Ready Features

- ✅ Comprehensive logging with file rotation
- ✅ Disk-based caching for API responses
- ✅ Error handling and retry logic
- ✅ Input validation and sanitization
- ✅ Configurable via environment variables
- ✅ Rich CLI interface with progress indicators

## 📋 Requirements

- Python 3.10 or higher
- Ollama with local models (when run via orchestrator) **OR** OpenAI API key (standalone mode)
- Optional: Tavily/Serper API key for enhanced web search
- Optional: News API key for news aggregation

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd Crewai_module

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file (or use the existing one):

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Recommended for better results
TAVILY_API_KEY=your_tavily_api_key_here
NEWS_API_KEY=your_news_api_key_here

# Optional
SEC_USER_AGENT=your-email@example.com
```

### 3. Run Analysis

**Option A: Command Line**
```bash
# Analyze Tesla stock
python main.py TSLA

# With verbose output
python main.py TSLA --verbose

# Disable caching
python main.py TSLA --no-cache
```

**Option B: API Server**
```bash
# Start API server
python api.py

# Submit analysis via API (in another terminal)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TSLA", "verbose": true}'

# Check status
curl http://localhost:8000/status/{job_id}

# Download report
curl http://localhost:8000/report/TSLA -o TSLA_report.md
```

See [API_USAGE.md](API_USAGE.md) for complete API documentation.

### 4. View Results

Reports are saved to `outputs/[TICKER]_investment_report.md`

## 📁 Project Structure

```
financial_crew/
├── config/               # Configuration management
│   ├── settings.py      # Pydantic settings
│   └── agents.yaml      # Agent configurations
├── agents/              # Agent implementations
│   ├── research_analyst.py
│   ├── financial_analyst.py
│   └── investment_advisor.py
├── tasks/               # Task definitions
│   ├── research_task.py
│   ├── financial_analysis_task.py
│   ├── filing_analysis_task.py
│   └── report_task.py
├── tools/               # Agent tools
│   ├── sec_tools.py    # SEC EDGAR integration
│   ├── rag_tools.py    # RAG retrieval
│   ├── web_search_tools.py
│   ├── financial_tools.py
│   └── sentiment_tools.py
├── rag/                 # RAG system
│   ├── document_processor.py
│   ├── embedder.py
│   └── vector_store.py
├── utils/               # Utilities
│   ├── logger.py       # Logging
│   ├── cache.py        # Caching
│   └── validators.py   # Input validation
├── crew.py             # Main orchestrator
├── main.py             # Entry point
└── requirements.txt
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | ✅ |
| `OPENAI_MODEL` | OpenAI model to use | gpt-4o-mini | ❌ |
| `TAVILY_API_KEY` | Tavily search API key | - | ❌ |
| `NEWS_API_KEY` | News API key | - | ❌ |
| `SEC_USER_AGENT` | Email for SEC API | - | ✅ |
| `LOG_LEVEL` | Logging level | INFO | ❌ |
| `CACHE_ENABLED` | Enable caching | true | ❌ |
| `CACHE_EXPIRE_HOURS` | Cache expiration | 24 | ❌ |
| `EMBEDDING_MODEL` | Sentence transformer model | all-MiniLM-L6-v2 | ❌ |
| `MAX_ITER` | Max agent iterations | 25 | ❌ |
| `ENABLE_DELEGATION` | Enable agent delegation | true | ❌ |

## 📊 Example Output

The system generates comprehensive investment reports including:

- **Executive Summary**: Investment thesis and recommendation
- **Financial Health**: Revenue, profitability, cash flow analysis
- **Market Sentiment**: News analysis and market perception
- **Regulatory Analysis**: Regulatory changes and risk factors
- **Insider Trading**: Recent insider activity patterns
- **Upcoming Events**: Earnings calls, product launches, key dates
- **Competitive Position**: Market positioning vs competitors
- **Investment Recommendation**: Buy/Hold/Sell with price targets

## 🤝 Agent Collaboration

The system features **inter-agent delegation**:

- Research Analyst can request competitor analysis
- Financial Analyst can delegate market research tasks
- Investment Advisor synthesizes insights from all agents
- Agents share context through CrewAI's memory system

## 🛠️ Advanced Features

### RAG System

The RAG system automatically:
1. Downloads SEC 10-Q filings from EDGAR
2. Processes and chunks documents (1000 chars with 200 overlap)
3. Generates embeddings using Sentence Transformers
4. Stores in ChromaDB for fast semantic search
5. Retrieves relevant sections for agent queries

### Caching

- API responses cached to disk
- Configurable expiration times
- Significant cost savings on repeated queries
- Cache stored in `data/cache/`

### Logging

- Structured logging with Loguru
- Daily rotating log files
- Separate error logs
- Located in `logs/` directory

## 📝 Development

### Adding Custom Tools

```python
from crewai_tools import BaseTool

class CustomTool(BaseTool):
    name: str = "Tool Name"
    description: str = "Tool description"
    
    def _run(self, input: str) -> str:
        # Tool logic here
        return result
```

### Modifying Agents

Edit `config/agents.yaml` to customize:
- Agent roles and goals
- Backstories
- Task descriptions
- Expected outputs

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure `.env` file has valid API keys
2. **Import Errors**: Run `pip install -r requirements.txt`
3. **SEC API Fails**: Set `SEC_USER_AGENT` to your email
4. **RAG Not Working**: Check `data/filings/` for downloaded files

### Logs

Check logs for detailed error information:
```bash
# View latest log
cat logs/financial_crew_$(date +%Y-%m-%d).log

# View errors
cat logs/errors_$(date +%Y-%m-%d).log
```

## 📜 License

MIT License - see LICENSE file for details

## 🙏 Credits

Built with:
- [CrewAI](https://github.com/joaomdmoura/crewAI) - Multi-agent orchestration
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [yfinance](https://github.com/ranaroussi/yfinance) - Financial data
- [Rich](https://github.com/Textualize/rich) - Terminal UI

## 📧 Support

For issues and questions:
- Check logs in `logs/` directory
- Review configuration in `.env`
- Ensure all API keys are valid

---

**Note**: This system is for informational purposes only. Not financial advice. Always conduct your own research before making investment decisions.
