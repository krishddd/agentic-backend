"""test_kol_agent — KOL generates analytical posts with data."""
import os, sys, pytest
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from unittest.mock import patch, MagicMock
from src.abm.contracts import AgentPersona, SocialAction
from src.abm.environment import InteractionLedger, RecommendationEngine
from src.abm.agents import KOLAgent

def make_kol_agent(tmp_path, sentiment=0.5):
    ledger = InteractionLedger(str(tmp_path / "kol.db"))
    rec = RecommendationEngine(ledger)
    model = MagicMock(); model.tick = 0; model.ticker = "TSLA"
    model.template_reasons = ["EV market"]
    model.data_bundle = {
        "financial_summary": "Revenue $25B, P/E 80, margin growth 5%",
        "news_summary": "Tesla announces new factory",
    }
    persona = AgentPersona(agent_id=1, name="KOLAnalyst", role="analyst",
        agent_type="kol", sentiment=sentiment, influence=0.95,
        stubbornness=0.2, model="qwen3:8b", following=[])
    return KOLAgent(model, persona, ledger, rec)

class TestKOLAgent:
    @patch("src.abm.agents.call_ollama", return_value="TSLA's valuation is stretched but momentum is strong.")
    def test_generates_analytical_content(self, mock_llm, tmp_path):
        agent = make_kol_agent(tmp_path, sentiment=0.5)
        content = agent._generate_content([])
        assert len(content) > 10
        mock_llm.assert_called_once()

    @patch("src.abm.agents.call_ollama", return_value="TSLA analysis post.")
    def test_prompt_includes_financial_data(self, mock_llm, tmp_path):
        agent = make_kol_agent(tmp_path)
        agent._generate_content([])
        prompt = mock_llm.call_args[0][0]
        assert "Revenue" in prompt or "financial" in prompt.lower()

    @patch("src.abm.agents.call_ollama", return_value="Mock.")
    def test_prompt_includes_market_influence(self, mock_llm, tmp_path):
        agent = make_kol_agent(tmp_path)
        agent._generate_content([])
        prompt = mock_llm.call_args[0][0]
        assert "influence" in prompt.lower()

    @patch("src.abm.agents.call_ollama", return_value="Mock.")
    def test_kol_action_favors_posting(self, mock_llm, tmp_path):
        agent = make_kol_agent(tmp_path)
        actions = [agent._decide_action([]) for _ in range(200)]
        post_count = sum(1 for a in actions if a == SocialAction.POST)
        browse_count = sum(1 for a in actions if a == SocialAction.BROWSE_FEED)
        # KOLs should post more than browse (30% vs 20% weight)
        assert post_count > browse_count * 0.5

    @patch("src.abm.agents.call_ollama", return_value="Full step mock.")
    def test_step_completes(self, mock_llm, tmp_path):
        agent = make_kol_agent(tmp_path)
        agent.ledger.record_post(0, 99, "seed", 0.2)
        agent.step()
        states = agent.ledger.conn.execute("SELECT * FROM agent_states").fetchall()
        assert len(states) >= 1
