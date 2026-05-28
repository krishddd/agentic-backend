"""test_llm_agent — LLM citizen generates coherent text."""
import os, sys, pytest
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from unittest.mock import patch, MagicMock
from src.abm.contracts import AgentPersona, SocialAction
from src.abm.environment import InteractionLedger, RecommendationEngine
from src.abm.agents import LLMCitizen

def make_llm_agent(tmp_path, sentiment=0.3):
    ledger = InteractionLedger(str(tmp_path / "llm.db"))
    rec = RecommendationEngine(ledger)
    model = MagicMock(); model.tick = 0; model.ticker = "NVDA"
    model.template_reasons = ["AI boom"]
    persona = AgentPersona(agent_id=1, name="LLMBot", role="trader",
        agent_type="llm_citizen", sentiment=sentiment, influence=0.5,
        stubbornness=0.3, model="qwen3:4b", following=[])
    return LLMCitizen(model, persona, ledger, rec)

class TestLLMAgent:
    @patch("src.abm.agents.call_ollama", return_value="NVDA is surging on AI demand.")
    def test_generates_content_via_llm(self, mock_llm, tmp_path):
        agent = make_llm_agent(tmp_path, sentiment=0.5)
        content = agent._generate_content([])
        assert len(content) > 5
        assert "NVDA" in content
        mock_llm.assert_called_once()

    @patch("src.abm.agents.call_ollama", return_value="Interesting point on valuation.")
    def test_generates_reply_via_llm(self, mock_llm, tmp_path):
        agent = make_llm_agent(tmp_path)
        reply = agent._generate_reply({"content": "NVDA rally"})
        assert len(reply) > 5
        mock_llm.assert_called_once()

    @patch("src.abm.agents.call_ollama", return_value="Mock post.")
    def test_decide_action_returns_social_action(self, mock_llm, tmp_path):
        agent = make_llm_agent(tmp_path)
        action = agent._decide_action([])
        assert isinstance(action, SocialAction)

    @patch("src.abm.agents.call_ollama", return_value="Mock response.")
    def test_step_completes(self, mock_llm, tmp_path):
        agent = make_llm_agent(tmp_path)
        agent.ledger.record_post(0, 99, "seed", 0.2)
        agent.step()
        states = agent.ledger.conn.execute("SELECT * FROM agent_states").fetchall()
        assert len(states) >= 1

    @patch("src.abm.agents.call_ollama", return_value="Bearish take on NVDA.")
    def test_prompt_contains_stance(self, mock_llm, tmp_path):
        agent = make_llm_agent(tmp_path, sentiment=-0.5)
        agent._generate_content([])
        call_args = mock_llm.call_args
        prompt = call_args[0][0]
        assert "bearish" in prompt.lower()
