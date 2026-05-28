"""test_simulation_lite — 50 agents × 10 ticks completes."""
import os, sys, pytest, random
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from unittest.mock import patch
from src.abm.simulation import MarketSentimentModel
from src.abm.contracts import AgentPersona, BUDGET_MODES

def _pop(n=50):
    agents = []
    for i in range(n):
        t = "kol" if i < 5 else ("llm_citizen" if i < 15 else "rule_based")
        agents.append(AgentPersona(
            agent_id=i, name=f"A{i}", role="retail",
            agent_type=t, sentiment=random.uniform(-0.8, 0.8),
            influence=0.9 if i < 5 else (0.5 if i < 15 else 0.3),
            stubbornness=random.uniform(0.2, 0.7), model="none",
            following=[j for j in range(n) if j != i and random.random() < 0.1]))
    return agents

class TestSimulationLite:
    @patch("src.abm.agents.call_ollama", return_value="Lite sim mock.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_50_agents_10_ticks_completes(self, mock_check, mock_llm, tmp_path):
        pop = _pop(50)
        model = MarketSentimentModel(
            ticker="LITE", data_bundle={}, graph_context="test",
            agent_population=pop, budget_mode="lite")
        ticks = BUDGET_MODES["lite"]["ticks"]
        result = model.run(ticks)
        assert result.total_ticks == ticks

    @patch("src.abm.agents.call_ollama", return_value="Mock.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_posts_generated(self, mock_check, mock_llm, tmp_path):
        pop = _pop(50)
        model = MarketSentimentModel(
            ticker="LITE", data_bundle={"research_output": "seed data"},
            graph_context="test", agent_population=pop, budget_mode="lite")
        ticks = BUDGET_MODES["lite"]["ticks"]
        result = model.run(ticks)
        assert result.total_actions > 0, "No actions during simulation"

    @patch("src.abm.agents.call_ollama", return_value="Mock.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_sentiment_trajectory_recorded(self, mock_check, mock_llm, tmp_path):
        pop = _pop(50)
        model = MarketSentimentModel(
            ticker="LITE", data_bundle={}, graph_context="test",
            agent_population=pop, budget_mode="lite")
        ticks = BUDGET_MODES["lite"]["ticks"]
        result = model.run(ticks)
        assert len(result.sentiment_trajectory) == ticks

    @patch("src.abm.agents.call_ollama", return_value="Mock.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_result_has_valid_fields(self, mock_check, mock_llm, tmp_path):
        pop = _pop(50)
        model = MarketSentimentModel(
            ticker="LITE", data_bundle={}, graph_context="test",
            agent_population=pop, budget_mode="lite")
        ticks = BUDGET_MODES["lite"]["ticks"]
        result = model.run(ticks)
        assert result.ticker == "LITE"
        assert result.total_agents == 50
        assert result.total_ticks == ticks
        dist = result.final_sentiment_distribution
        assert "bullish" in dist and "bearish" in dist and "neutral" in dist
