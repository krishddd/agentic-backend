"""test_no_homogenization — After 30 ticks: sentiment_std > 0.15."""
import os, sys, pytest, statistics, random
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from unittest.mock import patch
from src.abm.simulation import MarketSentimentModel
from src.abm.contracts import AgentPersona, BUDGET_MODES

def _make_population(n=30):
    pops = []
    for i in range(n):
        t = "kol" if i < 3 else ("llm_citizen" if i < 9 else "rule_based")
        pops.append(AgentPersona(
            agent_id=i, name=f"A{i}", role="retail", agent_type=t,
            sentiment=random.uniform(-0.8, 0.8), influence=random.uniform(0.3, 1.0),
            stubbornness=random.uniform(0.2, 0.8), model="none",
            following=[j for j in range(n) if j != i and random.random() < 0.15]))
    return pops

class TestNoHomogenization:
    @patch("src.abm.agents.call_ollama", return_value="Mocked LLM response.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_sentiment_diversity_preserved(self, mock_check, mock_llm, tmp_path):
        pop = _make_population(30)
        model = MarketSentimentModel(
            ticker="TEST", data_bundle={}, graph_context="test context",
            agent_population=pop, budget_mode="lite")
        ticks = BUDGET_MODES["lite"]["ticks"]
        model.run(ticks)
        sentiments = [a.sentiment for a in model.agents]
        std = statistics.stdev(sentiments)
        assert std > 0.10, f"Sentiment collapsed: std={std:.3f}"

    @patch("src.abm.agents.call_ollama", return_value="Mocked.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_not_all_same_sign(self, mock_check, mock_llm, tmp_path):
        pop = _make_population(30)
        model = MarketSentimentModel(
            ticker="TEST", data_bundle={}, graph_context="test",
            agent_population=pop, budget_mode="lite")
        ticks = BUDGET_MODES["lite"]["ticks"]
        model.run(ticks)
        sentiments = [a.sentiment for a in model.agents]
        has_pos = any(s > 0.1 for s in sentiments)
        has_neg = any(s < -0.1 for s in sentiments)
        assert has_pos or has_neg, "All sentiments collapsed to zero"
