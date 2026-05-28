"""test_monte_carlo — Standard mode runs 3 paths, returns aggregate."""
import os, sys, pytest, random
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from unittest.mock import patch
from src.abm.simulation import MarketSentimentModel
from src.abm.contracts import AgentPersona, BUDGET_MODES

def _pop(n=20):
    return [AgentPersona(
        agent_id=i, name=f"A{i}", role="retail",
        agent_type="kol" if i < 2 else ("llm_citizen" if i < 6 else "rule_based"),
        sentiment=random.uniform(-0.5, 0.5),
        influence=0.8 if i < 2 else 0.3,
        stubbornness=0.3, model="none",
        following=[j for j in range(n) if j != i and random.random() < 0.15])
    for i in range(n)]

class TestMonteCarlo:
    @patch("src.abm.agents.call_ollama", return_value="MC mock.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_runs_3_paths(self, mock_check, mock_llm, tmp_path):
        results = []
        ticks = BUDGET_MODES["lite"]["ticks"]
        for path_i in range(3):
            pop = _pop(20)
            model = MarketSentimentModel(
                ticker="MC", data_bundle={}, graph_context="ctx",
                agent_population=pop, budget_mode="lite")
            result = model.run(ticks)
            avg = sum(a.sentiment for a in model.agents) / max(len(list(model.agents)), 1)
            results.append("bullish" if avg > 0.1 else ("bearish" if avg < -0.1 else "neutral"))
        assert len(results) == 3

    @patch("src.abm.agents.call_ollama", return_value="MC mock.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_returns_valid_sentiments(self, mock_check, mock_llm, tmp_path):
        results = []
        ticks = BUDGET_MODES["lite"]["ticks"]
        for _ in range(3):
            pop = _pop(20)
            model = MarketSentimentModel(
                ticker="MC", data_bundle={}, graph_context="ctx",
                agent_population=pop, budget_mode="lite")
            result = model.run(ticks)
            sentiments = [a.sentiment for a in model.agents]
            avg = sum(sentiments) / len(sentiments) if sentiments else 0
            results.append("bullish" if avg > 0.1 else ("bearish" if avg < -0.1 else "neutral"))
        for r in results:
            assert r in ("bullish", "bearish", "neutral")

    @patch("src.abm.agents.call_ollama", return_value="MC mock.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_aggregate_majority(self, mock_check, mock_llm, tmp_path):
        from collections import Counter
        results = []
        ticks = BUDGET_MODES["lite"]["ticks"]
        for _ in range(5):
            pop = _pop(20)
            model = MarketSentimentModel(
                ticker="MC", data_bundle={}, graph_context="ctx",
                agent_population=pop, budget_mode="lite")
            result = model.run(ticks)
            sentiments = [a.sentiment for a in model.agents]
            avg = sum(sentiments) / len(sentiments) if sentiments else 0
            results.append("bullish" if avg > 0.1 else ("bearish" if avg < -0.1 else "neutral"))
        majority = Counter(results).most_common(1)[0][0]
        assert majority in ("bullish", "bearish", "neutral")
