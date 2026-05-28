"""test_follow_graph — tick=0 has pre-populated follows from GraphRAG."""
import os, sys, pytest, random
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from unittest.mock import patch
from src.abm.simulation import MarketSentimentModel
from src.abm.contracts import AgentPersona

def _pop(n=20):
    return [AgentPersona(
        agent_id=i, name=f"A{i}", role="retail",
        agent_type="kol" if i < 2 else ("llm_citizen" if i < 6 else "rule_based"),
        sentiment=random.uniform(-0.5, 0.5), influence=0.8 if i < 2 else 0.3,
        stubbornness=0.3, model="none",
        following=[j for j in range(n) if j != i and random.random() < 0.2])
    for i in range(n)]

class TestFollowGraph:
    @patch("src.abm.agents.call_ollama", return_value="Mock")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_follows_seeded_at_tick_zero(self, mock_check, mock_llm, tmp_path):
        pop = _pop(20)
        model = MarketSentimentModel(
            ticker="TEST", data_bundle={}, graph_context="ctx",
            agent_population=pop, budget_mode="lite")
        follows = model.ledger.conn.execute("SELECT COUNT(*) FROM follows").fetchone()
        assert follows[0] > 0, "Follow graph not seeded at tick 0"
        model.ledger.close()

    @patch("src.abm.agents.call_ollama", return_value="Mock")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_no_self_follows(self, mock_check, mock_llm, tmp_path):
        pop = _pop(20)
        model = MarketSentimentModel(
            ticker="TEST", data_bundle={}, graph_context="ctx",
            agent_population=pop, budget_mode="lite")
        self_follows = model.ledger.conn.execute(
            "SELECT COUNT(*) FROM follows WHERE follower_id = followed_id").fetchone()
        assert self_follows[0] == 0
        model.ledger.close()

    @patch("src.abm.agents.call_ollama", return_value="Mock")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_follow_ids_valid(self, mock_check, mock_llm, tmp_path):
        pop = _pop(20)
        ids = {p.agent_id for p in pop}
        model = MarketSentimentModel(
            ticker="TEST", data_bundle={}, graph_context="ctx",
            agent_population=pop, budget_mode="lite")
        follows = model.ledger.conn.execute("SELECT * FROM follows").fetchall()
        for f in follows:
            assert f["follower_id"] in ids
            assert f["followed_id"] in ids
        model.ledger.close()
