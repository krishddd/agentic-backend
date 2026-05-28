"""test_kol_followers — All agents follow KOLs at spawn."""
import os, sys, pytest
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from unittest.mock import patch
from src.abm.simulation import MarketSentimentModel
from src.abm.contracts import AgentPersona

def _pop(n=20):
    agents = []
    for i in range(n):
        t = "kol" if i < 2 else "rule_based"
        following = [j for j in range(2) if j != i] if i >= 2 else []
        agents.append(AgentPersona(
            agent_id=i, name=f"A{i}", role="analyst" if i < 2 else "retail",
            agent_type=t, sentiment=0.0, influence=0.9 if i < 2 else 0.3,
            stubbornness=0.3, model="none", following=following))
    return agents

class TestKOLFollowers:
    @patch("src.abm.agents.call_ollama", return_value="Mock")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_all_non_kols_follow_kols(self, mock_check, mock_llm, tmp_path):
        pop = _pop(20)
        model = MarketSentimentModel(
            ticker="T", data_bundle={}, graph_context="ctx",
            agent_population=pop, budget_mode="lite")
        kol_ids = {0, 1}
        for aid in range(2, 20):
            follows = model.ledger.conn.execute(
                "SELECT followed_id FROM follows WHERE follower_id=?", (aid,)).fetchall()
            followed_set = {f[0] for f in follows}
            for kid in kol_ids:
                assert kid in followed_set, f"Agent {aid} doesn't follow KOL {kid}"
        model.ledger.close()

    @patch("src.abm.agents.call_ollama", return_value="Mock")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_kols_have_high_follower_count(self, mock_check, mock_llm, tmp_path):
        pop = _pop(20)
        model = MarketSentimentModel(
            ticker="T", data_bundle={}, graph_context="ctx",
            agent_population=pop, budget_mode="lite")
        for kid in [0, 1]:
            cnt = model.ledger.conn.execute(
                "SELECT COUNT(*) FROM follows WHERE followed_id=?", (kid,)).fetchone()
            assert cnt[0] >= 15, f"KOL {kid} only has {cnt[0]} followers"
        model.ledger.close()
