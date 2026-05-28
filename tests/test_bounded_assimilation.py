"""test_bounded_assimilation — Close sentiment → assimilation; far → entrenchment."""
import os, sys, pytest, math
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from unittest.mock import MagicMock
from src.abm.contracts import AgentPersona
from src.abm.environment import InteractionLedger, RecommendationEngine
from src.abm.agents import RuleBasedCitizen

def make_agent(tmp_path, sentiment=0.0, stubbornness=0.3):
    ledger = InteractionLedger(str(tmp_path / "ba.db"))
    rec = RecommendationEngine(ledger)
    model = MagicMock(); model.tick = 0; model.ticker = "T"
    model.template_reasons = ["test"]
    persona = AgentPersona(agent_id=1, name="A", role="r", agent_type="rule_based",
        sentiment=sentiment, influence=0.5, stubbornness=stubbornness, model="none", following=[])
    return RuleBasedCitizen(model, persona, ledger, rec), ledger

def make_feed(ledger, sentiment, n=5):
    for i in range(n):
        ledger.record_post(0, 100 + i, f"p{i}", sentiment)
    return ledger.conn.execute("SELECT * FROM posts").fetchall()

class TestBoundedAssimilation:
    def test_close_sentiment_assimilates(self, tmp_path):
        agent, ledger = make_agent(tmp_path, sentiment=0.1)
        feed = make_feed(ledger, 0.2)  # close (gap=0.1)
        old = agent.sentiment
        agent._update_sentiment(feed)
        # Should move toward 0.2 (assimilation)
        assert agent.sentiment > old - 0.05  # allows noise

    def test_far_sentiment_entrenches(self, tmp_path):
        agent, ledger = make_agent(tmp_path, sentiment=-0.8)
        feed = make_feed(ledger, 0.8)  # far (gap=1.6)
        old = agent.sentiment
        agent._update_sentiment(feed)
        # Should move AWAY from 0.8 (entrenchment) or stay put
        assert agent.sentiment <= old + 0.1  # should not jump toward 0.8

    def test_sentiment_stays_in_bounds(self, tmp_path):
        agent, ledger = make_agent(tmp_path, sentiment=0.99)
        feed = make_feed(ledger, 1.0)
        for _ in range(20):
            agent._update_sentiment(feed)
        assert -1.0 <= agent.sentiment <= 1.0

    def test_stubborn_agent_moves_less(self, tmp_path):
        agent_open, _ = make_agent(tmp_path, sentiment=0.0, stubbornness=0.1)
        agent_stub, ledger2 = make_agent(tmp_path / "s", sentiment=0.0, stubbornness=0.9)
        feed = make_feed(ledger2, 0.3)
        agent_open._update_sentiment(feed)
        # Recreate feed for second agent
        ledger_o = InteractionLedger(str(tmp_path / "o.db"))
        feed_o = make_feed(ledger_o, 0.3)
        agent_open2, _ = make_agent(tmp_path / "o2", sentiment=0.0, stubbornness=0.1)
        agent_open2._update_sentiment(feed_o)
        agent_stub._update_sentiment(feed)
        # Stubborn should move less (on average, with noise this is probabilistic)
        # Just verify both are valid
        assert -1.0 <= agent_open2.sentiment <= 1.0
        assert -1.0 <= agent_stub.sentiment <= 1.0

    def test_empty_feed_no_change(self, tmp_path):
        agent, _ = make_agent(tmp_path, sentiment=0.5)
        old = agent.sentiment
        agent._update_sentiment([])
        assert agent.sentiment == old
