"""test_rule_agent — Template post generation.

RuleBasedCitizen uses weighted random action selection and template-based
content generation. Tests run through actual MarketSentimentModel for
correct mesa initialization.
"""
import os, sys, pytest
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from unittest.mock import patch
from src.abm.contracts import AgentPersona, BUDGET_MODES
from src.abm.simulation import MarketSentimentModel


def _pop_rule_heavy(n=20, base_sentiment=0.3):
    """Population weighted toward rule-based agents."""
    import random as _rnd
    return [AgentPersona(
        agent_id=i, name=f"R{i}", role="retail",
        agent_type="kol" if i < 1 else ("llm_citizen" if i < 3 else "rule_based"),
        sentiment=base_sentiment + _rnd.uniform(-0.1, 0.1),
        influence=0.8 if i < 1 else 0.3,
        stubbornness=0.3, model="none",
        following=[j for j in range(n) if j != i and _rnd.random() < 0.1])
    for i in range(n)]


class TestRuleAgent:
    @patch("src.abm.agents.call_ollama", return_value="LLM mock for rule test.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_agents_generate_posts(self, mock_check, mock_llm):
        """Rule agents + LLM/KOL should create posts during simulation."""
        pop = _pop_rule_heavy(20, base_sentiment=0.5)
        model = MarketSentimentModel(
            ticker="AAPL", data_bundle={"research_output": "Apple revenue up 15%"},
            graph_context="strong earnings report", agent_population=pop,
            budget_mode="lite")
        for _ in range(5):
            model.step()
        posts = model.ledger.conn.execute(
            "SELECT COUNT(*) FROM posts WHERE tick > 0").fetchone()
        assert posts[0] >= 1, "No agent posts generated"
        model.ledger.close()

    @patch("src.abm.agents.call_ollama", return_value="LLM mock.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_actions_recorded(self, mock_check, mock_llm):
        """After 3 ticks, rule agents should have recorded actions."""
        pop = _pop_rule_heavy(20)
        model = MarketSentimentModel(
            ticker="ACT", data_bundle={}, graph_context="ctx",
            agent_population=pop, budget_mode="lite")
        for _ in range(3):
            model.step()
        count = model.ledger.conn.execute("SELECT COUNT(*) FROM actions").fetchone()
        assert count[0] >= 15, f"Too few actions: {count[0]}"
        model.ledger.close()

    @patch("src.abm.agents.call_ollama", return_value="Mock.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_state_recorded_each_tick(self, mock_check, mock_llm):
        """Each agent records state after every step."""
        pop = _pop_rule_heavy(20)
        model = MarketSentimentModel(
            ticker="STP", data_bundle={}, graph_context="ctx",
            agent_population=pop, budget_mode="lite")
        model.step()
        states = model.ledger.conn.execute(
            "SELECT COUNT(DISTINCT agent_id) FROM agent_states WHERE tick=1").fetchone()
        assert states[0] == 20
        model.ledger.close()

    @patch("src.abm.agents.call_ollama", return_value="Mock.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_sentiments_evolve(self, mock_check, mock_llm):
        """Sentiments should change over time."""
        pop = _pop_rule_heavy(20)
        model = MarketSentimentModel(
            ticker="EVO", data_bundle={}, graph_context="ctx",
            agent_population=pop, budget_mode="lite")
        initials = [a.sentiment for a in model.agents]
        for _ in range(10):
            model.step()
        finals = [a.sentiment for a in model.agents]
        changed = sum(1 for i, f in zip(initials, finals) if abs(i - f) > 0.001)
        assert changed >= 1
        model.ledger.close()

    @patch("src.abm.agents.call_ollama", return_value="Mock.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_full_run_completes(self, mock_check, mock_llm):
        """A full lite run should complete without errors."""
        pop = _pop_rule_heavy(20)
        model = MarketSentimentModel(
            ticker="FULL", data_bundle={}, graph_context="ctx",
            agent_population=pop, budget_mode="lite")
        ticks = BUDGET_MODES["lite"]["ticks"]
        result = model.run(ticks)
        assert result.total_ticks == ticks
        assert result.total_agents == 20
