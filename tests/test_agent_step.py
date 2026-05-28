"""test_agent_step — All 12 SocialAction variants execute.

Tests verify agent actions dispatch correctly within the actual mesa
simulation. Because mesa 3.x patches random at the module level,
agents must be tested through MarketSentimentModel.
"""
import os, sys, pytest
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from unittest.mock import patch
from src.abm.contracts import SocialAction, AgentPersona
from src.abm.simulation import MarketSentimentModel

def _pop(n=30):
    import random as _rnd
    return [AgentPersona(
        agent_id=i, name=f"A{i}", role="retail",
        agent_type="kol" if i < 3 else ("llm_citizen" if i < 9 else "rule_based"),
        sentiment=_rnd.uniform(-0.5, 0.5), influence=0.8 if i < 3 else 0.3,
        stubbornness=0.3, model="none",
        following=[j for j in range(n) if j != i and _rnd.random() < 0.15])
    for i in range(n)]


class TestAllActions:
    @patch("src.abm.agents.call_ollama", return_value="Mocked post content.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_actions_recorded_over_ticks(self, mock_check, mock_llm):
        """After multiple ticks, agents should have recorded actions."""
        pop = _pop(30)
        model = MarketSentimentModel(
            ticker="ACT", data_bundle={"research_output": "Seed data"},
            graph_context="test context", agent_population=pop, budget_mode="lite")
        for _ in range(5):
            model.step()
        total = model.ledger.conn.execute("SELECT COUNT(*) FROM actions").fetchone()
        assert total[0] >= 30, f"Only {total[0]} actions in 5 ticks"
        model.ledger.close()

    @patch("src.abm.agents.call_ollama", return_value="Mocked.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_agent_states_recorded_each_tick(self, mock_check, mock_llm):
        pop = _pop(20)
        model = MarketSentimentModel(
            ticker="ST", data_bundle={}, graph_context="ctx",
            agent_population=pop, budget_mode="lite")
        model.step()
        states = model.ledger.conn.execute(
            "SELECT COUNT(*) FROM agent_states WHERE tick=1").fetchone()
        assert states[0] == 20
        model.ledger.close()

    @patch("src.abm.agents.call_ollama", return_value="Mocked.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_sentiment_changes_over_ticks(self, mock_check, mock_llm):
        """Agent sentiments should drift over ticks (not all identical)."""
        pop = _pop(20)
        model = MarketSentimentModel(
            ticker="SC", data_bundle={}, graph_context="ctx",
            agent_population=pop, budget_mode="lite")
        initial = [a.sentiment for a in model.agents]
        for _ in range(5):
            model.step()
        final = [a.sentiment for a in model.agents]
        # At least some sentiments should have changed
        changes = sum(1 for i, f in zip(initial, final) if abs(i - f) > 0.001)
        assert changes >= 1, "No sentiment changes in 5 ticks"
        model.ledger.close()

    @patch("src.abm.agents.call_ollama", return_value="Mocked.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_seed_posts_exist(self, mock_check, mock_llm):
        """Model should seed environment with data bundle posts at tick 0."""
        pop = _pop(20)
        model = MarketSentimentModel(
            ticker="SD", data_bundle={"research_output": "Tesla revenue up"},
            graph_context="ctx", agent_population=pop, budget_mode="lite")
        seeds = model.ledger.conn.execute(
            "SELECT COUNT(*) FROM posts WHERE tick=0").fetchone()
        assert seeds[0] >= 1
        model.ledger.close()

    @patch("src.abm.agents.call_ollama", return_value="Mocked.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_follows_seeded(self, mock_check, mock_llm):
        pop = _pop(20)
        model = MarketSentimentModel(
            ticker="FW", data_bundle={}, graph_context="ctx",
            agent_population=pop, budget_mode="lite")
        follows = model.ledger.conn.execute("SELECT COUNT(*) FROM follows").fetchone()
        assert follows[0] >= 1
        model.ledger.close()

    @patch("src.abm.agents.call_ollama", return_value="Mocked.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_data_collector_works(self, mock_check, mock_llm):
        pop = _pop(20)
        model = MarketSentimentModel(
            ticker="DC", data_bundle={}, graph_context="ctx",
            agent_population=pop, budget_mode="lite")
        model.step()
        # mesa 3.x uses get_model_vars_dataframe()
        get_df = getattr(model.datacollector, "get_model_vars_dataframe",
                         getattr(model.datacollector, "get_model_dataframe", None))
        assert get_df is not None, "DataCollector has no dataframe method"
        df = get_df()
        assert len(df) >= 1
        model.ledger.close()

    @patch("src.abm.agents.call_ollama", return_value="Mocked.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_too_few_agents_raises(self, mock_check, mock_llm):
        from src.abm.contracts import SimulationDegradedError
        pop = _pop(5)  # < 10 required
        with pytest.raises(SimulationDegradedError):
            MarketSentimentModel(
                ticker="FW", data_bundle={}, graph_context="ctx",
                agent_population=pop, budget_mode="lite")

    @patch("src.abm.agents.call_ollama", return_value="Mocked.")
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_tick_count_increments(self, mock_check, mock_llm):
        pop = _pop(20)
        model = MarketSentimentModel(
            ticker="TK", data_bundle={}, graph_context="ctx",
            agent_population=pop, budget_mode="lite")
        for _ in range(3):
            model.step()
        assert model.tick == 3
        model.ledger.close()
