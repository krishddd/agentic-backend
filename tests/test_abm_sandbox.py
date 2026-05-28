"""
Integration tests for Phase 2 — ABM Sandbox.

Tests the full stack: environment.py, agents.py, simulation.py.
Uses only RuleBasedCitizen agents (no LLM calls) to keep tests fast.
"""

import os
import pytest
import tempfile
import shutil
from unittest.mock import patch

from src.abm.contracts import (
    AgentPersona, AgentType, SimulationResult, BUDGET_MODES,
    SimulationDegradedError,
)
from src.abm.environment import InteractionLedger, RecommendationEngine
from src.abm.agents import RuleBasedCitizen, BaseMarketAgent
from src.abm.simulation import MarketSentimentModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def ledger(tmp_dir):
    db_path = os.path.join(tmp_dir, "test_ledger.db")
    led = InteractionLedger(db_path)
    yield led
    led.close()


def make_personas(n: int = 20) -> list:
    """Create n personas for testing — all rule-based, no LLM needed."""
    personas = []
    for i in range(n):
        personas.append(AgentPersona(
            agent_id=i,
            name=f"Agent_{i}",
            agent_type=AgentType.RULE_BASED,
            role="retail_investor",
            sentiment=(-0.5 + i / n),
            influence=0.1 + (0.8 * i / n),
            stubbornness=0.3 + (0.4 * i / n),
            following=list(range(max(0, i - 3), i)),
        ))
    return personas


# ---------------------------------------------------------------------------
# InteractionLedger
# ---------------------------------------------------------------------------

class TestInteractionLedger:
    def test_record_and_count_posts(self, ledger):
        ledger.record_post(1, 0, "Hello", 0.5)
        ledger.record_post(1, 1, "World", -0.3)
        ledger.record_post(2, 0, "Tick 2", 0.1)
        assert ledger.count_posts(1) == 2
        assert ledger.count_posts(2) == 3

    def test_record_action(self, ledger):
        ledger.record_action(1, 0, "like", target_post_id=1)
        row = ledger.conn.execute(
            "SELECT * FROM actions WHERE agent_id = 0").fetchone()
        assert row is not None
        assert row["action_type"] == "like"

    def test_record_follow(self, ledger):
        ledger.record_follow(0, 1, tick=1)
        ledger.record_follow(0, 1, tick=2)  # duplicate — ignored
        count = ledger.conn.execute(
            "SELECT COUNT(*) FROM follows").fetchone()[0]
        assert count == 1

    def test_record_agent_state(self, ledger):
        ledger.record_agent_state(0, 1, 0.5, 0.8)
        ledger.record_agent_state(0, 1, 0.6, 0.8)  # replace
        row = ledger.conn.execute(
            "SELECT sentiment FROM agent_states "
            "WHERE agent_id = 0 AND tick = 1").fetchone()
        assert abs(row[0] - 0.6) < 0.001

    def test_increment_post_stat(self, ledger):
        pid = ledger.record_post(1, 0, "test", 0.0)
        ledger.increment_post_stat(pid, "likes")
        ledger.increment_post_stat(pid, "likes")
        row = ledger.conn.execute(
            "SELECT likes FROM posts WHERE post_id = ?", (pid,)).fetchone()
        assert row[0] == 2

    def test_increment_invalid_field_raises(self, ledger):
        pid = ledger.record_post(1, 0, "test", 0.0)
        with pytest.raises(ValueError, match="Invalid stat field"):
            ledger.increment_post_stat(pid, "invalid_field")

    def test_wal_mode(self, ledger):
        mode = ledger.conn.execute(
            "PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_get_sentiment_snapshot(self, ledger):
        ledger.record_agent_state(0, 1, 0.5, 0.8)
        ledger.record_agent_state(1, 1, -0.5, 0.3)
        ledger.record_agent_state(2, 1, 0.0, 0.5)
        snap = ledger.get_sentiment_snapshot(1)
        assert abs(snap["bullish"] - 1/3) < 0.01
        assert abs(snap["bearish"] - 1/3) < 0.01
        assert abs(snap["neutral"] - 1/3) < 0.01

    def test_get_most_engaged_posts(self, ledger):
        pid1 = ledger.record_post(1, 0, "boring", 0.0)
        pid2 = ledger.record_post(1, 1, "viral", 0.5)
        for _ in range(5):
            ledger.increment_post_stat(pid2, "likes")
        top = ledger.get_most_engaged_posts(limit=1)
        assert len(top) == 1
        assert top[0]["post_id"] == pid2


# ---------------------------------------------------------------------------
# RecommendationEngine
# ---------------------------------------------------------------------------

class TestRecommendationEngine:
    def test_curate_feed_returns_list(self, ledger):
        engine = RecommendationEngine(ledger)
        ledger.record_post(1, 0, "A post", 0.5)
        feed = engine.curate_feed(agent_id=1, tick=1)
        assert isinstance(feed, list)

    def test_interest_feed_uses_follows(self, ledger):
        engine = RecommendationEngine(ledger)
        ledger.record_follow(1, 0, tick=0)
        ledger.record_post(1, 0, "From followed", 0.5)
        ledger.record_post(1, 99, "From stranger", 0.1)
        interest = engine._interest_feed(agent_id=1, tick=1, limit=10)
        assert len(interest) >= 1
        contents = [p["content"] for p in interest]
        assert "From followed" in contents


# ---------------------------------------------------------------------------
# MarketSentimentModel (rule-based only, no LLM)
# ---------------------------------------------------------------------------

class TestMarketSentimentModel:
    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_lite_simulation_completes(self, mock_check, tmp_dir):
        personas = make_personas(50)
        model = MarketSentimentModel(
            ticker="TEST",
            data_bundle={
                "research_output": "Test research data about TEST",
                "financial_summary": "Revenue grew 15% YoY",
                "news_summary": "TEST launches new product line",
            },
            graph_context="Market fundamentals are strong\nIndustry trends upward",
            agent_population=personas,
            budget_mode="lite",
            run_id=f"test_{tmp_dir.replace(os.sep, '_')[-20:]}",
        )
        result = model.run(ticks=3)

        assert isinstance(result, SimulationResult)
        assert result.ticker == "TEST"
        assert result.total_ticks == 3
        assert result.total_agents == 50
        assert result.total_actions > 0
        assert len(result.sentiment_trajectory) == 3
        assert result.coalition_summary == ""
        assert result.ledger_path.endswith("ledger.db")

    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_seed_posts_exist(self, mock_check, tmp_dir):
        personas = make_personas(20)
        model = MarketSentimentModel(
            ticker="SEED",
            data_bundle={
                "research_output": "Research about SEED",
                "financial_summary": "Financial data",
                "news_summary": "News summary",
            },
            graph_context="",
            agent_population=personas,
            budget_mode="lite",
            run_id="seed_test",
        )
        count = model.ledger.count_posts(0)
        assert count >= 1
        model.ledger.close()

    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_follow_graph_seeded(self, mock_check, tmp_dir):
        personas = make_personas(20)
        model = MarketSentimentModel(
            ticker="FOLLOW",
            data_bundle={"research_output": "x", "financial_summary": "y",
                         "news_summary": "z"},
            graph_context="",
            agent_population=personas,
            budget_mode="lite",
            run_id="follow_test",
        )
        follows = model.ledger.conn.execute(
            "SELECT COUNT(*) FROM follows").fetchone()[0]
        assert follows > 0
        model.ledger.close()

    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_too_few_agents_raises(self, mock_check, tmp_dir):
        personas = make_personas(5)
        with pytest.raises(SimulationDegradedError):
            MarketSentimentModel(
                ticker="FAIL",
                data_bundle={},
                graph_context="",
                agent_population=personas,
                budget_mode="lite",
                run_id="fail_test",
            )

    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_sentiment_not_homogenized(self, mock_check, tmp_dir):
        """After a few ticks, bounded assimilation should prevent collapse."""
        personas = make_personas(50)
        model = MarketSentimentModel(
            ticker="HETERO",
            data_bundle={"research_output": "x", "financial_summary": "y",
                         "news_summary": "z"},
            graph_context="Strong earnings\nGood fundamentals",
            agent_population=personas,
            budget_mode="lite",
            run_id="hetero_test",
        )
        result = model.run(ticks=5)
        last_snap = result.sentiment_trajectory[-1]
        nonzero = sum(1 for v in last_snap.values() if v > 0.01)
        assert nonzero >= 1

    @patch("src.abm.simulation.check_model_available", return_value=True)
    def test_data_collector_works(self, mock_check, tmp_dir):
        personas = make_personas(20)
        model = MarketSentimentModel(
            ticker="DC",
            data_bundle={"research_output": "x", "financial_summary": "y",
                         "news_summary": "z"},
            graph_context="",
            agent_population=personas,
            budget_mode="lite",
            run_id="dc_test",
        )
        model.step()
        model.step()
        df = model.datacollector.get_model_vars_dataframe()
        assert len(df) >= 2
        assert "avg_sentiment" in df.columns
        model.ledger.close()
