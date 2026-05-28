"""test_report_agent — Formatting helpers + MC metadata in prompt."""
import os, sys, pytest
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from unittest.mock import patch
from src.abm.report_agent import SimulationReportAgent
from src.abm.contracts import SimulationResult
from src.abm.environment import InteractionLedger

@pytest.fixture
def agent():
    return SimulationReportAgent(model="qwen3:8b")

@pytest.fixture
def sim_result():
    return SimulationResult(
        ticker="TST", total_agents=50, total_ticks=10,
        total_actions=200, sentiment_trajectory=[
            {"bullish": 0.4, "bearish": 0.3, "neutral": 0.3},
            {"bullish": 0.5, "bearish": 0.25, "neutral": 0.25},
        ],
        final_sentiment_distribution={"bullish": 0.55, "bearish": 0.2, "neutral": 0.25},
        coalition_summary="", ledger_path="")

class TestFormatPosts:
    def test_format_posts_empty(self, agent):
        assert agent._format_posts([]) == "(no posts)"

    def test_format_posts_dict(self, agent):
        posts = [{"content": "Test post", "score": 15}]
        out = agent._format_posts(posts)
        assert "[15 pts]" in out
        assert "Test post" in out

    def test_format_posts_truncates(self, agent):
        posts = [{"content": "x" * 200, "score": 5}]
        out = agent._format_posts(posts)
        assert len(out) < 200  # truncated to 120 chars

class TestFormatTrajectory:
    def test_format_trajectory_empty(self, agent):
        assert agent._format_trajectory([]) == "(no data)"

    def test_format_trajectory_with_data(self, agent):
        traj = [{"bullish": 0.4, "bearish": 0.3, "neutral": 0.3}]
        out = agent._format_trajectory(traj)
        assert "Tick 1" in out
        assert "bull=" in out

class TestFormatKOL:
    def test_format_kol_empty(self, agent):
        assert agent._format_kol([]) == "(no KOL posts)"

    def test_format_kol_with_data(self, agent):
        posts = [{"content": "KOL analysis", "influence": 0.95}]
        out = agent._format_kol(posts)
        assert "influence=0.95" in out

class TestAnalyze:
    @patch("src.abm.report_agent.call_ollama",
           return_value="DOMINANT NARRATIVE: Bullish consensus. CONFIDENCE: 85%")
    def test_analyze_produces_report(self, mock_llm, agent, sim_result, tmp_path):
        ledger = InteractionLedger(str(tmp_path / "rpt.db"))
        ledger.record_post(0, 1, "Bullish on TST", 0.6)
        ledger.record_agent_state(1, 0, 0.6, 0.9)
        ledger.close()
        report = agent.analyze(str(tmp_path / "rpt.db"), sim_result)
        assert "DOMINANT NARRATIVE" in report

    @patch("src.abm.report_agent.call_ollama",
           return_value="MC analysis across 5 paths.")
    def test_analyze_with_mc_metadata(self, mock_llm, agent, sim_result, tmp_path):
        ledger = InteractionLedger(str(tmp_path / "mc_rpt.db"))
        ledger.record_post(0, 1, "Test", 0.5)
        ledger.close()
        mc_meta = {
            "paths": 5, "majority": "bullish",
            "agreement": 0.8, "path_sentiments": ["bullish"] * 4 + ["neutral"]}
        report = agent.analyze(str(tmp_path / "mc_rpt.db"), sim_result, mc_meta)
        # Verify MC metadata was in the prompt
        prompt = mock_llm.call_args[0][0]
        assert "MONTE CARLO AGGREGATE" in prompt
        assert "5 independent" in prompt
