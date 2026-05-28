"""
Tests for Phase 1.5 — Orchestrator ABM Integration.

Tests the orchestrator's Monte Carlo ABM integration:
- WorkflowState new fields
- _step_abm_simulation (dry_run + mirofish disabled)
- _step_abm_report (dry_run + mirofish disabled)
- _aggregate_mc_results
- Due diligence pipeline with ABM steps (dry_run)
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import asdict

# Ensure project root is on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.abm.contracts import SimulationResult


# ---------------------------------------------------------------------------
# Test WorkflowState ABM fields
# ---------------------------------------------------------------------------

class TestWorkflowStateABMFields:
    def test_state_has_abm_fields(self):
        """WorkflowState should have all new ABM fields with defaults."""
        from orchestrator import WorkflowState
        state = WorkflowState()
        assert state.graph_context == ""
        assert state.simulation_result == {}
        assert state.simulation_report == ""
        assert state.validation_result == {}
        assert state.mirofish_skipped is False
        assert state.abm_budget_mode == "lite"


# ---------------------------------------------------------------------------
# Test ABM step methods
# ---------------------------------------------------------------------------

class TestABMSteps:
    @pytest.fixture
    def orchestrator(self):
        """Orchestrator with mirofish disabled."""
        with patch("orchestrator.load_dotenv"), \
             patch("orchestrator.PromptRouter"):
            from orchestrator import MultiAgentOrchestrator
            orch = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
            orch.config = {"mirofish": {"enabled": False}}
            orch.printer = MagicMock()
            return orch

    @pytest.fixture
    def orchestrator_mf(self):
        """Orchestrator with mirofish enabled (dry_run only)."""
        with patch("orchestrator.load_dotenv"), \
             patch("orchestrator.PromptRouter"):
            from orchestrator import MultiAgentOrchestrator, WorkflowState
            orch = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
            orch.config = {
                "mirofish": {
                    "enabled": True,
                    "default_budget_mode": "lite",
                }
            }
            orch.printer = MagicMock()
            return orch

    def test_abm_simulation_skips_when_disabled(self, orchestrator):
        from orchestrator import WorkflowState
        state = WorkflowState()
        orchestrator._step_abm_simulation(state, "TEST", dry_run=False)
        assert state.mirofish_skipped is True
        assert state.simulation_result == {}

    def test_abm_report_skips_when_mirofish_skipped(self, orchestrator):
        from orchestrator import WorkflowState
        state = WorkflowState(mirofish_skipped=True)
        orchestrator._step_abm_report(state, "TEST", dry_run=False)
        assert state.simulation_report == ""
        assert "abm_report" not in state.agent_sequence

    def test_abm_simulation_dry_run(self, orchestrator_mf):
        from orchestrator import WorkflowState
        state = WorkflowState(abm_budget_mode="lite")
        orchestrator_mf._step_abm_simulation(state, "AAPL", dry_run=True)
        assert state.mirofish_skipped is False
        assert state.simulation_result["ticker"] == "AAPL"
        assert state.simulation_result["mc_paths"] == 1
        assert state.simulation_result["mc_majority_sentiment"] == "bullish"
        assert state.simulation_result["mc_agreement"] == 1.0
        assert "abm_simulation" in state.agent_sequence

    def test_abm_report_dry_run(self, orchestrator_mf):
        from orchestrator import WorkflowState
        state = WorkflowState(
            simulation_result={"mc_paths": 1, "coalition_summary": ""},
            mirofish_skipped=False,
        )
        orchestrator_mf._step_abm_report(state, "AAPL", dry_run=True)
        assert len(state.simulation_report) > 0
        assert "DRY RUN" in state.simulation_report
        assert "abm_report" in state.agent_sequence


# ---------------------------------------------------------------------------
# Test _aggregate_mc_results
# ---------------------------------------------------------------------------

class TestAggregateMCResults:
    def test_single_path_aggregation(self):
        with patch("orchestrator.load_dotenv"), \
             patch("orchestrator.PromptRouter"):
            from orchestrator import MultiAgentOrchestrator
            orch = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)

        result = SimulationResult(
            ticker="TEST", total_ticks=10, total_agents=50,
            total_actions=100,
            final_sentiment_distribution={
                "bullish": 0.5, "bearish": 0.2, "neutral": 0.3},
            sentiment_trajectory=[],
            ledger_path="/tmp/test.db",
        )
        aggregated = orch._aggregate_mc_results([result])
        assert aggregated["mc_paths"] == 1
        assert aggregated["mc_majority_sentiment"] == "bullish"
        assert aggregated["mc_agreement"] == 1.0
        assert aggregated["ticker"] == "TEST"

    def test_multi_path_aggregation(self):
        with patch("orchestrator.load_dotenv"), \
             patch("orchestrator.PromptRouter"):
            from orchestrator import MultiAgentOrchestrator
            orch = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)

        results = [
            SimulationResult(
                ticker="MC", total_ticks=10, total_agents=50,
                total_actions=100,
                final_sentiment_distribution={
                    "bullish": 0.6, "bearish": 0.2, "neutral": 0.2},
                sentiment_trajectory=[], ledger_path="/tmp/p0.db"),
            SimulationResult(
                ticker="MC", total_ticks=10, total_agents=50,
                total_actions=110,
                final_sentiment_distribution={
                    "bullish": 0.5, "bearish": 0.3, "neutral": 0.2},
                sentiment_trajectory=[], ledger_path="/tmp/p1.db"),
            SimulationResult(
                ticker="MC", total_ticks=10, total_agents=50,
                total_actions=90,
                final_sentiment_distribution={
                    "bullish": 0.2, "bearish": 0.6, "neutral": 0.2},
                sentiment_trajectory=[], ledger_path="/tmp/p2.db"),
        ]
        agg = orch._aggregate_mc_results(results)
        assert agg["mc_paths"] == 3
        # 2 bullish + 1 bearish → majority = bullish
        assert agg["mc_majority_sentiment"] == "bullish"
        assert abs(agg["mc_agreement"] - 2/3) < 0.01
        assert len(agg["mc_path_sentiments"]) == 3
        assert len(agg["mc_all_ledger_paths"]) == 3


# ---------------------------------------------------------------------------
# Test dry_run pipeline includes ABM
# ---------------------------------------------------------------------------

class TestDueDiligenceWithABM:
    @patch("orchestrator.load_dotenv")
    @patch("orchestrator.PromptRouter")
    def test_dry_run_pipeline_includes_abm_steps(self, mock_router, mock_env):
        """Dry run due diligence with mirofish enabled should include ABM steps."""
        from orchestrator import MultiAgentOrchestrator
        mock_router_instance = MagicMock()
        mock_router.return_value = mock_router_instance

        orch = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
        orch.config = {
            "mirofish": {
                "enabled": True,
                "default_budget_mode": "lite",
            }
        }
        orch.printer = MagicMock()
        orch.router = mock_router_instance
        orch.base_dir = MagicMock()
        orch._marketing_path = "/tmp"

        intent = MagicMock()
        intent.primary_ticker = "TEST"
        intent.primary_company = "Test Corp"
        intent.tickers = ["TEST"]
        intent.raw_prompt = "Analyze TEST"
        intent.receiver_email = "test@example.com"

        state = orch._pipeline_due_diligence(intent, dry_run=True)

        assert "abm_simulation" in state.agent_sequence
        assert "abm_report" in state.agent_sequence
        assert state.simulation_result["mc_paths"] == 1
        assert len(state.simulation_report) > 0
        assert state.total_steps == 11

    @patch("orchestrator.load_dotenv")
    @patch("orchestrator.PromptRouter")
    def test_dry_run_pipeline_skips_abm_when_disabled(self, mock_router, mock_env):
        from orchestrator import MultiAgentOrchestrator
        orch = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
        orch.config = {"mirofish": {"enabled": False}}
        orch.printer = MagicMock()
        orch.router = MagicMock()
        orch.base_dir = MagicMock()
        orch._marketing_path = "/tmp"

        intent = MagicMock()
        intent.primary_ticker = "TEST"
        intent.primary_company = "Test Corp"
        intent.tickers = ["TEST"]
        intent.raw_prompt = "Analyze TEST"
        intent.receiver_email = "test@example.com"

        state = orch._pipeline_due_diligence(intent, dry_run=True)

        assert "abm_simulation" not in state.agent_sequence
        assert state.mirofish_skipped is True
        assert state.total_steps == 9
