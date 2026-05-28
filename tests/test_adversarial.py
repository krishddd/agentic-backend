"""
Tests for Phase 3 — Adversarial Validation.

Tests the JudgeValidator.adversarial_validate() method and
the orchestrator's _step_adversarial_validation integration.
All LLM calls are mocked for speed.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Ensure project root is on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Ensure Crewai_module is importable
_CREWAI = os.path.join(_ROOT, "Crewai_module")
if _CREWAI not in sys.path:
    sys.path.insert(0, _CREWAI)


# ---------------------------------------------------------------------------
# Mock LLM callers
# ---------------------------------------------------------------------------

def mock_llm_json_claims(prompt, model="qwen3:8b", max_tokens=500):
    """Returns JSON array of claims."""
    return '["Revenue grew 25% YoY", "P/E ratio is 18.5", "Free cash flow is $5.2B"]'


def mock_llm_supported(prompt, model="qwen3:8b", max_tokens=100):
    """Always returns SUPPORTED."""
    return "SUPPORTED — evidence confirms this claim"


def mock_llm_mixed(prompt, model="qwen3:8b", max_tokens=500):
    """Claims extraction returns JSON; check returns mixed verdicts."""
    if "Extract" in prompt or "extract" in prompt:
        return '["Revenue grew 25%", "Market cap is $800B", "CEO resigned"]'
    if "CEO resigned" in prompt:
        return "CONTRADICTED — CEO is still in position"
    if "Market cap" in prompt:
        return "UNSUPPORTED — no evidence about market cap"
    return "SUPPORTED — evidence confirms"


def mock_llm_corrected(prompt, model="qwen3:8b", max_tokens=1500):
    """Returns corrected report text."""
    if "Rewrite" in prompt:
        return "Corrected report with unsupported claims removed."
    return mock_llm_mixed(prompt, model, max_tokens)


# ---------------------------------------------------------------------------
# Test JudgeValidator adversarial methods
# ---------------------------------------------------------------------------

class TestJudgeValidatorAdversarial:
    @pytest.fixture
    def validator(self):
        from evaluation.judge_validator import JudgeValidator
        return JudgeValidator(judge_service=None)

    def test_extract_claims_json(self, validator):
        claims = validator._extract_claims(
            "Revenue grew 25%. P/E is 18.5.",
            caller=mock_llm_json_claims,
        )
        assert len(claims) == 3
        assert "Revenue grew 25% YoY" in claims

    def test_extract_claims_fallback(self, validator):
        def caller(prompt, model="qwen3:8b", max_tokens=500):
            return "- Revenue grew 25%\n- P/E is low\n"
        claims = validator._extract_claims("test report", caller=caller)
        assert len(claims) >= 1

    def test_check_claim_supported(self, validator):
        result = validator._check_claim(
            "Revenue grew 25%",
            {"financial": "Total Revenue: $10B, Growth: 25%"},
            caller=mock_llm_supported,
        )
        assert result["verdict"] == "supported"
        assert result["claim"] == "Revenue grew 25%"

    def test_check_claim_contradicted(self, validator):
        def caller(prompt, model="qwen3:8b", max_tokens=100):
            return "CONTRADICTED — evidence shows 15% growth, not 25%"
        result = validator._check_claim(
            "Revenue grew 25%",
            {"financial": "Growth: 15%"},
            caller=caller,
        )
        assert result["verdict"] == "contradicted"

    def test_check_claim_unsupported(self, validator):
        def caller(prompt, model="qwen3:8b", max_tokens=100):
            return "UNSUPPORTED — no data available"
        result = validator._check_claim(
            "Some claim",
            {"financial": "Unrelated data"},
            caller=caller,
        )
        assert result["verdict"] == "unsupported"

    def test_adversarial_validate_all_supported(self, validator):
        result = validator.adversarial_validate(
            synthesis_report="Revenue grew 25%. P/E is 18.5.",
            evidence={"financial": "Revenue: $10B, Growth: 25%, P/E: 18.5"},
            threshold=0.7,
            llm_caller=mock_llm_supported,
        )
        # mock_llm_supported always says SUPPORTED, but _extract_claims
        # called with same mock will fail JSON parse → fallback lines
        assert result["claim_survival_rate"] <= 1.0
        assert result["should_replace"] is False

    def test_adversarial_validate_below_threshold(self, validator):
        result = validator.adversarial_validate(
            synthesis_report="Revenue grew 25%. Market cap is $800B. CEO resigned.",
            evidence={"financial": "Revenue: $8B", "news": "CEO reaffirmed"},
            threshold=0.9,
            llm_caller=mock_llm_corrected,
        )
        # With mixed verdicts and high threshold, should_replace may be True
        assert "claim_survival_rate" in result
        assert "total_claims" in result
        assert "flagged_claims" in result
        assert isinstance(result["flagged_claims"], list)

    def test_adversarial_validate_empty_report(self, validator):
        def caller(prompt, model="qwen3:8b", max_tokens=500):
            return ""  # No claims extracted
        result = validator.adversarial_validate(
            synthesis_report="",
            evidence={},
            llm_caller=caller,
        )
        assert result["total_claims"] == 0
        assert result["claim_survival_rate"] == 1.0
        assert result["should_replace"] is False

    def test_generate_corrected_report(self, validator):
        verdicts = [
            {"claim": "Bad claim", "verdict": "contradicted",
             "reason": "Evidence says otherwise"},
        ]
        result = validator._generate_corrected_report(
            "Original report with bad claim.",
            verdicts,
            caller=mock_llm_corrected,
        )
        assert "Corrected" in result


# ---------------------------------------------------------------------------
# Test orchestrator integration
# ---------------------------------------------------------------------------

class TestOrchestratorAdversarial:
    @patch("orchestrator.load_dotenv")
    @patch("orchestrator.PromptRouter")
    def test_adversarial_step_dry_run(self, mock_router, mock_env):
        from orchestrator import MultiAgentOrchestrator, WorkflowState
        orch = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
        orch.config = {"mirofish": {"enabled": False}}
        orch.printer = MagicMock()

        state = WorkflowState(
            synthesis_report="Test report with claims.",
        )
        orch._step_adversarial_validation(state, dry_run=True)

        assert "adversarial_validation" in state.agent_sequence
        assert state.validation_result["claim_survival_rate"] == 0.85
        assert state.validation_result["total_claims"] == 12
        assert state.validation_result["supported_claims"] == 10
        assert len(state.validation_result["flagged_claims"]) == 2
        assert state.validation_result["should_replace"] is False

    @patch("orchestrator.load_dotenv")
    @patch("orchestrator.PromptRouter")
    def test_pipeline_includes_adversarial(self, mock_router, mock_env):
        """Dry run due diligence should include adversarial_validation step."""
        from orchestrator import MultiAgentOrchestrator
        orch = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
        orch.config = {"mirofish": {"enabled": True, "default_budget_mode": "lite"}}
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

        assert "adversarial_validation" in state.agent_sequence
        assert state.validation_result["claim_survival_rate"] == 0.85
        assert state.total_steps == 11

    @patch("orchestrator.load_dotenv")
    @patch("orchestrator.PromptRouter")
    def test_pipeline_step_count_without_mirofish(self, mock_router, mock_env):
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
        assert state.total_steps == 9
        assert "adversarial_validation" in state.agent_sequence
