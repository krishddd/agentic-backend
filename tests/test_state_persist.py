"""
Tests for Phase 4 — Pipeline State Persistence.

Tests PipelineStateStore: save, load, migrate, list_runs, cleanup, delete_run.
All tests use a temporary directory.
"""

import os
import sys
import json
import time
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict

# Ensure project root is on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.pipeline_state_store import PipelineStateStore


# ---------------------------------------------------------------------------
# Minimal state dataclass for testing (mirrors WorkflowState shape)
# ---------------------------------------------------------------------------

@dataclass
class MockState:
    workflow_name: str = "due_diligence"
    prompt: str = "Analyze TEST"
    company: str = "Test Corp"
    tickers: list = field(default_factory=lambda: ["TEST"])
    synthesis_report: str = "Test report content"
    simulation_result: dict = field(default_factory=dict)
    simulation_report: str = ""
    mirofish_skipped: bool = False
    validation_result: dict = field(default_factory=dict)
    abm_budget_mode: str = "lite"
    graph_context: str = ""


@dataclass
class MockSimContext:
    run_id: str = "test-run-001"
    ticker: str = "TEST"
    total_ticks: int = 10
    total_agents: int = 50
    ledger_path: str = "/tmp/test.db"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineStateStore:
    @pytest.fixture
    def store(self, tmp_path):
        return PipelineStateStore(base_dir=tmp_path)

    def test_save_and_load(self, store):
        state = MockState(synthesis_report="Full analysis of TEST Corp")
        path = store.save_state("run-001", state)
        assert path.exists()
        assert path.name == "state.json"

        loaded = store.load_state("run-001")
        assert loaded["schema_version"] >= 1
        assert loaded["state"]["workflow_name"] == "due_diligence"
        assert loaded["state"]["synthesis_report"] == "Full analysis of TEST Corp"
        assert loaded["saved_at"] is not None

    def test_save_with_sim_context(self, store):
        state = MockState()
        ctx = MockSimContext(run_id="sim-001", ticker="AAPL")
        store.save_state("run-002", state, sim_context=ctx)

        loaded = store.load_state("run-002")
        assert loaded["simulation_context"] is not None
        assert loaded["simulation_context"]["ticker"] == "AAPL"
        assert loaded["simulation_context"]["run_id"] == "sim-001"

    def test_save_without_sim_context(self, store):
        state = MockState()
        store.save_state("run-003", state)
        loaded = store.load_state("run-003")
        assert loaded["simulation_context"] is None

    def test_load_nonexistent_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load_state("nonexistent-run")

    def test_list_runs_empty(self, store):
        runs = store.list_runs()
        assert runs == []

    def test_list_runs(self, store):
        store.save_state("run-a", MockState(workflow_name="due_diligence"))
        store.save_state("run-b", MockState(workflow_name="competitor_intel"))

        runs = store.list_runs()
        assert len(runs) == 2
        ids = {r["run_id"] for r in runs}
        assert "run-a" in ids
        assert "run-b" in ids
        workflows = {r["workflow"] for r in runs}
        assert "due_diligence" in workflows
        assert "competitor_intel" in workflows

    def test_delete_run(self, store):
        store.save_state("run-del", MockState())
        assert store.delete_run("run-del") is True
        assert store.delete_run("run-del") is False  # already gone
        with pytest.raises(FileNotFoundError):
            store.load_state("run-del")

    def test_cleanup_old_runs(self, store):
        # Create a run with an old timestamp
        state = MockState()
        store.save_state("old-run", state)

        # Manually backdate saved_at
        sf = store.base_dir / "old-run" / "state.json"
        data = json.loads(sf.read_text())
        old_date = (datetime.now() - timedelta(days=10)).isoformat()
        data["saved_at"] = old_date
        sf.write_text(json.dumps(data))

        # Create a fresh run
        store.save_state("new-run", state)

        deleted = store.cleanup_old_runs(max_age_days=7)
        assert "old-run" in deleted
        assert "new-run" not in deleted
        assert not (store.base_dir / "old-run").exists()
        assert (store.base_dir / "new-run").exists()

    def test_cleanup_skips_corrupt_files(self, store):
        # Create a run with corrupt JSON
        run_dir = store.base_dir / "corrupt-run"
        run_dir.mkdir()
        (run_dir / "state.json").write_text("not valid json")

        deleted = store.cleanup_old_runs(max_age_days=0)
        assert "corrupt-run" not in deleted  # should be skipped, not crash


class TestMigration:
    @pytest.fixture
    def store(self, tmp_path):
        return PipelineStateStore(base_dir=tmp_path)

    def test_migrate_v0_to_current(self, store):
        # Write a v0 state (missing ABM fields)
        run_dir = store.base_dir / "old-v0"
        run_dir.mkdir()
        old_payload = {
            "schema_version": 0,
            "state": {
                "workflow_name": "due_diligence",
                "prompt": "Analyze OLD",
            },
            "saved_at": datetime.now().isoformat(),
        }
        (run_dir / "state.json").write_text(json.dumps(old_payload))

        loaded = store.load_state("old-v0")
        # Should have been migrated
        assert loaded["schema_version"] >= 1
        assert loaded["state"]["mirofish_skipped"] is False
        assert loaded["state"]["simulation_result"] == {}
        assert loaded["state"]["simulation_report"] == ""
        assert loaded["state"]["graph_context"] == ""
        assert loaded["state"]["validation_result"] == {}
        assert loaded["state"]["abm_budget_mode"] == "lite"

    def test_current_version_no_migration(self, store):
        state = MockState()
        store.save_state("current", state)
        loaded = store.load_state("current")
        # Should load without migration
        assert loaded["state"]["workflow_name"] == "due_diligence"


class TestOverwrite:
    @pytest.fixture
    def store(self, tmp_path):
        return PipelineStateStore(base_dir=tmp_path)

    def test_save_overwrites_existing(self, store):
        store.save_state("run-ow", MockState(prompt="version1"))
        store.save_state("run-ow", MockState(prompt="version2"))
        loaded = store.load_state("run-ow")
        assert loaded["state"]["prompt"] == "version2"
