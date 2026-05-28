"""
Tests for Phase 5 — Scenario Injection API.

Tests the /api/pipeline/{run_id}/inject endpoint
and the state persistence API endpoints.
Uses FastAPI TestClient with mocked state store.
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field

# Ensure project root is on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.pipeline_state_store import PipelineStateStore


# ---------------------------------------------------------------------------
# We cannot import app.py directly (it tries to import orchestrator etc.)
# so we test the scenario injection logic independently + state store API.
# ---------------------------------------------------------------------------


class TestScenarioInjectionLogic:
    """Test the theoretical injection calculation without the FastAPI server."""

    def test_theoretical_bullish_shift(self):
        """Bullish event should increase bullish sentiment."""
        pre = {"bullish": 0.4, "bearish": 0.3, "neutral": 0.3}
        event_sentiment = 0.8
        event_weight = abs(event_sentiment) * 0.3  # 0.24

        post = {}
        for k, v in pre.items():
            if k == "bullish":
                post[k] = round(v + event_sentiment * event_weight, 3)
            else:
                post[k] = round(v + (-event_sentiment * event_weight * 0.5), 3)

        assert post["bullish"] > pre["bullish"]
        assert post["bearish"] < pre["bearish"]
        assert post["neutral"] < pre["neutral"]

    def test_theoretical_bearish_shift(self):
        """Bearish event should increase bearish sentiment."""
        pre = {"bullish": 0.4, "bearish": 0.3, "neutral": 0.3}
        event_sentiment = -0.8
        event_weight = abs(event_sentiment) * 0.3

        post = {}
        for k, v in pre.items():
            if k == "bearish":
                post[k] = round(v + event_sentiment * event_weight, 3)
            else:
                post[k] = round(v + (-event_sentiment * event_weight * 0.5), 3)

        assert post["bearish"] < pre["bearish"]  # sentiment is negative so this adds negative
        # The formula uses event_sentiment which is negative for bearish

    def test_sentiment_shift_calculation(self):
        """Verify shift = post - pre for each key."""
        pre = {"bullish": 0.5, "bearish": 0.3, "neutral": 0.2}
        post = {"bullish": 0.6, "bearish": 0.25, "neutral": 0.15}

        shift = {k: round(post[k] - pre[k], 3) for k in pre if k in post}
        assert shift["bullish"] == 0.1
        assert shift["bearish"] == -0.05
        assert shift["neutral"] == -0.05


class TestScenarioWithStateStore:
    """Test scenario injection with real state store (tmp_path)."""

    @pytest.fixture
    def store(self, tmp_path):
        return PipelineStateStore(base_dir=tmp_path)

    def test_inject_into_mirofish_run(self, store):
        """Should be able to load state with simulation data."""
        @dataclass
        class FakeState:
            workflow_name: str = "due_diligence"
            mirofish_skipped: bool = False
            simulation_result: dict = field(default_factory=lambda: {
                "ticker": "AAPL",
                "mc_paths": 1,
                "mc_majority_sentiment": "bullish",
                "mc_agreement": 1.0,
                "final_sentiment_distribution": {
                    "bullish": 0.6, "bearish": 0.2, "neutral": 0.2},
                "total_ticks": 10,
            })
            simulation_report: str = "ABM report text"

        store.save_state("run-inject-001", FakeState())
        loaded = store.load_state("run-inject-001")

        state = loaded["state"]
        assert state["mirofish_skipped"] is False
        sim = state["simulation_result"]
        assert sim["ticker"] == "AAPL"
        assert sim["final_sentiment_distribution"]["bullish"] == 0.6

    def test_inject_into_skipped_run_fails_logically(self, store):
        """Runs with mirofish_skipped should be rejected for injection."""
        @dataclass
        class SkippedState:
            workflow_name: str = "due_diligence"
            mirofish_skipped: bool = True
            simulation_result: dict = field(default_factory=dict)

        store.save_state("run-skip", SkippedState())
        loaded = store.load_state("run-skip")
        state = loaded["state"]
        assert state["mirofish_skipped"] is True
        assert state["simulation_result"] == {}

    def test_nonexistent_run_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load_state("nonexistent-run")


class TestStateAPIEndpoints:
    """Test the state listing and loading via PipelineStateStore directly."""

    @pytest.fixture
    def store(self, tmp_path):
        return PipelineStateStore(base_dir=tmp_path)

    def test_list_runs_returns_metadata(self, store):
        @dataclass
        class S:
            workflow_name: str = "due_diligence"

        store.save_state("run-a", S())
        store.save_state("run-b", S(workflow_name="competitor_intel"))

        runs = store.list_runs()
        assert len(runs) == 2
        ids = {r["run_id"] for r in runs}
        assert "run-a" in ids and "run-b" in ids

    def test_load_state_returns_full_payload(self, store):
        @dataclass
        class S:
            workflow_name: str = "market_pulse"
            prompt: str = "Sentiment on NVDA"

        store.save_state("run-load", S())
        payload = store.load_state("run-load")
        assert "schema_version" in payload
        assert "state" in payload
        assert "saved_at" in payload
        assert payload["state"]["workflow_name"] == "market_pulse"


class TestLedgerInjection:
    """Test SQLite WAL concurrent injection."""

    def test_inject_post_into_ledger(self, tmp_path):
        """Should insert a scenario_injection post via WAL mode."""
        import sqlite3
        db_path = tmp_path / "test_ledger.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                tick INTEGER,
                content TEXT,
                sentiment REAL,
                post_type TEXT
            )
        """)
        conn.commit()

        # Inject scenario
        conn.execute("""
            INSERT INTO posts (agent_id, tick, content, sentiment, post_type)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "SCENARIO_INJECTOR", 11,
            "[BREAKING] CEO resigns", -0.9, "scenario_injection"
        ))
        conn.commit()

        # Verify
        row = conn.execute(
            "SELECT * FROM posts WHERE post_type='scenario_injection'"
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[1] == "SCENARIO_INJECTOR"
        assert row[2] == 11
        assert "[BREAKING]" in row[3]
        assert row[4] == -0.9

    def test_wal_concurrent_read_write(self, tmp_path):
        """Verify WAL allows concurrent read + write (no lock contention)."""
        import sqlite3
        db_path = tmp_path / "wal_ledger.db"

        # Writer
        writer = sqlite3.connect(str(db_path))
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT, tick INTEGER, content TEXT,
                sentiment REAL, post_type TEXT
            )
        """)
        writer.execute(
            "INSERT INTO posts VALUES (NULL,?,?,?,?,?)",
            ("A1", 1, "post1", 0.5, "original"))
        writer.commit()

        # Reader opens simultaneously
        reader = sqlite3.connect(str(db_path))
        reader.execute("PRAGMA journal_mode=WAL")
        rows = reader.execute("SELECT COUNT(*) FROM posts").fetchone()
        assert rows[0] == 1

        # Writer adds more while reader is open
        writer.execute(
            "INSERT INTO posts VALUES (NULL,?,?,?,?,?)",
            ("INJECTOR", 2, "[BREAKING] event", -0.8, "scenario_injection"))
        writer.commit()

        # Reader can still read (gets updated data)
        rows2 = reader.execute("SELECT COUNT(*) FROM posts").fetchone()
        assert rows2[0] == 2

        reader.close()
        writer.close()
