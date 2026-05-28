"""
Phase 4 — Pipeline State Persistence.

Provides save / load / migrate / cleanup for pipeline run state,
including ABM simulation context.
Uses PersistedSimulationContext from contracts (Fix #9).
"""

import json
import shutil
import logging
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, fields

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version — bump when WorkflowState or SimulationContext change shape.
# ---------------------------------------------------------------------------
try:
    from src.abm.contracts import SCHEMA_VERSION
except ImportError:
    SCHEMA_VERSION = 1


class PipelineStateStore:
    """Persist / restore pipeline run state to ``reports/output/<run_id>/``."""

    STATE_FILE = "state.json"

    def __init__(self, base_dir: str | Path | None = None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent / "reports" / "output"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_state(self, run_id: str, state, sim_context=None) -> Path:
        """Persist a WorkflowState (+ optional simulation context) to disk.

        Args:
            run_id: Unique run identifier (used as directory name).
            state:  WorkflowState dataclass instance.
            sim_context: Optional PersistedSimulationContext dataclass.

        Returns:
            Path to the written state.json file.
        """
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        state_path = run_dir / self.STATE_FILE

        # Serialise state — use asdict if dataclass, else dict()
        try:
            state_dict = asdict(state)
        except TypeError:
            state_dict = dict(state) if hasattr(state, "__iter__") else {}

        payload = {
            "schema_version": SCHEMA_VERSION,
            "state": state_dict,
            "simulation_context": (
                asdict(sim_context) if sim_context else None),
            "saved_at": datetime.now().isoformat(),
        }

        with self._lock:
            state_path.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )

        logger.info("Saved pipeline state → %s", state_path)
        return state_path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_state(self, run_id: str) -> dict:
        """Load persisted state for *run_id*.

        Automatically migrates older schema versions.

        Returns:
            Full payload dict with keys:
            ``schema_version``, ``state``, ``simulation_context``, ``saved_at``.

        Raises:
            FileNotFoundError: if the run directory or state file is missing.
        """
        state_path = self.base_dir / run_id / self.STATE_FILE
        if not state_path.exists():
            raise FileNotFoundError(
                f"No saved state for run '{run_id}' at {state_path}")

        payload = json.loads(state_path.read_text(encoding="utf-8"))

        if payload.get("schema_version", 0) < SCHEMA_VERSION:
            payload = self._migrate(payload)
            # Re-save migrated version
            state_path.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )

        return payload

    # ------------------------------------------------------------------
    # List runs
    # ------------------------------------------------------------------

    def list_runs(self) -> list[dict]:
        """Return metadata for every persisted run (id, saved_at, workflow)."""
        runs = []
        for child in sorted(self.base_dir.iterdir(), reverse=True):
            sf = child / self.STATE_FILE
            if not sf.exists():
                continue
            try:
                data = json.loads(sf.read_text(encoding="utf-8"))
                runs.append({
                    "run_id": child.name,
                    "saved_at": data.get("saved_at", ""),
                    "workflow": data.get("state", {}).get(
                        "workflow_name", ""),
                    "schema_version": data.get("schema_version", 0),
                })
            except (json.JSONDecodeError, OSError):
                continue
        return runs

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def _migrate(self, payload: dict) -> dict:
        """Forward-migrate from older schema versions."""
        v = payload.get("schema_version", 0)
        logger.info("Migrating state from schema v%d → v%d", v, SCHEMA_VERSION)

        state = payload.setdefault("state", {})

        if v < 1:
            # Fields added in Phase 1.5 / 2
            state.setdefault("mirofish_skipped", False)
            state.setdefault("simulation_result", {})
            state.setdefault("simulation_report", "")
            state.setdefault("graph_context", "")
            state.setdefault("validation_result", {})
            state.setdefault("abm_budget_mode", "lite")

        payload["schema_version"] = SCHEMA_VERSION
        return payload

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_old_runs(self, max_age_days: int = 7) -> list[str]:
        """Remove run directories older than *max_age_days*.

        Uses ``saved_at`` from the state file (not filesystem mtime).
        Thread-safe via ``self._lock``.

        Returns list of deleted run_id values.
        """
        deleted: list[str] = []
        now = datetime.now()

        with self._lock:
            for run_dir in list(self.base_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                sf = run_dir / self.STATE_FILE
                if not sf.exists():
                    continue
                try:
                    saved_at = datetime.fromisoformat(
                        json.loads(sf.read_text(encoding="utf-8"))["saved_at"])
                    if (now - saved_at).days > max_age_days:
                        shutil.rmtree(run_dir)
                        deleted.append(run_dir.name)
                        logger.info("Cleaned up old run: %s", run_dir.name)
                except (json.JSONDecodeError, KeyError, ValueError, OSError):
                    continue

        return deleted

    # ------------------------------------------------------------------
    # Delete single run
    # ------------------------------------------------------------------

    def delete_run(self, run_id: str) -> bool:
        """Delete a specific run's persisted state."""
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            return False
        shutil.rmtree(run_dir)
        logger.info("Deleted run: %s", run_id)
        return True
