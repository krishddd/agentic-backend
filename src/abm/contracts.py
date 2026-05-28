"""
MiroFish ABM Data Contracts — Pro Edition.

All shared dataclasses, enums, and exceptions for the ABM sandbox.
SCHEMA_VERSION = 2 — Pro upgrade with dual-channel, contagion, coalitions,
phase transitions, 5-point sentiment, and volume-weighted influence.

Usage:
    from src.abm.contracts import (
        AgentType, SocialAction, PlatformChannel, AgentPersona,
        SimulationResult, ContagionEvent, CoalitionCluster,
        PhaseTransition, CatalystShock,
        PersistedSimulationContext, ValidationResult,
        SimulationDegradedError, ModelNotFoundError,
        SCHEMA_VERSION, BUDGET_MODES, SENTIMENT_LABELS,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------
SCHEMA_VERSION: int = 2

# ---------------------------------------------------------------------------
# 5-point sentiment label mapping (continuous score -> discrete label)
# ---------------------------------------------------------------------------
SENTIMENT_LABELS: Dict[str, tuple] = {
    "strongly_bearish": (-1.0, -0.6),
    "bearish":          (-0.6, -0.2),
    "neutral":          (-0.2,  0.2),
    "bullish":          ( 0.2,  0.6),
    "strongly_bullish": ( 0.6,  1.0),
}


def sentiment_to_label(score: float) -> str:
    """Convert continuous sentiment score to 5-point label."""
    for label, (lo, hi) in SENTIMENT_LABELS.items():
        if lo <= score < hi:
            return label
    return "strongly_bullish" if score >= 0.6 else "strongly_bearish"


# ---------------------------------------------------------------------------
# Budget mode presets — single source of truth for agent counts + ticks
# ---------------------------------------------------------------------------
BUDGET_MODES: Dict[str, Dict[str, int]] = {
    "lite":     {"total": 50,  "kol": 5,  "llm": 10, "rule": 35, "ticks": 10, "mc_paths": 5},
    "standard": {"total": 100, "kol": 10, "llm": 20, "rule": 70, "ticks": 20, "mc_paths": 3},
    "deep":     {"total": 200, "kol": 20, "llm": 40, "rule": 140, "ticks": 30, "mc_paths": 5},
}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AgentType(Enum):
    """Classification of agent LLM usage."""
    RULE_BASED = "rule_based"     # No LLM — template + randomization
    LLM_CITIZEN = "llm_citizen"   # Small LLM for text generation
    KOL = "kol"                   # Key Opinion Leader — LLM-powered, high influence


class SocialAction(Enum):
    """The 12 social actions available to every agent each tick."""
    POST = "post"                 # Create original content
    REPLY = "reply"               # Reply to existing post
    REPOST = "repost"             # Share someone else's post
    LIKE = "like"                 # Endorse a post
    FOLLOW = "follow"             # Start following an agent
    UNFOLLOW = "unfollow"         # Stop following
    SHIFT_OPINION = "shift"       # Internal state change
    BROWSE_FEED = "browse"        # Read curated feed (no visible action)
    QUOTE_POST = "quote"          # Repost with commentary
    BLOCK = "block"               # Sever connection
    SEARCH = "search"             # Look for specific topic
    DO_NOTHING = "idle"           # Skip this tick


class PlatformChannel(Enum):
    """Dual-platform simulation channels (MiroFish Phase 3).

    MICROBLOG: Fast-paced, short-form (Twitter/X archetype).
               Favours virality, rapid contagion, KOL amplification.
    FORUM:     Long-form, threaded discussion (Reddit archetype).
               Favours analysis, debate, slower opinion shifts.
    """
    MICROBLOG = "microblog"
    FORUM = "forum"


# ---------------------------------------------------------------------------
# Event & Analysis Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ContagionEvent:
    """A detected viral cascade moment during simulation.

    Flagged when sentiment shift exceeds the volatility-adjusted threshold:
    ``shift > max(0.15, 2 * rolling_std(last_5_ticks))``
    """
    tick: int
    trigger_agent_id: int
    trigger_agent_name: str
    magnitude: float              # absolute shift in avg sentiment
    direction: str                # "bullish" or "bearish"
    affected_count: int           # agents whose sentiment shifted
    channel: str = ""             # which platform triggered the cascade


@dataclass
class CoalitionCluster:
    """A group of agents with aligned sentiment positions.

    Identified via elbow-method k-selection (k=2..6) on agent sentiments,
    not a fixed k=3.
    """
    cluster_id: int
    agent_ids: List[int]
    mean_sentiment: float
    label: str                    # auto-labelled via sentiment_to_label()
    size: int = 0

    def __post_init__(self):
        self.size = len(self.agent_ids)


@dataclass
class PhaseTransition:
    """A critical tipping point where small perturbation causes macro shift.

    Detected when rolling std deviation drops sharply (convergence) or
    sentiment distribution modality changes.
    """
    tick: int
    pre_sentiment_avg: float
    post_sentiment_avg: float
    delta: float
    trigger: str = ""             # description of what caused it


@dataclass
class CatalystShock:
    """An injected event-driven shock at a specific tick.

    Real market ABMs seed discrete shocks (earnings beat/miss, Fed
    announcement, insider filing) to test resilience. Without this,
    simulations only model baseline drift.
    """
    tick: int
    event_type: str               # "earnings_beat", "earnings_miss", "fed_rate", etc.
    magnitude: float              # -1.0 to 1.0, intensity of the shock
    description: str = ""


# ---------------------------------------------------------------------------
# Agent Persona
# ---------------------------------------------------------------------------

@dataclass
class AgentPersona:
    """Blueprint for a single simulated agent — Pro Edition.

    Created by ``GraphKnowledgeManager.generate_agent_population()`` and
    consumed by the Mesa model's ``_spawn_agents()``.

    ``following`` is pre-populated by ``_wire_follow_graph()`` so the
    network topology is realistic from tick 0.

    Pro additions:
    - ``influence_weight``: Volume-weighted market impact (institutional > retail)
    - ``platform_preference``: Dual-channel preference (microblog vs forum)
    - ``memory_buffer``: Interaction history with exponential decay
    """
    agent_id: int
    name: str
    agent_type: AgentType
    role: str                     # e.g. "retail_investor", "analyst", "insider"
    sentiment: float              # -1.0 (bearish) to 1.0 (bullish)
    influence: float              # 0.0 to 1.0 (KOLs have high influence)
    stubbornness: float           # 0.0 (easily swayed) to 1.0 (immovable)
    model: str = ""               # Ollama model name (empty for rule-based)
    backstory: str = ""
    following: List[int] = field(default_factory=list)
    # --- Pro fields ---
    influence_weight: float = 1.0         # volume-weighted (institutional=10, retail=1)
    platform_preference: str = "microblog"  # "microblog" or "forum"
    memory_buffer: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not -1.0 <= self.sentiment <= 1.0:
            raise ValueError(f"sentiment must be in [-1, 1], got {self.sentiment}")
        if not 0.0 <= self.influence <= 1.0:
            raise ValueError(f"influence must be in [0, 1], got {self.influence}")
        if not 0.0 <= self.stubbornness <= 1.0:
            raise ValueError(f"stubbornness must be in [0, 1], got {self.stubbornness}")


# ---------------------------------------------------------------------------
# Simulation Result
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Raw result from a single Mesa simulation run — Pro Edition.

    ``coalition_summary``, ``contagion_events``, and ``top_kol_positions``
    start empty — they are filled post-hoc by the ``SimulationReportAgent``
    in ``_step_abm_report()`` to break the circular dependency between
    simulation output and LLM analysis.

    Pro additions:
    - 5-point sentiment distribution
    - Per-channel trajectory
    - Contagion events, coalitions, phase transitions
    - Dominant path and variance for MC analysis
    """
    ticker: str
    total_ticks: int
    total_agents: int
    total_actions: int
    final_sentiment_distribution: Dict[str, float]     # 5-point scale
    sentiment_trajectory: List[Dict[str, float]]       # per-tick snapshots
    coalition_summary: str = ""                         # filled by ReportAgent
    contagion_events: List[str] = field(default_factory=list)   # compat
    top_kol_positions: Dict[str, str] = field(default_factory=dict)
    information_gain: float = 0.0
    ledger_path: str = ""
    # --- Pro fields ---
    contagion_events_detail: List[dict] = field(default_factory=list)
    coalitions: List[dict] = field(default_factory=list)
    phase_transitions: List[dict] = field(default_factory=list)
    channel_trajectories: Dict[str, List[Dict[str, float]]] = field(
        default_factory=dict)                          # per-channel breakdown
    dominant_path_index: int = 0
    path_variance: float = 0.0
    # --- Grandmaster fields ---
    agent_states: List[dict] = field(default_factory=list)   # per-agent final state


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

@dataclass
class PersistedSimulationContext:
    """Replaces the v5 ``PersistedAgentContext``.

    ABM agents are Mesa objects and cannot be individually serialized.
    We persist the ledger path + results + report text instead.
    """
    run_id: str
    ledger_path: str              # path to SQLite db
    simulation_result: dict       # ``dataclasses.asdict(SimulationResult)``
    simulation_report: str        # ReportAgent narrative output


@dataclass
class ValidationResult:
    """Output of ``JudgeValidator.adversarial_validate()``."""
    claim_survival_rate: float
    total_claims: int
    supported_claims: int
    flagged_claims: List[Dict[str, str]]
    corrected_report: str
    should_replace: bool


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SimulationDegradedError(Exception):
    """Raised when too few agents spawn or simulation produces no actions."""
    pass


class ModelNotFoundError(Exception):
    """Raised when a required Ollama model is not pulled locally."""
    pass
