"""
Post-simulation analysis agent — Pro Edition.

Reads the SQLite ledger after the simulation completes, builds a
structured prompt with Pro additions: dual-channel breakdown, contagion
events, coalition map, phase transitions, confidence intervals, and
multi-ledger MC synthesis.

Fills coalition_summary into SimulationResult post-hoc to break the
circular dependency between simulation output and LLM analysis.
"""

import logging
from typing import Dict, List, Optional

from src.abm.contracts import SimulationResult
from src.abm.environment import InteractionLedger
from src.abm.llm_utils import call_ollama

logger = logging.getLogger(__name__)


class SimulationReportAgent:
    """Analyzes a completed simulation ledger and produces a narrative report.
    
    Pro Edition: multi-ledger MC synthesis, contagion events, coalition map,
    phase transitions, per-channel breakdown, statistical summary.
    """

    def __init__(self, model: str = "qwen3:8b"):
        self.model = model

    def analyze(
        self,
        ledger_path: str,
        sim_result: SimulationResult,
        mc_metadata: Optional[Dict] = None,
        all_ledger_paths: Optional[List[str]] = None,
    ) -> str:
        """Read ledger(s), produce narrative report with Pro analysis.

        Pro: synthesizes all MC path ledgers when available, not just path 0.

        Args:
            ledger_path: Path to the primary SQLite ledger database.
            sim_result: Raw simulation result.
            mc_metadata: Optional dict with MC aggregate info.
            all_ledger_paths: Optional list of all MC path ledger paths.

        Returns:
            Narrative analysis string from the LLM.
        """
        ledger = InteractionLedger(ledger_path)
        aux_ledgers = []
        try:
            top_posts = ledger.get_most_engaged_posts(limit=20)
            kol_posts = ledger.get_posts_by_influence(min_influence=0.7)

            # Pro: get per-channel top posts
            micro_posts = ledger.get_most_engaged_posts(
                limit=5, channel="microblog")
            forum_posts = ledger.get_most_engaged_posts(
                limit=5, channel="forum")

            # Pro: multi-ledger MC synthesis — gather divergence info
            mc_divergence_section = ""
            if all_ledger_paths and len(all_ledger_paths) > 1:
                mc_divergence_section = self._analyze_mc_divergence(
                    all_ledger_paths, aux_ledgers)

            # Build Monte Carlo section if multi-run
            mc_section = ""
            if mc_metadata and mc_metadata.get("paths", 1) > 1:
                mc_section = (
                    f"\n\nMONTE CARLO AGGREGATE "
                    f"({mc_metadata['paths']} independent simulation "
                    f"branches):\n"
                    f"- Majority sentiment: {mc_metadata['majority']}\n"
                    f"- Agreement across paths: "
                    f"{mc_metadata['agreement']:.0%}\n"
                    f"- Per-path outcomes: "
                    f"{mc_metadata['path_sentiments']}\n"
                    f"- Dominant path index: "
                    f"{mc_metadata.get('dominant_path_index', 0)}\n"
                    f"- Path variance: "
                    f"{mc_metadata.get('path_variance', 0):.4f}"
                )
                if mc_divergence_section:
                    mc_section += f"\n{mc_divergence_section}"

            # Pro: contagion events section
            contagion_section = self._format_contagion_events(
                sim_result.contagion_events_detail)

            # Pro: coalition section
            coalition_section = self._format_coalitions(
                sim_result.coalitions)

            # Pro: phase transitions section
            phase_section = self._format_phase_transitions(
                sim_result.phase_transitions)

            # Pro: channel breakdown
            channel_section = self._format_channel_trajectories(
                sim_result.channel_trajectories)

            prompt = (
                "You are a senior market analyst reviewing a sentiment "
                "simulation. Produce a DETAILED technical report.\n\n"
                f"SIMULATION: {sim_result.ticker}, "
                f"{sim_result.total_agents} participants, "
                f"{sim_result.total_ticks} periods, "
                f"{sim_result.total_actions} interactions.\n"
                f"Final sentiment (5-point scale): "
                f"{sim_result.final_sentiment_distribution}"
                f"{mc_section}\n\n"
                f"CONTAGION EVENTS DETECTED:\n{contagion_section}\n\n"
                f"COALITION CLUSTERS:\n{coalition_section}\n\n"
                f"PHASE TRANSITIONS:\n{phase_section}\n\n"
                f"DUAL-CHANNEL BREAKDOWN:\n{channel_section}\n\n"
                f"TOP POSTS (by engagement):\n"
                f"{self._format_posts(top_posts)}\n\n"
                f"MICROBLOG TOP POSTS:\n"
                f"{self._format_posts(micro_posts, max_items=5)}\n\n"
                f"FORUM TOP POSTS:\n"
                f"{self._format_posts(forum_posts, max_items=5)}\n\n"
                f"SENTIMENT TRAJECTORY (5-point):\n"
                f"{self._format_trajectory(sim_result.sentiment_trajectory)}"
                f"\n\n"
                f"KEY OPINION LEADER POSTS:\n"
                f"{self._format_kol(kol_posts)}\n\n"
                "Produce DETAILED analysis with sections:\n"
                "1. DOMINANT NARRATIVE — what consensus emerged\n"
                "2. DISSENT — which groups resisted\n"
                "3. CONTAGION EVENTS — cascade moments and triggers\n"
                "4. COALITION MAP — cluster analysis and alignment\n"
                "5. PHASE TRANSITIONS — critical tipping points\n"
                "6. PLATFORM DYNAMICS — microblog vs forum differences\n"
                "7. MONTE CARLO ASSESSMENT — path consistency, variance, "
                "dominant trajectory\n"
                "8. STATISTICAL SUMMARY — final distribution, CI, "
                "skewness\n"
                "9. CONFIDENCE (0-100%)"
            )

            return call_ollama(
                prompt, model=self.model, max_tokens=4000, timeout=180
            )
        finally:
            ledger.close()
            for al in aux_ledgers:
                try:
                    al.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Pro: MC divergence analysis
    # ------------------------------------------------------------------

    def _analyze_mc_divergence(
        self,
        all_ledger_paths: List[str],
        aux_ledgers: list,
    ) -> str:
        """Analyze where MC paths diverged by comparing final snapshots.
        
        Bug fix: previously only used path 0 ledger. Now synthesizes all.
        """
        path_summaries = []
        for i, lp in enumerate(all_ledger_paths):
            try:
                al = InteractionLedger(lp)
                aux_ledgers.append(al)
                # Get last tick from this ledger
                max_tick_row = al.conn.execute(
                    "SELECT MAX(tick) FROM agent_states").fetchone()
                max_tick = max_tick_row[0] if max_tick_row[0] else 0
                if max_tick > 0:
                    snap = al.get_sentiment_snapshot(max_tick)
                    path_summaries.append(
                        f"  Path {i}: {snap}")
            except Exception as e:
                path_summaries.append(f"  Path {i}: error ({e})")

        if path_summaries:
            return (
                "MC PATH DIVERGENCE:\n"
                + "\n".join(path_summaries)
            )
        return ""

    # ------------------------------------------------------------------
    # Pro: formatting helpers for new sections
    # ------------------------------------------------------------------

    @staticmethod
    def _format_contagion_events(events: list) -> str:
        if not events:
            return "(no contagion events detected)"
        lines = []
        for e in events:
            lines.append(
                f"  Tick {e.get('tick', '?')}: {e.get('direction', '?')} "
                f"cascade, magnitude={e.get('magnitude', 0):.3f}, "
                f"threshold={e.get('threshold_used', 0):.3f}")
        return "\n".join(lines)

    @staticmethod
    def _format_coalitions(coalitions: list) -> str:
        if not coalitions:
            return "(no coalitions identified)"
        lines = []
        for c in coalitions:
            lines.append(
                f"  Cluster {c.get('cluster_id', '?')}: "
                f"{c.get('label', '?')} ({c.get('size', 0)} agents), "
                f"avg_sentiment={c.get('mean_sentiment', 0):.3f}")
        return "\n".join(lines)

    @staticmethod
    def _format_phase_transitions(transitions: list) -> str:
        if not transitions:
            return "(no phase transitions detected)"
        lines = []
        for pt in transitions:
            lines.append(
                f"  Tick {pt.get('tick', '?')}: "
                f"pre={pt.get('pre_sentiment_avg', 0):.3f} -> "
                f"post={pt.get('post_sentiment_avg', 0):.3f}, "
                f"trigger: {pt.get('trigger', 'unknown')}")
        return "\n".join(lines)

    @staticmethod
    def _format_channel_trajectories(
        channel_trajs: Dict[str, list]
    ) -> str:
        if not channel_trajs:
            return "(no channel data)"
        lines = []
        for ch, trajs in channel_trajs.items():
            if trajs and any(trajs):
                last = trajs[-1] if trajs else {}
                b = last.get("bullish", 0) + last.get(
                    "strongly_bullish", 0)
                br = last.get("bearish", 0) + last.get(
                    "strongly_bearish", 0)
                lines.append(
                    f"  {ch.upper()}: final bull={b:.0%} bear={br:.0%}")
        return "\n".join(lines) or "(no channel data)"

    # ------------------------------------------------------------------
    # Original formatting helpers (kept for compatibility)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_posts(posts, max_items: int = 10) -> str:
        lines = []
        for p in posts[:max_items]:
            content = (p["content"] if isinstance(p, dict)
                       else p[3])
            score = (p["score"] if isinstance(p, dict) and "score" in p
                     else "?")
            lines.append(f"  [{score} pts] {str(content)[:120]}")
        return "\n".join(lines) or "(no posts)"

    @staticmethod
    def _format_trajectory(trajectory: list) -> str:
        lines = []
        for i, snap in enumerate(trajectory):
            sb = snap.get("strongly_bullish", 0)
            b = snap.get("bullish", 0)
            n = snap.get("neutral", 0)
            br = snap.get("bearish", 0)
            sbr = snap.get("strongly_bearish", 0)
            lines.append(
                f"  Tick {i + 1}: "
                f"s.bull={sb:.0%} bull={b:.0%} neut={n:.0%} "
                f"bear={br:.0%} s.bear={sbr:.0%}"
            )
        return "\n".join(lines) or "(no data)"

    @staticmethod
    def _format_kol(kol_posts, max_items: int = 10) -> str:
        lines = []
        for p in kol_posts[:max_items]:
            content = (p["content"] if isinstance(p, dict)
                       else p[3])
            infl = (p["influence"]
                    if isinstance(p, dict) and "influence" in p
                    else "?")
            lines.append(
                f"  [influence={infl}] {str(content)[:120]}"
            )
        return "\n".join(lines) or "(no KOL posts)"
