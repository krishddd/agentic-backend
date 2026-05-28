"""
Mesa Model — MarketSentimentModel simulation orchestrator.

Manages agent spawning, follow-graph seeding, environment setup, tick
execution, data collection, and result compilation. Connection lifecycle
is handled with try/finally to ensure ledger.close() always runs.
"""

import mesa
import logging
from datetime import datetime
from typing import List

from src.abm.agents import RuleBasedCitizen, LLMCitizen, KOLAgent
from src.abm.environment import InteractionLedger, RecommendationEngine
from src.abm.contracts import (
    SimulationResult, SimulationDegradedError,
    ModelNotFoundError, AgentPersona, BUDGET_MODES,
    CatalystShock,
)
from src.abm.llm_utils import check_model_available
from src.abm import analysis as abm_analysis

logger = logging.getLogger(__name__)


class MarketSentimentModel(mesa.Model):
    """Mesa ABM model for market sentiment simulation.

    Orchestrates a population of hybrid agents (rule-based, LLM, KOL)
    interacting on a simulated social platform with a SQLite ledger.
    """

    def __init__(
        self,
        ticker: str,
        data_bundle: dict,
        graph_context: str,
        agent_population: List[AgentPersona],
        budget_mode: str = "lite",
        run_id: str = "",
        llm_citizen_model: str = "llama3.2:latest",
        kol_model: str = "llama3.2:latest",
        recommendation_interest_weight: float = 0.7,
        random_seed: int = 0,
    ):
        super().__init__()
        self.ticker = ticker
        self.tick = 0
        self.data_bundle = data_bundle
        self.random_seed = random_seed

        # Generate run_id if not provided
        self.run_id = run_id or (
            datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + ticker
        )

        # Pre-flight model check
        required_models = {llm_citizen_model, kol_model}
        missing = [m for m in required_models
                   if not check_model_available(m)]
        if missing:
            raise ModelNotFoundError(
                f"Missing Ollama models: {missing}. "
                f"Run: ollama pull <model>"
            )

        mode = BUDGET_MODES[budget_mode]

        # LLM call estimate for logging
        est_llm_calls = int(
            (mode["llm"] * 0.25 + mode["kol"] * 0.30) * mode["ticks"] + 1
        )
        est_time = est_llm_calls * 2 / 60  # ~2s per Ollama call
        logger.info(
            f"ABM [{budget_mode}]: {mode['total']} agents × "
            f"{mode['ticks']} ticks, ~{est_llm_calls} LLM calls, "
            f"~{est_time:.1f} min"
        )

        # Environment — connection managed with close()
        db_path = f"results/simulations/{self.run_id}/ledger.db"
        self.ledger = InteractionLedger(db_path)
        self.recommender = RecommendationEngine(
            self.ledger, recommendation_interest_weight
        )

        # Template reasons extracted from graph context (for rule-based)
        self.template_reasons = self._extract_reasons(graph_context)

        # Spawn agents by influence rank
        self._spawn_agents(
            agent_population, mode, llm_citizen_model, kol_model
        )

        # Pre-populate follow graph in ledger from persona.following
        self._seed_follow_graph(agent_population[:mode["total"]])

        # Seed environment with initial market data posts
        self._seed_environment(data_bundle)

        # Pro: catalyst shocks to inject at specific ticks
        self.catalyst_shocks: List[CatalystShock] = []
        self._generate_catalysts(data_bundle)

        # Pro: contagion events and phase transitions tracked per-tick
        self.contagion_events: list = []
        self.phase_transitions: list = []

        # Live sentiment trajectory — agents can read for mid-sim foresight
        self.sentiment_trajectory: list = []

        # Mesa data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "avg_sentiment": lambda m: m._avg_sentiment(),
                "sentiment_std": lambda m: m._sentiment_std(),
                "total_posts": lambda m: m.ledger.count_posts(m.tick),
            },
            agent_reporters={
                "sentiment": "sentiment",
                "influence": "influence",
            },
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _spawn_agents(self, population: List[AgentPersona], mode: dict,
                      citizen_model: str, kol_model: str):
        """Spawn agents sorted by influence: top → KOL, mid → LLM, rest → rule."""
        if len(population) < 10:
            raise SimulationDegradedError(
                f"Need ≥10 personas from GraphRAG, got {len(population)}"
            )
        sorted_pop = sorted(
            population, key=lambda p: p.influence, reverse=True
        )
        for i, persona in enumerate(sorted_pop[:mode["total"]]):
            if i < mode["kol"]:
                persona.model = kol_model
                KOLAgent(self, persona, self.ledger, self.recommender)
            elif i < mode["kol"] + mode["llm"]:
                persona.model = citizen_model
                LLMCitizen(self, persona, self.ledger, self.recommender)
            else:
                RuleBasedCitizen(
                    self, persona, self.ledger, self.recommender
                )

    def _seed_follow_graph(self, population: List[AgentPersona]):
        """Write pre-populated follow relationships to ledger at tick 0.

        These come from _wire_follow_graph() in GraphKnowledgeManager.
        """
        for persona in population:
            for followed_id in persona.following:
                self.ledger.record_follow(
                    persona.agent_id, followed_id, tick=0
                )

    def _seed_environment(self, data_bundle: dict):
        """Inject initial market data as seed posts at tick 0.
        Pro: uses 500 chars per source (up from 200) for richer context.
        """
        seeds = [
            (data_bundle.get(k, "")[:500], ch)
            for k, ch in [
                ("research_output", "forum"),
                ("financial_summary", "forum"),
                ("news_summary", "microblog"),
                ("sec_analysis", "forum"),
                ("sentiment_data", "microblog"),
            ]
        ]
        for content, channel in seeds:
            if content.strip():
                self.ledger.record_post(
                    0, -1, content, 0.0, channel=channel)

    def _generate_catalysts(self, data_bundle: dict):
        """Pro: Auto-generate catalyst shocks from data bundle.
        
        Scans news/financial data for shock keywords and schedules
        injections at specific ticks during simulation.
        """
        news = data_bundle.get("news_summary", "").lower()
        fin = data_bundle.get("financial_summary", "").lower()
        combined = news + " " + fin

        shock_patterns = [
            ("earnings beat", "earnings_beat", 0.3),
            ("earnings miss", "earnings_miss", -0.3),
            ("revenue growth", "positive_earnings", 0.2),
            ("revenue decline", "negative_earnings", -0.2),
            ("fed rate", "fed_rate", -0.15),
            ("rate cut", "fed_rate_cut", 0.2),
            ("sec investigation", "sec_investigation", -0.25),
            ("lawsuit", "legal_risk", -0.2),
            ("acquisition", "acquisition", 0.15),
            ("downgrade", "analyst_downgrade", -0.2),
            ("upgrade", "analyst_upgrade", 0.2),
            ("insider sell", "insider_selling", -0.15),
        ]

        import random as _rng
        for pattern, event_type, magnitude in shock_patterns:
            if pattern in combined:
                # Schedule at a random tick in the first 70% of simulation
                max_tick = max(3, int(BUDGET_MODES.get(
                    "lite", {}).get("ticks", 10) * 0.7))
                tick = _rng.randint(2, max_tick)
                self.catalyst_shocks.append(CatalystShock(
                    tick=tick,
                    event_type=event_type,
                    magnitude=magnitude,
                    description=f"Auto-detected: '{pattern}' in data",
                ))

    @staticmethod
    def _extract_reasons(graph_context: str) -> list:
        """Extract short reason phrases from graph context for templates."""
        if not graph_context:
            return ["market fundamentals", "recent earnings",
                    "industry trends"]
        lines = [line.strip() for line in graph_context.split('\n')
                 if len(line.strip()) > 10]
        return lines[:20] or ["market conditions"]

    # ------------------------------------------------------------------
    # Simulation execution
    # ------------------------------------------------------------------

    def step(self):
        """Execute one tick: shuffle agents, inject catalysts, run, analyze."""
        self.tick += 1

        # Pro: inject catalyst shocks at scheduled ticks
        for shock in self.catalyst_shocks:
            if shock.tick == self.tick:
                self._inject_catalyst(shock)

        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

    def _inject_catalyst(self, shock: CatalystShock):
        """Pro: Inject an event-driven shock at this tick.
        
        Shifts all agents' sentiment by a fraction of the shock magnitude,
        modulated by their stubbornness.
        """
        logger.info(
            f"Catalyst injected at tick {self.tick}: "
            f"{shock.event_type} (magnitude={shock.magnitude:.2f})")
        for agent in self.agents:
            openness = 1.0 - agent.persona.stubbornness
            shift = shock.magnitude * openness * 0.5
            agent.sentiment = max(-1.0, min(1.0,
                                            agent.sentiment + shift))
        # Record as a system post
        self.ledger.record_post(
            self.tick, -1,
            f"[CATALYST] {shock.event_type}: {shock.description}",
            shock.magnitude, channel="microblog")

    def run(self, ticks: int) -> SimulationResult:
        """Run the full simulation. Closes ledger in finally block."""
        try:
            self.sentiment_trajectory = []
            for _ in range(ticks):
                self.step()
                snap = self.ledger.get_sentiment_snapshot(self.tick)
                self.sentiment_trajectory.append(snap)
                logger.info(
                    f"Tick {self.tick}/{ticks}: "
                    f"avg={self._avg_sentiment():.2f}, "
                    f"std={self._sentiment_std():.2f}"
                )

                # Pro: per-tick contagion detection
                new_contagion = abm_analysis.detect_contagion_events(
                    self.sentiment_trajectory)
                # Only add events from the latest tick
                for evt in new_contagion:
                    if evt["tick"] == len(self.sentiment_trajectory):
                        if evt not in self.contagion_events:
                            self.contagion_events.append(evt)
                            logger.info(
                                f"Contagion detected at tick {evt['tick']}: "
                                f"{evt['direction']} shift={evt['magnitude']:.3f}")

            # Pro: post-simulation phase transition detection
            self.phase_transitions = abm_analysis.detect_phase_transitions(
                self.sentiment_trajectory)
            for pt in self.phase_transitions:
                logger.info(
                    f"Phase transition at tick {pt['tick']}: "
                    f"delta={pt['delta']:.3f}")

            return self._compile_raw_result()
        finally:
            self.ledger.close()

    # ------------------------------------------------------------------
    # Result compilation
    # ------------------------------------------------------------------

    def _compile_raw_result(self) -> SimulationResult:
        """Build SimulationResult with Pro + Grandmaster fields filled."""
        trajectory = []
        for t in range(1, self.tick + 1):
            trajectory.append(self.ledger.get_sentiment_snapshot(t))

        # Pro: per-channel trajectories
        channel_trajectories = {"microblog": [], "forum": []}
        for t in range(1, self.tick + 1):
            ch_snap = self.ledger.get_channel_sentiment_snapshot(t)
            for ch in ("microblog", "forum"):
                channel_trajectories[ch].append(ch_snap.get(ch, {}))

        total_actions = self.ledger.conn.execute(
            "SELECT COUNT(*) FROM actions"
        ).fetchone()[0]

        # Pro: coalition identification
        agent_sents = self.ledger.get_agent_sentiments_at_tick(self.tick)
        coalitions = abm_analysis.identify_coalitions(agent_sents)

        # Grandmaster: per-agent state snapshot for graph/chat/predictions
        agent_states = []
        for a in self.agents:
            coalition_id = -1
            for c in coalitions:
                if a.unique_id in c.get("agent_ids", []):
                    coalition_id = c["cluster_id"]
                    break
            agent_states.append({
                "agent_id": a.unique_id,
                "persona_name": getattr(a.persona, "name", f"Agent_{a.unique_id}"),
                "role": a.persona.agent_type.value if hasattr(a.persona.agent_type, "value") else str(a.persona.agent_type),
                "sentiment": round(a.sentiment, 4),
                "influence": round(getattr(a.persona, "influence", 0.5), 3),
                "stubbornness": round(getattr(a.persona, "stubbornness", 0.3), 3),
                "channel_preference": getattr(a, "channel", "microblog"),
                "memory_buffer": list(a.memory)[-5:] if hasattr(a, "memory") else [],
                "coalition_id": coalition_id,
            })

        return SimulationResult(
            ticker=self.ticker,
            total_ticks=self.tick,
            total_agents=len(list(self.agents)),
            total_actions=total_actions,
            final_sentiment_distribution=(
                trajectory[-1] if trajectory else {}
            ),
            sentiment_trajectory=trajectory,
            coalition_summary="",
            contagion_events=[str(e) for e in self.contagion_events],
            top_kol_positions=self._get_kol_positions(),
            information_gain=self._compute_info_gain(trajectory),
            ledger_path=f"results/simulations/{self.run_id}/ledger.db",
            # Pro fields
            contagion_events_detail=self.contagion_events,
            coalitions=coalitions,
            phase_transitions=self.phase_transitions,
            channel_trajectories=channel_trajectories,
            # Grandmaster fields
            agent_states=agent_states,
        )

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def _avg_sentiment(self) -> float:
        vals = [a.sentiment for a in self.agents]
        return sum(vals) / len(vals) if vals else 0.0

    def _sentiment_std(self) -> float:
        vals = [a.sentiment for a in self.agents]
        if not vals:
            return 0.0
        avg = sum(vals) / len(vals)
        return (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5

    @staticmethod
    def _compute_info_gain(trajectory: list) -> float:
        """Measure how much the sentiment distribution shifted."""
        if len(trajectory) < 2:
            return 0.0
        first = trajectory[0].get("bullish", 0.5)
        last = trajectory[-1].get("bullish", 0.5)
        return abs(last - first)

    def _get_kol_positions(self) -> dict:
        """Capture each KOL agent's final state for top_kol_positions."""
        from src.abm.agents import KOLAgent
        from src.abm.contracts import sentiment_to_label
        positions = {}
        for a in self.agents:
            if isinstance(a, KOLAgent):
                positions[f"KOL_{a.unique_id}"] = {
                    "sentiment": round(a.sentiment, 4),
                    "influence": round(a.influence, 3),
                    "stance": sentiment_to_label(a.sentiment),
                    "last_memory": a.memory[-1] if a.memory else "",
                }
        return positions
