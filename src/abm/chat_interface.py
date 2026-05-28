"""
Deep Interaction — Chat Interface (Grandmaster Edition).

Enables post-simulation conversations with:
1. **Individual agents** — respond in-character using their persona,
   sentiment history, memory buffer, and coalition membership
2. **ReportAgent** — answer follow-up questions using toolset to
   query the post-simulation ledger and graph

Both build LLM prompts dynamically from the simulation state.
"""

import logging
from typing import Any, Dict, List, Optional

from src.abm.llm_utils import call_ollama
from src.abm.contracts import SimulationResult, AgentPersona

logger = logging.getLogger(__name__)


class AgentChatInterface:
    """Chat with any individual agent from a completed simulation.

    Reconstructs the agent's perspective using their persona data,
    memory buffer, sentiment trajectory, and coalition membership.
    """

    def __init__(
        self,
        personas: List[AgentPersona],
        sim_result: SimulationResult,
        coalitions: Optional[List[dict]] = None,
        model: str = "qwen3:8b",
    ):
        self.personas = {p.agent_id: p for p in personas}
        self.sim_result = sim_result
        self.coalitions = coalitions or []
        self.model = model
        self._agent_coalition_map = self._build_coalition_map()

    def _build_coalition_map(self) -> Dict[int, dict]:
        """Map agent IDs to their coalition membership."""
        mapping = {}
        for c in self.coalitions:
            for aid in c.get("agent_ids", []):
                mapping[aid] = {
                    "cluster_id": c.get("cluster_id"),
                    "label": c.get("label", "unknown"),
                    "size": c.get("size", 0),
                    "mean_sentiment": c.get("mean_sentiment", 0.0),
                }
        return mapping

    def chat(
        self,
        agent_id: int,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Chat with a specific agent.

        Args:
            agent_id: ID of the agent to chat with.
            user_message: User's message.
            conversation_history: Optional prior turns.

        Returns:
            Dict with 'response', 'agent_name', 'sentiment', 'coalition'.
        """
        persona = self.personas.get(agent_id)
        if not persona:
            return {
                "response": f"Agent {agent_id} not found in simulation.",
                "error": True,
            }

        # Build agent's perspective prompt
        coalition = self._agent_coalition_map.get(agent_id, {})

        # Format conversation history
        history_text = ""
        if conversation_history:
            for turn in conversation_history[-5:]:  # last 5 turns
                role = turn.get("role", "user")
                content = turn.get("content", "")
                history_text += f"\n{role}: {content}"

        # Memory — read from agent_states (runtime data), not persona (init-time)
        memory_text = ""
        agent_state = next(
            (s for s in getattr(self.sim_result, 'agent_states', [])
             if s.get('agent_id') == agent_id), {})
        memories = agent_state.get('memory_buffer', [])
        if memories:
            memory_text = "\n".join(f"  - {m}" for m in memories[-8:])

        prompt = (
            f"You are {persona.name}, a {persona.role} in a market "
            f"sentiment simulation about {self.sim_result.ticker}.\n\n"
            f"YOUR PROFILE:\n"
            f"  Role: {persona.role}\n"
            f"  Influence: {persona.influence:.2f}\n"
            f"  Current sentiment: {persona.sentiment:+.3f} "
            f"({'bullish' if persona.sentiment > 0.2 else 'bearish' if persona.sentiment < -0.2 else 'neutral'})\n"
            f"  Stubbornness: {persona.stubbornness:.2f}\n"
            f"  Platform: {persona.platform_preference}\n"
        )

        if persona.backstory:
            prompt += f"  Backstory: {persona.backstory}\n"

        if coalition:
            prompt += (
                f"\nYOUR COALITION:\n"
                f"  Cluster: {coalition.get('label', 'unknown')} "
                f"(ID={coalition.get('cluster_id')})\n"
                f"  Members: {coalition.get('size', 0)} agents\n"
                f"  Group sentiment: {coalition.get('mean_sentiment', 0):+.3f}\n"
            )

        if memory_text:
            prompt += f"\nYOUR MEMORIES:\n{memory_text}\n"

        prompt += (
            f"\nSIMULATION CONTEXT:\n"
            f"  Ticker: {self.sim_result.ticker}\n"
            f"  Duration: {self.sim_result.total_ticks} ticks\n"
            f"  Total agents: {self.sim_result.total_agents}\n"
            f"  Final consensus: {self.sim_result.final_sentiment_distribution}\n"
        )

        if history_text:
            prompt += f"\nCONVERSATION SO FAR:{history_text}\n"

        prompt += (
            f"\nUser asks: {user_message}\n\n"
            "Respond IN CHARACTER as this agent. Stay true to your "
            "role, sentiment, and memories. Be specific about why "
            "you hold your current view. Reference your experiences "
            "from the simulation."
        )

        response = call_ollama(
            prompt, model=self.model, max_tokens=500, timeout=60)

        return {
            "response": response,
            "agent_name": persona.name,
            "agent_id": agent_id,
            "agent_role": persona.role,
            "sentiment": persona.sentiment,
            "coalition": coalition.get("label", "none"),
            "error": False,
        }

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all available agents for chat."""
        agents = []
        for aid, persona in sorted(self.personas.items()):
            coalition = self._agent_coalition_map.get(aid, {})
            agents.append({
                "agent_id": aid,
                "name": persona.name,
                "role": persona.role,
                "sentiment": round(persona.sentiment, 3),
                "influence": round(persona.influence, 3),
                "platform": persona.platform_preference,
                "coalition": coalition.get("label", "none"),
            })
        return agents


class ReportAgentChat:
    """Conversational interface to the ReportAgent.

    Uses a tool-augmented LLM to answer follow-up questions about
    the simulation. The agent can query the ledger, graph, and
    analysis results.
    """

    def __init__(
        self,
        sim_result: SimulationResult,
        coalitions: Optional[List[dict]] = None,
        contagion_events: Optional[List[dict]] = None,
        phase_transitions: Optional[List[dict]] = None,
        trend_predictions: Optional[dict] = None,
        report_text: str = "",
        model: str = "qwen3:8b",
    ):
        self.sim_result = sim_result
        self.coalitions = coalitions or []
        self.contagion_events = contagion_events or []
        self.phase_transitions = phase_transitions or []
        self.trend_predictions = trend_predictions
        self.report_text = report_text
        self.model = model

    def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Answer questions about the simulation.

        The ReportAgent builds context from all available simulation
        data and uses it to answer the user's question.

        Args:
            user_message: User's question.
            conversation_history: Optional prior turns.

        Returns:
            Dict with 'response', 'sources_used'.
        """
        # Build comprehensive context from simulation data
        context = self._build_context(user_message)

        # Format history
        history_text = ""
        if conversation_history:
            for turn in conversation_history[-5:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                history_text += f"\n{role}: {content}"

        prompt = (
            "You are the ReportAgent, a senior market analyst who just "
            "completed a comprehensive sentiment simulation analysis. "
            "Answer the user's question using the data below.\n\n"
            f"{context}\n"
        )

        if history_text:
            prompt += f"\nPREVIOUS CONVERSATION:{history_text}\n"

        prompt += (
            f"\nUser asks: {user_message}\n\n"
            "Provide a data-driven answer. Reference specific numbers, "
            "ticks, agents, or events from the simulation. If the user "
            "asks a hypothetical ('what if'), reason through it using "
            "the simulation dynamics."
        )

        response = call_ollama(
            prompt, model=self.model, max_tokens=800, timeout=90)

        return {
            "response": response,
            "sources_used": self._identify_sources_used(user_message),
            "error": False,
        }

    def _build_context(self, question: str) -> str:
        """Build relevant context for the question."""
        lines = []

        # Always include simulation summary
        lines.append(
            f"SIMULATION: {self.sim_result.ticker}, "
            f"{self.sim_result.total_agents} agents, "
            f"{self.sim_result.total_ticks} ticks, "
            f"{self.sim_result.total_actions} actions")
        lines.append(
            f"FINAL SENTIMENT: {self.sim_result.final_sentiment_distribution}")

        # Coalition data
        if self.coalitions:
            lines.append(f"\nCOALITIONS ({len(self.coalitions)} clusters):")
            for c in self.coalitions:
                lines.append(
                    f"  Cluster {c.get('cluster_id')}: "
                    f"{c.get('label')} ({c.get('size')} agents), "
                    f"avg={c.get('mean_sentiment', 0):+.3f}")

        # Contagion events
        if self.contagion_events:
            lines.append(
                f"\nCONTAGION EVENTS ({len(self.contagion_events)}):")
            for e in self.contagion_events:
                lines.append(
                    f"  Tick {e.get('tick')}: {e.get('direction')} "
                    f"cascade, magnitude={e.get('magnitude', 0):.3f}")

        # Phase transitions
        if self.phase_transitions:
            lines.append(
                f"\nPHASE TRANSITIONS ({len(self.phase_transitions)}):")
            for pt in self.phase_transitions:
                lines.append(
                    f"  Tick {pt.get('tick')}: "
                    f"delta={pt.get('delta', 0):+.4f}, "
                    f"trigger={pt.get('trigger')}")

        # Trend predictions
        if self.trend_predictions:
            lines.append(f"\nTREND PREDICTIONS:")
            forecasts = self.trend_predictions.get("forecasts", [])
            for f in forecasts:
                lines.append(
                    f"  T+{f.get('horizon')}: "
                    f"{f.get('predicted_sentiment', 0):+.4f} "
                    f"[{f.get('confidence_low', 0):+.4f}, "
                    f"{f.get('confidence_high', 0):+.4f}] "
                    f"({f.get('confidence_pct', 0)}% confidence)")

        # Trajectory
        lines.append("\nSENTIMENT TRAJECTORY:")
        for i, snap in enumerate(self.sim_result.sentiment_trajectory):
            b = snap.get("bullish", 0) + snap.get("strongly_bullish", 0)
            br = snap.get("bearish", 0) + snap.get("strongly_bearish", 0)
            lines.append(
                f"  Tick {i+1}: bull={b:.0%} bear={br:.0%}")

        # Report excerpt
        if self.report_text:
            lines.append(
                f"\nREPORT EXCERPT:\n{self.report_text[:1500]}")

        return "\n".join(lines)

    def _identify_sources_used(self, question: str) -> List[str]:
        """Identify which data sources are relevant to the question."""
        q_lower = question.lower()
        sources = []
        if any(kw in q_lower for kw in
               ["coalition", "cluster", "group", "faction"]):
            sources.append("coalitions")
        if any(kw in q_lower for kw in
               ["contagion", "cascade", "viral", "spread"]):
            sources.append("contagion_events")
        if any(kw in q_lower for kw in
               ["phase", "transition", "tipping", "critical"]):
            sources.append("phase_transitions")
        if any(kw in q_lower for kw in
               ["predict", "forecast", "future", "trend", "outlook"]):
            sources.append("trend_predictions")
        if any(kw in q_lower for kw in
               ["sentiment", "bull", "bear", "neutral"]):
            sources.append("sentiment_trajectory")
        if not sources:
            sources.append("simulation_summary")
        return sources
