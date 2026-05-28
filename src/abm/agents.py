"""
ABM Agent Types — BaseMarketAgent + three concrete subclasses.

- RuleBasedCitizen: No LLM, template-based text, fast.
- LLMCitizen:       Small LLM for text generation, rule-based action selection.
- KOLAgent:         High-influence LLM agent with access to the data bundle.

All agents inherit the Hegselmann-Krause bounded assimilation sentiment
model with Gaussian decay and stochastic noise.
"""

import mesa
import math
import random
import logging
from typing import List

from src.abm.contracts import SocialAction, AgentPersona
from src.abm.environment import InteractionLedger, RecommendationEngine
from src.abm.llm_utils import call_ollama

logger = logging.getLogger(__name__)

# Note: Template strings are now built inline in RuleBasedCitizen._generate_content
# to avoid a mesa import side-effect that corrupts module-level mutable lists.


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseMarketAgent(mesa.Agent):
    """Abstract base for all simulated market participants — Grandmaster Edition.

    Each tick: read feed → update sentiment → decide action → execute → record.

    Grandmaster additions:
    - Dynamic temporal memory with importance-weighted decay
    - Cross-platform awareness (discover hot posts from other channel)
    - Volume-weighted influence in sentiment assimilation
    - Dual-channel post tagging (microblog/forum)
    """

    MEMORY_SIZE = 15      # keep last N interactions (up from 5)
    MEMORY_DECAY = 0.7    # exponential decay base (w = DECAY^age)
    CROSS_PLATFORM_CHANCE = 0.2  # 20% chance to also see other channel

    def __init__(self, model: "mesa.Model", persona: AgentPersona,
                 ledger: InteractionLedger,
                 recommender: RecommendationEngine):
        super().__init__(model)
        self.persona = persona
        self.ledger = ledger
        self.recommender = recommender
        self.sentiment = persona.sentiment
        self.influence = persona.influence
        self.memory: List[str] = list(persona.memory_buffer)  # Pro: memory
        self.channel = persona.platform_preference  # Pro: dual-channel
        # Grandmaster: importance scores for memory entries
        self.memory_importance: List[float] = [
            1.0] * len(self.memory)

    def step(self):
        feed = self.recommender.curate_feed(
            self.persona.agent_id, self.model.tick,
            home_channel=self.channel)  # Pro: dual-channel feed

        # Grandmaster: cross-platform discovery
        if random.random() < self.CROSS_PLATFORM_CHANCE:
            other_ch = "forum" if self.channel == "microblog" else "microblog"
            cross_feed = self.recommender.curate_feed(
                self.persona.agent_id, self.model.tick,
                home_channel=other_ch)
            # Mix in top 2 cross-platform posts
            feed = feed + cross_feed[:2]

        self._update_sentiment(feed)
        action = self._decide_action(feed)
        self._execute_action(action, feed)
        self._record_state()

    # ------------------------------------------------------------------
    # Sentiment — Hegselmann-Krause bounded confidence with Gaussian decay
    # ------------------------------------------------------------------

    def _update_sentiment(self, feed):
        """Non-linear bounded assimilation using Gaussian decay.

        Based on the Hegselmann-Krause bounded confidence model.
        Uses a continuous Gaussian weight that smoothly decays influence
        as opinion distance grows. Adds stochastic noise (epsilon) to
        simulate irrational human mood.

        Prevents: homogenization, oscillation from hard cutoffs.
        Enables: echo chambers, polarization, contagion events.
        """
        if not feed:
            return

        feed_sentiments = []
        for p in feed:
            try:
                # sqlite3.Row supports key access
                sent = p["sentiment"]
            except (KeyError, TypeError, IndexError):
                try:
                    # fallback: dict access
                    sent = p.get("sentiment") if isinstance(p, dict) else None
                except Exception:
                    sent = None
            if sent is not None and isinstance(sent, (int, float)):
                feed_sentiments.append(float(sent))

        if not feed_sentiments:
            return

        # Volume-weighted average: institutional agents have higher impact
        weight_total = 0.0
        weighted_sum = 0.0
        for i, fs in enumerate(feed_sentiments):
            # Use influence_weight if available from feed, fallback to 1.0
            w = 1.0
            if i < len(feed):
                try:
                    w = float(self._get_field(feed[i], "influence", 1.0))
                except (TypeError, AttributeError, ValueError):
                    w = 1.0
            weight_total += w
            weighted_sum += fs * w

        avg_feed_sentiment = (
            weighted_sum / weight_total if weight_total > 0
            else sum(feed_sentiments) / len(feed_sentiments))
        gap = avg_feed_sentiment - self.sentiment

        # Base openness (alpha) — how receptive this agent is
        openness = (1.0 - self.persona.stubbornness) * 0.15

        # Tolerance threshold (tau): Gaussian decay width
        tolerance = 0.4

        # Gaussian weight: 1.0 if gap is 0, decays toward 0 as gap widens
        weight = math.exp(-(gap ** 2) / (2 * tolerance ** 2))

        if abs(gap) < 0.5:
            # Assimilation Zone: Move toward the feed average
            drift = gap * openness * weight
        else:
            # Entrenchment Zone: Boomerang effect (move away)
            # Inverted and dampened so agents don't radicalize too fast
            drift = -(gap * openness * 0.5)

        # Stochastic noise to prevent permanent mathematical gridlock
        noise = random.uniform(-0.02, 0.02)

        # Apply and clamp between -1.0 (max bearish) and 1.0 (max bullish)
        new_sentiment = self.sentiment + drift + noise
        self.sentiment = max(-1.0, min(1.0, new_sentiment))

        # Grandmaster: Add to memory with importance scoring
        importance = min(1.0, abs(gap) * 2)  # bigger gaps = more important
        summary = (f"tick={self.model.tick}: feed_avg="
                   f"{avg_feed_sentiment:.2f}, gap={gap:.2f}")
        self.memory.append(summary)
        self.memory_importance.append(importance)
        # Trim with importance-weighted eviction
        if len(self.memory) > self.MEMORY_SIZE:
            self._evict_least_important()
    # ------------------------------------------------------------------
    # Grandmaster: Memory management
    # ------------------------------------------------------------------

    def _evict_least_important(self):
        """Remove least important memory entry (preserve recent important ones)."""
        if len(self.memory) <= 1:
            return
        # Score = importance * recency (newer = higher index = higher recency)
        scores = []
        for i, (mem, imp) in enumerate(
                zip(self.memory, self.memory_importance)):
            recency = (i + 1) / len(self.memory)  # 0→1
            scores.append(imp * 0.6 + recency * 0.4)
        # Remove the entry with lowest combined score
        min_idx = scores.index(min(scores))
        self.memory.pop(min_idx)
        self.memory_importance.pop(min_idx)

    def _recall_memory(self, max_entries: int = 5) -> str:
        """Recall top-importance memories for LLM prompt context.

        Returns formatted memory string, sorted by importance descending.
        """
        if not self.memory:
            return "(no memories)"
        # Pair memories with importance, sort by importance desc
        paired = list(zip(self.memory, self.memory_importance))
        paired.sort(key=lambda x: x[1], reverse=True)
        top = paired[:max_entries]
        lines = [f"  [{imp:.1f}] {mem}" for mem, imp in top]
        return "AGENT MEMORY:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Action selection (override in subclasses)
    # ------------------------------------------------------------------

    def _decide_action(self, feed) -> SocialAction:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Action execution — dispatches all 12 SocialActions
    # ------------------------------------------------------------------

    @staticmethod
    def _get_field(item, key: str, fallback=None):
        """Safely extract a field from a feed item (Row, dict, or tuple)."""
        try:
            return item[key]
        except (KeyError, TypeError, IndexError):
            return fallback

    def _execute_action(self, action: SocialAction, feed: list):
        tick = self.model.tick
        aid = self.persona.agent_id

        if action == SocialAction.POST:
            content = self._generate_content(feed) or ""
            post_id = self.ledger.record_post(
                tick, aid, content, self.sentiment,
                channel=self.channel)  # Pro: dual-channel
            self.ledger.record_action(
                tick, aid, "post", target_post_id=post_id)
            # Pro: add to memory
            self.memory.append(f"posted: {content[:50]}")
            if len(self.memory) > self.MEMORY_SIZE:
                self.memory = self.memory[-self.MEMORY_SIZE:]

        elif action == SocialAction.REPLY:
            if feed:
                target = random.choice(feed)
                pid = self._get_field(target, "post_id")
                if pid is not None:
                    content = self._generate_reply(target)
                    self.ledger.record_action(
                        tick, aid, "reply", target_post_id=pid,
                        content=content)
                    self.ledger.increment_post_stat(pid, "replies")

        elif action == SocialAction.REPOST:
            if feed:
                target = random.choice(feed)
                pid = self._get_field(target, "post_id")
                if pid is not None:
                    self.ledger.record_action(
                        tick, aid, "repost", target_post_id=pid)
                    self.ledger.increment_post_stat(pid, "reposts")

        elif action == SocialAction.LIKE:
            if feed:
                target = random.choice(feed)
                pid = self._get_field(target, "post_id")
                if pid is not None:
                    self.ledger.record_action(
                        tick, aid, "like", target_post_id=pid)
                    self.ledger.increment_post_stat(pid, "likes")

        elif action == SocialAction.FOLLOW:
            if feed:
                target = random.choice(feed)
                t_aid = self._get_field(target, "agent_id")
                if t_aid is not None and t_aid != aid:
                    self.ledger.record_follow(aid, t_aid, tick)
                    self.ledger.record_action(
                        tick, aid, "follow", target_agent_id=t_aid)

        elif action == SocialAction.UNFOLLOW:
            self.ledger.record_action(tick, aid, "unfollow")

        elif action == SocialAction.QUOTE_POST:
            if feed:
                target = random.choice(feed)
                pid = self._get_field(target, "post_id")
                if pid is not None:
                    content = self._generate_content(feed)
                    self.ledger.record_action(
                        tick, aid, "quote", target_post_id=pid,
                        content=content)
                    self.ledger.increment_post_stat(pid, "reposts")

        elif action == SocialAction.SHIFT_OPINION:
            self.ledger.record_action(tick, aid, "shift")

        elif action in (SocialAction.BROWSE_FEED, SocialAction.SEARCH,
                        SocialAction.DO_NOTHING, SocialAction.BLOCK):
            self.ledger.record_action(tick, aid, action.value)

    # ------------------------------------------------------------------
    # State recording
    # ------------------------------------------------------------------

    def _record_state(self):
        """Snapshot sentiment and influence to agent_states table."""
        self.ledger.record_agent_state(
            self.persona.agent_id, self.model.tick,
            self.sentiment, self.influence)

    # ------------------------------------------------------------------
    # Content generation (override in subclasses)
    # ------------------------------------------------------------------

    def _generate_content(self, feed) -> str:
        raise NotImplementedError

    def _generate_reply(self, target_post) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Rule-Based Citizen — No LLM, template + randomization
# ---------------------------------------------------------------------------

class RuleBasedCitizen(BaseMarketAgent):
    """Fastest agent type. Uses templates for text, weighted random for actions."""

    def _decide_action(self, feed) -> SocialAction:
        weights = [0.40, 0.20, 0.15, 0.10, 0.10, 0.05]
        actions = [
            SocialAction.BROWSE_FEED, SocialAction.LIKE,
            SocialAction.REPOST, SocialAction.REPLY,
            SocialAction.FOLLOW, SocialAction.POST,
        ]
        return random.choices(actions, weights=weights)[0]

    def _generate_content(self, feed) -> str:
        ticker = self.model.ticker
        reasons = [r for r in self.model.template_reasons if r]
        reason = random.choice(reasons) if reasons else "market conditions"

        if self.sentiment > 0.2:
            templates = (
                f"I think {ticker} has strong growth potential. {reason}",
                f"{ticker} looking good. {reason} — staying long.",
                f"Bullish on {ticker}. {reason}",
            )
        elif self.sentiment < -0.2:
            templates = (
                f"Concerned about {ticker}. {reason}",
                f"{ticker} is overvalued IMO. {reason}",
                f"Bearish on {ticker}. {reason} — reducing position.",
            )
        else:
            templates = (
                f"Watching {ticker} closely. {reason}",
                f"Mixed signals on {ticker}. {reason}",
            )
        return random.choice(templates)

    def _generate_reply(self, target_post) -> str:
        ticker = self.model.ticker
        if self.sentiment > 0.2:
            return f"Agree — {ticker} looks strong."
        elif self.sentiment < -0.2:
            return f"Not sure about that. I'm cautious on {ticker}."
        return f"Interesting take on {ticker}."


# ---------------------------------------------------------------------------
# LLM Citizen — Small LLM for text, rule-based action selection
# ---------------------------------------------------------------------------

class LLMCitizen(BaseMarketAgent):
    """Mid-tier agent. Uses Ollama for text generation, rules for actions."""

    def _decide_action(self, feed) -> SocialAction:
        weights = [0.30, 0.15, 0.15, 0.15, 0.10, 0.10, 0.05]
        actions = [
            SocialAction.BROWSE_FEED, SocialAction.POST,
            SocialAction.REPLY, SocialAction.LIKE,
            SocialAction.REPOST, SocialAction.FOLLOW,
            SocialAction.QUOTE_POST,
        ]
        return random.choices(actions, weights=weights)[0]

    def _generate_content(self, feed) -> str:
        feed_text = self._summarize_feed(feed, max_items=3)
        stance = ("bullish" if self.sentiment > 0.2
                  else "bearish" if self.sentiment < -0.2
                  else "neutral")
        prompt = (
            f"You are {self.persona.name}, a {self.persona.role}. "
            f"Sentiment: {stance}. Recent posts:\n{feed_text}\n"
            f"Write a 1-2 sentence social media post about "
            f"{self.model.ticker}."
        )
        return call_ollama(prompt, model=self.persona.model, max_tokens=80)

    def _generate_reply(self, target_post) -> str:
        content = self._get_field(target_post, "content", "")
        prompt = (
            f"You are {self.persona.name}. Reply briefly (1 sentence) "
            f"to: '{str(content)[:200]}'"
        )
        return call_ollama(prompt, model=self.persona.model, max_tokens=60)

    def _summarize_feed(self, feed, max_items: int = 3) -> str:
        items = feed[:max_items]
        lines = []
        for p in items:
            c = self._get_field(p, "content", "")
            lines.append(f"- {str(c)[:100]}")
        return "\n".join(lines) or "(no recent posts)"


# ---------------------------------------------------------------------------
# KOL Agent — Key Opinion Leader, high influence, analytical posts
# ---------------------------------------------------------------------------

class KOLAgent(LLMCitizen):
    """Highest-influence agent. Uses data bundle for analytical posts."""

    def _decide_action(self, feed) -> SocialAction:
        weights = [0.20, 0.30, 0.15, 0.15, 0.10, 0.05, 0.05]
        actions = [
            SocialAction.BROWSE_FEED, SocialAction.POST,
            SocialAction.REPLY, SocialAction.LIKE,
            SocialAction.REPOST, SocialAction.FOLLOW,
            SocialAction.QUOTE_POST,
        ]
        return random.choices(actions, weights=weights)[0]

    def _assess_momentum(self) -> str:
        """Lightweight mid-sim momentum check for KOL foresight.

        Uses the model's live sentiment_trajectory to compute velocity
        and acceleration. Returns a signal string only when momentum
        is significant (|accel| > 0.005). Pure math, no LLM call.
        """
        traj = getattr(self.model, "sentiment_trajectory", [])
        if len(traj) < 5:
            return ""
        # Convert last 5 snapshots to net-sentiment scores
        scores = []
        for s in traj[-5:]:
            bull = s.get("bullish", 0) + s.get("strongly_bullish", 0)
            bear = s.get("bearish", 0) + s.get("strongly_bearish", 0)
            scores.append(bull - bear)
        velocities = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
        if len(velocities) < 2:
            return ""
        accel = velocities[-1] - velocities[-2]
        if abs(accel) < 0.005:
            return ""
        direction = "accelerating" if accel > 0 else "decelerating"
        return (f"\nMOMENTUM SIGNAL: Sentiment is {direction} "
                f"(Δ²={accel:+.3f}). Consider adjusting stance.\n")

    def _generate_content(self, feed) -> str:
        feed_text = self._summarize_feed(feed, max_items=3)
        data = self.model.data_bundle
        # Pro: data-aware prompt with actual pipeline data
        fin_summary = data.get("financial_summary", "")[:500]
        news_summary = data.get("news_summary", "")[:300]
        sec_data = data.get("sec_analysis", "")[:200]
        sent_data = data.get("sentiment_data", "")[:150]
        # Pro: memory context with exponential decay
        memory_ctx = ""
        if self.memory:
            weighted = []
            for i, m in enumerate(reversed(self.memory)):
                w = self.MEMORY_DECAY ** i
                weighted.append(f"  [{w:.1f}] {m}")
            memory_ctx = f"\nYour recent memory:\n" + "\n".join(weighted[:3])
        # Foresight: mid-sim momentum signal (pure math, no LLM call)
        foresight = self._assess_momentum()
        stance = ("strongly bullish" if self.sentiment > 0.6
                  else "bullish" if self.sentiment > 0.2
                  else "bearish" if self.sentiment < -0.2
                  else "strongly bearish" if self.sentiment < -0.6
                  else "neutral")
        prompt = (
            f"You are {self.persona.name}, a {self.persona.role} "
            f"with significant market influence.\n"
            f"Financial data: {fin_summary}\n"
            f"Latest news: {news_summary}\n"
        )
        if sec_data:
            prompt += f"SEC/Risk data: {sec_data}\n"
        if sent_data:
            prompt += f"Market sentiment: {sent_data}\n"
        prompt += (
            f"Recent discussion:\n{feed_text}\n"
            f"{memory_ctx}\n"
            f"{foresight}"
            f"Your stance: {stance} (sentiment={self.sentiment:.2f}).\n"
            f"Write an analytical take (2-3 sentences) on "
            f"{self.model.ticker}. Reference specific data points."
        )
        return call_ollama(prompt, model=self.persona.model, max_tokens=200)

