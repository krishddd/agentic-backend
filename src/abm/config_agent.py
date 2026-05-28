"""
Simulation Config Agent — Grandmaster Edition.

LLM-powered agent that reads pipeline data and auto-configures
simulation parameters. Analyzes data richness, ticker type, detected
events, and news sentiment to set optimal ABM configuration.

Output: SimulationConfig dataclass with all parameters needed by
MarketSentimentModel.
"""

import logging
import json
import re
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.abm.llm_utils import call_ollama
from src.abm.contracts import CatalystShock

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Auto-generated simulation configuration."""
    n_agents: int = 50
    n_ticks: int = 10
    budget_mode: str = "lite"
    mc_paths: int = 1
    microblog_ratio: float = 0.6     # % agents on microblog
    forum_ratio: float = 0.4         # % agents on forum
    initial_sentiment_bias: float = 0.0  # -1 to 1
    catalyst_shocks: List[CatalystShock] = field(default_factory=list)
    data_richness_score: float = 0.0  # 0-1 quality metric
    prediction_targets: List[str] = field(default_factory=list)
    reasoning: str = ""              # LLM's rationale


class SimulationConfigAgent:
    """Auto-configure simulation parameters from pipeline data.

    Analyzes the data bundle (research, financial, news, SEC) to
    determine optimal simulation settings without manual configuration.
    """

    def __init__(self, model: str = "llama3.2:latest"):
        self.model = model

    def configure(
        self,
        data_bundle: dict,
        ticker: str,
        user_prompt: str = "",
    ) -> SimulationConfig:
        """Analyze pipeline data and generate optimal sim config.

        Args:
            data_bundle: Pipeline outputs (research, financial, news, SEC).
            ticker: Target ticker symbol.
            user_prompt: Original user query for prediction extraction.

        Returns:
            SimulationConfig with auto-tuned parameters.
        """
        config = SimulationConfig()

        # Step 1: Measure data richness
        config.data_richness_score = self._measure_data_richness(data_bundle)

        # Step 2: Scale params by data richness
        config = self._scale_parameters(config)

        # Step 3: Detect initial sentiment from data
        config.initial_sentiment_bias = self._detect_sentiment_bias(
            data_bundle)

        # Step 4: Auto-detect catalyst shocks from news/financial
        config.catalyst_shocks = self._detect_catalysts(data_bundle, config)

        # Step 5: Extract prediction targets from user prompt
        config.prediction_targets = self._extract_predictions(user_prompt)

        # Step 6: Get LLM reasoning (optional, fast)
        config.reasoning = self._get_llm_reasoning(
            data_bundle, ticker, config)

        logger.info(
            f"[ConfigAgent] {ticker}: {config.n_agents} agents × "
            f"{config.n_ticks} ticks, mode={config.budget_mode}, "
            f"richness={config.data_richness_score:.2f}, "
            f"bias={config.initial_sentiment_bias:+.2f}, "
            f"{len(config.catalyst_shocks)} catalysts, "
            f"{len(config.prediction_targets)} predictions")

        return config

    # ------------------------------------------------------------------
    # Step 1: Data richness measurement
    # ------------------------------------------------------------------

    def _measure_data_richness(self, data_bundle: dict) -> float:
        """Score data quality 0-1 based on available sources and depth."""
        scores = []
        sources = [
            ("research_output", 2000),    # expect 2k+ chars
            ("financial_summary", 3000),   # expect 3k+ chars
            ("news_summary", 1000),
            ("sec_analysis", 2000),
            ("sentiment_data", 200),
        ]
        for key, expected_len in sources:
            text = data_bundle.get(key, "")
            if text:
                ratio = min(1.0, len(text) / expected_len)
                scores.append(ratio)
            else:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    # ------------------------------------------------------------------
    # Step 2: Scale parameters
    # ------------------------------------------------------------------

    def _scale_parameters(self, config: SimulationConfig) -> SimulationConfig:
        """Scale agent count, ticks, and MC paths by data richness."""
        richness = config.data_richness_score

        if richness >= 0.8:
            config.n_agents = 80
            config.n_ticks = 15
            config.budget_mode = "standard"
            config.mc_paths = 3
        elif richness >= 0.5:
            config.n_agents = 50
            config.n_ticks = 10
            config.budget_mode = "lite"
            config.mc_paths = 1
        else:
            config.n_agents = 30
            config.n_ticks = 8
            config.budget_mode = "lite"
            config.mc_paths = 1

        return config

    # ------------------------------------------------------------------
    # Step 3: Sentiment bias detection
    # ------------------------------------------------------------------

    def _detect_sentiment_bias(self, data_bundle: dict) -> float:
        """Detect initial market sentiment from pipeline data."""
        sentiment_data = data_bundle.get("sentiment_data", "")
        if not sentiment_data:
            return 0.0

        positive_keywords = [
            "bullish", "strong buy", "outperform", "growth",
            "beat", "exceeded", "upgrade", "positive", "momentum",
        ]
        negative_keywords = [
            "bearish", "sell", "underperform", "decline",
            "miss", "warning", "downgrade", "negative", "risk",
        ]

        text = sentiment_data.lower()
        pos_count = sum(1 for kw in positive_keywords if kw in text)
        neg_count = sum(1 for kw in negative_keywords if kw in text)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return round((pos_count - neg_count) / total * 0.5, 3)

    # ------------------------------------------------------------------
    # Step 4: Catalyst detection
    # ------------------------------------------------------------------

    def _detect_catalysts(
        self, data_bundle: dict, config: SimulationConfig
    ) -> List[CatalystShock]:
        """Auto-detect potential catalyst shocks from data."""
        catalysts = []
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
            ("upgrade", "analyst_upgrade", 0.2),
            ("downgrade", "analyst_downgrade", -0.2),
            ("lawsuit", "legal_risk", -0.2),
            ("acquisition", "acquisition", 0.15),
            ("layoff", "restructuring", -0.15),
            ("partnership", "strategic_partnership", 0.1),
            ("data breach", "cybersecurity_incident", -0.25),
            ("regulatory", "regulatory_action", -0.15),
        ]

        seen_types = set()
        for pattern, event_type, magnitude in shock_patterns:
            if pattern in combined and event_type not in seen_types:
                seen_types.add(event_type)
                # Place at random tick in simulation
                tick = random.randint(2, max(3, config.n_ticks - 2))
                catalysts.append(CatalystShock(
                    tick=tick,
                    event_type=event_type,
                    magnitude=magnitude,
                ))

        return catalysts

    # ------------------------------------------------------------------
    # Step 5: Prediction target extraction
    # ------------------------------------------------------------------

    def _extract_predictions(self, user_prompt: str) -> List[str]:
        """Extract what the user wants to predict from their query."""
        if not user_prompt:
            return ["sentiment_direction", "consensus_strength"]

        prompt_lower = user_prompt.lower()
        targets = []

        prediction_map = {
            "price": "price_direction",
            "stock": "price_direction",
            "sentiment": "sentiment_direction",
            "bull": "bullish_probability",
            "bear": "bearish_probability",
            "risk": "risk_assessment",
            "volatil": "volatility_forecast",
            "consensus": "consensus_strength",
            "trend": "trend_momentum",
            "outlook": "outlook_forecast",
            "target": "price_target_range",
        }

        for keyword, prediction in prediction_map.items():
            if keyword in prompt_lower and prediction not in targets:
                targets.append(prediction)

        if not targets:
            targets = ["sentiment_direction", "consensus_strength"]

        return targets

    # ------------------------------------------------------------------
    # Step 6: LLM reasoning
    # ------------------------------------------------------------------

    def _get_llm_reasoning(
        self, data_bundle: dict, ticker: str, config: SimulationConfig
    ) -> str:
        """Get brief LLM reasoning for the config choices."""
        try:
            # Build compact summary
            data_summary = ""
            for key in ["research_output", "financial_summary", "news_summary"]:
                text = data_bundle.get(key, "")
                if text:
                    data_summary += f"\n{key}: {text[:300]}..."

            prompt = (
                f"Given this data about {ticker}:\n{data_summary}\n\n"
                f"The simulation will use {config.n_agents} agents, "
                f"{config.n_ticks} ticks, mode={config.budget_mode}.\n"
                f"Bias: {config.initial_sentiment_bias:+.2f}, "
                f"{len(config.catalyst_shocks)} catalyst shocks detected.\n\n"
                "In 2 sentences, explain why these settings are appropriate."
            )
            return call_ollama(prompt, model=self.model, max_tokens=150,
                               timeout=30)
        except Exception as e:
            logger.warning(f"Config reasoning failed: {e}")
            return ""
