"""
Trend Predictor — Grandmaster Edition.

Extrapolates simulation trajectories to predict future sentiment
trends. Uses multiple forecasting methods:
1. Linear regression on sentiment trajectory
2. Momentum-based forecasting (acceleration/deceleration)
3. Coalition stability analysis
4. Monte Carlo forward projection

Output: PredictionResult with confidence intervals for T+5/T+10/T+30.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrendForecast:
    """Single time-horizon forecast."""
    horizon: int              # ticks into the future
    predicted_sentiment: float
    confidence_low: float
    confidence_high: float
    confidence_pct: float     # 0-100
    method: str               # which method produced this


@dataclass
class PredictionResult:
    """Complete prediction output."""
    ticker: str
    current_sentiment: float
    trend_direction: str       # "bullish", "bearish", "neutral", "volatile"
    momentum: float            # acceleration of sentiment change
    coalition_stability: float  # 0-1 how stable clusters are
    forecasts: List[TrendForecast] = field(default_factory=list)
    narrative: str = ""        # LLM-generated prediction narrative
    risk_factors: List[str] = field(default_factory=list)


class TrendPredictor:
    """Predict future sentiment trends from simulation data.

    Combines statistical extrapolation with structural analysis
    (coalition stability, momentum) for grounded predictions.
    """

    def predict(
        self,
        ticker: str,
        sentiment_trajectory: List[Dict[str, float]],
        coalitions: Optional[List[dict]] = None,
        contagion_events: Optional[List[dict]] = None,
        mc_path_sentiments: Optional[List[str]] = None,
    ) -> PredictionResult:
        """Generate multi-horizon predictions.

        Args:
            ticker: Target ticker symbol.
            sentiment_trajectory: Per-tick sentiment distributions.
            coalitions: Coalition cluster data.
            contagion_events: Detected contagion cascade events.
            mc_path_sentiments: Monte Carlo path outcomes.

        Returns:
            PredictionResult with forecasts at T+5, T+10, T+30.
        """
        # Convert trajectory to numeric scores
        scores = self._trajectory_to_scores(sentiment_trajectory)

        if not scores:
            return PredictionResult(
                ticker=ticker,
                current_sentiment=0.0,
                trend_direction="neutral",
                momentum=0.0,
                coalition_stability=0.5,
            )

        current = scores[-1]

        # Calculate momentum
        momentum = self._compute_momentum(scores)

        # Coalition stability
        stability = self._compute_coalition_stability(coalitions)

        # Determine trend direction
        direction = self._classify_direction(scores, momentum)

        # Generate forecasts at multiple horizons
        forecasts = []
        for horizon in [5, 10, 30]:
            forecast = self._forecast_horizon(
                scores, horizon, momentum, stability)
            forecasts.append(forecast)

        # Identify risk factors
        risks = self._identify_risks(
            scores, momentum, stability, contagion_events)

        return PredictionResult(
            ticker=ticker,
            current_sentiment=round(current, 4),
            trend_direction=direction,
            momentum=round(momentum, 4),
            coalition_stability=round(stability, 3),
            forecasts=forecasts,
            risk_factors=risks,
        )

    # ------------------------------------------------------------------
    # Trajectory conversion
    # ------------------------------------------------------------------

    def _trajectory_to_scores(
        self, trajectory: List[Dict[str, float]]
    ) -> List[float]:
        """Convert distributions to net-sentiment scores."""
        scores = []
        for snap in trajectory:
            bull = snap.get("bullish", 0) + snap.get("strongly_bullish", 0)
            bear = snap.get("bearish", 0) + snap.get("strongly_bearish", 0)
            scores.append(bull - bear)
        return scores

    # ------------------------------------------------------------------
    # Momentum computation
    # ------------------------------------------------------------------

    def _compute_momentum(self, scores: List[float]) -> float:
        """Compute sentiment acceleration (second derivative)."""
        if len(scores) < 3:
            return 0.0

        # First derivatives (velocity)
        velocities = [scores[i] - scores[i - 1]
                      for i in range(1, len(scores))]

        # Second derivative (acceleration) — recent window
        window = min(5, len(velocities) - 1)
        if window < 1:
            return velocities[-1] if velocities else 0.0

        recent_accel = [velocities[i] - velocities[i - 1]
                        for i in range(len(velocities) - window,
                                       len(velocities))]

        return sum(recent_accel) / len(recent_accel)

    # ------------------------------------------------------------------
    # Coalition stability
    # ------------------------------------------------------------------

    def _compute_coalition_stability(
        self, coalitions: Optional[List[dict]]
    ) -> float:
        """Measure how stable coalition clusters are (0=fragile, 1=solid)."""
        if not coalitions or len(coalitions) < 2:
            return 0.5  # default

        sizes = [c.get("size", 0) for c in coalitions]
        total = sum(sizes)
        if total == 0:
            return 0.5

        # Herfindahl index — higher = more concentrated = more stable
        hhi = sum((s / total) ** 2 for s in sizes)

        # Sentiment spread between clusters
        sentiments = [c.get("mean_sentiment", 0.0) for c in coalitions]
        if len(sentiments) >= 2:
            spread = max(sentiments) - min(sentiments)
            # High spread = less stable (polarization)
            polarization_penalty = min(0.3, spread * 0.3)
        else:
            polarization_penalty = 0.0

        stability = min(1.0, max(0.0, hhi - polarization_penalty))
        return stability

    # ------------------------------------------------------------------
    # Direction classification
    # ------------------------------------------------------------------

    def _classify_direction(
        self, scores: List[float], momentum: float
    ) -> str:
        """Classify overall trend direction."""
        if len(scores) < 2:
            return "neutral"

        # Net change over trajectory
        net_change = scores[-1] - scores[0]
        # Recent trend (last 3 ticks)
        recent_change = (scores[-1] - scores[-min(3, len(scores))]
                         if len(scores) >= 2 else 0)

        # Volatility
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        volatility = math.sqrt(variance)

        if volatility > 0.15:
            return "volatile"
        elif net_change > 0.05 or recent_change > 0.03:
            return "bullish"
        elif net_change < -0.05 or recent_change < -0.03:
            return "bearish"
        else:
            return "neutral"

    # ------------------------------------------------------------------
    # Horizon forecasting
    # ------------------------------------------------------------------

    def _forecast_horizon(
        self,
        scores: List[float],
        horizon: int,
        momentum: float,
        stability: float,
    ) -> TrendForecast:
        """Forecast sentiment at a specific horizon using ensemble."""
        # Method 1: Linear regression extrapolation
        lr_pred = self._linear_extrapolation(scores, horizon)

        # Method 2: Momentum-based projection
        current = scores[-1] if scores else 0.0
        velocity = (scores[-1] - scores[-2]) if len(scores) >= 2 else 0.0
        mom_pred = current + velocity * horizon + 0.5 * momentum * horizon ** 2
        # Dampen extreme extrapolations
        mom_pred = max(-1.0, min(1.0, mom_pred))

        # Method 3: Mean-reverting (weighted by stability)
        mean_score = sum(scores) / len(scores) if scores else 0.0
        reversion_weight = max(0.1, 1.0 - stability)
        mr_pred = current * (1 - reversion_weight) + mean_score * reversion_weight

        # Ensemble: weighted average
        ensemble_pred = (lr_pred * 0.4 + mom_pred * 0.3 + mr_pred * 0.3)
        ensemble_pred = max(-1.0, min(1.0, ensemble_pred))

        # Confidence interval
        base_uncertainty = 0.05 * math.sqrt(horizon)
        stability_factor = max(0.3, 1.0 - stability * 0.5)
        uncertainty = base_uncertainty * stability_factor

        conf_low = max(-1.0, ensemble_pred - uncertainty)
        conf_high = min(1.0, ensemble_pred + uncertainty)
        conf_pct = max(10.0, min(95.0, 80.0 - horizon * 1.5))

        # Stress test for long horizons (T+20 and beyond)
        if horizon >= 20:
            stress_floor = self._stress_test_projection(
                scores, horizon, stability)
            conf_low = max(-1.0, min(conf_low, stress_floor))

        return TrendForecast(
            horizon=horizon,
            predicted_sentiment=round(ensemble_pred, 4),
            confidence_low=round(conf_low, 4),
            confidence_high=round(conf_high, 4),
            confidence_pct=round(conf_pct, 1),
            method="ensemble (linear + momentum + mean-revert)",
        )

    def _linear_extrapolation(
        self, scores: List[float], horizon: int
    ) -> float:
        """Simple linear regression extrapolation."""
        n = len(scores)
        if n < 2:
            return scores[0] if scores else 0.0

        # OLS: y = a + bx
        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n
        numerator = sum((i - x_mean) * (scores[i] - y_mean)
                        for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if abs(denominator) < 1e-10:
            return scores[-1]

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        prediction = intercept + slope * (n - 1 + horizon)
        return max(-1.0, min(1.0, prediction))

    def _stress_test_projection(
        self,
        scores: List[float],
        horizon: int,
        stability: float,
        n_sims: int = 50,
    ) -> float:
        """Monte Carlo stress test: inject random shocks into forward projection.

        Simulates n_sims forward paths with random negative shocks whose
        probability is inversely proportional to coalition stability.
        Returns the 10th percentile outcome (stress floor).

        This is pure math (no LLM calls), ~0.1ms for 50 sims.
        """
        if len(scores) < 3:
            return scores[-1] if scores else 0.0

        current = scores[-1]
        velocity = scores[-1] - scores[-2]

        results = []
        for _ in range(n_sims):
            val = current
            for t in range(horizon):
                # Forward with noisy velocity
                val += velocity * random.gauss(1.0, 0.3)
                # Random shock: probability inversely proportional to stability
                # Fragile coalitions (stability=0.09) → 7% shock chance/tick
                # Stable coalitions (stability=0.80) → 1.6% shock chance/tick
                if random.random() < (1.0 - stability) * 0.08:
                    val += random.gauss(-0.15, 0.1)  # negative shock
                val = max(-1.0, min(1.0, val))
            results.append(val)

        results.sort()
        # 10th percentile = stress floor
        return results[int(n_sims * 0.1)]

    # ------------------------------------------------------------------
    # Risk identification
    # ------------------------------------------------------------------

    def _identify_risks(
        self,
        scores: List[float],
        momentum: float,
        stability: float,
        contagion_events: Optional[List[dict]] = None,
    ) -> List[str]:
        """Identify prediction risk factors."""
        risks = []

        # Momentum reversal risk
        if abs(momentum) > 0.01:
            direction = "bullish" if momentum > 0 else "bearish"
            risks.append(
                f"Momentum {direction} ({momentum:+.3f}) — "
                "reversal possible if catalysts change")

        # Low stability risk
        if stability < 0.3:
            risks.append(
                f"Low coalition stability ({stability:.2f}) — "
                "opinion fragmentation may cause rapid shifts")

        # High volatility
        if scores:
            mean_s = sum(scores) / len(scores)
            var = sum((s - mean_s) ** 2 for s in scores) / len(scores)
            if var > 0.02:
                risks.append(
                    f"High sentiment volatility (var={var:.3f}) — "
                    "predictions have wider uncertainty")

        # Contagion risk
        if contagion_events and len(contagion_events) > 2:
            risks.append(
                f"{len(contagion_events)} contagion cascades detected — "
                "market susceptible to viral sentiment shifts")

        # Echo chamber risk
        if stability > 0.8 and len(scores) > 5:
            recent_var = sum(
                (scores[i] - scores[i-1]) ** 2
                for i in range(max(1, len(scores)-5), len(scores))
            ) / 5
            if recent_var < 0.001:
                risks.append(
                    "Echo chamber formation — "
                    "sentiment may be artificially stable")

        return risks
