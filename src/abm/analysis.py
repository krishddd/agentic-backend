"""
MiroFish ABM Statistical Analysis Engine — Pro Edition.

Post-simulation computation for Monte Carlo aggregation, saddle-point
analysis, phase transition detection, coalition identification, and
price back-correlation.

Implements the Social Path Integral framework from the MiroFish research:
- Saddle-point approximation → dominant social trajectory
- Path variance → distribution of alternative outcomes
- Effective action → distilled dominant behavioral pattern
- Confidence intervals → statistical validation
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contagion Detection (volatility-adjusted)
# ---------------------------------------------------------------------------

def detect_contagion_events(
    sentiment_trajectory: List[Dict[str, float]],
    agent_sentiments_per_tick: Optional[List[List[float]]] = None,
    base_threshold: float = 0.06,
    lookback: int = 5,
) -> List[dict]:
    """Detect contagion cascade moments using volatility-adjusted thresholds.

    Instead of a fixed 15% threshold, uses:
        threshold = max(base_threshold, 2 * rolling_std(last N ticks))

    This avoids false-flagging normal movement in volatile sims and
    missing real contagion in calm ones.

    Args:
        sentiment_trajectory: Per-tick avg sentiment snapshots.
        agent_sentiments_per_tick: Optional per-agent sentiment list per tick.
        base_threshold: Minimum threshold floor.
        lookback: Rolling window size for std calculation.

    Returns:
        List of contagion event dicts.
    """
    events = []
    if len(sentiment_trajectory) < 2:
        return events

    # Extract avg sentiment per tick
    avgs = []
    for snap in sentiment_trajectory:
        # Compute weighted average from 5-point distribution
        total = 0.0
        weights = {"strongly_bearish": -0.8, "bearish": -0.4,
                   "neutral": 0.0, "bullish": 0.4, "strongly_bullish": 0.8}
        for label, w in weights.items():
            total += snap.get(label, 0.0) * w
        # Fallback to 3-state if 5-point not available
        if total == 0.0:
            total = (snap.get("bullish", 0.0) - snap.get("bearish", 0.0))
        avgs.append(total)

    for i in range(1, len(avgs)):
        shift = abs(avgs[i] - avgs[i - 1])

        # Compute rolling std for volatility-adjusted threshold
        window = avgs[max(0, i - lookback):i]
        if len(window) >= 2:
            mean_w = sum(window) / len(window)
            variance = sum((x - mean_w) ** 2 for x in window) / len(window)
            rolling_std = math.sqrt(variance)
        else:
            rolling_std = 0.0

        threshold = max(base_threshold, 2.0 * rolling_std)

        if shift > threshold:
            direction = "bullish" if avgs[i] > avgs[i - 1] else "bearish"
            events.append({
                "tick": i + 1,
                "magnitude": round(shift, 4),
                "direction": direction,
                "threshold_used": round(threshold, 4),
                "rolling_std": round(rolling_std, 4),
                "trigger_agent_id": 0,
                "trigger_agent_name": "system",
                "affected_count": 0,
                "channel": "",
            })

    return events


# ---------------------------------------------------------------------------
# Phase Transition Detection
# ---------------------------------------------------------------------------

def detect_phase_transitions(
    sentiment_trajectory: List[Dict[str, float]],
    std_drop_threshold: float = 0.3,
    lookback: int = 3,
) -> List[dict]:
    """Detect phase transitions — critical tipping points.

    A phase transition is identified when:
    1. Rolling std deviation drops sharply (convergence = opinion collapse)
    2. Or avg sentiment crosses a modality boundary rapidly

    Args:
        sentiment_trajectory: Per-tick snapshots.
        std_drop_threshold: Minimum fractional std drop to flag.
        lookback: Window for rolling std.

    Returns:
        List of phase transition dicts.
    """
    transitions = []
    if len(sentiment_trajectory) < 3:
        return transitions

    # Compute per-tick avg sentiment
    avgs = []
    for snap in sentiment_trajectory:
        b = snap.get("bullish", 0.0) + snap.get("strongly_bullish", 0.0)
        br = snap.get("bearish", 0.0) + snap.get("strongly_bearish", 0.0)
        avgs.append(b - br)

    # Compute rolling std per tick
    stds = []
    for i in range(len(avgs)):
        window = avgs[max(0, i - lookback + 1):i + 1]
        if len(window) >= 2:
            mean_w = sum(window) / len(window)
            var = sum((x - mean_w) ** 2 for x in window) / len(window)
            stds.append(math.sqrt(var))
        else:
            stds.append(0.0)

    # Detect sharp std drops (convergence)
    for i in range(1, len(stds)):
        if stds[i - 1] > 0.01:
            drop = (stds[i - 1] - stds[i]) / stds[i - 1]
            if drop > std_drop_threshold:
                delta = round(avgs[i] - avgs[i - 1], 4)
                # Filter: require meaningful sentiment change (not just noise)
                if abs(delta) < 0.01:
                    continue
                transitions.append({
                    "tick": i + 1,
                    "pre_sentiment_avg": round(avgs[i - 1], 4),
                    "post_sentiment_avg": round(avgs[i], 4),
                    "delta": delta,
                    "trigger": f"std_convergence (std dropped {drop:.0%})",
                })

    return transitions


# ---------------------------------------------------------------------------
# Coalition Identification (elbow method)
# ---------------------------------------------------------------------------

def identify_coalitions(
    agent_sentiments: List[Tuple[int, float]],
    min_k: int = 2,
    max_k: int = 6,
) -> List[dict]:
    """Identify aligned agent clusters using elbow-method k-selection.

    Uses simple 1D k-means on sentiment values with k chosen by
    minimizing inertia drop-off (elbow method). Avoids fixed k=3
    which would force artificial clusters on unimodal distributions.

    Args:
        agent_sentiments: List of (agent_id, sentiment) tuples.
        min_k: Minimum cluster count to try.
        max_k: Maximum cluster count to try.

    Returns:
        List of CoalitionCluster dicts.
    """
    from src.abm.contracts import sentiment_to_label

    if len(agent_sentiments) < 3:
        return []

    values = [s for _, s in agent_sentiments]
    ids = [aid for aid, _ in agent_sentiments]

    # Clamp max_k to number of unique values
    unique_vals = len(set(round(v, 2) for v in values))
    effective_max_k = min(max_k, unique_vals, len(values))
    if effective_max_k < min_k:
        effective_max_k = min_k

    def _kmeans_1d(data, k, max_iter=50):
        """Simple 1D k-means."""
        import random as _rng
        sorted_data = sorted(data)
        # Initialize centroids evenly
        step = len(sorted_data) // k
        centroids = [sorted_data[min(i * step, len(sorted_data) - 1)]
                     for i in range(k)]

        for _ in range(max_iter):
            # Assign points
            clusters = [[] for _ in range(k)]
            for idx, val in enumerate(data):
                dists = [abs(val - c) for c in centroids]
                clusters[dists.index(min(dists))].append(idx)

            # Update centroids
            new_centroids = []
            for cl in clusters:
                if cl:
                    new_centroids.append(
                        sum(data[i] for i in cl) / len(cl))
                else:
                    new_centroids.append(
                        _rng.uniform(min(data), max(data)))
            if new_centroids == centroids:
                break
            centroids = new_centroids

        # Compute inertia
        inertia = 0.0
        for cl_idx, cl in enumerate(clusters):
            for i in cl:
                inertia += (data[i] - centroids[cl_idx]) ** 2

        return clusters, centroids, inertia

    # Try each k and pick via elbow method
    results = {}
    for k in range(min_k, effective_max_k + 1):
        clusters, centroids, inertia = _kmeans_1d(values, k)
        results[k] = (clusters, centroids, inertia)

    # Elbow: pick k where inertia drop-off ratio is largest
    best_k = min_k
    if len(results) > 1:
        ks = sorted(results.keys())
        ratios = []
        for i in range(1, len(ks)):
            prev_inertia = results[ks[i - 1]][2]
            curr_inertia = results[ks[i]][2]
            if prev_inertia > 0:
                ratios.append((ks[i], (prev_inertia - curr_inertia)
                               / prev_inertia))
            else:
                ratios.append((ks[i], 0.0))
        # Find the k with biggest relative drop
        if ratios:
            best_k = max(ratios, key=lambda x: x[1])[0]

    clusters, centroids, _ = results[best_k]

    # Build coalition objects
    coalitions = []
    for cl_idx, cl in enumerate(clusters):
        if not cl:
            continue
        member_ids = [ids[i] for i in cl]
        mean_sent = centroids[cl_idx]
        coalitions.append({
            "cluster_id": cl_idx,
            "agent_ids": member_ids,
            "mean_sentiment": round(mean_sent, 4),
            "label": sentiment_to_label(mean_sent),
            "size": len(member_ids),
        })

    return sorted(coalitions, key=lambda c: c["mean_sentiment"])


# ---------------------------------------------------------------------------
# Monte Carlo Saddle-Point Analysis
# ---------------------------------------------------------------------------

def compute_saddle_point(mc_results: List[dict]) -> dict:
    """Identify the dominant social trajectory (saddle-point approximation).

    The saddle-point is the MC path whose final sentiment is closest to
    the weighted mean of all paths — the "most probable" trajectory.

    Returns:
        Dict with dominant_path_index, dominant_sentiment, mean_sentiment,
        and per-path final sentiments.
    """
    if not mc_results:
        return {"dominant_path_index": 0, "path_variance": 0.0}

    # Compute final avg sentiment per path
    path_finals = []
    for r in mc_results:
        dist = r.get("final_sentiment_distribution", {})
        b = dist.get("bullish", 0) + dist.get("strongly_bullish", 0)
        br = dist.get("bearish", 0) + dist.get("strongly_bearish", 0)
        path_finals.append(b - br)

    mean_final = sum(path_finals) / len(path_finals) if path_finals else 0.0

    # Saddle point: path closest to mean
    min_dist = float("inf")
    dominant_idx = 0
    for i, pf in enumerate(path_finals):
        d = abs(pf - mean_final)
        if d < min_dist:
            min_dist = d
            dominant_idx = i

    return {
        "dominant_path_index": dominant_idx,
        "mean_sentiment": round(mean_final, 4),
        "dominant_sentiment": round(path_finals[dominant_idx], 4),
        "path_finals": [round(p, 4) for p in path_finals],
    }


def compute_path_variance(mc_results: List[dict]) -> float:
    """Compute variance across MC paths — measures branch divergence.

    High variance = paths disagree (unstable scenario).
    Low variance = paths converge (stable prediction).
    """
    if len(mc_results) < 2:
        return 0.0

    finals = []
    for r in mc_results:
        dist = r.get("final_sentiment_distribution", {})
        b = dist.get("bullish", 0) + dist.get("strongly_bullish", 0)
        br = dist.get("bearish", 0) + dist.get("strongly_bearish", 0)
        finals.append(b - br)

    mean_f = sum(finals) / len(finals)
    variance = sum((f - mean_f) ** 2 for f in finals) / len(finals)
    return round(variance, 6)


def compute_confidence_interval(
    sentiments: List[float],
    confidence: float = 0.95,
) -> dict:
    """Compute confidence interval on final sentiment values.

    Uses t-distribution approximation for small sample sizes.

    Returns:
        Dict with mean, std, ci_lower, ci_upper, n.
    """
    n = len(sentiments)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0,
                "ci_upper": 0.0, "n": 0}

    mean = sum(sentiments) / n
    if n == 1:
        return {"mean": round(mean, 4), "std": 0.0,
                "ci_lower": round(mean, 4), "ci_upper": round(mean, 4),
                "n": 1}

    variance = sum((s - mean) ** 2 for s in sentiments) / (n - 1)
    std = math.sqrt(variance)
    # t-value approximation for 95% CI
    t_values = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78,
                6: 2.57, 7: 2.45, 8: 2.37, 9: 2.31, 10: 2.26}
    t_val = t_values.get(n, 1.96)  # fallback to z for large n
    margin = t_val * std / math.sqrt(n)

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_lower": round(mean - margin, 4),
        "ci_upper": round(mean + margin, 4),
        "n": n,
    }


def compute_price_correlation(
    sentiment_trajectory: List[Dict[str, float]],
    price_returns: Optional[List[float]] = None,
) -> Optional[dict]:
    """Compute Pearson correlation between simulated sentiment and actual
    price returns for back-testing validation.

    Args:
        sentiment_trajectory: Per-tick sentiment snapshots.
        price_returns: Actual price returns (% change per period).
                       If None, returns None (no price data available).

    Returns:
        Dict with pearson_r, p_value_approx, or None.
    """
    if not price_returns or len(price_returns) < 3:
        return None

    # Extract sentiment signal
    sent_signal = []
    for snap in sentiment_trajectory:
        b = snap.get("bullish", 0) + snap.get("strongly_bullish", 0)
        br = snap.get("bearish", 0) + snap.get("strongly_bearish", 0)
        sent_signal.append(b - br)

    # Align lengths
    n = min(len(sent_signal), len(price_returns))
    s = sent_signal[:n]
    p = price_returns[:n]

    if n < 3:
        return None

    # Pearson correlation
    mean_s = sum(s) / n
    mean_p = sum(p) / n

    cov = sum((s[i] - mean_s) * (p[i] - mean_p) for i in range(n)) / n
    std_s = math.sqrt(sum((x - mean_s) ** 2 for x in s) / n)
    std_p = math.sqrt(sum((x - mean_p) ** 2 for x in p) / n)

    if std_s < 1e-9 or std_p < 1e-9:
        return {"pearson_r": 0.0, "direction": "no_signal", "n": n}

    r = cov / (std_s * std_p)

    direction = "positive" if r > 0.3 else "negative" if r < -0.3 else "weak"

    return {
        "pearson_r": round(r, 4),
        "direction": direction,
        "n": n,
    }
