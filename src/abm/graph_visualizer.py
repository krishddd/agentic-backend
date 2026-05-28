"""
Graph Visualization — Grandmaster Edition.

Generates interactive graph data in multiple formats for visualization
of entity relationships, coalition clusters, sentiment polarity, and
temporal evolution of the simulation network.

Exports:
    - D3.js-compatible JSON (nodes + links + metadata)
    - Mermaid diagram syntax
    - Adjacency matrix
    - Temporal snapshots (graph state at each tick)
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


class GraphVisualizer:
    """Generate visualization data from the knowledge graph and simulation.

    Supports three output formats plus temporal evolution tracking.
    """

    # Sentiment → color mapping
    SENTIMENT_COLORS = {
        "strongly_bullish": "#00C853",
        "bullish": "#69F0AE",
        "neutral": "#B0BEC5",
        "bearish": "#FF8A80",
        "strongly_bearish": "#D50000",
    }

    INFLUENCE_TIERS = [
        (0.8, "tier1_kol", 40),
        (0.5, "tier2_analyst", 28),
        (0.2, "tier3_active", 20),
        (0.0, "tier4_retail", 14),
    ]

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    # ------------------------------------------------------------------
    # D3.js JSON export
    # ------------------------------------------------------------------

    def to_d3_json(
        self,
        coalitions: Optional[List[dict]] = None,
        agent_sentiments: Optional[Dict[int, float]] = None,
    ) -> dict:
        """Export graph as D3.js-compatible JSON.

        Args:
            coalitions: Optional coalition cluster data from analysis.
            agent_sentiments: Optional {agent_id: sentiment_score} for coloring.

        Returns:
            Dict with 'nodes' and 'links' arrays.
        """
        # Build coalition lookup
        coalition_map: Dict[int, int] = {}
        if coalitions:
            for c in coalitions:
                cid = c.get("cluster_id", 0)
                for aid in c.get("agent_ids", []):
                    coalition_map[aid] = cid

        nodes = []
        node_index = {}
        for i, (node, data) in enumerate(self.graph.nodes(data=True)):
            node_index[node] = i

            # Determine visual properties
            role = data.get("role", data.get("type", "entity"))
            influence = data.get("influence", 0.3)
            sentiment = 0.0

            # Try to get agent sentiment
            agent_id = data.get("agent_id")
            if agent_id is not None and agent_sentiments:
                sentiment = agent_sentiments.get(agent_id, 0.0)

            # Size by influence
            size = 14
            tier = "tier4_retail"
            for threshold, tier_name, tier_size in self.INFLUENCE_TIERS:
                if influence >= threshold:
                    tier = tier_name
                    size = tier_size
                    break

            # Color by sentiment
            color = self._sentiment_to_color(sentiment)

            nodes.append({
                "id": str(node),
                "label": str(node)[:30],
                "role": role,
                "influence": round(influence, 3),
                "sentiment": round(sentiment, 3),
                "tier": tier,
                "size": size,
                "color": color,
                "coalition": coalition_map.get(
                    agent_id, -1) if agent_id is not None else -1,
                "description": str(
                    data.get("description", data.get("backstory", ""))
                )[:200],
            })

        links = []
        for u, v, data in self.graph.edges(data=True):
            if u in node_index and v in node_index:
                links.append({
                    "source": str(u),
                    "target": str(v),
                    "relation": data.get(
                        "relation", data.get("label", "related_to")),
                    "weight": data.get("weight", 1.0),
                })

        return {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "total_nodes": len(nodes),
                "total_edges": len(links),
                "coalitions": len(coalitions) if coalitions else 0,
            },
        }

    # ------------------------------------------------------------------
    # Mermaid diagram
    # ------------------------------------------------------------------

    def to_mermaid(self, max_nodes: int = 30) -> str:
        """Export graph as Mermaid diagram syntax.

        Args:
            max_nodes: Maximum nodes to include (for readability).

        Returns:
            Mermaid graph definition string.
        """
        lines = ["graph LR"]

        # Sort nodes by degree (most connected first)
        degree_sorted = sorted(
            self.graph.degree(), key=lambda x: x[1], reverse=True)

        included_nodes = set()
        for node, degree in degree_sorted[:max_nodes]:
            included_nodes.add(node)

        # Add edges between included nodes
        added_edges = set()
        for u, v, data in self.graph.edges(data=True):
            if u in included_nodes and v in included_nodes:
                edge_key = (str(u), str(v))
                if edge_key not in added_edges:
                    relation = data.get(
                        "relation", data.get("label", "related"))
                    # Sanitize for mermaid
                    u_safe = self._mermaid_safe(str(u))
                    v_safe = self._mermaid_safe(str(v))
                    r_safe = self._mermaid_safe(relation)
                    lines.append(
                        f'    {u_safe}["{u_safe}"] '
                        f'-->|"{r_safe}"| '
                        f'{v_safe}["{v_safe}"]')
                    added_edges.add(edge_key)

        return "\n".join(lines) if len(lines) > 1 else "graph LR\n    A[No graph data]"

    # ------------------------------------------------------------------
    # Adjacency matrix
    # ------------------------------------------------------------------

    def to_adjacency_matrix(self) -> dict:
        """Export graph as adjacency matrix.

        Returns:
            Dict with 'labels' (node names) and 'matrix' (2D int array).
        """
        nodes = list(self.graph.nodes())
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}

        matrix = [[0] * n for _ in range(n)]
        for u, v in self.graph.edges():
            if u in node_idx and v in node_idx:
                i, j = node_idx[u], node_idx[v]
                matrix[i][j] = 1
                matrix[j][i] = 1  # undirected

        return {
            "labels": [str(n) for n in nodes],
            "matrix": matrix,
            "size": n,
        }

    # ------------------------------------------------------------------
    # Temporal evolution
    # ------------------------------------------------------------------

    def build_temporal_snapshots(
        self,
        sentiment_trajectory: List[Dict[str, float]],
        coalitions_per_tick: Optional[List[List[dict]]] = None,
    ) -> List[dict]:
        """Build temporal snapshots of graph state per tick.

        Args:
            sentiment_trajectory: Per-tick sentiment distributions.
            coalitions_per_tick: Optional coalition data per tick.

        Returns:
            List of per-tick snapshot dicts with sentiment + coalition data.
        """
        snapshots = []
        for tick, snap in enumerate(sentiment_trajectory):
            snapshot = {
                "tick": tick + 1,
                "sentiment_distribution": snap,
                "dominant_sentiment": max(snap, key=snap.get) if snap else "neutral",
            }

            # Add coalition data if available
            if coalitions_per_tick and tick < len(coalitions_per_tick):
                snapshot["coalitions"] = coalitions_per_tick[tick]
                snapshot["n_coalitions"] = len(coalitions_per_tick[tick])

            snapshots.append(snapshot)

        return snapshots

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sentiment_to_color(self, score: float) -> str:
        """Map continuous sentiment score to hex color."""
        if score >= 0.6:
            return self.SENTIMENT_COLORS["strongly_bullish"]
        elif score >= 0.2:
            return self.SENTIMENT_COLORS["bullish"]
        elif score >= -0.2:
            return self.SENTIMENT_COLORS["neutral"]
        elif score >= -0.6:
            return self.SENTIMENT_COLORS["bearish"]
        else:
            return self.SENTIMENT_COLORS["strongly_bearish"]

    @staticmethod
    def _mermaid_safe(text: str) -> str:
        """Sanitize text for Mermaid diagram syntax."""
        return (text.replace('"', "'")
                    .replace("[", "(")
                    .replace("]", ")")
                    .replace("{", "(")
                    .replace("}", ")")
                    .replace("<", "")
                    .replace(">", "")
                    .replace("\n", " ")
                    .strip()[:40])

    # ------------------------------------------------------------------
    # Factory: Build from simulation result (no pre-built graph needed)
    # ------------------------------------------------------------------

    @staticmethod
    def build_from_sim_result(
        ticker: str,
        agent_states: list,
        coalitions: list,
        pipeline_data: Optional[Dict[str, str]] = None,
    ) -> dict:
        """Build D3.js graph data directly from simulation results.

        Creates a graph with:
        - Agent nodes (colored by coalition, sized by influence)
        - Ticker node (central hub)
        - Entity nodes (extracted from pipeline text)
        - Edges (agent→ticker, agent→coalition, ticker→entity)

        Args:
            ticker: Target ticker symbol.
            agent_states: Per-agent state dicts from SimulationResult.
            coalitions: Coalition cluster data.
            pipeline_data: Dict of pipeline reports (research, financial, etc.)

        Returns:
            D3.js-compatible dict with 'nodes', 'links', 'metadata'.
        """
        viz = GraphVisualizer.__new__(GraphVisualizer)
        nodes = []
        links = []

        # Coalition colors
        coalition_colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
            "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
        ]

        # 1. Central ticker node
        nodes.append({
            "id": ticker,
            "label": ticker,
            "role": "ticker",
            "influence": 1.0,
            "sentiment": 0.0,
            "tier": "tier1_kol",
            "size": 50,
            "color": "#FFD700",
            "coalition": -1,
            "description": f"Target: {ticker}",
        })

        # 2. Agent nodes from agent_states
        for agent in agent_states:
            aid = agent.get("agent_id", 0)
            cid = agent.get("coalition_id", -1)
            sentiment = agent.get("sentiment", 0.0)
            influence = agent.get("influence", 0.3)

            # Size by influence
            size = 14
            for threshold, _, tier_size in GraphVisualizer.INFLUENCE_TIERS:
                if influence >= threshold:
                    size = tier_size
                    break

            color = coalition_colors[cid % len(coalition_colors)] if cid >= 0 else "#B0BEC5"

            nodes.append({
                "id": f"agent_{aid}",
                "label": agent.get("persona_name", f"Agent {aid}")[:25],
                "role": agent.get("role", "citizen"),
                "influence": round(influence, 3),
                "sentiment": round(sentiment, 3),
                "tier": agent.get("role", "tier4_retail"),
                "size": size,
                "color": color,
                "coalition": cid,
                "description": f"{agent.get('role', 'citizen')} on {agent.get('channel_preference', 'microblog')}",
            })

            # Edge: agent → ticker
            links.append({
                "source": f"agent_{aid}",
                "target": ticker,
                "relation": "discusses",
                "weight": influence,
            })

        # 3. Coalition hub nodes
        for coal in coalitions:
            cid = coal.get("cluster_id", 0)
            label = coal.get("label", "neutral")
            size = coal.get("size", 0)
            color = coalition_colors[cid % len(coalition_colors)]
            nodes.append({
                "id": f"coalition_{cid}",
                "label": f"Coalition {cid}: {label} ({size})",
                "role": "coalition",
                "influence": 0.6,
                "sentiment": round(coal.get("mean_sentiment", 0.0), 3),
                "tier": "coalition",
                "size": 30,
                "color": color,
                "coalition": cid,
                "description": f"{label} cluster: {size} agents, avg sentiment {coal.get('mean_sentiment', 0):.3f}",
            })

            # Edges: coalition members → coalition hub
            for aid in coal.get("agent_ids", []):
                links.append({
                    "source": f"agent_{aid}",
                    "target": f"coalition_{cid}",
                    "relation": "member_of",
                    "weight": 0.5,
                })

        # 4. Entity nodes from pipeline data (rule-based extraction)
        if pipeline_data:
            entities = GraphVisualizer._extract_entities_from_text(
                pipeline_data, ticker)
            for entity in entities[:20]:  # cap at 20
                nodes.append({
                    "id": f"entity_{entity['name']}",
                    "label": entity["name"][:25],
                    "role": entity["type"],
                    "influence": 0.3,
                    "sentiment": 0.0,
                    "tier": "entity",
                    "size": 16,
                    "color": "#78909C",
                    "coalition": -1,
                    "description": entity.get("context", ""),
                })
                links.append({
                    "source": ticker,
                    "target": f"entity_{entity['name']}",
                    "relation": entity.get("relation", "related_to"),
                    "weight": 0.4,
                })

        return {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "ticker": ticker,
                "total_nodes": len(nodes),
                "total_edges": len(links),
                "coalitions": len(coalitions),
                "agents": len(agent_states),
            },
        }

    @staticmethod
    def _extract_entities_from_text(
        pipeline_data: Dict[str, str], ticker: str
    ) -> List[dict]:
        """Rule-based entity extraction from pipeline reports."""
        import re
        entities = []
        seen = set()
        all_text = " ".join(v for v in pipeline_data.values() if v)

        # Extract company names (capitalized multi-word near keywords)
        competitor_patterns = [
            r"compet(?:itor|es|ition)\w*\s+(?:include|such as|like|from)\s+([\w\s,]+)",
            r"([\w]+)\s+(?:vs\.?|versus|compared to)\s+" + re.escape(ticker),
        ]
        for pat in competitor_patterns:
            for match in re.finditer(pat, all_text, re.IGNORECASE):
                names = match.group(1).split(",")
                for name in names[:5]:
                    name = name.strip().split(" and ")[0].strip()
                    if len(name) > 2 and name not in seen and name.upper() != ticker:
                        seen.add(name)
                        entities.append({
                            "name": name[:30],
                            "type": "competitor",
                            "relation": "competes_with",
                            "context": f"Competitor of {ticker}",
                        })

        # Extract financial metrics mentioned
        metric_patterns = [
            (r"revenue.*?(\$[\d.,]+\s*[BMK]?\w*)", "revenue", "has_metric"),
            (r"P/E.*?ratio.*?([\d.]+)", "P/E Ratio", "has_valuation"),
            (r"market\s+cap.*?(\$[\d.,]+\s*[BMT]?\w*)", "Market Cap", "has_metric"),
            (r"net\s+(?:income|loss).*?(\$[\d.,]+\s*[BMK]?\w*)", "Net Income", "has_metric"),
        ]
        for pat, metric_name, relation in metric_patterns:
            match = re.search(pat, all_text, re.IGNORECASE)
            if match and metric_name not in seen:
                seen.add(metric_name)
                entities.append({
                    "name": metric_name,
                    "type": "metric",
                    "relation": relation,
                    "context": match.group(0)[:100],
                })

        # Extract sector/industry
        sector_match = re.search(
            r"(?:sector|industry)[:\s]+([\w\s/]+?)(?:\.|,|\n)", all_text, re.IGNORECASE)
        if sector_match:
            sector = sector_match.group(1).strip()[:30]
            if sector and sector not in seen:
                seen.add(sector)
                entities.append({
                    "name": sector,
                    "type": "sector",
                    "relation": "operates_in",
                    "context": f"Industry of {ticker}",
                })

        return entities
