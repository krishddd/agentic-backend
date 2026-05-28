"""
GraphRAG Engine — Grandmaster Edition.

Provides natural language querying over the knowledge graph to inject
relevant context into agent prompts during simulation. Uses semantic
similarity matching on node/edge descriptions to retrieve relevant
subgraphs.

Usage:
    from src.abm.graphrag import GraphRAGEngine
    rag = GraphRAGEngine(graph)
    context = rag.query("What competitors threaten PANW?")
    agent_ctx = rag.get_agent_context(agent_id=5, persona=persona)
"""

import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


class GraphRAGEngine:
    """Query the knowledge graph with natural language and retrieve context.

    Two retrieval modes:
    1. **Global query** — find subgraph relevant to a question
    2. **Agent context** — build personalized context for a specific agent
       based on their role, backstory, and graph neighborhood
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self._node_texts: Dict[str, str] = {}
        self._edge_texts: List[Tuple[str, str, str]] = []
        self._build_index()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self):
        """Build searchable text index from graph nodes and edges."""
        for node, data in self.graph.nodes(data=True):
            text_parts = [str(node)]
            for key in ("description", "backstory", "role", "label", "type"):
                if key in data:
                    text_parts.append(str(data[key]))
            self._node_texts[str(node)] = " ".join(text_parts).lower()

        for u, v, data in self.graph.edges(data=True):
            label = data.get("relation", data.get("label", "related_to"))
            text = f"{u} {label} {v}".lower()
            self._edge_texts.append((str(u), str(v), text))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 10) -> str:
        """Natural language query → relevant graph context string.

        Uses TF-IDF-like keyword matching to find relevant nodes and
        edges. Returns a formatted context block suitable for LLM prompts.

        Args:
            question: Natural language question.
            top_k: Maximum number of relevant items to return.

        Returns:
            Formatted context string with relevant entities and relations.
        """
        query_tokens = self._tokenize(question)
        if not query_tokens:
            return "(no relevant graph context)"

        # Score nodes
        node_scores: List[Tuple[str, float]] = []
        for node_id, text in self._node_texts.items():
            score = self._score_match(query_tokens, text)
            if score > 0:
                node_scores.append((node_id, score))
        node_scores.sort(key=lambda x: x[1], reverse=True)

        # Score edges
        edge_scores: List[Tuple[Tuple[str, str, str], float]] = []
        for u, v, text in self._edge_texts:
            score = self._score_match(query_tokens, text)
            if score > 0:
                edge_scores.append(((u, v, text), score))
        edge_scores.sort(key=lambda x: x[1], reverse=True)

        # Build context
        lines = ["KNOWLEDGE GRAPH CONTEXT:"]

        # Top entities
        top_nodes = node_scores[:top_k]
        if top_nodes:
            lines.append("\nRelevant Entities:")
            for node_id, score in top_nodes:
                data = self.graph.nodes.get(node_id, {})
                role = data.get("role", data.get("type", "entity"))
                desc = data.get("description", data.get("backstory", ""))
                lines.append(f"  - {node_id} ({role}): {desc[:200]}")

        # Top relationships
        top_edges = edge_scores[:top_k]
        if top_edges:
            lines.append("\nRelevant Relationships:")
            for (u, v, text), score in top_edges:
                edge_data = self.graph.edges.get((u, v), {})
                relation = edge_data.get("relation", edge_data.get(
                    "label", "related_to"))
                lines.append(f"  - {u} --[{relation}]--> {v}")

        return "\n".join(lines) if len(lines) > 1 else "(no relevant context)"

    def get_agent_context(
        self,
        agent_id: int,
        persona_name: str = "",
        persona_role: str = "",
        depth: int = 2,
    ) -> str:
        """Build personalized graph context for a specific agent.

        Retrieves the ego-graph (neighborhood) around the agent's
        associated entity in the knowledge graph.

        Args:
            agent_id: Agent's numeric ID.
            persona_name: Agent's name (used to find matching graph node).
            persona_role: Agent's role for context enrichment.
            depth: Neighborhood depth to traverse.

        Returns:
            Formatted context string for the agent's LLM prompt.
        """
        # Find matching node
        match_node = None
        name_lower = persona_name.lower()
        for node in self.graph.nodes():
            if str(node).lower() == name_lower:
                match_node = node
                break

        if match_node is None:
            # Fallback: query by role
            if persona_role:
                return self.query(f"{persona_role} {persona_name}", top_k=5)
            return ""

        # Get ego graph
        try:
            ego = nx.ego_graph(self.graph, match_node, radius=depth)
        except Exception:
            return ""

        lines = [f"AGENT KNOWLEDGE ({persona_name}, {persona_role}):"]

        # Direct connections
        neighbors = list(ego.neighbors(match_node))
        if neighbors:
            lines.append(f"  Connected to {len(neighbors)} entities:")
            for n in neighbors[:10]:
                edge_data = ego.edges.get((match_node, n), {})
                relation = edge_data.get("relation", "connected_to")
                lines.append(f"    - {n} ({relation})")

        # 2-hop relationships
        if depth >= 2:
            second_hop = set()
            for n in neighbors:
                for nn in ego.neighbors(n):
                    if nn != match_node and nn not in neighbors:
                        second_hop.add((n, nn))
            if second_hop:
                lines.append(f"  Extended network ({len(second_hop)} links):")
                for n, nn in list(second_hop)[:8]:
                    lines.append(f"    - {n} → {nn}")

        return "\n".join(lines)

    def get_collective_memory(self, ticker: str) -> str:
        """Extract collective market memory from graph for a ticker.

        Gathers all relationships and attributes associated with the
        ticker entity to build a shared context that all agents receive.

        Returns:
            Formatted collective memory string.
        """
        context = self.query(ticker, top_k=20)
        if not context or context.startswith("(no"):
            return ""

        # Add graph-level statistics
        stats_lines = [
            f"\nGRAPH STATISTICS:",
            f"  Nodes: {self.graph.number_of_nodes()}",
            f"  Edges: {self.graph.number_of_edges()}",
        ]

        # Top influencers by degree
        if self.graph.number_of_nodes() > 0:
            centrality = nx.degree_centrality(self.graph)
            top = sorted(centrality.items(), key=lambda x: x[1],
                         reverse=True)[:5]
            if top:
                stats_lines.append("  Top entities by connectivity:")
                for node, cent in top:
                    stats_lines.append(f"    - {node} (centrality={cent:.3f})")

        return context + "\n" + "\n".join(stats_lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace tokenizer with stopword removal."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "and", "but", "or", "not", "no", "so",
            "if", "then", "than", "that", "this", "these", "those",
            "what", "which", "who", "whom", "how", "when", "where", "why",
        }
        tokens = text.lower().split()
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    @staticmethod
    def _score_match(query_tokens: List[str], text: str) -> float:
        """TF-IDF-like scoring: count query token hits in text."""
        text_tokens = set(text.split())
        hits = sum(1 for t in query_tokens if t in text_tokens)
        if hits == 0:
            return 0.0
        # Normalize by query length, boost for higher coverage
        coverage = hits / len(query_tokens)
        return coverage * math.log(1 + hits)
