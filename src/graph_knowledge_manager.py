
import logging
import networkx as nx
import os
import json
import random
from typing import List, Tuple, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.abm.contracts import AgentPersona, AgentType

logger = logging.getLogger(__name__)

class GraphKnowledgeManager:
    """
    Manages the Knowledge Graph using NetworkX.
    Extracts entities and relations from text, builds the graph, and queries it.
    """

    def __init__(self, llm: BaseChatModel = None, graph_path: str = "results/knowledge_graph.graphml"):
        self.llm = llm
        self.graph_path = graph_path
        self.graph = nx.Graph()
        self._load_graph()

    def _load_graph(self):
        """Loads the graph from disk if it exists."""
        if os.path.exists(self.graph_path):
            try:
                self.graph = nx.read_graphml(self.graph_path)
                logger.info(f"Loaded knowledge graph with {self.graph.number_of_nodes()} nodes.")
            except Exception as e:
                logger.error(f"Failed to load graph: {e}")
        else:
            logger.info("Initializing new empty knowledge graph.")

    def save_graph(self):
        """Saves the graph to disk."""
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        try:
            nx.write_graphml(self.graph, self.graph_path)
            logger.info(f"Saved knowledge graph with {self.graph.number_of_nodes()} nodes.")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    def extract_triplets(self, text: str) -> List[Dict[str, str]]:
        """
        Uses LLM to extract (Subject, Relation, Object) triplets from text.
        """
        if self.llm is None:
            logger.warning("extract_triplets called but no LLM configured — returning empty.")
            return []

        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Extract knowledge triplets from the following text.
Each triplet should consist of a Subject (Entity), a Relation, and an Object (Entity).
Focus on relationships between people, organizations, locations, and key concepts.
Ignore generic or vague statements.

Text: {text}

Return ONLY a JSON object with a single key "triplets" containing a list of objects like:
{{"triplets": [{{"subject": "Elon Musk", "relation": "CEO of", "object": "Tesla"}}]}}
"""
        )
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            # Chunk text if too large (simplified here)
            result = chain.invoke({"text": text[:4000]})
            return result.get("triplets", [])
        except Exception as e:
            logger.error(f"Triplet extraction failed: {e}")
            return []

    def build_graph_from_documents(self, documents: List[str]):
        """
        Iterates over documents, extracts triplets, and updates the graph.
        """
        logger.info(f"Building graph from {len(documents)} documents...")
        count = 0
        for doc in documents:
            triplets = self.extract_triplets(doc)
            for triplet in triplets:
                subj = triplet.get("subject")
                obj_ = triplet.get("object")
                rel = triplet.get("relation")
                
                if subj and obj_ and rel:
                    self.graph.add_node(subj, type="entity")
                    self.graph.add_node(obj_, type="entity")
                    
                    # Add edge with relation as attribute, store list of relations if multiple exist
                    if self.graph.has_edge(subj, obj_):
                        existing_rels = self.graph[subj][obj_].get('relations', [])
                        if rel not in existing_rels:
                            existing_rels.append(rel)
                            self.graph[subj][obj_]['relations'] = existing_rels
                    else:
                        self.graph.add_edge(subj, obj_, relations=[rel])
            count += 1
            if count % 10 == 0:
                logger.info(f"Processed {count}/{len(documents)} documents for graph.")
        
        self.save_graph()

    def get_context_for_query(self, query: str, hops: int = 1) -> str:
        """
        1. Extract entities from the query.
        2. Find them in the graph.
        3. Traverse 'hops' degrees.
        4. Return a text summary of connections.
        """
        # 1. Extract entities from query using LLM
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Extract the key entities (Subject/Object) from this query. Return JSON list ['Entity1', 'Entity2']. Query: {query}"
        )
        try:
            chain = prompt | self.llm | JsonOutputParser()
            query_entities = chain.invoke({"query": query})
            if isinstance(query_entities, dict):
                 query_entities = query_entities.get('entities', [])
        except Exception:
            return ""

        context_lines = []
        found_entities = []

        # 2. Find in graph
        for entity in query_entities:
            # Simple fuzzy match or direct lookup
            # For this MVP, we try direct match and case-insensitive match
            match = None
            if entity in self.graph:
                match = entity
            else:
                # Naive case insensitive search
                for node in self.graph.nodes():
                    if node.lower() == entity.lower():
                        match = node
                        break
            
            if match:
                found_entities.append(match)
                # 3. Traverse
                subgraph = nx.ego_graph(self.graph, match, radius=hops)
                
                for u, v, data in subgraph.edges(data=True):
                    rels = ", ".join(data.get('relations', []))
                    context_lines.append(f"{u} --[{rels}]--> {v}")

        if not context_lines:
            return ""

        # Dedup
        context_lines = list(set(context_lines))
        return "Graph Knowledge Context:\n" + "\n".join(context_lines)

    # ------------------------------------------------------------------
    # ABM Population Generation (Phase 1)
    # ------------------------------------------------------------------

    # Keywords used to classify graph entities into agent roles
    _ROLE_KEYWORDS: Dict[str, List[str]] = {
        "analyst":         ["analyst", "research", "rating", "forecast", "valuation"],
        "insider":         ["insider", "executive", "ceo", "cfo", "director", "board"],
        "media":           ["media", "news", "cnbc", "bloomberg", "reuters", "journalist"],
        "regulator":       ["sec", "regulation", "compliance", "filing", "fda", "fed"],
        "competitor":      ["competitor", "rival", "competing", "market share"],
        "institutional":   ["fund", "hedge", "institutional", "etf", "blackrock", "vanguard"],
        "retail_investor": [],  # fallback role
    }

    def generate_agent_population(
        self, ticker: str, n_agents: int = 100
    ) -> List[AgentPersona]:
        """Generate diverse agent personas from the knowledge graph.

        Returns ``List[AgentPersona]`` — never ``List[dict]``.

        **Step 1** — Extract graph nodes and classify each into a market
        role (analyst, insider, media, etc.) via relation keywords.

        **Step 2** — If the graph has fewer entities than ``n_agents``,
        pad with synthetic "retail_investor" personas so the simulation
        always reaches the requested scale.

        **Step 3** — Call ``_wire_follow_graph()`` to pre-populate each
        persona's ``following`` list for realistic tick-0 network topology.
        """
        entities = self._graph_entities_to_roles(ticker)
        personas: List[AgentPersona] = []

        for i, entity in enumerate(entities[:n_agents]):
            agent_type = self._assign_type(i, n_agents)
            # Influence derived from graph degree centrality
            influence = entity.get("influence", random.uniform(0.1, 0.4))
            # Clamp to valid range
            influence = max(0.0, min(1.0, influence))

            personas.append(AgentPersona(
                agent_id=i,
                name=entity["name"],
                agent_type=agent_type,
                role=entity["role"],
                sentiment=entity.get("sentiment", random.uniform(-0.5, 0.5)),
                influence=influence,
                stubbornness=random.uniform(0.2, 0.8),
                backstory=entity.get("backstory", ""),
                following=[],
                platform_preference=random.choice(
                    ["microblog"] * 3 + ["forum"] * 2),  # ~60/40 split
            ))

        # Pad with synthetic retail investors if graph is too small
        while len(personas) < n_agents:
            idx = len(personas)
            personas.append(AgentPersona(
                agent_id=idx,
                name=f"Retail_{idx}",
                agent_type=self._assign_type(idx, n_agents),
                role="retail_investor",
                sentiment=random.uniform(-0.5, 0.5),
                influence=random.uniform(0.05, 0.2),
                stubbornness=random.uniform(0.2, 0.8),
                backstory=f"Retail investor following {ticker}",
                following=[],
                platform_preference=random.choice(
                    ["microblog"] * 3 + ["forum"] * 2),  # ~60/40 split
            ))

        # Pre-populate follow graph for realistic tick-0 topology
        self._wire_follow_graph(personas)

        logger.info(
            f"Generated {len(personas)} agent personas for {ticker} "
            f"({sum(1 for p in personas if p.agent_type == AgentType.KOL)} KOL, "
            f"{sum(1 for p in personas if p.agent_type == AgentType.LLM_CITIZEN)} LLM, "
            f"{sum(1 for p in personas if p.agent_type == AgentType.RULE_BASED)} rule-based)"
        )
        return personas

    def _graph_entities_to_roles(self, ticker: str) -> List[Dict[str, Any]]:
        """Convert graph nodes into role-classified entity dicts.

        Each entity gets:
        - ``name``: the node label
        - ``role``: classified from relation keywords
        - ``influence``: derived from degree centrality (0–1)
        - ``sentiment``: random initial value
        - ``backstory``: edges summary for context

        Returns entities sorted by influence descending (highest-degree
        nodes first, so they become KOLs in ``_assign_type``).
        """
        if self.graph.number_of_nodes() == 0:
            return []

        # Degree centrality → influence score
        centrality = nx.degree_centrality(self.graph)
        entities: List[Dict[str, Any]] = []

        for node in self.graph.nodes():
            # Collect all relation labels touching this node
            all_relations: List[str] = []
            for _, _, data in self.graph.edges(node, data=True):
                all_relations.extend(data.get("relations", []))
            relations_text = " ".join(all_relations).lower()

            # Classify role: check node name first, then relation text
            role = "retail_investor"  # fallback
            node_lower = node.lower()

            # Pass 1 — match on entity name (highest priority)
            for candidate_role, keywords in self._ROLE_KEYWORDS.items():
                if any(kw in node_lower for kw in keywords):
                    role = candidate_role
                    break

            # Pass 2 — if still unclassified, match on relation text
            if role == "retail_investor" and relations_text:
                for candidate_role, keywords in self._ROLE_KEYWORDS.items():
                    if any(kw in relations_text for kw in keywords):
                        role = candidate_role
                        break

            # Build a short backstory from the node's edges
            edge_summaries = []
            for _, neighbor, data in self.graph.edges(node, data=True):
                rels = ", ".join(data.get("relations", []))
                edge_summaries.append(f"{node} --[{rels}]--> {neighbor}")
            backstory = "; ".join(edge_summaries[:3])  # cap at 3

            entities.append({
                "name": str(node),
                "role": role,
                "influence": centrality.get(node, 0.1),
                "sentiment": random.uniform(-0.5, 0.5),
                "backstory": backstory,
            })

        # Sort by influence descending so high-degree nodes → KOLs
        entities.sort(key=lambda e: e["influence"], reverse=True)
        return entities

    @staticmethod
    def _assign_type(index: int, total: int) -> AgentType:
        """Assign agent type by position in influence-sorted list.

        Distribution: ~10% KOL, ~20% LLM, ~70% rule-based.
        """
        kol_cutoff = max(1, int(total * 0.10))
        llm_cutoff = kol_cutoff + max(1, int(total * 0.20))
        if index < kol_cutoff:
            return AgentType.KOL
        elif index < llm_cutoff:
            return AgentType.LLM_CITIZEN
        return AgentType.RULE_BASED

    @staticmethod
    def _wire_follow_graph(personas: List[AgentPersona]) -> None:
        """Build a realistic power-law follow topology.

        Three rules applied at population generation time:

        1. **KOL gravity** — every agent follows every KOL (public figures).
        2. **Community clustering** — agents with the same ``role`` follow
           up to 5 same-role peers.
        3. **Weak ties** — 10% random cross-community follows to allow
           information to bridge echo chambers.

        The resulting ``persona.following`` lists are later written to the
        SQLite ledger by ``MarketSentimentModel._seed_follow_graph()``.
        """
        kol_ids = [p.agent_id for p in personas if p.influence > 0.7]

        # Group agents by role
        role_groups: Dict[str, List[int]] = {}
        for p in personas:
            role_groups.setdefault(p.role, []).append(p.agent_id)

        for p in personas:
            following_set: set = set()

            # Rule 1 — follow all KOLs (except self)
            for kid in kol_ids:
                if kid != p.agent_id:
                    following_set.add(kid)

            # Rule 2 — follow same-role peers (cap at 5)
            peers = [pid for pid in role_groups.get(p.role, [])
                     if pid != p.agent_id]
            for peer in peers[:5]:
                following_set.add(peer)

            # Rule 3 — random weak ties (10% of remaining agents)
            others = [o.agent_id for o in personas
                      if o.agent_id != p.agent_id
                      and o.agent_id not in following_set]
            n_random = max(1, int(len(others) * 0.10))
            if others:
                for pick in random.sample(others, min(n_random, len(others))):
                    following_set.add(pick)

            p.following = sorted(following_set)

    # ------------------------------------------------------------------
    # Grandmaster: Reality Seed Extraction
    # ------------------------------------------------------------------

    def extract_reality_seeds(self, data_bundle: dict, ticker: str):
        """Parse pipeline data into structured graph entities.

        Extracts real entities (companies, people, metrics, events) from
        the pipeline research/financial/news/SEC data and builds the
        knowledge graph from ground truth rather than random generation.

        Args:
            data_bundle: Dict with research_output, financial_summary,
                         news_summary, sec_analysis, sentiment_data.
            ticker: The target ticker symbol.
        """
        # Always ensure ticker is a central node
        self.graph.add_node(ticker, type="company", role="target",
                            description=f"Target company: {ticker}")

        # Extract from each pipeline source
        sources = {
            "research": data_bundle.get("research_output", ""),
            "financial": data_bundle.get("financial_summary", ""),
            "news": data_bundle.get("news_summary", ""),
            "sec": data_bundle.get("sec_analysis", ""),
        }

        for source_label, text in sources.items():
            if not text or len(text) < 50:
                continue

            # Rule-based entity extraction (no LLM needed — fast)
            entities = self._extract_entities_rule_based(text, ticker)
            for entity in entities:
                ename = entity["name"]
                if ename == ticker:
                    continue
                self.graph.add_node(
                    ename,
                    type=entity.get("type", "entity"),
                    role=entity.get("role", "related"),
                    description=entity.get("context", "")[:200],
                    source=source_label,
                )
                self.graph.add_edge(
                    ticker, ename,
                    relations=[entity.get("relation", "mentioned_with")],
                    source=source_label,
                )

            # LLM-based triplet extraction (if LLM available)
            if self.llm and len(text) > 200:
                try:
                    triplets = self.extract_triplets(text[:3000])
                    for t in triplets[:20]:  # cap
                        subj = t.get("subject", "")
                        obj_ = t.get("object", "")
                        rel = t.get("relation", "")
                        if subj and obj_ and rel:
                            self.graph.add_node(subj, type="entity")
                            self.graph.add_node(obj_, type="entity")
                            if self.graph.has_edge(subj, obj_):
                                rels = self.graph[subj][obj_].get(
                                    "relations", [])
                                if rel not in rels:
                                    rels.append(rel)
                                    self.graph[subj][obj_]["relations"] = rels
                            else:
                                self.graph.add_edge(
                                    subj, obj_, relations=[rel])
                except Exception as e:
                    logger.warning(f"LLM triplet extraction failed: {e}")

        self.save_graph()
        logger.info(
            f"Reality seeds extracted: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges")

    def _extract_entities_rule_based(
        self, text: str, ticker: str
    ) -> List[Dict[str, str]]:
        """Fast rule-based entity extraction from pipeline text.

        Extracts: company names (uppercase 2-5 letter tokens), people
        (title-case multi-word), numeric metrics, and known keywords.
        """
        import re
        entities = []
        seen = set()

        # Pattern 1: Ticker-like symbols (2-5 uppercase letters)
        ticker_pattern = re.findall(r'\b([A-Z]{2,5})\b', text)
        for t in ticker_pattern:
            skip = {"AND", "THE", "FOR", "WAS", "ARE", "INC", "LTD",
                    "ETF", "SEC", "CEO", "CFO", "COO", "IPO", "GDP",
                    "API", "USA", "LLC", "ALL", "NOT", "BUT", "CAN",
                    "HAS", "HAD", "ITS", "OUR", "NEW", "NOW", "MAY"}
            if t not in skip and t != ticker and t not in seen:
                seen.add(t)
                entities.append({
                    "name": t, "type": "company",
                    "relation": "competitor_or_peer",
                    "role": "competitor",
                    "context": f"Mentioned alongside {ticker}",
                })

        # Pattern 2: Named people (Title Case sequences)
        people_pattern = re.findall(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b', text)
        for person in people_pattern[:10]:
            if person not in seen and len(person) > 5:
                seen.add(person)
                # Classify role from context
                role = "retail_investor"
                context_window = text[max(0,
                    text.find(person) - 100):text.find(person) + 100].lower()
                for r, kws in self._ROLE_KEYWORDS.items():
                    if any(kw in context_window for kw in kws):
                        role = r
                        break
                entities.append({
                    "name": person, "type": "person",
                    "relation": "associated_with",
                    "role": role,
                    "context": context_window[:200],
                })

        # Pattern 3: Key metrics (revenue, margin, etc.)
        metric_patterns = [
            (r'revenue[:\s]+\$?([\d.]+\s*[BMK]?)',
             "revenue_metric", "has_metric"),
            (r'margin[:\s]+([\d.]+%)',
             "margin_metric", "has_metric"),
            (r'growth[:\s]+([\d.]+%)',
             "growth_metric", "has_metric"),
            (r'market\s+cap[:\s]+\$?([\d.]+\s*[BMT]?)',
             "market_cap", "has_metric"),
        ]
        for pattern, metric_type, relation in metric_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:3]:
                metric_name = f"{ticker}_{metric_type}_{match}"
                if metric_name not in seen:
                    seen.add(metric_name)
                    entities.append({
                        "name": metric_name, "type": "metric",
                        "relation": relation,
                        "role": "data_point",
                        "context": f"{metric_type}: {match}",
                    })

        return entities

    # ------------------------------------------------------------------
    # Grandmaster: Memory Injection
    # ------------------------------------------------------------------

    def inject_collective_memory(self, ticker: str) -> str:
        """Build shared collective memory from graph for all agents.

        Returns a structured text block that represents the market's
        collective knowledge about the ticker, derived from the graph.
        """
        if self.graph.number_of_nodes() == 0:
            return ""

        lines = [f"COLLECTIVE MARKET MEMORY FOR {ticker}:"]

        # Central entity info
        if ticker in self.graph:
            neighbors = list(self.graph.neighbors(ticker))
            lines.append(f"  {ticker} has {len(neighbors)} known relationships")

            # Categorize relationships
            competitors = []
            people = []
            metrics = []
            for n in neighbors:
                ndata = self.graph.nodes.get(n, {})
                ntype = ndata.get("type", "entity")
                if ntype == "company" or ndata.get("role") == "competitor":
                    competitors.append(str(n))
                elif ntype == "person":
                    people.append(str(n))
                elif ntype == "metric":
                    metrics.append(str(n))

            if competitors:
                lines.append(
                    f"  Competitors/Peers: {', '.join(competitors[:10])}")
            if people:
                lines.append(
                    f"  Key People: {', '.join(people[:10])}")
            if metrics:
                lines.append(
                    f"  Key Metrics: {', '.join(metrics[:5])}")

        # Top influencers by degree
        if self.graph.number_of_nodes() > 1:
            centrality = nx.degree_centrality(self.graph)
            top = sorted(centrality.items(), key=lambda x: x[1],
                         reverse=True)[:5]
            lines.append("  Most Connected Entities:")
            for node, cent in top:
                lines.append(f"    - {node} (connectivity={cent:.3f})")

        return "\n".join(lines)

    def inject_individual_memory(
        self, persona: AgentPersona
    ) -> List[str]:
        """Seed a persona's memory buffer from relevant graph edges.

        Builds initial memory entries based on the persona's role and
        any matching graph nodes, so agents start with grounded knowledge.

        Args:
            persona: The agent persona to enrich.

        Returns:
            List of memory strings to prepend to persona.memory_buffer.
        """
        memories = []

        # Find matching graph node
        name_lower = persona.name.lower()
        match_node = None
        for node in self.graph.nodes():
            if str(node).lower() == name_lower:
                match_node = node
                break

        if match_node:
            # Add direct relationship memories
            for _, neighbor, data in self.graph.edges(match_node, data=True):
                rels = ", ".join(data.get("relations", ["connected_to"]))
                memories.append(f"I know: {match_node} {rels} {neighbor}")
                if len(memories) >= 5:
                    break

        # Role-based knowledge injection
        role_knowledge = {
            "analyst": "I analyze financial data, valuations, and market trends.",
            "insider": "I have deep knowledge of company operations and strategy.",
            "media": "I track and report on market news and corporate events.",
            "regulator": "I monitor regulatory compliance and SEC filings.",
            "competitor": "I understand competitive dynamics and market positioning.",
            "institutional": "I manage large positions and track institutional flows.",
            "retail_investor": "I follow market sentiment and social media trends.",
        }
        role_mem = role_knowledge.get(persona.role, "")
        if role_mem:
            memories.insert(0, role_mem)

        return memories[:8]  # cap at 8 initial memories
