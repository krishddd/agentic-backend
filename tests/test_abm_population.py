"""
Tests for Phase 1 — GraphKnowledgeManager ABM population generation.

Tests the new methods without requiring an LLM or real graph file:
- generate_agent_population()
- _graph_entities_to_roles()
- _assign_type()
- _wire_follow_graph()
"""

import pytest
import networkx as nx
from unittest.mock import MagicMock

from src.abm.contracts import AgentPersona, AgentType
from src.graph_knowledge_manager import GraphKnowledgeManager


@pytest.fixture
def gm_with_graph():
    """Create a GraphKnowledgeManager with a mock LLM and a small test graph."""
    mock_llm = MagicMock()
    gm = GraphKnowledgeManager.__new__(GraphKnowledgeManager)
    gm.llm = mock_llm
    gm.graph_path = ""
    gm.graph = nx.Graph()

    # Build a small financial knowledge graph
    # Nodes: company, people, concepts
    gm.graph.add_node("Tesla", type="entity")
    gm.graph.add_node("Elon Musk", type="entity")
    gm.graph.add_node("SEC", type="entity")
    gm.graph.add_node("CNBC", type="entity")
    gm.graph.add_node("Jim Cramer", type="entity")
    gm.graph.add_node("Ford", type="entity")
    gm.graph.add_node("Retail Investors", type="entity")
    gm.graph.add_node("ARK Invest", type="entity")
    gm.graph.add_node("Battery Tech", type="entity")
    gm.graph.add_node("Revenue Growth", type="entity")

    # Edges with relations
    gm.graph.add_edge("Elon Musk", "Tesla", relations=["CEO of"])
    gm.graph.add_edge("Tesla", "SEC", relations=["filing", "regulation"])
    gm.graph.add_edge("CNBC", "Tesla", relations=["news coverage"])
    gm.graph.add_edge("Jim Cramer", "Tesla", relations=["analyst rating"])
    gm.graph.add_edge("Ford", "Tesla", relations=["competitor"])
    gm.graph.add_edge("Retail Investors", "Tesla", relations=["invested in"])
    gm.graph.add_edge("ARK Invest", "Tesla", relations=["fund holding"])
    gm.graph.add_edge("Tesla", "Battery Tech", relations=["research"])
    gm.graph.add_edge("Tesla", "Revenue Growth", relations=["forecast"])
    gm.graph.add_edge("Jim Cramer", "CNBC", relations=["analyst on"])

    return gm


@pytest.fixture
def gm_empty():
    """GraphKnowledgeManager with an empty graph."""
    gm = GraphKnowledgeManager.__new__(GraphKnowledgeManager)
    gm.llm = MagicMock()
    gm.graph_path = ""
    gm.graph = nx.Graph()
    return gm


# ---------------------------------------------------------------------------
# _assign_type
# ---------------------------------------------------------------------------

class TestAssignType:
    def test_first_10_percent_are_kol(self):
        assert GraphKnowledgeManager._assign_type(0, 100) == AgentType.KOL
        assert GraphKnowledgeManager._assign_type(9, 100) == AgentType.KOL

    def test_next_20_percent_are_llm(self):
        assert GraphKnowledgeManager._assign_type(10, 100) == AgentType.LLM_CITIZEN
        assert GraphKnowledgeManager._assign_type(29, 100) == AgentType.LLM_CITIZEN

    def test_rest_are_rule_based(self):
        assert GraphKnowledgeManager._assign_type(30, 100) == AgentType.RULE_BASED
        assert GraphKnowledgeManager._assign_type(99, 100) == AgentType.RULE_BASED

    def test_small_population_has_at_least_one_of_each(self):
        # With 5 agents: 1 KOL, 1 LLM, 3 rule-based
        assert GraphKnowledgeManager._assign_type(0, 5) == AgentType.KOL
        assert GraphKnowledgeManager._assign_type(1, 5) == AgentType.LLM_CITIZEN
        assert GraphKnowledgeManager._assign_type(2, 5) == AgentType.RULE_BASED


# ---------------------------------------------------------------------------
# _graph_entities_to_roles
# ---------------------------------------------------------------------------

class TestGraphEntitiesToRoles:
    def test_returns_list_of_dicts(self, gm_with_graph):
        entities = gm_with_graph._graph_entities_to_roles("TSLA")
        assert isinstance(entities, list)
        assert all(isinstance(e, dict) for e in entities)

    def test_entity_has_required_keys(self, gm_with_graph):
        entities = gm_with_graph._graph_entities_to_roles("TSLA")
        for e in entities:
            assert "name" in e
            assert "role" in e
            assert "influence" in e
            assert "sentiment" in e

    def test_sorted_by_influence_descending(self, gm_with_graph):
        entities = gm_with_graph._graph_entities_to_roles("TSLA")
        influences = [e["influence"] for e in entities]
        assert influences == sorted(influences, reverse=True)

    def test_role_classification(self, gm_with_graph):
        entities = gm_with_graph._graph_entities_to_roles("TSLA")
        entity_map = {e["name"]: e["role"] for e in entities}
        # SEC should be classified as regulator (keyword: "sec")
        assert entity_map.get("SEC") == "regulator"
        # CNBC should be classified as media (keyword: "cnbc")
        assert entity_map.get("CNBC") == "media"

    def test_empty_graph_returns_empty(self, gm_empty):
        assert gm_empty._graph_entities_to_roles("TSLA") == []


# ---------------------------------------------------------------------------
# generate_agent_population
# ---------------------------------------------------------------------------

class TestGenerateAgentPopulation:
    def test_returns_correct_count(self, gm_with_graph):
        pop = gm_with_graph.generate_agent_population("TSLA", n_agents=20)
        assert len(pop) == 20

    def test_returns_agent_personas(self, gm_with_graph):
        pop = gm_with_graph.generate_agent_population("TSLA", n_agents=10)
        assert all(isinstance(p, AgentPersona) for p in pop)

    def test_unique_agent_ids(self, gm_with_graph):
        pop = gm_with_graph.generate_agent_population("TSLA", n_agents=50)
        ids = [p.agent_id for p in pop]
        assert len(ids) == len(set(ids))

    def test_pads_when_graph_too_small(self, gm_with_graph):
        # Graph has 10 nodes, request 50 agents — should pad with retail
        pop = gm_with_graph.generate_agent_population("TSLA", n_agents=50)
        assert len(pop) == 50
        retail = [p for p in pop if p.role == "retail_investor"]
        assert len(retail) > 0

    def test_empty_graph_still_returns_agents(self, gm_empty):
        pop = gm_empty.generate_agent_population("TSLA", n_agents=20)
        assert len(pop) == 20
        # All should be synthetic retail investors
        assert all(p.role == "retail_investor" for p in pop)

    def test_sentiment_in_range(self, gm_with_graph):
        pop = gm_with_graph.generate_agent_population("TSLA", n_agents=30)
        for p in pop:
            assert -1.0 <= p.sentiment <= 1.0

    def test_influence_in_range(self, gm_with_graph):
        pop = gm_with_graph.generate_agent_population("TSLA", n_agents=30)
        for p in pop:
            assert 0.0 <= p.influence <= 1.0

    def test_stubbornness_in_range(self, gm_with_graph):
        pop = gm_with_graph.generate_agent_population("TSLA", n_agents=30)
        for p in pop:
            assert 0.0 <= p.stubbornness <= 1.0

    def test_type_distribution(self, gm_with_graph):
        pop = gm_with_graph.generate_agent_population("TSLA", n_agents=100)
        kols = sum(1 for p in pop if p.agent_type == AgentType.KOL)
        llms = sum(1 for p in pop if p.agent_type == AgentType.LLM_CITIZEN)
        rules = sum(1 for p in pop if p.agent_type == AgentType.RULE_BASED)
        assert kols >= 1
        assert llms >= 1
        assert rules >= 1
        assert kols + llms + rules == 100


# ---------------------------------------------------------------------------
# _wire_follow_graph
# ---------------------------------------------------------------------------

class TestWireFollowGraph:
    def test_following_populated(self, gm_with_graph):
        pop = gm_with_graph.generate_agent_population("TSLA", n_agents=30)
        # At least some agents should have followers
        agents_with_follows = [p for p in pop if len(p.following) > 0]
        assert len(agents_with_follows) > 0

    def test_no_self_follows(self, gm_with_graph):
        pop = gm_with_graph.generate_agent_population("TSLA", n_agents=30)
        for p in pop:
            assert p.agent_id not in p.following

    def test_following_is_sorted(self, gm_with_graph):
        pop = gm_with_graph.generate_agent_population("TSLA", n_agents=30)
        for p in pop:
            assert p.following == sorted(p.following)

    def test_following_no_duplicates(self, gm_with_graph):
        pop = gm_with_graph.generate_agent_population("TSLA", n_agents=30)
        for p in pop:
            assert len(p.following) == len(set(p.following))

    def test_following_ids_are_valid(self, gm_with_graph):
        pop = gm_with_graph.generate_agent_population("TSLA", n_agents=30)
        valid_ids = {p.agent_id for p in pop}
        for p in pop:
            for fid in p.following:
                assert fid in valid_ids

    def test_kols_are_widely_followed(self, gm_with_graph):
        """KOLs with influence > 0.7 should be followed by most agents."""
        pop = gm_with_graph.generate_agent_population("TSLA", n_agents=50)
        kol_ids = {p.agent_id for p in pop if p.influence > 0.7}
        if not kol_ids:
            pytest.skip("No KOL with influence > 0.7 in this run")
        for kol_id in kol_ids:
            followers = sum(1 for p in pop if kol_id in p.following)
            # KOLs should be followed by at least 50% of non-KOL agents
            non_kol_count = sum(1 for p in pop if p.agent_id != kol_id)
            assert followers >= non_kol_count * 0.5, (
                f"KOL {kol_id} has {followers} followers "
                f"out of {non_kol_count} non-self agents"
            )
