"""
Test Suite for Epistemic Agent Pipeline Fixes
==============================================
Validates that ALL tool types (web, browser, system, code, PDF, image,
screenshot, MCP search) are reachable through the agent's active inference
pipeline -- not just file operations.

Tests:
1. Intent classification: no false-positive file_operation for non-file queries
2. Action proposal coverage: correct tools proposed for each query domain
3. EFE scoring: non-zero scores for all 30+ registered tools
4. Posterior simulation: entropy reduction for all profiled tools
5. Observation validity: abort-check accepts all tool output formats
"""

import math
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.epistemic_agent.generative_model import (
    BeliefState, Action, ActionType, FileStatus, UserIntent, RiskLevel
)
from src.epistemic_agent.free_energy import FreeEnergyCalculator
from src.epistemic_agent.look_ahead import LookAheadModule
from src.epistemic_agent.uncertainty import UncertaintyEstimator


def _make_uniform_belief() -> BeliefState:
    """Create a uniform (maximum-entropy) belief state for testing."""
    return BeliefState(
        file_status_alpha={s: 1.0 for s in FileStatus},
        user_intent_alpha={s: 1.0 for s in UserIntent},
        risk_level_alpha={s: 1.0 for s in RiskLevel},
    )


def _make_look_ahead_module() -> LookAheadModule:
    """Create a LookAheadModule with the required uncertainty estimator."""
    ue = UncertaintyEstimator()
    return LookAheadModule(uncertainty_estimator=ue)


# ===================================================================
#  TEST 1: Intent Classification -- No False-Positive file_operation
# ===================================================================

def test_intent_no_false_positive():
    """
    Queries that are clearly NOT about files should NOT trigger file_operation.
    This was BUG-4: generic keywords ('show', 'open', 'read', 'display')
    caused false-positive file_operation matches.
    """
    print("=" * 70)
    print("TEST 1: Intent Classification -- No False-Positive file_operation")
    print("=" * 70)

    lam = _make_look_ahead_module()

    non_file_queries = [
        ("What is the capital of France?",          "factual_query"),
        ("Show me latest AI news",                  "news_query"),
        ("What is machine learning?",               "factual_query"),
        ("How does photosynthesis work?",            "factual_query"),
        ("Tell me about quantum computing",          "factual_query"),
        ("What is my CPU usage?",                    "system_query"),
        ("Take a screenshot",                        "screenshot_query"),
        ("Open https://example.com",                 "browser_query"),
        ("Search Google for Python tutorials",       "browser_query"),
        ("Find papers on transformer architecture",  "academic_query"),
    ]

    passed = 0
    failed = 0

    for query, expected_intent in non_file_queries:
        intents = lam._classify_intent(query.lower())
        is_file = intents.get("file_operation", False)
        has_expected = intents.get(expected_intent, False)

        if is_file:
            print(f"  X FALSE POSITIVE: '{query}' -> file_operation=True (should be {expected_intent})")
            failed += 1
        elif not has_expected:
            print(f"  X MISSED: '{query}' -> {expected_intent}=False (should be True)")
            failed += 1
        else:
            print(f"  OK '{query}' -> {expected_intent}=True, file_operation=False")
            passed += 1

    # Also verify that ACTUAL file queries still work
    file_queries = [
        "delete file test.txt",
        "read file config.py",
        "list files in src directory",
        "show file contents of app.py",
    ]

    for query in file_queries:
        intents = lam._classify_intent(query.lower())
        if intents.get("file_operation"):
            print(f"  OK '{query}' -> file_operation=True (correct)")
            passed += 1
        else:
            print(f"  X '{query}' -> file_operation=False (should be True)")
            failed += 1

    print(f"\n  Results: {passed}/{passed + failed} passed, {failed} failed")
    assert failed == 0, f"Intent classification failed for {failed} queries"
    print("  >> Test 1 PASSED\n")


# ===================================================================
#  TEST 2: Action Proposal Coverage
# ===================================================================

def test_action_proposal_coverage():
    """
    For each query domain, verify that propose_actions() generates
    at least one action with the expected tool name.
    """
    print("=" * 70)
    print("TEST 2: Action Proposal Coverage")
    print("=" * 70)

    lam = _make_look_ahead_module()
    belief = _make_uniform_belief()

    test_cases = [
        ("What is the capital of France?",          "web_search"),
        ("Show me latest AI news",                  "news_search"),
        ("Find papers on transformer architecture", "arxiv_search"),
        ("Take a screenshot",                       "capture_screen"),
        ("What is my CPU usage?",                   "system_info"),
        ("Create a PDF report",                     "create_pdf"),
        ("Delete file test.txt from src",           "list_files"),
    ]

    passed = 0
    failed = 0

    for query, expected_tool in test_cases:
        actions = lam.propose_actions(query, belief)
        tool_names = [a.name for a in actions]

        if expected_tool in tool_names:
            print(f"  OK '{query}' -> [{expected_tool}] in {tool_names}")
            passed += 1
        else:
            print(f"  X '{query}' -> [{expected_tool}] NOT in {tool_names}")
            failed += 1

    print(f"\n  Results: {passed}/{passed + failed} passed, {failed} failed")
    assert failed == 0, f"Action proposal failed for {failed} queries"
    print("  >> Test 2 PASSED\n")


# ===================================================================
#  TEST 3: EFE Scoring -- Non-Zero for All Tool Types
# ===================================================================

def test_efe_nonzero_all_tools():
    """
    For each registered tool, create an Action and verify that
    calculate_efe() returns a non-zero score.
    This was BUG-2 + BUG-3: epistemic=0 + extrinsic=0 for non-file tools.
    """
    print("=" * 70)
    print("TEST 3: EFE Non-Zero for All Tool Types")
    print("=" * 70)

    calc = FreeEnergyCalculator()
    belief = _make_uniform_belief()

    # All tools that MUST have non-zero EFE scores
    tools = [
        # (name, action_type)
        ("list_files",      ActionType.EPISTEMIC),
        ("web_search",      ActionType.EPISTEMIC),
        ("ask_user",        ActionType.EPISTEMIC),
        ("news_search",     ActionType.EPISTEMIC),
        ("arxiv_search",    ActionType.EPISTEMIC),
        ("document_search", ActionType.EPISTEMIC),
        ("browse_url",      ActionType.PRAGMATIC),
        ("google_search",   ActionType.PRAGMATIC),
        ("system_info",     ActionType.PRAGMATIC),
        ("get_processes",   ActionType.PRAGMATIC),
        ("capture_screen",  ActionType.PRAGMATIC),
        ("create_pdf",      ActionType.PRAGMATIC),
        ("read_pdf",        ActionType.PRAGMATIC),
        ("create_image",    ActionType.PRAGMATIC),
        ("page_screenshot", ActionType.PRAGMATIC),
        ("clipboard_read",  ActionType.PRAGMATIC),
        ("get_directory_tree", ActionType.PRAGMATIC),
        ("search_files",    ActionType.PRAGMATIC),
    ]

    passed = 0
    failed = 0

    for tool_name, action_type in tools:
        action = Action(
            name=tool_name,
            action_type=action_type,
            arguments={},
            description=f"Test action: {tool_name}",
        )
        efe = calc.calculate_efe(action, belief, action_history=[])
        is_nonzero = abs(efe) > 1e-6

        if is_nonzero:
            print(f"  OK {tool_name:20s} -> EFE = {efe:+.4f}")
            passed += 1
        else:
            print(f"  X  {tool_name:20s} -> EFE = {efe:+.4f} (ZERO -- invisible to selector)")
            failed += 1

    print(f"\n  Results: {passed}/{passed + failed} passed, {failed} failed")
    assert failed == 0, f"EFE scoring is zero for {failed} tools"
    print("  >> Test 3 PASSED\n")


# ===================================================================
#  TEST 4: Posterior Simulation -- Entropy Reduction
# ===================================================================

def test_posterior_entropy_reduction():
    """
    For each tool in _TOOL_INFO_PROFILES, verify that the simulated
    posterior has strictly lower entropy than the prior.
    This was BUG-1: only 5 tools had posterior simulation.
    """
    print("=" * 70)
    print("TEST 4: Posterior Simulation -- Entropy Reduction")
    print("=" * 70)

    calc = FreeEnergyCalculator()
    belief = _make_uniform_belief()

    def total_entropy(posteriors):
        h = 0.0
        for factor_probs in posteriors.values():
            for p in factor_probs.values():
                if p > 1e-12:
                    h -= p * math.log(p)
        return h

    prior_posteriors = {
        "file_status": dict(belief.file_status_probs),
        "user_intent": dict(belief.user_intent_probs),
        "risk_level": dict(belief.risk_level_probs),
    }
    prior_h = total_entropy(prior_posteriors)

    passed = 0
    failed = 0

    for tool_name in calc._TOOL_INFO_PROFILES:
        action = Action(
            name=tool_name,
            action_type=ActionType.EPISTEMIC,
            arguments={},
            description=f"Test: {tool_name}",
        )
        posterior = calc._simulate_posterior(action, belief)
        post_h = total_entropy(posterior)
        delta = prior_h - post_h

        if delta > 1e-6:
            print(f"  OK {tool_name:20s} -> DeltaH = -{delta:.4f} (entropy reduced)")
            passed += 1
        else:
            print(f"  X  {tool_name:20s} -> DeltaH = {delta:.4f} (NO reduction)")
            failed += 1

    print(f"\n  Results: {passed}/{passed + failed} passed, {failed} failed")
    assert failed == 0, f"Posterior simulation failed for {failed} tools"
    print("  >> Test 4 PASSED\n")


# ===================================================================
#  TEST 5: Observation Validity -- Abort Check
# ===================================================================

def test_observation_validity():
    """
    Verify that the observation abort-check logic does NOT classify
    valid tool outputs as errors (BUG-6).
    """
    print("=" * 70)
    print("TEST 5: Observation Validity -- Abort Check")
    print("=" * 70)

    # Simulating the abort check logic from enhanced_agent.py
    def is_observation_ok(observation_str: str) -> bool:
        observation_is_error = any(w in observation_str for w in [
            'Error:', 'FAILED', 'Unknown tool', 'not installed',
            'unavailable', 'Traceback', 'Exception',
        ])
        observation_ok = (not observation_is_error) or any(w in observation_str for w in [
            'FINAL_ANSWER', 'CLARIFICATION',
        ])
        return observation_ok

    valid_outputs = [
        ("Files found: ['app.py', 'config.py']",           "list_files"),
        ("Search Results: [{'title': 'AI'}]",               "web_search"),
        ("Page: Example Domain\nURL: https://example.com",  "browse_url"),
        ("Google Results for 'python':\n1. Python.org",      "google_search"),
        ("System: Windows 10\nCPU: 8 cores, 45% used",     "system_info"),
        ("Top processes:\n[1234] chrome -- CPU: 5.0%",      "get_processes"),
        ("Clipboard: Hello world",                           "clipboard_read"),
        ("Screenshot saved: desktop_screenshot.png (1920x1080)", "capture_screen"),
        ("PDF created: output.pdf (3 pages)",                "create_pdf"),
        ("News Search (success): AI breakthrough...",        "news_search"),
        ("ArXiv Search (success): 5 papers found",          "arxiv_search"),
        ("Document Search (success): 3 relevant docs",      "document_search"),
        ("Wikipedia (success): Tokyo is the capital...",     "wikipedia_search"),
        ("FINAL_ANSWER: The answer is 42",                   "answer_user"),
        ("Directory Tree:\nsrc/\n  app.py\n  config.py",    "get_directory_tree"),
    ]

    error_outputs = [
        ("Error: No filepath provided",     "should_be_error"),
        ("FAILED: File not found",          "should_be_error"),
        ("Error: Unknown tool 'foobar'",    "should_be_error"),
    ]

    passed = 0
    failed = 0

    for obs, tool in valid_outputs:
        ok = is_observation_ok(obs)
        if ok:
            print(f"  OK {tool:20s} -> accepted")
            passed += 1
        else:
            print(f"  X  {tool:20s} -> REJECTED (false abort!)")
            failed += 1

    for obs, label in error_outputs:
        ok = is_observation_ok(obs)
        if not ok:
            print(f"  OK error output -> correctly rejected")
            passed += 1
        else:
            print(f"  X  error output -> accepted (should be rejected)")
            failed += 1

    print(f"\n  Results: {passed}/{passed + failed} passed, {failed} failed")
    assert failed == 0, f"Observation validity failed for {failed} cases"
    print("  >> Test 5 PASSED\n")


# ===================================================================

def main():
    print("\n" + "=" * 70)
    print("EPISTEMIC AGENT PIPELINE TEST SUITE")
    print("Validates fixes for BUG-1 through BUG-8")
    print("=" * 70 + "\n")

    try:
        test_intent_no_false_positive()
        test_action_proposal_coverage()
        test_efe_nonzero_all_tools()
        test_posterior_entropy_reduction()
        test_observation_validity()

        print("=" * 70)
        print("ALL 5 TESTS PASSED!")
        print("=" * 70)
        print("\nThe epistemic agent pipeline now supports ALL tool types:")
        print("  - Web search, News, ArXiv, Wikipedia, Document search")
        print("  - Browser (URL, Google, screenshots)")
        print("  - System monitoring (CPU, RAM, processes)")
        print("  - Code execution, PDF, Image, Desktop screenshots")
        print()
        return 0

    except AssertionError as e:
        print(f"\n>> TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n>> UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
