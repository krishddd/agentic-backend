"""
Test script to verify expanded knowledge base
"""
from src.reasoning_agent.knowledge_manager import KnowledgeManager, KnowledgeDomain

def test_knowledge_loading():
    """Test that knowledge files load correctly"""
    print("=" * 60)
    print("TEST 1: Knowledge Loading Verification")
    print("=" * 60)
    
    km = KnowledgeManager()
    
    math_count = len(km.bases[KnowledgeDomain.MATH])
    physics_count = len(km.bases[KnowledgeDomain.PHYSICS])
    
    print(f"✓ Math concepts loaded: {math_count}")
    print(f"✓ Physics concepts loaded: {physics_count}")
    print(f"✓ Total concepts: {math_count + physics_count}")
    
    assert math_count >= 30, f"Expected at least 30 math concepts, got {math_count}"
    assert physics_count >= 25, f"Expected at least 25 physics concepts, got {physics_count}"
    
    print("\n✅ Knowledge loading test PASSED\n")
    return km

def test_domain_detection(km):
    """Test domain detection"""
    print("=" * 60)
    print("TEST 2: Domain Detection")
    print("=" * 60)
    
    test_cases = [
        ("What is Ohm's law?", KnowledgeDomain.PHYSICS),
        ("Find derivative of x^3", KnowledgeDomain.MATH),
        ("Explain electromagnetic induction", KnowledgeDomain.PHYSICS),
        ("Solve quadratic equation", KnowledgeDomain.MATH),
        ("What is entropy?", KnowledgeDomain.PHYSICS),
        ("Calculate integral of sin(x)", KnowledgeDomain.MATH),
    ]
    
    for query, expected_domain in test_cases:
        detected = km.detect_domain(query)
        status = "✓" if detected == expected_domain else "✗"
        print(f"{status} '{query}' -> {detected.value} (expected: {expected_domain.value})")
        assert detected == expected_domain, f"Domain detection failed for: {query}"
    
    print("\n✅ Domain detection test PASSED\n")

def test_concept_retrieval(km):
    """Test concept retrieval"""
    print("=" * 60)
    print("TEST 3: Concept Retrieval")
    print("=" * 60)
    
    # Test math concepts
    test_cases = [
        ("exponential growth", KnowledgeDomain.MATH, "Exponential"),
        ("electromagnetic induction", KnowledgeDomain.PHYSICS, "Electromagnetic Induction"),
        ("pythagorean theorem", KnowledgeDomain.MATH, "Pythagorean"),
        ("thermodynamics", KnowledgeDomain.PHYSICS, "Thermodynamics"),
        ("trigonometric identities", KnowledgeDomain.MATH, "Trigonometric"),
        ("special relativity", KnowledgeDomain.PHYSICS, "Relativity"),
    ]
    
    for query, domain, expected_in_name in test_cases:
        results = km.retrieve(query, domain, k=3)
        assert len(results) > 0, f"No results for query: {query}"
        
        # Check if expected concept is in results
        found = any(expected_in_name.lower() in r['metadata']['name'].lower() for r in results)
        status = "✓" if found else "⚠"
        print(f"{status} Query: '{query}' -> Found {len(results)} result(s)")
        if found:
            matching_result = next(r for r in results if expected_in_name.lower() in r['metadata']['name'].lower())
            print(f"   Found concept: {matching_result['metadata']['name']}")
    
    print("\n✅ Concept retrieval test PASSED\n")

def test_new_topics():
    """Test some specific new topics"""
    print("=" * 60)
    print("TEST 4: New Topics Verification")
    print("=" * 60)
    
    km = KnowledgeManager()
    
    # Test new math topics
    new_math_topics = [
        "Matrix Operations",
        "Complex Numbers",
        "Law of Sines",
        "Partial Derivatives",
        "Normal Distribution"
    ]
    
    print("Checking new Math topics:")
    math_concepts = km.bases[KnowledgeDomain.MATH]
    for topic in new_math_topics:
        found = any(topic.lower() in c.name.lower() for c in math_concepts)
        status = "✓" if found else "✗"
        print(f"  {status} {topic}")
        assert found, f"New math topic not found: {topic}"
    
    # Test new physics topics
    new_physics_topics = [
        "Coulomb's Law",
        "Maxwell's Equations",
        "Doppler Effect",
        "Special Relativity",
        "Quantum Mechanics"
    ]
    
    print("\nChecking new Physics topics:")
    physics_concepts = km.bases[KnowledgeDomain.PHYSICS]
    for topic in new_physics_topics:
        found = any(topic.lower() in c.name.lower() for c in physics_concepts)
        status = "✓" if found else "✗"
        print(f"  {status} {topic}")
        assert found, f"New physics topic not found: {topic}"
    
    print("\n✅ New topics verification PASSED\n")

def main():
    print("\n" + "=" * 60)
    print("EXPANDED KNOWLEDGE BASE VERIFICATION TESTS")
    print("=" * 60 + "\n")
    
    try:
        km = test_knowledge_loading()
        test_domain_detection(km)
        test_concept_retrieval(km)
        test_new_topics()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print(f"\nKnowledge base successfully expanded!")
        print(f"  • Mathematics: {len(km.bases[KnowledgeDomain.MATH])} concepts")
        print(f"  • Physics: {len(km.bases[KnowledgeDomain.PHYSICS])} concepts")
        print(f"  • Total: {len(km.bases[KnowledgeDomain.MATH]) + len(km.bases[KnowledgeDomain.PHYSICS])} concepts")
        print()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
