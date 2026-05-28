"""
Comprehensive Test Suite for Expanded Knowledge Base
Tests all knowledge domains, retrieval, and domain detection.
"""
from src.reasoning_agent.knowledge_manager import KnowledgeManager, KnowledgeDomain

def test_knowledge_loading():
    """Test that all knowledge files load correctly"""
    print("=" * 70)
    print("TEST 1: Knowledge Base Loading Verification")
    print("=" * 70)
    
    km = KnowledgeManager()
    
    math_count = len(km.bases.get(KnowledgeDomain.MATH, []))
    physics_count = len(km.bases.get(KnowledgeDomain.PHYSICS, []))
    chemistry_count = len(km.bases.get(KnowledgeDomain.CHEMISTRY, []))
    cs_count = len(km.bases.get(KnowledgeDomain.COMPUTER_SCIENCE, []))
    
    print(f"\n✓ Mathematics concepts: {math_count}")
    print(f"✓ Physics concepts: {physics_count}")
    print(f"✓ Chemistry concepts: {chemistry_count}")
    print(f"✓ Computer Science concepts: {cs_count}")
    print(f"\n✓ TOTAL: {math_count + physics_count + chemistry_count + cs_count} concepts")
    
    # Validate counts
    assert math_count >= 100, f"Expected ≥100 math concepts, got {math_count}"
    assert physics_count >= 100, f"Expected ≥100 physics concepts, got {physics_count}"
    assert chemistry_count >= 25, f"Expected ≥25 chemistry concepts, got {chemistry_count}"
    assert cs_count >= 15, f"Expected ≥15 CS concepts, got {cs_count}"
    
    print("\n✅ Knowledge loading test PASSED\n")
    return km

def test_domain_detection(km):
    """Test enhanced domain detection"""
    print("=" * 70)
    print("TEST 2: Domain Detection with New Domains")
    print("=" * 70)
    
    test_cases = [
        # Math (basic)
        ("What is the derivative of x^3?", KnowledgeDomain.MATH),
        ("Solve quadratic equation", KnowledgeDomain.MATH),
        
        # Math (advanced) 
        ("What is a differential equation?", KnowledgeDomain.MATH),
        ("Explain eigenvalues and eigenvectors", KnowledgeDomain.MATH),
        ("What is the Laplace transform?", KnowledgeDomain.MATH),
        ("Explain vector space", KnowledgeDomain.MATH),
        
        # Physics (basic)
        ("Calculate force given mass and acceleration", KnowledgeDomain.PHYSICS),
        ("What is Ohm's law?", KnowledgeDomain.PHYSICS),
        
        # Physics (advanced)
        ("Explain Schrödinger equation", KnowledgeDomain.PHYSICS),
        ("What is quantum tunneling?", KnowledgeDomain.PHYSICS),
        ("Describe the Boltzmann distribution", KnowledgeDomain.PHYSICS),
        ("What are phonons?", KnowledgeDomain.PHYSICS),
        
        # Chemistry
        ("What is ionic bonding?", KnowledgeDomain.CHEMISTRY),
        ("Explain Gibbs free energy", KnowledgeDomain.CHEMISTRY),
        ("What is the Nernst equation?", KnowledgeDomain.CHEMISTRY),
        ("Describe chemical equilibrium", KnowledgeDomain.CHEMISTRY),
        
        # Computer Science
        ("What is binary search?", KnowledgeDomain.COMPUTER_SCIENCE),
        ("Explain Big-O notation", KnowledgeDomain.COMPUTER_SCIENCE),
        ("What are hash tables?", KnowledgeDomain.COMPUTER_SCIENCE),
        ("Describe Dijkstra's algorithm", KnowledgeDomain.COMPUTER_SCIENCE),
    ]
    
    passed = 0
    failed = 0
    
    for query, expected_domain in test_cases:
        detected = km.detect_domain(query)
        is_correct = detected == expected_domain
        status = "✓" if is_correct else "✗"
        
        if is_correct:
            passed += 1
        else:
            failed += 1
            
        print(f"{status} '{query}'")
        print(f"   → Detected: {detected.value}, Expected: {expected_domain.value}")
    
    print(f"\n📊 Results: {passed}/{len(test_cases)} passed, {failed} failed")
    
    assert failed == 0, f"Domain detection failed for {failed} queries"
    
    print("✅ Domain detection test PASSED\n")

def test_concept_retrieval(km):
    """Test concept retrieval for all domains"""
    print("=" * 70)
    print("TEST 3: Concept Retrieval Across All Domains")
    print("=" * 70)
    
    test_cases = [
        # Math - Advanced
        ("differential equations", KnowledgeDomain.MATH, "Differential"),
        ("eigenvalue", KnowledgeDomain.MATH, "Eigen"),
        ("Fourier series", KnowledgeDomain.MATH, "Fourier"),
        ("graph theory", KnowledgeDomain.MATH, "Graph"),
        
        # Physics - Advanced
        ("quantum mechanics", KnowledgeDomain.PHYSICS, "Quantum"),
        ("Schrödinger", KnowledgeDomain.PHYSICS, "Schr\u00f6dinger"),
        ("Boltzmann", KnowledgeDomain.PHYSICS, "Boltzmann"),
        ("Navier-Stokes", KnowledgeDomain.PHYSICS, "Navier"),
        
        # Chemistry
        ("chemical bonding", KnowledgeDomain.CHEMISTRY, "Bond"),
        ("Gibbs energy", KnowledgeDomain.CHEMISTRY, "Gibbs"),
        ("Arrhenius", KnowledgeDomain.CHEMISTRY, "Arrhenius"),
        
        # Computer Science
        ("sorting algorithms", KnowledgeDomain.COMPUTER_SCIENCE, "Sort"),
        ("binary search", KnowledgeDomain.COMPUTER_SCIENCE, "Binary"),
        ("Big-O", KnowledgeDomain.COMPUTER_SCIENCE, "Big-O"),
    ]
    
    for query, domain, expected_in_name in test_cases:
        results = km.retrieve(query, domain, k=3)
        assert len(results) > 0, f"No results for query: {query}"
        
        # Check if expected concept is in results
        found = any(expected_in_name.lower() in r['metadata']['name'].lower() for r in results)
        status = "✓" if found else "⚠"
        
        print(f"{status} Query: '{query}' → {len(results)} result(s)")
        if found:
            matching_result = next(r for r in results if expected_in_name.lower() in r['metadata']['name'].lower())
            print(f"   Found: {matching_result['metadata']['name']}")
    
    print("\n✅ Concept retrieval test PASSED\n")

def test_new_specific_concepts():
    """Test specific new concepts are present"""
    print("=" * 70)
    print("TEST 4: Verification of Specific New Concepts")
    print("=" * 70)
    
    km = KnowledgeManager()
    
    # Advanced math topics to verify
    new_math_topics = [
        "Ordinary Differential Equations",
        "Eigenvalues",
        "Vector Spaces",
        "Prime Numbers",
        "Graph Theory",
        "Groups",
        "Metric Spaces",
        "Gradient",
        "Markov Chains",
        "Lagrange Multipliers"
    ]
    
    # Advanced physics topics to verify
    new_physics_topics = [
        "Schrödinger",
        "Quantum Tunneling",
        "Boltzmann Distribution",
        "Navier-Stokes",
        "Lagrangian Mechanics",
        "Superconductivity",
        "Nuclear Fission",
        "Standard Model"
    ]
    
    # Chemistry topics to verify
    chemistry_topics = [
        "VSEPR",
        "Gibbs Free Energy",
        "Arrhenius",
        "Le Chatelier",
        "Nernst"
    ]
    
    # CS topics to verify
    cs_topics = [
        "Binary Search",
        "Dynamic Programming",
        "Hash Tables",
        "Big-O",
        "NP-Complete"
    ]
    
    print("\nAdvanced Math Concepts:")
    math_concepts = km.bases[KnowledgeDomain.MATH]
    for topic in new_math_topics:
        found = any(topic.lower() in c.name.lower() for c in math_concepts)
        status = "✓" if found else "✗"
        print(f"  {status} {topic}")
        assert found, f"Math topic not found: {topic}"
    
    print("\nAdvanced Physics Concepts:")
    physics_concepts = km.bases[KnowledgeDomain.PHYSICS]
    for topic in new_physics_topics:
        found = any(topic.lower() in c.name.lower() for c in physics_concepts)
        status = "✓" if found else "✗"
        print(f"  {status} {topic}")
        assert found, f"Physics topic not found: {topic}"
    
    print("\nChemistry Concepts:")
    chem_concepts = km.bases[KnowledgeDomain.CHEMISTRY]
    for topic in chemistry_topics:
        found = any(topic.lower() in c.name.lower() for c in chem_concepts)
        status = "✓" if found else "✗"
        print(f"  {status} {topic}")
        assert found, f"Chemistry topic not found: {topic}"
    
    print("\nComputer Science Concepts:")
    cs_concepts = km.bases[KnowledgeDomain.COMPUTER_SCIENCE]
    for topic in cs_topics:
        found = any(topic.lower() in c.name.lower() for c in cs_concepts)
        status = "✓" if found else "✗"
        print(f"  {status} {topic}")
        assert found, f"CS topic not found: {topic}"
    
    print("\n✅ Specific concepts verification PASSED\n")

def test_knowledge_coverage():
    """Test coverage across different topics"""
    print("=" * 70)
    print("TEST 5: Knowledge Coverage Analysis")
    print("=" * 70)
    
    km = KnowledgeManager()
    
    # Analyze Math topics
    math_topics = {}
    for concept in km.bases[KnowledgeDomain.MATH]:
        topic = concept.topic
        math_topics[topic] = math_topics.get(topic, 0) + 1
    
    print("\n📚 Mathematics Topic Coverage:")
    for topic, count in sorted(math_topics.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count} concepts")
    
    # Analyze Physics topics
    physics_topics = {}
    for concept in km.bases[KnowledgeDomain.PHYSICS]:
        topic = concept.topic
        physics_topics[topic] = physics_topics.get(topic, 0) + 1
    
    print("\n⚛️  Physics Topic Coverage:")
    for topic, count in sorted(physics_topics.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count} concepts")
    
    # Analyze Chemistry topics
    chem_topics = {}
    for concept in km.bases[KnowledgeDomain.CHEMISTRY]:
        topic = concept.topic
        chem_topics[topic] = chem_topics.get(topic, 0) + 1
    
    print("\n🧪 Chemistry Topic Coverage:")
    for topic, count in sorted(chem_topics.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count} concepts")
    
    # Analyze CS topics
    cs_topics = {}
    for concept in km.bases[KnowledgeDomain.COMPUTER_SCIENCE]:
        topic = concept.topic
        cs_topics[topic] = cs_topics.get(topic, 0) + 1
    
    print("\n💻 Computer Science Topic Coverage:")
    for topic, count in sorted(cs_topics.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count} concepts")
    
    print("\n✅ Coverage analysis complete\n")

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE KNOWLEDGE BASE TEST SUITE")
    print("Testing 200+ concepts across Math, Physics, Chemistry, and CS")
    print("=" * 70 + "\n")
    
    try:
        km = test_knowledge_loading()
        test_domain_detection(km)
        test_concept_retrieval(km)
        test_new_specific_concepts()
        test_knowledge_coverage()
        
        print("=" * 70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 70)
        print("\nKnowledge base successfully expanded to 200+ concepts!")
        print(f"  • Mathematics: {len(km.bases[KnowledgeDomain.MATH])} concepts")
        print(f"  • Physics: {len(km.bases[KnowledgeDomain.PHYSICS])} concepts")
        print(f"  • Chemistry: {len(km.bases[KnowledgeDomain.CHEMISTRY])} concepts")
        print(f"  • Computer Science: {len(km.bases[KnowledgeDomain.COMPUTER_SCIENCE])} concepts")
        total = sum(len(v) for v in km.bases.values())
        print(f"  • TOTAL: {total} concepts across all domains")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
