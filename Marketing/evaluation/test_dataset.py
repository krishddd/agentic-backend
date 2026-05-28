"""
Test Dataset Framework

Systematic test cases for regression testing and quality validation.
Industry standard: OpenAI Evals approach with golden datasets
"""
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum
import json
from pathlib import Path
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)


class TestCategory(Enum):
    """Test case categories."""
    HAPPY_PATH = "happy_path"  # Typical successful scenarios
    EDGE_CASE = "edge_case"  # Boundary conditions
    ERROR_HANDLING = "error_handling"  # Expected failures
    ADVERSARIAL = "adversarial"  # Stress tests
    REGRESSION = "regression"  # Previously failed cases


class TestCase(BaseModel):
    """Single test case."""
    test_id: str = Field(description="Unique test identifier")
    agent: str = Field(description="Agent to test (secretary/analyst)")
    category: TestCategory
    description: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    quality_threshold: float = Field(ge=0, le=1, default=0.8)
    tags: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""


class TestResult(BaseModel):
    """Test execution result."""
    test_id: str
    passed: bool
    score: float
    duration_ms: float
    actual_output: Dict[str, Any]
    expected_output: Dict[str, Any]
    diff: Dict[str, Any]
    error: Optional[str] = None
    executed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class TestDataset:
    """
    Test dataset for systematic agent evaluation.
    
    Industry standard: OpenAI Evals golden dataset approach
    Enables regression testing and quality benchmarking.
    """
    
    def __init__(self, dataset_path: str = "data/test_datasets"):
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        self.test_cases: Dict[str, TestCase] = {}
        self.test_results: List[TestResult] = []
        
        # Load existing test cases
        self._load_test_cases()
    
    def add_test_case(self, test_case: TestCase):
        """Add new test case to dataset."""
        self.test_cases[test_case.test_id] = test_case
        self._save_test_cases()
        logger.info(f"Added test case: {test_case.test_id} ({test_case.category.value})")
    
    def get_test_cases(
        self,
        agent: Optional[str] = None,
        category: Optional[TestCategory] = None,
        tags: Optional[List[str]] = None
    ) -> List[TestCase]:
        """
        Get filtered test cases.
        
        Args:
            agent: Filter by agent (secretary/analyst)
            category: Filter by category
            tags: Filter by tags (OR matching)
        
        Returns:
            List of matching test cases
        """
        cases = list(self.test_cases.values())
        
        if agent:
            cases = [c for c in cases if c.agent == agent]
        
        if category:
            cases = [c for c in cases if c.category == category]
        
        if tags:
            cases = [c for c in cases if any(tag in c.tags for tag in tags)]
        
        return cases
    
    def run_test(
        self,
        test_case: TestCase,
        executor: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> TestResult:
        """
        Run single test case.
        
        Args:
            test_case: Test to run
            executor: Function that executes agent with input_data
        
        Returns:
            Test result
        """
        import time
        
        start_time = time.time()
        error = None
        actual_output = {}
        passed = False
        score = 0.0
        
        try:
            # Execute agent
            actual_output = executor(test_case.input_data)
            
            # Compare outputs
            diff = self._compare_outputs(test_case.expected_output, actual_output)
            
            # Calculate score
            score = self._calculate_similarity(test_case.expected_output, actual_output)
            
            # Check if passed
            passed = score >= test_case.quality_threshold
            
        except Exception as e:
            error = str(e)
            logger.error(f"Test {test_case.test_id} failed with error: {e}")
            diff = {"error": error}
        
        duration_ms = (time.time() - start_time) * 1000
        
        result = TestResult(
            test_id=test_case.test_id,
            passed=passed,
            score=score,
            duration_ms=duration_ms,
            actual_output=actual_output,
            expected_output=test_case.expected_output,
            diff=diff,
            error=error
        )
        
        self.test_results.append(result)
        return result
    
    def run_test_suite(
        self,
        agent: str,
        executor: Callable[[Dict[str, Any]], Dict[str, Any]],
        category: Optional[TestCategory] = None
    ) -> Dict[str, Any]:
        """
        Run complete test suite for an agent.
        
        Args:
            agent: Agent to test
            executor: Execution function
            category: Optional category filter
        
        Returns:
            Suite results summary
        """
        test_cases = self.get_test_cases(agent=agent, category=category)
        
        if not test_cases:
            logger.warning(f"No test cases found for agent={agent}, category={category}")
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "results": []
            }
        
        results = []
        for test_case in test_cases:
            result = self.run_test(test_case, executor)
            results.append(result)
            
            status = "✓ PASS" if result.passed else "✗ FAIL"
            logger.info(
                f"{status} {test_case.test_id}: {result.score:.2f} "
                f"({result.duration_ms:.0f}ms)"
            )
        
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        
        summary = {
            "agent": agent,
            "category": category.value if category else "all",
            "total": len(results),
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": passed_count / len(results) if results else 0.0,
            "avg_score": sum(r.score for r in results) / len(results) if results else 0.0,
            "avg_duration_ms": sum(r.duration_ms for r in results) / len(results) if results else 0.0,
            "results": [r.dict() for r in results]
        }
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    def _compare_outputs(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare expected vs actual outputs."""
        diff = {
            "missing": [],
            "extra": [],
            "different": []
        }
        
        # Check for missing keys
        for key in expected:
            if key not in actual:
                diff["missing"].append(key)
            elif expected[key] != actual.get(key):
                diff["different"].append({
                    "key": key,
                    "expected": expected[key],
                    "actual": actual.get(key)
                })
        
        # Check for extra keys
        for key in actual:
            if key not in expected:
                diff["extra"].append(key)
        
        return diff
    
    def _calculate_similarity(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any]
    ) -> float:
        """Calculate similarity score (0-1)."""
        if not expected:
            return 1.0 if not actual else 0.0
        
        matching = 0
        total = len(expected)
        
        for key, value in expected.items():
            if key in actual:
                if isinstance(value, bool):
                    # Boolean exact match
                    if actual[key] == value:
                        matching += 1
                elif isinstance(value, (int, float)):
                    # Numeric close match (within 10%)
                    if abs(actual[key] - value) / max(abs(value), 1) < 0.1:
                        matching += 1
                elif isinstance(value, str):
                    # String contains or exact match
                    if value.lower() in str(actual[key]).lower():
                        matching += 1
                else:
                    # Other types - exact match
                    if actual[key] == value:
                        matching += 1
        
        return matching / total if total > 0 else 0.0
    
    def _load_test_cases(self):
        """Load test cases from disk."""
        test_file = self.dataset_path / "test_cases.json"
        
        if test_file.exists():
            try:
                with open(test_file, 'r') as f:
                    data = json.load(f)
                    for case_data in data:
                        case = TestCase(**case_data)
                        self.test_cases[case.test_id] = case
                logger.info(f"Loaded {len(self.test_cases)} test cases")
            except Exception as e:
                logger.error(f"Failed to load test cases: {e}")
    
    def _save_test_cases(self):
        """Save test cases to disk."""
        test_file = self.dataset_path / "test_cases.json"
        
        try:
            with open(test_file, 'w') as f:
                data = [case.dict() for case in self.test_cases.values()]
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Saved {len(self.test_cases)} test cases")
        except Exception as e:
            logger.error(f"Failed to save test cases: {e}")
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save test results."""
        results_dir = self.dataset_path / "results"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = results_dir / f"test_run_{timestamp}.json"
        
        try:
            with open(result_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Saved test results to {result_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


# Pre-defined golden test cases
GOLDEN_TEST_CASES = [
    # Secretary - Happy Path
    TestCase(
        test_id="sec_happy_001",
        agent="secretary",
        category=TestCategory.HAPPY_PATH,
        description="Standard email processing with all required fields",
        input_data={
            "Client_Name": "John Doe",
            "Company_Name": "Tech Corp",
            "Raw_Discussion_Notes": "Client needs pricing information for enterprise plan",
            "Client_Requirements": "Pricing details and demo scheduling",
            "Scheduling_Next_Steps": "Follow-up call next Tuesday"
        },
        expected_output={
            "email_sent": True,
            "sheets_updated": True,
            "contains_pricing": True,
            "calendar_invite_created": True
        },
        quality_threshold=0.85,
        tags=["pricing", "demo", "routine"]
    ),
    
    # Secretary - Edge Case
    TestCase(
        test_id="sec_edge_001",
        agent="secretary",
        category=TestCategory.EDGE_CASE,
        description="Missing scheduling info - should handle gracefully",
        input_data={
            "Client_Name": "Jane Smith",
            "Company_Name": "StartupX",
            "Raw_Discussion_Notes": "Interested in product",
            "Client_Requirements": "General inquiry",
            "Scheduling_Next_Steps": ""  # Empty
        },
        expected_output={
            "email_sent": True,
            "sheets_updated": True,
            "calendar_invite_created": False  # Should skip
        },
        quality_threshold=0.80,
        tags=["edge_case", "missing_data"]
    ),
    
    # Analyst - Happy Path
    TestCase(
        test_id="ana_happy_001",
        agent="analyst",
        category=TestCategory.HAPPY_PATH,
        description="High-value deal with competitor mention",
        input_data={
            "Company_Name": "BigCorp Inc",
            "Raw_Discussion_Notes": "Ready to sign $500K deal. Mentioned they're also talking to Acme Corp",
            "Client_Requirements": "Enterprise solution with SLA"
        },
        expected_output={
            "deal_score_high": True,  # Score >= 8
            "competitor_detected": True,
            "alert_sent": True,
            "sheets_updated": True
        },
        quality_threshold=0.90,
        tags=["high_value", "competitor", "urgent"]
    ),
    
    # Analyst - Edge Case
    TestCase(
        test_id="ana_edge_001",
        agent="analyst",
        category=TestCategory.EDGE_CASE,
        description="Low-value inquiry",
        input_data={
            "Company_Name": "Small Business",
            "Raw_Discussion_Notes": "Just browsing, might be interested later",
            "Client_Requirements": "General information"
        },
        expected_output={
            "deal_score_high": False,  # Score < 8
            "competitor_detected": False,
            "alert_sent": False,  # No alert for low value
            "sheets_updated": True
        },
        quality_threshold=0.75,
        tags=["low_value", "browsing"]
    )
]


def initialize_golden_dataset():
    """Initialize dataset with golden test cases."""
    dataset = TestDataset()
    
    for test_case in GOLDEN_TEST_CASES:
        dataset.add_test_case(test_case)
    
    logger.info(f"Initialized golden dataset with {len(GOLDEN_TEST_CASES)} cases")
    return dataset
