"""Test dataset framework for systematic agent evaluation (OpenAI Evals style)."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)


class TestCategory(str, Enum):
    """Categories of test cases."""
    TYPICAL = "typical"  # Normal use cases
    EDGE_CASE = "edge_case"  # Boundary conditions
    ADVERSARIAL = "adversarial"  # Stress tests, attacks
    REGRESSION = "regression"  # Previously failed cases


@dataclass
class TestCase:
    """Individual test case for agent evaluation."""
    test_id: str
    category: TestCategory
    input: Dict[str, Any]
    expected_output: Dict[str, Any]
    description: str
    tags: List[str]
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TestSuite:
    """Collection of test cases for a specific task type."""
    suite_id: str
    name: str
    description: str
    task_type: str  # e.g., "stock_analysis"
    test_cases: List[TestCase]
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "metadata": self.metadata or {}
        }


class TestDatasetManager:
    """Manages test datasets for agent evaluation."""
    
    def __init__(self, dataset_dir: Path = None):
        """Initialize test dataset manager.
        
        Args:
            dataset_dir: Directory containing test datasets
        """
        self.dataset_dir = dataset_dir or Path("data/eval_datasets")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.test_suites: Dict[str, TestSuite] = {}
    
    def create_default_test_suite(self) -> TestSuite:
        """Create default test suite for stock analysis."""
        test_cases = []
        
        # ============ TYPICAL CASES ============
        test_cases.extend([
            TestCase(
                test_id="typical_001",
                category=TestCategory.TYPICAL,
                input={"stock_symbol": "AAPL"},
                expected_output={
                    "sections_present": ["Executive Summary", "Financial Analysis", "Risk Assessment", "Investment Recommendation"],
                    "contains_metrics": True,
                    "recommendation_type": ["BUY", "HOLD", "SELL"],
                    "minimum_word_count": 500
                },
                description="Standard analysis for large-cap tech stock (AAPL)",
                tags=["tech", "large-cap", "typical"]
            ),
            TestCase(
                test_id="typical_002",
                category=TestCategory.TYPICAL,
                input={"stock_symbol": "TSLA"},
                expected_output={
                    "sections_present": ["Executive Summary", "Financial Analysis", "Risk Assessment"],
                    "contains_metrics": True,
                    "mentions_ev_market": True,
                    "minimum_word_count": 500
                },
                description="Analysis for high-volatility growth stock (TSLA)",
                tags=["tech", "growth", "high-volatility"]
            ),
            TestCase(
                test_id="typical_003",
                category=TestCategory.TYPICAL,
                input={"stock_symbol": "JNJ"},
                expected_output={
                    "sections_present": ["Executive Summary", "Financial Analysis", "Risk Assessment"],
                    "mentions_healthcare": True,
                    "mentions_dividend": True
                },
                description="Analysis for defensive healthcare stock (JNJ)",
                tags=["healthcare", "defensive", "dividend"]
            ),
        ])
        
        # ============ EDGE CASES ============
        test_cases.extend([
            TestCase(
                test_id="edge_001",
                category=TestCategory.EDGE_CASE,
                input={"stock_symbol": "GME"},
                expected_output={
                    "mentions_volatility": True,
                    "risk_warning_present": True,
                    "acknowledges_social_media": True
                },
                description="Meme stock with extreme volatility",
                tags=["meme-stock", "extreme-volatility", "edge-case"]
            ),
            TestCase(
                test_id="edge_002",
                category=TestCategory.EDGE_CASE,
                input={"stock_symbol": "BRK.A"},
                expected_output={
                    "handles_class_a_notation": True,
                    "mentions_berkshire": True
                },
                description="Stock with unusual ticker format (Class A shares)",
                tags=["unusual-ticker", "edge-case"]
            ),
            TestCase(
                test_id="edge_003",
                category=TestCategory.EDGE_CASE,
                input={"stock_symbol": "ARKK"},
                expected_output={
                    "identifies_as_etf": True,
                    "describes_holdings": True
                },
                description="ETF instead of individual stock",
                tags=["etf", "edge-case"]
            ),
        ])
        
        # ============ ADVERSARIAL CASES ============
        test_cases.extend([
            TestCase(
                test_id="adv_001",
                category=TestCategory.ADVERSARIAL,
                input={"stock_symbol": "INVALID123"},
                expected_output={
                    "error_handling": "graceful",
                    "suggests_valid_ticker": True,
                    "no_hallucination": True
                },
                description="Invalid ticker symbol",
                tags=["invalid-input", "error-handling"]
            ),
            TestCase(
                test_id="adv_002",
                category=TestCategory.ADVERSARIAL,
                input={"stock_symbol": "aapl"},  # lowercase
                expected_output={
                    "normalizes_ticker": True,
                    "produces_valid_analysis": True
                },
                description="Lowercase ticker (should normalize to uppercase)",
                tags=["input-validation", "normalization"]
            ),
            TestCase(
                test_id="adv_003",
                category=TestCategory.ADVERSARIAL,
                input={"stock_symbol": "SPCE"},
                expected_output={
                    "mentions_bankruptcy_risk": True,
                    "acknowledges_uncertainty": True,
                    "conservative_recommendation": True
                },
                description="Recently bankrupt/troubled company",
                tags=["bankruptcy", "high-risk", "adversarial"]
            ),
        ])
        
        # ============ REGRESSION CASES ============
        test_cases.extend([
            TestCase(
                test_id="reg_001",
                category=TestCategory.REGRESSION,
                input={"stock_symbol": "NVDA"},
                expected_output={
                    "mentions_ai_chips": True,
                    "mentions_datacenter": True,
                    "financial_metrics_present": True
                },
                description="Previously missed AI chip narrative (regression test)",
                tags=["regression", "ai-sector"]
            ),
        ])
        
        suite = TestSuite(
            suite_id="financial_crew_v1",
            name="Financial Crew Standard Test Suite",
            description="Comprehensive test suite for stock analysis including typical, edge, and adversarial cases",
            task_type="stock_analysis",
            test_cases=test_cases,
            metadata={
                "version": "1.0.0",
                "created_date": "2026-01-02",
                "total_cases": len(test_cases)
            }
        )
        
        return suite
    
    def save_test_suite(self, suite: TestSuite):
        """Save test suite to JSON file.
        
        Args:
            suite: TestSuite to save
        """
        filepath = self.dataset_dir / f"{suite.suite_id}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(suite.to_dict(), f, indent=2)
        
        logger.info(f"Saved test suite '{suite.name}' to {filepath} ({len(suite.test_cases)} test cases)")
        self.test_suites[suite.suite_id] = suite
        
        return filepath
    
    def load_test_suite(self, suite_id: str) -> Optional[TestSuite]:
        """Load test suite from JSON file.
        
        Args:
            suite_id: Suite ID to load
            
        Returns:
            TestSuite or None if not found
        """
        filepath = self.dataset_dir / f"{suite_id}.json"
        
        if not filepath.exists():
            logger.warning(f"Test suite file not found: {filepath}")
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct TestSuite
        test_cases = [
            TestCase(
                test_id=tc['test_id'],
                category=TestCategory(tc['category']),
                input=tc['input'],
                expected_output=tc['expected_output'],
                description=tc['description'],
                tags=tc['tags'],
                metadata=tc.get('metadata')
            )
            for tc in data['test_cases']
        ]
        
        suite = TestSuite(
            suite_id=data['suite_id'],
            name=data['name'],
            description=data['description'],
            task_type=data['task_type'],
            test_cases=test_cases,
            metadata=data.get('metadata')
        )
        
        self.test_suites[suite_id] = suite
        logger.info(f"Loaded test suite '{suite.name}' ({len(suite.test_cases)} test cases)")
        
        return suite
    
    def get_test_cases_by_category(self, suite_id: str, category: TestCategory) -> List[TestCase]:
        """Get all test cases in a category.
        
        Args:
            suite_id: Suite ID
            category: Test category
            
        Returns:
            List of test cases
        """
        suite = self.test_suites.get(suite_id)
        if not suite:
            suite = self.load_test_suite(suite_id)
        
        if not suite:
            return []
        
        return [tc for tc in suite.test_cases if tc.category == category]
    
    def get_test_cases_by_tags(self, suite_id: str, tags: List[str]) -> List[TestCase]:
        """Get test cases matching any of the provided tags.
        
        Args:
            suite_id: Suite ID
            tags: List of tags to match
            
        Returns:
            List of matching test cases
        """
        suite = self.test_suites.get(suite_id)
        if not suite:
            suite = self.load_test_suite(suite_id)
        
        if not suite:
            return []
        
        return [tc for tc in suite.test_cases if any(tag in tc.tags for tag in tags)]
    
    def add_test_case(self, suite_id: str, test_case: TestCase):
        """Add a test case to an existing suite.
        
        Args:
            suite_id: Suite ID
            test_case: TestCase to add
        """
        suite = self.test_suites.get(suite_id)
        if not suite:
            suite = self.load_test_suite(suite_id)
        
        if not suite:
            raise ValueError(f"Test suite '{suite_id}' not found")
        
        suite.test_cases.append(test_case)
        self.save_test_suite(suite)
        
        logger.info(f"Added test case '{test_case.test_id}' to suite '{suite_id}'")
    
    def run_test_suite(self, suite_id: str, run_function, save_results: bool = True) -> Dict[str, Any]:
        """Run all test cases in a suite.
        
        Args:
            suite_id: Suite ID to run
            run_function: Function that takes input and returns output
            save_results: Whether to save results to file
            
        Returns:
            Test run results
        """
        suite = self.test_suites.get(suite_id)
        if not suite:
            suite = self.load_test_suite(suite_id)
        
        if not suite:
            raise ValueError(f"Test suite '{suite_id}' not found")
        
        logger.info(f"Running test suite '{suite.name}' ({len(suite.test_cases)} cases)")
        
        results = {
            "suite_id": suite_id,
            "suite_name": suite.name,
            "total_cases": len(suite.test_cases),
            "passed": 0,
            "failed": 0,
            "test_results": []
        }
        
        for test_case in suite.test_cases:
            logger.info(f"Running test case: {test_case.test_id}")
            
            try:
                # Run the test
                output = run_function(test_case.input)
                
                # Validate output against expected
                passed = self._validate_output(output, test_case.expected_output)
                
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                
                results["test_results"].append({
                    "test_id": test_case.test_id,
                    "category": test_case.category.value,
                    "description": test_case.description,
                    "passed": passed,
                    "output": output,
                    "expected": test_case.expected_output
                })
                
            except Exception as e:
                logger.error(f"Test case {test_case.test_id} failed with exception: {e}")
                results["failed"] += 1
                results["test_results"].append({
                    "test_id": test_case.test_id,
                    "category": test_case.category.value,
                    "passed": False,
                    "error": str(e)
                })
        
        # Calculate pass rate
        results["pass_rate"] = results["passed"] / results["total_cases"] if results["total_cases"] > 0 else 0
        
        logger.info(f"Test suite completed: {results['passed']}/{results['total_cases']} passed ({results['pass_rate']*100:.1f}%)")
        
        if save_results:
            self._save_test_results(results)
        
        return results
    
    def _validate_output(self, actual: Any, expected: Dict[str, Any]) -> bool:
        """Validate actual output against expected criteria.
        
        Args:
            actual: Actual output from agent
            expected: Expected output criteria
            
        Returns:
            True if validation passes
        """        # Simple validation logic
        # In production, this would be more sophisticated
        
        # For now, just check if the output is not None/empty
        if not actual:
            return False
        
        # You would implement specific validators based on expected criteria
        # For example:
        # - Check if certain sections are present in text
        # - Verify minimum word count
        # - Check for specific keywords/mentions
        # - Validate structure/format
        
        return True  # Simplified for now
    
    def _save_test_results(self, results: Dict[str, Any]):
        """Save test results to file.
        
        Args:
            results: Test run results
        """
        results_dir = self.dataset_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = results_dir / f"test_run_{results['suite_id']}_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved test results to {filepath}")
