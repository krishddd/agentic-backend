"""
LLM Judge Validation

Validates reliability and consistency of LLM-as-a-Judge evaluations.
Industry standard: OpenAI Evals approach - validate judge before trusting scores
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import statistics
from services.llm_judge_service import LLMJudgeService
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class JudgeValidationResult:
    """Results of judge validation."""
    metric_name: str
    consistency_score: float  # 0-1, higher = more consistent
    variance: float  # Lower = better
    human_agreement: Optional[float] = None  # If human labels available
    bias_detected: bool = False
    reliable: bool = True
    notes: List[str] = None
    
    def __post_init__(self):
        if self.notes is None:
            self.notes = []


class LLMJudgeValidator:
    """
    Validates LLM-as-a-Judge reliability.
    
    Industry standard: OpenAI Evals approach
    Ensures judge scores are consistent and trustworthy before using for evaluation.
    """
    
    def __init__(self, judge_service: LLMJudgeService):
        self.judge = judge_service
        self.validation_results: Dict[str, JudgeValidationResult] = {}
    
    def validate_consistency(
        self,
        metric_name: str,
        test_cases: List[Dict[str, str]],
        num_runs: int = 5
    ) -> JudgeValidationResult:
        """
        Validate judge consistency by running same eval multiple times.
        
        Args:
            metric_name: Metric to test (helpfulness, correctness, etc.)
            test_cases: List of test cases {'query': ..., 'response': ...}
            num_runs: How many times to run each test
        
        Returns:
            Validation result with consistency score
        """
        logger.info(f"Validating {metric_name} consistency ({num_runs} runs each)...")
        
        all_scores = []  # All scores for variance calculation
        case_variances = []  # Variance per test case
        
        for i, test_case in enumerate(test_cases):
            query = test_case['query']
            response = test_case['response']
            rubric = test_case.get('rubric')
            
            # Run same evaluation multiple times
            scores = []
            for run in range(num_runs):
                result = self._run_judge(metric_name, query, response, rubric)
                score = result.get('score', 0.0)
                scores.append(score)
                all_scores.append(score)
            
            # Calculate variance for this test case
            if len(scores) > 1:
                variance = statistics.variance(scores)
                case_variances.append(variance)
                
                logger.debug(
                    f"Case {i+1}: scores={scores}, "
                    f"mean={statistics.mean(scores):.2f}, var={variance:.4f}"
                )
        
        # Calculate overall consistency
        overall_variance = statistics.variance(all_scores) if len(all_scores) > 1 else 0.0
        avg_case_variance = statistics.mean(case_variances) if case_variances else 0.0
        
        # Consistency score: lower variance = higher consistency
        # Scale: variance of 0 = 1.0, variance of 1+ = 0.0
        consistency_score = max(0.0, 1.0 - avg_case_variance)
        
        # Determine if reliable
        reliable = consistency_score >= 0.7 and avg_case_variance < 0.3
        
        notes = []
        if avg_case_variance > 0.5:
            notes.append("High variance - judge scores are inconsistent")
        if consistency_score < 0.7:
            notes.append("Low consistency - may not be reliable for automation")
        if reliable:
            notes.append("Judge is sufficiently consistent for automated use")
        
        result = JudgeValidationResult(
            metric_name=metric_name,
            consistency_score=consistency_score,
            variance=avg_case_variance,
            reliable=reliable,
            notes=notes
        )
        
        self.validation_results[metric_name] = result
        
        logger.info(
            f"{metric_name} validation: consistency={consistency_score:.2f}, "
            f"variance={avg_case_variance:.4f}, reliable={reliable}"
        )
        
        return result
    
    def validate_against_human(
        self,
        metric_name: str,
        test_cases: List[Dict[str, Any]]
    ) -> JudgeValidationResult:
        """
        Validate judge against human labels.
        
        Args:
            metric_name: Metric to test
            test_cases: Cases with 'human_score' ground truth
        
        Returns:
            Validation with human agreement score
        """
        logger.info(f"Validating {metric_name} against {len(test_cases)} human labels...")
        
        agreements = []
        scores_llm = []
        scores_human = []
        
        for test_case in test_cases:
            query = test_case['query']
            response = test_case['response']
            human_score = test_case['human_score']
            rubric = test_case.get('rubric')
            
            # Get LLM judge score
            result = self._run_judge(metric_name, query, response, rubric)
            llm_score = result.get('score', 0.0)
            
            scores_llm.append(llm_score)
            scores_human.append(human_score)
            
            # Check agreement (within 0.5 points on 1-5 scale, or 0.1 on 0-1 scale)
            if human_score <= 1.0:  # 0-1 scale
                threshold = 0.1
            else:  # 1-5 scale
                threshold = 0.5
            
            agreed = abs(llm_score - human_score) <= threshold
            agreements.append(agreed)
        
        # Calculate agreement rate
        agreement_rate = sum(agreements) / len(agreements) if agreements else 0.0
        
        # Calculate correlation
        if len(scores_llm) > 1:
            correlation = self._calculate_correlation(scores_llm, scores_human)
        else:
            correlation = 0.0
        
        reliable = agreement_rate >= 0.7
        
        notes = []
        if agreement_rate < 0.7:
            notes.append(f"Low agreement with humans ({agreement_rate:.1%})")
        if correlation < 0.6:
            notes.append(f"Low correlation with human scores ({correlation:.2f})")
        if reliable:
            notes.append("Good agreement with human judgment")
        
        result = JudgeValidationResult(
            metric_name=metric_name,
            consistency_score=0.0,  # Not measured here
            variance=0.0,
            human_agreement=agreement_rate,
            reliable=reliable,
            notes=notes
        )
        
        self.validation_results[metric_name] = result
        
        logger.info(
            f"{metric_name} human validation: agreement={agreement_rate:.1%}, "
            f"correlation={correlation:.2f}, reliable={reliable}"
        )
        
        return result
    
    def detect_bias(
        self,
        metric_name: str,
        test_cases_by_group: Dict[str, List[Dict[str, str]]]
    ) -> JudgeValidationResult:
        """
        Detect bias in judge scores across different groups.
        
        Args:
            metric_name: Metric to test
            test_cases_by_group: Dict of group_name -> test cases
                Example: {"positive_sentiment": [...], "negative_sentiment": [...]}
        
        Returns:
            Validation with bias detection results
        """
        logger.info(f"Detecting bias in {metric_name}...")
        
        group_scores = {}
        
        for group_name, cases in test_cases_by_group.items():
            scores = []
            for case in cases:
                result = self._run_judge(
                    metric_name,
                    case['query'],
                    case['response'],
                    case.get('rubric')
                )
                scores.append(result.get('score', 0.0))
            
            group_scores[group_name] = {
                'mean': statistics.mean(scores) if scores else 0.0,
                'scores': scores
            }
        
        # Check for significant differences between groups
        means = [data['mean'] for data in group_scores.values()]
        mean_diff = max(means) - min(means) if means else 0.0
        
        # Bias detected if difference > 0.3 (on 0-1 scale)
        bias_detected = mean_diff > 0.3
        
        notes = []
        for group, data in group_scores.items():
            notes.append(f"{group}: mean={data['mean']:.2f}")
        
        if bias_detected:
            notes.append(f"⚠️ Bias detected: {mean_diff:.2f} difference between groups")
        else:
            notes.append("No significant bias detected")
        
        result = JudgeValidationResult(
            metric_name=metric_name,
            consistency_score=0.0,
            variance=0.0,
            bias_detected=bias_detected,
            reliable=not bias_detected,
            notes=notes
        )
        
        self.validation_results[metric_name] = result
        
        logger.info(f"{metric_name} bias detection: bias={bias_detected}, diff={mean_diff:.2f}")
        
        return result
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations."""
        return {
            "metrics_validated": len(self.validation_results),
            "reliable_metrics": sum(
                1 for r in self.validation_results.values() if r.reliable
            ),
            "results": {
                name: {
                    "consistency_score": result.consistency_score,
                    "variance": result.variance,
                    "human_agreement": result.human_agreement,
                    "bias_detected": result.bias_detected,
                    "reliable": result.reliable,
                    "notes": result.notes
                }
                for name, result in self.validation_results.items()
            }
        }
    
    def _run_judge(
        self,
        metric_name: str,
        query: str,
        response: str,
        rubric: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run judge evaluation."""
        if metric_name == "helpfulness":
            return self.judge.judge_helpfulness(query, response, rubric)
        elif metric_name == "correctness":
            return self.judge.judge_correctness(query, response, rubric)
        elif metric_name == "safety":
            return self.judge.judge_safety(query, response, rubric)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    def _calculate_correlation(
        self,
        scores1: List[float],
        scores2: List[float]
    ) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(scores1) != len(scores2) or len(scores1) < 2:
            return 0.0
        
        mean1 = statistics.mean(scores1)
        mean2 = statistics.mean(scores2)
        
        numerator = sum((s1 - mean1) * (s2 - mean2) for s1, s2 in zip(scores1, scores2))
        
        denom1 = sum((s1 - mean1) ** 2 for s1 in scores1)
        denom2 = sum((s2 - mean2) ** 2 for s2 in scores2)
        
        if denom1 == 0 or denom2 == 0:
            return 0.0
        
        return numerator / (denom1 * denom2) ** 0.5


# Validation test cases for common metrics
HELPFULNESS_TEST_CASES = [
    {
        'query': 'What is the capital of France?',
        'response': 'The capital of France is Paris.',
        'human_score': 5.0
    },
    {
        'query': 'What is the capital of France?',
        'response': 'France is a country in Europe.',
        'human_score': 2.0
    },
    {
        'query': 'Explain quantum computing',
        'response': 'Quantum computing uses quantum mechanics principles like superposition and entanglement to perform computations that are impossible for classical computers.',
        'human_score': 4.5
    }
]

CORRECTNESS_TEST_CASES = [
    {
        'query': '2 + 2 = ?',
        'response': '4',
        'human_score': 5.0
    },
    {
        'query': '2 + 2 = ?',
        'response': '5',
        'human_score': 1.0
    }
]
