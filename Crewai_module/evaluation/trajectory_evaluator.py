"""Trajectory evaluation service for analyzing agent decision-making paths."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from evaluation.models import TrajectoryQualityMetrics
from utils.logger import get_logger

logger = get_logger(__name__)


class StepType(str, Enum):
    """Types of steps in agent execution."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    TOOL_CALL = "tool_call"
    RAG_RETRIEVAL = "rag_retrieval"
    DECISION = "decision"
    OUTPUT_GENERATION = "output_generation"


@dataclass
class ExecutionStep:
    """Represents a single step in agent execution."""
    step_id: str
    step_type: StepType
    action: str
    tool_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    success: bool = True
    duration_ms: float = 0.0
    reasoning: Optional[str] = None


@dataclass
class OptimalPath:
    """Represents the optimal execution path for a task."""
    task_type: str
    expected_steps: List[StepType]
    min_steps: int
    max_steps: int
    required_tools: List[str]
    optional_tools: List[str]


class TrajectoryEvaluator:
    """Evaluates agent execution trajectories and compares to optimal paths."""
    
    # Define optimal paths for common tasks
    OPTIMAL_PATHS = {
        "stock_analysis": OptimalPath(
            task_type="stock_analysis",
            expected_steps=[
                StepType.RESEARCH,
                StepType.RAG_RETRIEVAL,
                StepType.ANALYSIS,
                StepType.DECISION,
                StepType.OUTPUT_GENERATION
            ],
            min_steps=4,
            max_steps=8,
            required_tools=["web_search", "rag_search"],
            optional_tools=["sec_filing_tool", "financial_data_tool"]
        )
    }
    
    def __init__(self):
        """Initialize trajectory evaluator."""
        self.evaluations: List[Dict[str, Any]] = []
    
    def evaluate_trajectory(self,
                           actual_steps: List[ExecutionStep],
                           task_type: str = "stock_analysis") -> Dict[str, Any]:
        """Evaluate an execution trajectory against optimal path.
        
        Args:
            actual_steps: List of actual execution steps
            task_type: Type of task being evaluated
            
        Returns:
            Trajectory evaluation results
        """
        optimal = self.OPTIMAL_PATHS.get(task_type)
        if not optimal:
            logger.warning(f"No optimal path defined for task type: {task_type}")
            return self._simple_evaluation(actual_steps)
        
        # Analyze trajectory
        results = {
            "trajectory_optimality": self._calculate_optimality(actual_steps, optimal),
            "step_efficiency": self._calculate_step_efficiency(actual_steps, optimal),
            "tool_selection_quality": self._evaluate_tool_selection(actual_steps, optimal),
            "path_deviation": self._calculate_path_deviation(actual_steps, optimal),
            "redundant_loops": self._detect_redundant_loops(actual_steps),
            "missing_critical_steps": self._find_missing_steps(actual_steps, optimal),
            "reasoning_quality": self._evaluate_reasoning_chain(actual_steps),
            "intermediate_accuracy": self._evaluate_intermediate_steps(actual_steps),
            "decision_confidence": self._calculate_decision_confidence(actual_steps)
        }
        
        # Generate detailed analysis
        results["analysis"] = self._generate_trajectory_analysis(actual_steps, optimal, results)
        results["recommendations"] = self._generate_recommendations(results)
        
        self.evaluations.append(results)
        return results
    
    def _calculate_optimality(self, actual: List[ExecutionStep], optimal: OptimalPath) -> float:
        """Calculate how optimal the trajectory is (0-1 scale).
        
        Formula: optimal_steps / max(actual_steps, optimal.min_steps)
        """
        actual_count = len(actual)
        if actual_count < optimal.min_steps:
            # Too few steps - missing critical operations
            return actual_count / optimal.min_steps
        elif actual_count > optimal.max_steps:
            # Too many steps - inefficient
            return optimal.max_steps / actual_count
        else:
            # Within acceptable range
            return 0.95 + (0.05 * (1 - (actual_count - optimal.min_steps) / (optimal.max_steps - optimal.min_steps)))
    
    def _calculate_step_efficiency(self, actual: List[ExecutionStep], optimal: OptimalPath) -> float:
        """Calculate step efficiency score."""
        # Count steps by type
        actual_types = [step.step_type for step in actual]
        expected_types_set = set(optimal.expected_steps)
        actual_types_set = set(actual_types)
        
        # How many expected steps were performed?
        matched = len(expected_types_set.intersection(actual_types_set))
        efficiency = matched / len(expected_types_set) if expected_types_set else 1.0
        
        return efficiency
    
    def _evaluate_tool_selection(self, actual: List[ExecutionStep], optimal: OptimalPath) -> float:
        """Evaluate quality of tool selection."""
        tool_calls = [step for step in actual if step.step_type == StepType.TOOL_CALL]
        if not tool_calls:
            return 0.5  # Neutral - no tools used
        
        tools_used = set(step.tool_name for step in tool_calls if step.tool_name)
        required_tools = set(optimal.required_tools)
        optional_tools = set(optimal.optional_tools)
        valid_tools = required_tools.union(optional_tools)
        
        # Score based on:
        # 1. Used all required tools
        # 2. Only used valid tools (no wrong tool selections)
        required_coverage = len(tools_used.intersection(required_tools)) / len(required_tools) if required_tools else 1.0
        invalid_tool_usage = len(tools_used - valid_tools) / len(tools_used) if tools_used else 0.0
        
        score = (required_coverage * 0.7) + ((1 - invalid_tool_usage) * 0.3)
        return score
    
    def _calculate_path_deviation(self, actual: List[ExecutionStep], optimal: OptimalPath) -> Dict[str, Any]:
        """Calculate deviation from optimal path."""
        actual_sequence = [step.step_type for step in actual]
        expected_sequence = optimal.expected_steps
        
        # Find Levenshtein distance (edit distance) between sequences
        deviations = self._sequence_diff(actual_sequence, expected_sequence)
        
        return {
            "deviation_count": len(deviations),
            "deviation_details": deviations,
            "sequence_similarity": 1 - (len(deviations) / max(len(actual_sequence), len(expected_sequence)))
        }
    
    def _detect_redundant_loops(self, actual: List[ExecutionStep]) -> Dict[str, Any]:
        """Detect redundant loops or repeated operations."""
        redundancies = []
        
        # Check for repeated identical actions
        for i in range(len(actual)):
            for j in range(i + 1, len(actual)):
                if (actual[i].action == actual[j].action and
                    actual[i].tool_name == actual[j].tool_name):
                    redundancies.append({
                        "step_indices": [i, j],
                        "action": actual[i].action,
                        "tool": actual[i].tool_name
                    })
        
        return {
            "redundant_loops_detected": len(redundancies) > 0,
            "redundancy_count": len(redundancies),
            "redundancy_details": redundancies
        }
    
    def _find_missing_steps(self, actual: List[ExecutionStep], optimal: OptimalPath) -> List[str]:
        """Find critical steps that were skipped."""
        actual_types = set(step.step_type for step in actual)
        expected_types = set(optimal.expected_steps)
        missing = expected_types - actual_types
        
        return [step_type.value for step_type in missing]
    
    def _evaluate_reasoning_chain(self, actual: List[ExecutionStep]) -> Dict[str, Any]:
        """Evaluate the quality of reasoning chain."""
        steps_with_reasoning = [step for step in actual if step.reasoning]
        
        if not steps_with_reasoning:
            return {
                "reasoning_provided": False,
                "reasoning_coherence": 0.0,
                "chain_of_thought_quality": 0.0
            }
        
        # Simple heuristics for reasoning quality
        avg_reasoning_length = sum(len(step.reasoning) for step in steps_with_reasoning) / len(steps_with_reasoning)
        reasoning_coverage = len(steps_with_reasoning) / len(actual)
        
        # High-quality reasoning should be:
        # 1. Present for most steps (coverage)
        # 2. Non-trivial (length > 50 chars)
        # 3. Consistent (all steps have it or none do)
        
        quality_score = min(1.0, reasoning_coverage * (1 if avg_reasoning_length > 50 else 0.7))
        
        return {
            "reasoning_provided": True,
            "reasoning_coverage": reasoning_coverage,
            "avg_reasoning_length": avg_reasoning_length,
            "chain_of_thought_quality": quality_score
        }
    
    def _evaluate_intermediate_steps(self, actual: List[ExecutionStep]) -> List[Dict[str, Any]]:
        """Evaluate each intermediate step."""
        intermediate_scores = []
        
        for i, step in enumerate(actual):
            step_eval = {
                "step_index": i,
                "step_type": step.step_type.value,
                "action": step.action,
                "success": step.success,
                "appropriate": self._is_step_appropriate(step, i, actual),
                "efficient": step.duration_ms < 5000,  # Simple heuristic
                "score": 1.0 if step.success else 0.0
            }
            
            # Adjust score based on appropriateness and efficiency
            if not step_eval["appropriate"]:
                step_eval["score"] *= 0.7
            if not step_eval["efficient"]:
                step_eval["score"] *= 0.9
            
            intermediate_scores.append(step_eval)
        
        return intermediate_scores
    
    def _calculate_decision_confidence(self, actual: List[ExecutionStep]) -> float:
        """Calculate overall decision confidence score."""
        # Based on:
        # 1. Success rate
        # 2. No redundant loops
        # 3. Efficient execution
        
        success_rate = sum(1 for step in actual if step.success) / len(actual) if actual else 0.0
        has_redundancy = any(
            actual[i].action == actual[j].action 
            for i in range(len(actual)) 
            for j in range(i + 1, len(actual))
        )
        avg_duration = sum(step.duration_ms for step in actual) / len(actual) if actual else 0.0
        
        confidence = success_rate * 0.6
        if not has_redundancy:
            confidence += 0.2
        if avg_duration < 3000:  # Fast execution
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _is_step_appropriate(self, step: ExecutionStep, index: int, all_steps: List[ExecutionStep]) -> bool:
        """Check if a step is appropriate at this point in execution."""
        # Simple heuristic: certain steps should come in order
        # e.g., research before analysis, analysis before output
        
        steps_before = all_steps[:index]
        steps_before_types = [s.step_type for s in steps_before]
        
        # Rules:
        # - Analysis should come after research
        # - Output should come after analysis
        # - Tool calls should have a purpose (after decision or during research)
        
        if step.step_type == StepType.ANALYSIS:
            return StepType.RESEARCH in steps_before_types or StepType.RAG_RETRIEVAL in steps_before_types
        elif step.step_type == StepType.OUTPUT_GENERATION:
            return StepType.ANALYSIS in steps_before_types
        
        return True  # Default to appropriate
    
    def _sequence_diff(self, actual: List[StepType], expected: List[StepType]) -> List[Dict[str, Any]]:
        """Calculate differences between two sequences."""
        deviations = []
        
        # Simple comparison: find missing and extra steps
        actual_set = set(actual)
        expected_set = set(expected)
        
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        
        for step in missing:
            deviations.append({"type": "missing", "step": step.value})
        for step in extra:
            deviations.append({"type": "extra", "step": step.value})
        
        return deviations
    
    def _simple_evaluation(self, actual: List[ExecutionStep]) -> Dict[str, Any]:
        """Simple evaluation when no optimal path is defined."""
        return {
            "trajectory_optimality": 0.8,  # Assume good by default
            "step_efficiency": len([s for s in actual if s.success]) / len(actual) if actual else 0.0,
            "tool_selection_quality": 0.8,
            "path_deviation": {"deviation_count": 0, "sequence_similarity": 1.0},
            "redundant_loops": {"redundant_loops_detected": False, "redundancy_count": 0},
            "missing_critical_steps": [],
            "reasoning_quality": self._evaluate_reasoning_chain(actual),
            "intermediate_accuracy": self._evaluate_intermediate_steps(actual),
            "decision_confidence": self._calculate_decision_confidence(actual)
        }
    
    def _generate_trajectory_analysis(self, 
                                     actual: List[ExecutionStep],
                                     optimal: OptimalPath,
                                     results: Dict[str, Any]) -> str:
        """Generate human-readable trajectory analysis."""
        analysis_parts = []
        
        # Overall assessment
        optimality = results["trajectory_optimality"]
        if optimality >= 0.9:
            analysis_parts.append("✅ Excellent trajectory - very close to optimal path.")
        elif optimality >= 0.7:
            analysis_parts.append("✓ Good trajectory with minor inefficiencies.")
        else:
            analysis_parts.append("⚠️ Suboptimal trajectory detected.")
        
        # Step efficiency
        efficiency = results["step_efficiency"]
        analysis_parts.append(f"Completed {int(efficiency*100)}% of expected critical steps.")
        
        # Missing steps
        if results["missing_critical_steps"]:
            analysis_parts.append(f"⚠️ Missing critical steps: {', '.join(results['missing_critical_steps'])}")
        
        # Redundancies
        if results["redundant_loops"]["redundant_loops_detected"]:
            count = results["redundant_loops"]["redundancy_count"]
            analysis_parts.append(f"⚠️ Detected {count} redundant operations.")
        
        # Tool usage
        tool_quality = results["tool_selection_quality"]
        if tool_quality < 0.7:
            analysis_parts.append("⚠️ Some tool selections may have been incorrect.")
        
        return " ".join(analysis_parts)
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on trajectory evaluation."""
        recommendations = []
        
        if results["trajectory_optimality"] < 0.8:
            recommendations.append("Review agent reasoning to improve path efficiency")
        
        if results["missing_critical_steps"]:
            recommendations.append(f"Ensure these steps are included: {', '.join(results['missing_critical_steps'])}")
        
        if results["redundant_loops"]["redundant_loops_detected"]:
            recommendations.append("Implement caching or memoization to avoid redundant operations")
        
        if results["tool_selection_quality"] < 0.7:
            recommendations.append("Review tool selection logic and available tool descriptions")
        
        if not recommendations:
            recommendations.append("Trajectory is optimal - no improvements needed")
        
        return recommendations
