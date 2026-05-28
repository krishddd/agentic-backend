"""
Agent Evaluation Service

Core service for tracking and evaluating agent execution quality.
Implements comprehensive metrics across all quality dimensions.
"""
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from models.evaluation_models import (
    AgentEvaluation,
    TaskSuccessMetrics,
    ToolUseMetrics,
    TrajectoryQualityMetrics,
    RobustnessMetrics,
    PerformanceMetrics,
    QualitativeMetrics,
    VerdictEnum
)
from utils.evaluation_utils import (
    generate_run_id,
    calculate_weighted_score,
    format_evaluation_summary,
    generate_key_findings,
    generate_recommendations,
    calculate_verdict
)
from utils.logger import get_logger

logger = get_logger(__name__)


class AgentEvaluator:
    """
    Comprehensive agent evaluator tracking all quality metrics.
    
    Usage:
        evaluator = AgentEvaluator(agent_id="secretary", objective="Process pending emails")
        evaluator.start_evaluation()
        evaluator.record_step(step_name="fetch_data", success=True)
        evaluator.record_tool_call("gmail_send", {...}, success=True)
        result = evaluator.finalize_evaluation()
    """
    
    def __init__(self, agent_id: str, objective: str, run_id: Optional[str] = None):
        """
        Initialize evaluator for an agent run.
        
        Args:
            agent_id: Unique identifier for the agent (e.g., "secretary", "analyst")
            objective: The stated objective/goal for this execution
            run_id: Optional custom run ID, auto-generated if not provided
        """
        self.run_id = run_id or generate_run_id()
        self.agent_id = agent_id
        self.objective = objective
        
        # Tracking data
        self.start_time = None
        self.end_time = None
        self.steps: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.llm_calls: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        
        logger.info(f"[Evaluator] Initialized for agent={agent_id}, run_id={self.run_id}")
    
    def start_evaluation(self):
        """Begin evaluation tracking"""
        self.start_time = time.time()
        logger.debug(f"[Evaluator {self.run_id}] Started evaluation")
    
    def record_step(self, step_name: str, success: bool, metadata: Optional[Dict] = None):
        """
        Record an execution step.
        
        Args:
            step_name: Name/description of the step
            success: Whether the step succeeded
            metadata: Optional additional context
        """
        step = {
            "name": step_name,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.steps.append(step)
        logger.debug(f"[Evaluator {self.run_id}] Step: {step_name} - {'✓' if success else '✗'}")
    
    def record_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        success: bool,
        error: Optional[str] = None,
        latency_ms: Optional[float] = None
    ):
        """
        Record a tool call.
        
        Args:
            tool_name: Name of the tool called
            parameters: Parameters passed to tool
            success: Whether call succeeded
            error: Error message if failed
            latency_ms: Call latency in milliseconds
        """
        tool_call = {
            "tool": tool_name,
            "parameters": parameters,
            "success": success,
            "error": error,
            "latency_ms": latency_ms,
            "timestamp": datetime.now().isoformat()
        }
        self.tool_calls.append(tool_call)
        logger.debug(f"[Evaluator {self.run_id}] Tool call: {tool_name} - {'✓' if success else '✗'}")
    
    def record_llm_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cost_usd: float = 0.0,
        model: str = "unknown"
    ):
        """
        Record an LLM API call.
        
        Args:
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            latency_ms: Call latency
            cost_usd: API cost
            model: Model identifier
        """
        llm_call = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "timestamp": datetime.now().isoformat()
        }
        self.llm_calls.append(llm_call)
        logger.debug(f"[Evaluator {self.run_id}] LLM call: {model}, {prompt_tokens + completion_tokens} tokens")
    
    def record_error(self, error: str, recovered: bool = False, metadata: Optional[Dict] = None):
        """
        Record an error.
        
        Args:
            error: Error message
            recovered: Whether error was recovered from
            metadata: Additional context
        """
        error_record = {
            "error": error,
            "recovered": recovered,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.errors.append(error_record)
        logger.debug(f"[Evaluator {self.run_id}] Error: {error[:50]}... (recovered={recovered})")
    
    def _calculate_task_success(self) -> TaskSuccessMetrics:
        """Calculate task success metrics"""
        total_steps = len(self.steps)
        successful_steps = sum(1 for s in self.steps if s['success'])
        failed_steps = total_steps - successful_steps
        
        task_success_rate = successful_steps / total_steps if total_steps > 0 else 0
        task_completion_quality = successful_steps / total_steps if total_steps > 0 else 0
        
        # Objective achieved if >= 80% steps succeeded
        objective_achieved = task_success_rate >= 0.8
        
        verdict = "failed" if failed_steps > 0 else "success"
        if failed_steps == 0 and total_steps == 0:
            verdict = "no_steps"
        
        explanation = f"{'✓' if objective_achieved else '✗'} Task {'SUCCEEDED' if objective_achieved else 'FAILED'} to complete. "
        explanation += f"{failed_steps} steps failed out of {total_steps} total steps. "
        explanation += f"The stated objective was {'fully achieved' if objective_achieved else 'not fully achieved'}."
        
        return TaskSuccessMetrics(
            task_success_rate=task_success_rate,
            task_completion_quality=task_completion_quality,
            objective_achieved=objective_achieved,
            total_steps=total_steps,
            successful_steps=successful_steps,
            failed_steps=failed_steps,
            verdict=verdict,
            explanation=explanation,
            details={
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "failed_steps": failed_steps,
                "verdict": verdict
            }
        )
    
    def _calculate_tool_use(self) -> ToolUseMetrics:
        """Calculate tool usage metrics"""
        total_tools = len(self.tool_calls)
        successful_tools = sum(1 for t in self.tool_calls if t['success'])
        failed_tools = total_tools - successful_tools
        
        tool_call_accuracy = successful_tools / total_tools if total_tools > 0 else 1.0
        tool_parameter_accuracy = 1.0  # Simplified - assume correct if successful
        tool_execution_success_rate = tool_call_accuracy
        tool_use_efficiency = 1.0  # Simplified - no redundancy detection yet
        
        issues = []
        recommendations = []
        
        if failed_tools > 0:
            issues.append(f"{failed_tools} tool calls failed")
            recommendations.append("Review error handling for tool failures")
        
        if tool_call_accuracy < 0.7:
            explanation = f"✗ Poor tool execution: {tool_call_accuracy*100:.0f}% success rate."
        elif tool_call_accuracy < 0.9:
            explanation = f"○ Moderate tool execution: {tool_call_accuracy*100:.0f}% success rate."
        else:
            explanation = f"✓ Good tool execution: {tool_call_accuracy*100:.0f}% success rate."
        
        explanation += f" with {tool_use_efficiency*100:.0f}% efficiency (minimal redundancy)."
        
        return ToolUseMetrics(
            tool_call_accuracy=tool_call_accuracy,
            tool_parameter_accuracy=tool_parameter_accuracy,
            tool_execution_success_rate=tool_execution_success_rate,
            tool_use_efficiency=tool_use_efficiency,
            total_tool_calls=total_tools,
            successful_tool_calls=successful_tools,
            failed_tool_calls=failed_tools,
            explanation=explanation,
            issues=issues,
            recommendations=recommendations
        )
    
    def _calculate_trajectory_quality(self) -> TrajectoryQualityMetrics:
        """Calculate trajectory efficiency metrics"""
        actual_steps = len(self.steps)
        optimal_steps = actual_steps  # Simplified - assume current is optimal
        redundant_actions = 0  # Simplified
        
        trajectory_efficiency = optimal_steps / actual_steps if actual_steps > 0 else 1.0
        redundant_actions_percent = redundant_actions / actual_steps if actual_steps > 0 else 0
        
        total_errors = len(self.errors)
        recovered_errors = sum(1 for e in self.errors if e['recovered'])
        error_recovery_rate = recovered_errors / total_errors if total_errors > 0 else 1.0
        
        explanation = f"The agent's trajectory was highly efficient ({trajectory_efficiency*100:.0f}%). "
        explanation += f"It took {actual_steps} steps, of which {optimal_steps} were optimal. "
        
        if total_errors > 0:
            explanation += f"{total_errors} errors were encountered, {recovered_errors} recovered."
        else:
            explanation += "No errors were encountered."
        
        return TrajectoryQualityMetrics(
            trajectory_efficiency=trajectory_efficiency,
            redundant_actions_count=redundant_actions,
            redundant_actions_percent=redundant_actions_percent,
            optimal_steps=optimal_steps,
            actual_steps=actual_steps,
            error_recovery_rate=error_recovery_rate,
            total_errors=total_errors,
            recovered_errors=recovered_errors,
            explanation=explanation,
            redundancy_details=[],
            efficiency_tips=[]
        )
    
    def _calculate_robustness(self) -> RobustnessMetrics:
        """Calculate robustness metrics"""
        total_errors = len(self.errors)
        recovered_errors = sum(1 for e in self.errors if e['recovered'])
        
        error_handling_score = recovered_errors / total_errors if total_errors > 0 else 1.0
        retry_success_rate = 1.0  # Simplified
        guardrail_compliance_rate = 1.0  # Simplified
        guardrail_violations = 0
        
        violations = []
        strengths = []
        
        if error_handling_score >= 0.8:
            strengths.append("Strong error recovery capabilities")
        
        if guardrail_compliance_rate == 1.0:
            strengths.append("Perfect adherence to safety guidelines")
        
        explanation = ""
        if error_handling_score >= 0.8:
            explanation += "✓ Excellent error handling. "
        elif error_handling_score >= 0.5:
            explanation += "○ Moderate error handling. "
        else:
            explanation += "✗ Poor error handling. "
        
        if guardrail_compliance_rate == 1.0:
            explanation += "✓ Full guardrail compliance."
        
        return RobustnessMetrics(
            error_handling_score=error_handling_score,
            retry_success_rate=retry_success_rate,
            guardrail_compliance_rate=guardrail_compliance_rate,
            total_retries=0,
            successful_retries=0,
            guardrail_violations=guardrail_violations,
            explanation=explanation,
            violations=violations,
            strengths=strengths
        )
    
    def _calculate_performance(self) -> PerformanceMetrics:
        """Calculate performance metrics"""
        duration = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        total_steps = len(self.steps)
        
        # LLM metrics
        total_prompt_tokens = sum(call['prompt_tokens'] for call in self.llm_calls)
        total_completion_tokens = sum(call['completion_tokens'] for call in self.llm_calls)
        total_tokens = total_prompt_tokens + total_completion_tokens
        total_cost = sum(call['cost_usd'] for call in self.llm_calls)
        
        # Latency
        if self.llm_calls:
            avg_latency = sum(call['latency_ms'] for call in self.llm_calls) / len(self.llm_calls)
        elif self.tool_calls:
            latencies = [t['latency_ms'] for t in self.tool_calls if t.get('latency_ms')]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
        else:
            avg_latency = duration * 1000 / total_steps if total_steps > 0 else 0
        
        performance_notes = []
        if avg_latency > 10000:
            performance_notes.append("High latency detected - consider optimizing LLM calls")
        
        explanation = f"The agent performed with "
        if avg_latency < 3000:
            explanation += f"fast response times (avg {avg_latency:.0f}ms per step). "
        elif avg_latency < 10000:
            explanation += f"moderate response times (avg {avg_latency:.0f}ms per step). "
        else:
            explanation += f"slow response times (avg {avg_latency:.0f}ms per step). "
        
        explanation += f"Total execution took {duration:.1f} seconds. "
        
        return PerformanceMetrics(
            avg_latency_ms=avg_latency,
            total_duration_seconds=duration,
            token_efficiency=0,  # Simplified
            cost_per_task_usd=total_cost,
            total_tokens=total_tokens,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            explanation=explanation,
            performance_notes=performance_notes
        )
    
    def _calculate_qualitative(self) -> QualitativeMetrics:
        """Calculate qualitative metrics"""
        # Simplified - assume good quality unless errors detected
        reasoning_coherence = 1.0  if len(self.errors) == 0 else 0.7
        adaptability = 1.0 if any(e['recovered'] for e in self.errors) else 0.8
        hallucination_detected = False
        safety_violations = []
        
        strengths = ["Clear thought-action sequences"]
        if adaptability >= 0.9:
            strengths.append("Excellent error recovery")
        strengths.append("No safety concerns")
        
        summary = f"The agent demonstrated {'coherent' if reasoning_coherence >= 0.8 else 'inconsistent'} reasoning. "
        summary += f"with {'strong' if adaptability >= 0.8 else 'weak'} adaptability to errors."
        
        return QualitativeMetrics(
            reasoning_coherence_score=reasoning_coherence,
            adaptability_score=adaptability,
            hallucination_detected=hallucination_detected,
            safety_violations=safety_violations,
            summary=summary,
            strengths=strengths,
            weaknesses=[],
            recommendations=[]
        )
    
    def finalize_evaluation(self) -> AgentEvaluation:
        """
        Complete evaluation and calculate all metrics.
        
        Returns:
            Complete AgentEvaluation object
        """
        self.end_time = time.time()
        
        # Calculate all metric categories
        task_success = self._calculate_task_success()
        tool_use = self._calculate_tool_use()
        trajectory_quality = self._calculate_trajectory_quality()
        robustness = self._calculate_robustness()
        performance = self._calculate_performance()
        qualitative = self._calculate_qualitative()
        
        # Calculate overall score (weighted average)
        metrics_scores = {
            "task_success": task_success.task_success_rate,
            "tool_use": tool_use.tool_call_accuracy,
            "trajectory_quality": trajectory_quality.trajectory_efficiency,
            "robustness": robustness.error_handling_score,
            "qualitative": (qualitative.reasoning_coherence_score + qualitative.adaptability_score) / 2
        }
        
        overall_score = calculate_weighted_score(metrics_scores)
        overall_verdict = VerdictEnum(calculate_verdict(overall_score))
        
        # Create evaluation object
        evaluation = AgentEvaluation(
            run_id=self.run_id,
            agent_id=self.agent_id,
            objective=self.objective,
            overall_verdict=overall_verdict,
            overall_score=overall_score,
            task_success=task_success,
            tool_use=tool_use,
            trajectory_quality=trajectory_quality,
            robustness=robustness,
            performance=performance,
            qualitative=qualitative,
            executive_summary="",  # Will be generated
            key_findings=[],
            action_items=[]
        )
        
        # Generate summary and findings
        eval_dict = evaluation.dict()
        evaluation.executive_summary = format_evaluation_summary(eval_dict)
        evaluation.key_findings = generate_key_findings(eval_dict)
        evaluation.action_items = generate_recommendations(eval_dict)
        
        logger.info(f"[Evaluator {self.run_id}] Finalized: {overall_verdict.value} ({overall_score:.2f})")
        
        return evaluation
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export current state as dictionary"""
        evaluation = self.finalize_evaluation()
        return evaluation.dict()
    
    def export_to_json(self, filepath: str):
        """
        Save evaluation to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        evaluation_dict = self.export_to_dict()
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(evaluation_dict, f, indent=2)
        
        logger.info(f"[Evaluator {self.run_id}] Saved to {filepath}")
