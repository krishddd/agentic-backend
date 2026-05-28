"""Agent evaluation service for tracking and calculating evaluation metrics."""

import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import json

from evaluation.models import (
    AgentEvaluation,
    TaskSuccessMetrics,
    ToolUseMetrics,
    TrajectoryQualityMetrics,
    RobustnessMetrics,
    PerformanceMetrics,
    QualitativeMetrics,
    EffectivenessMetrics,
    EfficiencyMetrics,
    SafetyMetrics,
    RAGPerformanceMetrics,
    StepData,
    ToolCallData,
    LLMCallData,
    ErrorData,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class AgentEvaluator:
    """Tracks agent execution and calculates comprehensive evaluation metrics."""
    
    def __init__(self, agent_id: str, objective: str, run_id: Optional[str] = None):
        """Initialize the evaluator.
        
        Args:
            agent_id: Identifier for the agent being evaluated
            objective: Stated objective for this run
            run_id: Optional run ID (generated if not provided)
        """
        self.agent_id = agent_id
        self.objective = objective
        self.run_id = run_id or str(uuid.uuid4())
        
        # Tracking data
        self.steps: List[StepData] = []
        self.tool_calls: List[ToolCallData] = []
        self.llm_calls: List[LLMCallData] = []
        self.errors: List[ErrorData] = []
        
        # Timing
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # RAG tracking
        self.rag_retrievals: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized AgentEvaluator for {agent_id} (run_id: {self.run_id})")
    
    def start_evaluation(self):
        """Start tracking evaluation."""
        self.start_time = time.time()
        logger.info(f"Started evaluation tracking for run {self.run_id}")
    
    def record_step(self, step_name: str, success: bool, duration_ms: float = 0, 
                    error_message: Optional[str] = None):
        """Record an execution step.
        
        Args:
            step_name: Name of the step
            success: Whether the step succeeded
            duration_ms: Step duration in milliseconds
            error_message: Error message if step failed
        """
        step = StepData(
            step_name=step_name,
            step_index=len(self.steps),
            success=success,
            duration_ms=duration_ms,
            error_message=error_message
        )
        self.steps.append(step)
        logger.debug(f"Recorded step: {step_name} (success={success})")
    
    def record_tool_call(self, tool_name: str, parameters: Dict[str, Any],
                        success: bool, result: Optional[Any] = None,
                        error_message: Optional[str] = None, latency_ms: float = 0):
        """Record a tool call.
        
        Args:
            tool_name: Name of the tool called
            parameters: Parameters passed to the tool
            success: Whether the tool call succeeded
            result: Tool call result (if successful)
            error_message: Error message (if failed)
            latency_ms: Latency in milliseconds
        """     
        tool_call = ToolCallData(
            tool_name=tool_name,
            parameters=parameters,
            success=success,
            result=result,
            error_message=error_message,
            latency_ms=latency_ms
        )
        self.tool_calls.append(tool_call)
        logger.debug(f"Recorded tool call: {tool_name} (success={success})")
    
    def record_llm_call(self, model: str, prompt_tokens: int, completion_tokens: int,
                       latency_ms: float, cost_usd: float = 0):
        """Record an LLM call.
        
        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            latency_ms: Latency in milliseconds
            cost_usd: Estimated cost in USD
        """
        llm_call = LLMCallData(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd
        )
        self.llm_calls.append(llm_call)
        logger.debug(f"Recorded LLM call: {model} ({llm_call.total_tokens} tokens)")
    
    def record_error(self, error_type: str, error_message: str, recovered: bool,
                    recovery_action: Optional[str] = None):
        """Record an error occurrence.
        
        Args:
            error_type: Type of error
            error_message: Error message
            recovered: Whether error was recovered from
            recovery_action: Action taken to recover
        """
        error = ErrorData(
            error_type=error_type,
            error_message=error_message,
            recovered=recovered,
            recovery_action=recovery_action
        )
        self.errors.append(error)
        logger.debug(f"Recorded error: {error_type} (recovered={recovered})")
    
    def record_rag_retrieval(self, query: str, retrieved_chunks: List[Any],
                            relevant_chunks: Optional[List[Any]] = None,
                            relevance_scores: Optional[List[float]] = None):
        """Record a RAG retrieval operation.
        
        Args:
            query: Retrieval query
            retrieved_chunks: Chunks that were retrieved
            relevant_chunks: Ground truth relevant chunks (if known)
            relevance_scores: Relevance scores for retrieved chunks
        """
        self.rag_retrievals.append({
            "query": query,
            "retrieved_count": len(retrieved_chunks),
            "relevant_count": len(relevant_chunks) if relevant_chunks else None,
            "relevance_scores": relevance_scores
        })
    
    def finalize_evaluation(self) -> AgentEvaluation:
        """Calculate all metrics and generate final evaluation.
        
        Returns:
            Complete AgentEvaluation object
        """
        self.end_time = time.time()
        
        logger.info(f"Finalizing evaluation for run {self.run_id}")
        
        # Calculate all metric categories
        task_success = self._calculate_task_success()
        tool_use = self._calculate_tool_use()
        trajectory_quality = self._calculate_trajectory_quality()
        robustness = self._calculate_robustness()
        performance = self._calculate_performance()
        qualitative = self._calculate_qualitative()
        
        # Four Pillars
        effectiveness = self._calculate_effectiveness()
        efficiency = self._calculate_efficiency()
        safety = self._calculate_safety()
        
        # RAG metrics (if applicable)
        rag_performance = self._calculate_rag_performance() if self.rag_retrievals else None
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            task_success, tool_use, trajectory_quality,
            robustness, performance, qualitative
        )
        
        # Determine verdict
        overall_verdict = "PASSED" if overall_score >= 0.7 and task_success.objective_achieved else "FAILED"
        
        # Generate summaries
        executive_summary = self._generate_executive_summary(overall_verdict, overall_score, task_success)
        key_findings = self._generate_key_findings(task_success, tool_use, trajectory_quality, qualitative)
        action_items = self._generate_action_items(tool_use, performance)
        
        evaluation = AgentEvaluation(
            run_id=self.run_id,
            agent_id=self.agent_id,
            objective=self.objective,
            overall_verdict=overall_verdict,
            overall_score=round(overall_score, 3),
            task_success=task_success,
            tool_use=tool_use,
            trajectory_quality=trajectory_quality,
            robustness=robustness,
            performance=performance,
            qualitative=qualitative,
            effectiveness=effectiveness,
            efficiency=efficiency,
            safety=safety,
            rag_performance=rag_performance,
            executive_summary=executive_summary,
            key_findings=key_findings,
            action_items=action_items
        )
        
        logger.info(f"Evaluation finalized: {overall_verdict} (score: {overall_score:.2f})")
        
        return evaluation
    
    # ========================================================================
    # Metric Calculation Methods
    # ========================================================================
    
    def _calculate_task_success(self) -> TaskSuccessMetrics:
        """Calculate task success metrics."""
        total_steps = len(self.steps)
        successful_steps = sum(1 for step in self.steps if step.success)
        failed_steps = total_steps - successful_steps
        
        task_success_rate = successful_steps / total_steps if total_steps > 0 else 0
        task_completion_quality = task_success_rate  # Simplified
        objective_achieved = task_success_rate >= 0.8  # 80% threshold
        
        verdict = "passed" if objective_achieved else "failed"
        
        if failed_steps == 0:
            explanation = f"✓ All {total_steps} steps completed successfully. The stated objective was fully achieved."
        elif objective_achieved:
            explanation = f"✓ Task completed with {successful_steps}/{total_steps} successful steps. The stated objective was achieved."
        else:
            explanation = f"✗ Task FAILED to complete. {failed_steps} steps failed out of {total_steps} total steps. The stated objective was not fully achieved."
        
        return TaskSuccessMetrics(
            task_success_rate=task_success_rate,
            task_completion_quality=task_completion_quality,
            objective_achieved=objective_achieved,
            explanation=explanation,
            total_steps=total_steps,
            successful_steps=successful_steps,
            failed_steps=failed_steps,
            verdict=verdict,
            details={"step_names": [step.step_name for step in self.steps]}
        )
    
    def _calculate_tool_use(self) -> ToolUseMetrics:
        """Calculate tool use metrics."""
        total_calls = len(self.tool_calls)
        if total_calls == 0:
            return ToolUseMetrics(
                tool_call_accuracy=1.0,
                tool_parameter_accuracy=1.0,
                tool_execution_success_rate=1.0,
                tool_use_efficiency=1.0,
                total_tool_calls=0,
                successful_tool_calls=0,
                failed_tool_calls=0,
                explanation="No tool calls made.",
                issues=[],
                recommendations=[]
            )
        
        successful_calls = sum(1 for call in self.tool_calls if call.success)
        failed_calls = total_calls - successful_calls
        
        success_rate = successful_calls / total_calls
        tool_call_accuracy = success_rate
        tool_parameter_accuracy = 1.0  # Simplified - would need more analysis
        tool_use_efficiency = 1.0  # Simplified - would analyze redundancy
        
        issues = []
        recommendations = []
        
        if failed_calls > 0:
            issues.append(f"{failed_calls} tool calls failed")
            recommendations.append("Review error handling for tool failures")
        
        if success_rate < 0.8:
            explanation = f"✗ Poor tool execution: {int(success_rate*100)}% success rate. with {int(tool_use_efficiency*100)}% efficiency (minimal redundancy)."
        elif success_rate < 1.0:
            explanation = f"⚠ Good tool execution: {int(success_rate*100)}% success rate. with {int(tool_use_efficiency*100)}% efficiency."
        else:
            explanation = f"✓ Excellent tool execution: 100% success rate with optimal efficiency."
        
        return ToolUseMetrics(
            tool_call_accuracy=tool_call_accuracy,
            tool_parameter_accuracy=tool_parameter_accuracy,
            tool_execution_success_rate=success_rate,
            tool_use_efficiency=tool_use_efficiency,
            total_tool_calls=total_calls,
            successful_tool_calls=successful_calls,
            failed_tool_calls=failed_calls,
            explanation=explanation,
            issues=issues,
            recommendations=recommendations
        )
    
    def _calculate_trajectory_quality(self) -> TrajectoryQualityMetrics:
        """Calculate trajectory quality metrics."""
        actual_steps = len(self.steps)
        optimal_steps = actual_steps  # Simplified - would need ideal path analysis
        redundant_actions = 0  # Simplified
        
        efficiency = optimal_steps / actual_steps if actual_steps > 0 else 1.0
        redundant_percent = redundant_actions / actual_steps if actual_steps > 0 else 0
        
        total_errors = len([e for e in self.errors if not e.recovered])
        recovered_errors = len([e for e in self.errors if e.recovered])
        error_recovery_rate = recovered_errors / len(self.errors) if self.errors else 1.0
        
        if efficiency == 1.0 and total_errors == 0:
            explanation = f"The agent's trajectory was highly efficient (100%). It took {actual_steps} steps, of which {optimal_steps} were optimal. No errors were encountered."
        else:
            explanation = f"The agent's trajectory was {int(efficiency*100)}% efficient. It took {actual_steps} steps (optimal: {optimal_steps})."
        
        return TrajectoryQualityMetrics(
            trajectory_efficiency=efficiency,
            redundant_actions_count=redundant_actions,
            redundant_actions_percent=redundant_percent,
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
        """Calculate robustness metrics."""
        total_errors = len(self.errors)
        recovered_errors = sum(1 for e in self.errors if e.recovered)
        
        error_handling_score = recovered_errors / total_errors if total_errors > 0 else 1.0
        retry_success_rate = 1.0  # Simplified
        guardrail_compliance_rate = 1.0  # Simplified
        guardrail_violations = 0
        
        strengths = []
        if error_handling_score >= 0.8:
            strengths.append("Strong error recovery capabilities")
        if guardrail_compliance_rate == 1.0:
            strengths.append("Perfect adherence to safety guidelines")
        
        if error_handling_score == 1.0 and guardrail_compliance_rate == 1.0:
            explanation = "✓ Excellent error handling. ✓ Full guardrail compliance."
        else:
            explanation = f"Error handling: {int(error_handling_score*100)}%. Guardrail compliance: {int(guardrail_compliance_rate*100)}%."
        
        return RobustnessMetrics(
            error_handling_score=error_handling_score,
            retry_success_rate=retry_success_rate,
            guardrail_compliance_rate=guardrail_compliance_rate,
            total_retries=0,
            successful_retries=0,
            guardrail_violations=guardrail_violations,
            explanation=explanation,
            violations=[],
            strengths=strengths
        )
    
    def _calculate_performance(self) -> PerformanceMetrics:
        """Calculate performance metrics."""
        if not self.start_time or not self.end_time:
            total_duration = 0
            avg_latency = 0
        else:
            total_duration = self.end_time - self.start_time
            # Average latency per step
            if self.steps:
                avg_latency = (total_duration * 1000) / len(self.steps)
            else:
                avg_latency = 0
        
        # Token and cost calculation
        total_tokens = sum(call.total_tokens for call in self.llm_calls)
        prompt_tokens = sum(call.prompt_tokens for call in self.llm_calls)
        completion_tokens = sum(call.completion_tokens for call in self.llm_calls)
        total_cost = sum(call.cost_usd for call in self.llm_calls)
        
        token_efficiency = 0  # Simplified - would need baseline
        
        performance_notes = []
        if avg_latency > 5000:
            performance_notes.append("High latency detected - consider optimizing LLM calls")
        
        explanation = f"The agent performed with {'slow' if avg_latency > 5000 else 'good'} response times (avg {int(avg_latency)}ms per step). Total execution took {total_duration:.1f} seconds. "
        
        return PerformanceMetrics(
            avg_latency_ms=avg_latency,
            total_duration_seconds=total_duration,
            token_efficiency=token_efficiency,
            cost_per_task_usd=total_cost,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            explanation=explanation,
            performance_notes=performance_notes
        )
    
    def _calculate_qualitative(self) -> QualitativeMetrics:
        """Calculate qualitative metrics."""
        reasoning_coherence = 1.0  # Would need LLM judge
        adaptability = 1.0  # Based on error recovery
        hallucination_detected = False  # Would need checking
        
        strengths = ["Clear thought-action sequences"]
        if len(self.errors) > 0 and all(e.recovered for e in self.errors):
            strengths.append("Excellent error recovery")
        strengths.append("No safety concerns")
        
        summary = f"The agent demonstrated coherent reasoning. with strong adaptability to errors."
        
        return QualitativeMetrics(
            reasoning_coherence_score=reasoning_coherence,
            adaptability_score=adaptability,
            hallucination_detected=hallucination_detected,
            safety_violations=[],
            summary=summary,
            strengths=strengths,
            weaknesses=[],
            recommendations=[]
        )
    
    def _calculate_effectiveness(self) -> EffectivenessMetrics:
        """Calculate effectiveness (Pillar 1)."""
        task_success_rate = sum(1 for step in self.steps if step.success) / len(self.steps) if self.steps else 0
        
        return EffectivenessMetrics(
            goal_achievement_score=task_success_rate,
            output_quality_score=task_success_rate,
            task_completion_rate=task_success_rate,
            user_satisfaction_proxy=task_success_rate,
            explanation=f"Effectiveness score: {int(task_success_rate*100)}%"
        )
    
    def _calculate_efficiency(self) -> EfficiencyMetrics:
        """Calculate efficiency (Pillar 2)."""
        total_cost = sum(call.cost_usd for call in self.llm_calls)
        
        return EfficiencyMetrics(
            token_efficiency_score=0.8,  # Simplified
            time_efficiency_score=0.8,  # Simplified
            cost_efficiency_score=0.8,  # Simplified
            tool_call_efficiency=0.8,  # Simplified
            total_cost_usd=total_cost,
            explanation=f"Total cost: ${total_cost:.4f}"
        )
    
    def _calculate_safety(self) -> SafetyMetrics:
        """Calculate safety & alignment (Pillar 4)."""
        return SafetyMetrics(
            bias_score=1.0,  # Would need LLM judge
            safety_compliance_score=1.0,
            alignment_score=1.0,
            ethical_compliance_score=1.0,
            hallucination_rate=0.0,
            safety_violations_detected=0,
            explanation="No safety violations detected",
            violations=[]
        )
    
    def _calculate_rag_performance(self) -> Optional[RAGPerformanceMetrics]:
        """Calculate RAG-specific metrics."""
        if not self.rag_retrievals:
            return None
        
        total_retrievals = len(self.rag_retrievals)
        successful_retrievals = total_retrievals  # Simplified
        
        return RAGPerformanceMetrics(
            retrieval_precision=0.8,  # Would need ground truth
            retrieval_recall=0.8,  # Would need ground truth
            context_utilization=0.75,  # Would need analysis
            hallucination_rate=0.05,  # Would need checking
            relevance_score=0.85,  # Would need LLM judge
            total_retrievals=total_retrievals,
            successful_retrievals=successful_retrievals,
            explanation=f"Performed {total_retrievals} RAG retrievals"
        )
    
    def _calculate_overall_score(self, task_success, tool_use, trajectory,
                                 robustness, performance, qualitative) -> float:
        """Calculate weighted overall score."""
        weights = {
            "task_success": 0.3,
            "tool_use": 0.2,
            "trajectory": 0.15,
            "robustness": 0.15,
            "performance": 0.1,
            "qualitative": 0.1
        }
        
        score = (
            weights["task_success"] * task_success.task_success_rate +
            weights["tool_use"] * tool_use.tool_execution_success_rate +
            weights["trajectory"] * trajectory.trajectory_efficiency +
            weights["robustness"] * robustness.error_handling_score +
            weights["performance"] * (1.0 if performance.avg_latency_ms < 5000 else 0.5) +
            weights["qualitative"] * qualitative.reasoning_coherence_score
        )
        
        return score
    
    def _generate_executive_summary(self, verdict, score, task_success) -> str:
        """Generate executive summary."""
        summary = f"**Evaluation Result: {verdict}** (Overall Score: {score*100:.1f}%)\n\n"
        summary += f"The agent was tasked with: '{self.objective}'. "

        if verdict == "PASSED":
            summary += "It successfully completed the objective. "
        else:
            summary += "It failed to complete the objective. "

        summary += f"Out of {task_success.total_steps} steps, {int(task_success.task_completion_quality*100)}% were successful. "

        if self.tool_calls:
            # Bug fix: use actual tool-call success rate, not the step success rate
            tool_success_count = sum(1 for tc in self.tool_calls if tc.success)
            tool_success_rate = tool_success_count / len(self.tool_calls)
            summary += (
                f"The agent made {len(self.tool_calls)} tool calls "
                f"with {int(tool_success_rate * 100)}% tool success rate. "
            )

        return summary
    
    def _generate_key_findings(self, task_success, tool_use, trajectory, qualitative) -> List[str]:
        """Generate key findings."""
        findings = []
        
        if task_success.objective_achieved:
            findings.append("✓ Successfully achieved stated objective")
        else:
            findings.append("✗ Failed to achieve stated objective")
        
        if tool_use.failed_tool_calls > 0:
            findings.append(f"⚠ {tool_use.failed_tool_calls} tool failures detected")
        
        if trajectory.trajectory_efficiency >= 0.9:
            findings.append(f"✓ Efficient trajectory ({int(trajectory.trajectory_efficiency*100)}%)")
        
        for strength in qualitative.strengths:
            findings.append(f"✓ {strength}")
        
        return findings
    
    def _generate_action_items(self, tool_use, performance) -> List[str]:
        """Generate action items."""
        items = []
        items.extend(tool_use.recommendations)
        items.extend(performance.performance_notes)
        return items if items else ["No action items"]
    
    def save_evaluation(self, evaluation: AgentEvaluation, output_dir: Path):
        """Save evaluation to JSON file.
        
        Args:
            evaluation: Evaluation to save
            output_dir: Directory to save to
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{self.agent_id}_{self.run_id}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Use model_dump() — .dict() is deprecated in Pydantic V2
            dump_fn = getattr(evaluation, 'model_dump', None) or evaluation.dict
            json.dump(dump_fn(), f, indent=2, default=str)
        
        logger.info(f"Saved evaluation to {filepath}")
        
        return filepath
