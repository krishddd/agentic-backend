"""Evaluation data models for Financial Crew agent evaluation."""

from datetime import datetime
from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field


# ============================================================================
# Core Metric Models
# ============================================================================

class TaskSuccessMetrics(BaseModel):
    """Metrics for task success evaluation."""
    task_success_rate: float = Field(ge=0, le=1, description="Percentage of tasks completed successfully")
    task_completion_quality: float = Field(ge=0, le=1, description="Quality of completed tasks")
    objective_achieved: bool = Field(description="Whether the stated objective was achieved")
    explanation: str = Field(description="Human-readable explanation of task success")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    total_steps: int = Field(ge=0, description="Total number of steps executed")
    successful_steps: int = Field(ge=0, description="Number of successful steps")
    failed_steps: int = Field(ge=0, description="Number of failed steps")
    verdict: Literal["passed", "failed"] = Field(description="Overall verdict")


class ToolUseMetrics(BaseModel):
    """Metrics for tool usage evaluation."""
    tool_call_accuracy: float = Field(ge=0, le=1, description="Accuracy of tool selection")
    tool_parameter_accuracy: float = Field(ge=0, le=1, description="Correctness of tool parameters")
    tool_execution_success_rate: float = Field(ge=0, le=1, description="Tool execution success rate")
    tool_use_efficiency: float = Field(ge=0, le=1, description="Efficiency of tool usage")
    total_tool_calls: int = Field(ge=0, description="Total number of tool calls")
    successful_tool_calls: int = Field(ge=0, description="Successful tool calls")
    failed_tool_calls: int = Field(ge=0, description="Failed tool calls")
    explanation: str = Field(description="Explanation of tool usage")
    issues: List[str] = Field(default_factory=list, description="Issues encountered")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class TrajectoryQualityMetrics(BaseModel):
    """Metrics for trajectory quality evaluation."""
    trajectory_efficiency: float = Field(ge=0, le=1, description="Efficiency of execution path")
    redundant_actions_count: int = Field(ge=0, description="Number of redundant actions")
    redundant_actions_percent: float = Field(ge=0, le=1, description="Percentage of redundant actions")
    optimal_steps: int = Field(ge=0, description="Optimal number of steps")
    actual_steps: int = Field(ge=0, description="Actual number of steps taken")
    error_recovery_rate: float = Field(ge=0, le=1, description="Rate of successful error recovery")
    total_errors: int = Field(ge=0, description="Total errors encountered")
    recovered_errors: int = Field(ge=0, description="Errors successfully recovered from")
    explanation: str = Field(description="Explanation of trajectory quality")
    redundancy_details: List[str] = Field(default_factory=list, description="Details of redundancies")
    efficiency_tips: List[str] = Field(default_factory=list, description="Efficiency improvement tips")


class RobustnessMetrics(BaseModel):
    """Metrics for robustness evaluation."""
    error_handling_score: float = Field(ge=0, le=1, description="Error handling capability")
    retry_success_rate: float = Field(ge=0, le=1, description="Success rate of retries")
    guardrail_compliance_rate: float = Field(ge=0, le=1, description="Guardrail compliance rate")
    total_retries: int = Field(ge=0, description="Total retry attempts")
    successful_retries: int = Field(ge=0, description="Successful retries")
    guardrail_violations: int = Field(ge=0, description="Guardrail violations")
    explanation: str = Field(description="Explanation of robustness")
    violations: List[str] = Field(default_factory=list, description="List of violations")
    strengths: List[str] = Field(default_factory=list, description="Robustness strengths")


class PerformanceMetrics(BaseModel):
    """Metrics for performance evaluation."""
    avg_latency_ms: float = Field(ge=0, description="Average latency in milliseconds")
    total_duration_seconds: float = Field(ge=0, description="Total execution duration")
    token_efficiency: float = Field(ge=0, description="Token usage efficiency score")
    cost_per_task_usd: float = Field(ge=0, description="Estimated cost in USD")
    total_tokens: int = Field(ge=0, default=0, description="Total tokens used")
    prompt_tokens: int = Field(ge=0, default=0, description="Prompt tokens")
    completion_tokens: int = Field(ge=0, default=0, description="Completion tokens")
    explanation: str = Field(description="Performance explanation")
    performance_notes: List[str] = Field(default_factory=list, description="Performance notes")


class QualitativeMetrics(BaseModel):
    """Metrics for qualitative evaluation."""
    reasoning_coherence_score: float = Field(ge=0, le=1, description="Reasoning coherence score")
    adaptability_score: float = Field(ge=0, le=1, description="Adaptability score")
    hallucination_detected: bool = Field(description="Whether hallucinations were detected")
    safety_violations: List[str] = Field(default_factory=list, description="Safety violations")
    summary: str = Field(description="Qualitative summary")
    strengths: List[str] = Field(default_factory=list, description="Qualitative strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Qualitative weaknesses")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


# ============================================================================
# Four Pillars Framework Models
# ============================================================================

class EffectivenessMetrics(BaseModel):
    """Pillar 1: Effectiveness - How well does the agent achieve its objectives?"""
    goal_achievement_score: float = Field(ge=0, le=1, description="Goal achievement score")
    output_quality_score: float = Field(ge=0, le=1, description="Output quality score")
    task_completion_rate: float = Field(ge=0, le=1, description="Task completion rate")
    user_satisfaction_proxy: float = Field(ge=0, le=1, description="Estimated user satisfaction")
    explanation: str = Field(description="Effectiveness explanation")


class EfficiencyMetrics(BaseModel):
    """Pillar 2: Efficiency - How efficiently does the agent use resources?"""
    token_efficiency_score: float = Field(ge=0, le=1, description="Token usage efficiency")
    time_efficiency_score: float = Field(ge=0, le=1, description="Time efficiency")
    cost_efficiency_score: float = Field(ge=0, le=1, description="Cost efficiency")
    tool_call_efficiency: float = Field(ge=0, le=1, description="Tool call efficiency")
    total_cost_usd: float = Field(ge=0, description="Total cost in USD")
    explanation: str = Field(description="Efficiency explanation")


class SafetyMetrics(BaseModel):
    """Pillar 4: Safety & Alignment - Is the agent safe and aligned?"""
    bias_score: float = Field(ge=0, le=1, description="Bias detection score (0=biased, 1=unbiased)")
    safety_compliance_score: float = Field(ge=0, le=1, description="Safety compliance")
    alignment_score: float = Field(ge=0, le=1, description="Alignment with intended values")
    ethical_compliance_score: float = Field(ge=0, le=1, description="Ethical guideline compliance")
    hallucination_rate: float = Field(ge=0, le=1, description="Hallucination rate")
    safety_violations_detected: int = Field(ge=0, description="Number of safety violations")
    explanation: str = Field(description="Safety explanation")
    violations: List[str] = Field(default_factory=list, description="Detected violations")


# ============================================================================
# RAG-Specific Models
# ============================================================================

class RAGPerformanceMetrics(BaseModel):
    """Metrics specific to RAG (Retrieval-Augmented Generation) components."""
    retrieval_precision: float = Field(ge=0, le=1, description="Retrieval precision")
    retrieval_recall: float = Field(ge=0, le=1, description="Retrieval recall")
    context_utilization: float = Field(ge=0, le=1, description="How well retrieved context is used")
    hallucination_rate: float = Field(ge=0, le=1, description="Rate of hallucinations in RAG output")
    relevance_score: float = Field(ge=0, le=1, description="Relevance of retrieved documents")
    total_retrievals: int = Field(ge=0, default=0, description="Total retrieval operations")
    successful_retrievals: int = Field(ge=0, default=0, description="Successful retrievals")
    explanation: str = Field(description="RAG performance explanation")


# ============================================================================
# User Feedback Models
# ============================================================================

class UserFeedbackMetrics(BaseModel):
    """User feedback aggregation metrics."""
    thumbs_up: int = Field(ge=0, default=0, description="Thumbs up count")
    thumbs_down: int = Field(ge=0, default=0, description="Thumbs down count")
    avg_csat_score: Optional[float] = Field(ge=1, le=5, default=None, description="Average CSAT score (1-5)")
    nps_score: Optional[float] = Field(ge=-100, le=100, default=None, description="Net Promoter Score")
    total_feedback_count: int = Field(ge=0, default=0, description="Total feedback submissions")
    comments: List[str] = Field(default_factory=list, description="User comments")


# ============================================================================
# Complete Evaluation Result
# ============================================================================

class AgentEvaluation(BaseModel):
    """Complete evaluation result for a single agent run."""
    run_id: str = Field(description="Unique run identifier")
    agent_id: str = Field(description="Agent identifier")
    objective: str = Field(description="Stated objective for this run")
    overall_verdict: Literal["PASSED", "FAILED"] = Field(description="Overall verdict")
    overall_score: float = Field(ge=0, le=1, description="Overall weighted score")
    
    # Core metrics
    task_success: TaskSuccessMetrics
    tool_use: ToolUseMetrics
    trajectory_quality: TrajectoryQualityMetrics
    robustness: RobustnessMetrics
    performance: PerformanceMetrics
    qualitative: QualitativeMetrics
    
    # Four Pillars
    effectiveness: Optional[EffectivenessMetrics] = None
    efficiency: Optional[EfficiencyMetrics] = None
    safety: Optional[SafetyMetrics] = None
    
    # RAG-specific
    rag_performance: Optional[RAGPerformanceMetrics] = None
    
    # User feedback
    user_feedback: Optional[UserFeedbackMetrics] = None
    
    # Summary
    executive_summary: str = Field(description="Executive summary of evaluation")
    key_findings: List[str] = Field(description="Key findings from evaluation")
    action_items: List[str] = Field(description="Recommended action items")
    
    # Metadata
    evaluated_at: datetime = Field(default_factory=datetime.now, description="Evaluation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# Execution Tracking Models
# ============================================================================

class StepData(BaseModel):
    """Data for a single execution step."""
    step_name: str
    step_index: int
    success: bool
    duration_ms: float
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ToolCallData(BaseModel):
    """Data for a single tool call."""
    tool_name: str
    parameters: Dict[str, Any]
    success: bool
    result: Optional[Any] = None
    error_message: Optional[str] = None
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)


class LLMCallData(BaseModel):
    """Data for a single LLM call."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorData(BaseModel):
    """Data for an error occurrence."""
    error_type: str
    error_message: str
    recovered: bool
    recovery_action: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
