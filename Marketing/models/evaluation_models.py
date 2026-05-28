"""
Evaluation Models for Agent Quality Assessment

Based on Google's Agent Quality framework and existing evaluation patterns.
Implements the Four Pillars: Effectiveness, Efficiency, Robustness, Safety
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class VerdictEnum(str, Enum):
    """Evaluation verdict"""
    PASSED = "PASSED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


# Four Pillars Framework

class TaskSuccessMetrics(BaseModel):
    """Task Success and Effectiveness Metrics"""
    task_success_rate: float = Field(ge=0, le=1, description="Successful steps / total steps")
    task_completion_quality: float = Field(ge=0, le=1, description="Quality of completion")
    objective_achieved: bool = Field(description="Was the stated objective achieved?")
    total_steps: int = Field(ge=0)
    successful_steps: int = Field(ge=0)
    failed_steps: int = Field(ge=0)
    verdict: str = Field(default="failed")
    explanation: str = Field(default="")
    details: Dict[str, Any] = Field(default_factory=dict)


class ToolUseMetrics(BaseModel):
    """Tool Usage Quality Metrics"""
    tool_call_accuracy: float = Field(ge=0, le=1, description="Successful tool calls / total")
    tool_parameter_accuracy: float = Field(ge=0, le=1, description="Correct parameters")
    tool_execution_success_rate: float = Field(ge=0, le=1, description="Execution success rate")
    tool_use_efficiency: float = Field(ge=0, le=1, description="Minimal redundancy")
    total_tool_calls: int = Field(ge=0)
    successful_tool_calls: int = Field(ge=0)
    failed_tool_calls: int = Field(ge=0)
    explanation: str = Field(default="")
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class TrajectoryQualityMetrics(BaseModel):
    """Trajectory Efficiency and Quality Metrics"""
    trajectory_efficiency: float = Field(ge=0, le=1, description="Optimal steps / actual steps")
    redundant_actions_count: int = Field(ge=0)
    redundant_actions_percent: float = Field(ge=0, le=1)
    optimal_steps: int = Field(ge=0)
    actual_steps: int = Field(ge=0)
    error_recovery_rate: float = Field(ge=0, le=1)
    total_errors: int = Field(ge=0)
    recovered_errors: int = Field(ge=0)
    explanation: str = Field(default="")
    redundancy_details: List[str] = Field(default_factory=list)
    efficiency_tips: List[str] = Field(default_factory=list)


class RobustnessMetrics(BaseModel):
    """Robustness and Error Handling Metrics"""
    error_handling_score: float = Field(ge=0, le=1)
    retry_success_rate: float = Field(ge=0, le=1)
    guardrail_compliance_rate: float = Field(ge=0, le=1)
    total_retries: int = Field(ge=0)
    successful_retries: int = Field(ge=0)
    guardrail_violations: int = Field(ge=0)
    explanation: str = Field(default="")
    violations: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)


class PerformanceMetrics(BaseModel):
    """Performance and Resource Metrics"""
    avg_latency_ms: float = Field(ge=0)
    total_duration_seconds: float = Field(ge=0)
    token_efficiency: float = Field(ge=0, le=1, default=0)
    cost_per_task_usd: float = Field(ge=0, default=0)
    total_tokens: int = Field(ge=0, default=0)
    prompt_tokens: int = Field(ge=0, default=0)
    completion_tokens: int = Field(ge=0, default=0)
    explanation: str = Field(default="")
    performance_notes: List[str] = Field(default_factory=list)


class QualitativeMetrics(BaseModel):
    """Qualitative Assessment Metrics"""
    reasoning_coherence_score: float = Field(ge=0, le=1)
    adaptability_score: float = Field(ge=0, le=1)
    hallucination_detected: bool = Field(default=False)
    safety_violations: List[str] = Field(default_factory=list)
    summary: str = Field(default="")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


# Enhanced Metrics from PDF

class EffectivenessMetrics(BaseModel):
    """Effectiveness - Goal Achievement (Pillar 1)"""
    task_success_rate: float = Field(ge=0, le=1)
    task_completion_quality: float = Field(ge=0, le=1)
    objective_achieved: bool
    conversion_rate: Optional[float] = Field(None, ge=0, le=1)
    user_satisfaction: Optional[float] = Field(None, ge=0, le=1)


class EfficiencyMetrics(BaseModel):
    """Efficiency - Operational Cost (Pillar 2)"""
    total_steps: int = Field(ge=0)
    optimal_steps: int = Field(ge=0)
    efficiency_ratio: float = Field(ge=0, le=1)
    redundant_actions_count: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    wall_clock_time_ms: int = Field(ge=0)
    trajectory_complexity: float = Field(ge=0)


class SafetyMetrics(BaseModel):
    """Safety & Alignment (Pillar 4)"""
    guardrail_violations: int = Field(ge=0)
    prompt_injection_detected: bool = Field(default=False)
    pii_leakage_detected: bool = Field(default=False)
    harmful_content_detected: bool = Field(default=False)
    bias_score: float = Field(ge=0, le=1, default=0)
    safety_score: float = Field(ge=0, le=1)


class TrajectoryAdherenceMetrics(BaseModel):
    """Advanced Trajectory Metrics"""
    intended_path_followed: bool
    path_deviation_score: float = Field(ge=0, le=1)
    correct_tool_sequence: bool
    unnecessary_tool_calls: int = Field(ge=0)


class RAGPerformanceMetrics(BaseModel):
    """RAG-Specific Metrics"""
    retrieval_accuracy: float = Field(ge=0, le=1)
    context_utilization: float = Field(ge=0, le=1)
    source_correctness: bool
    hallucination_despite_rag: bool = Field(default=False)


class UserFeedbackMetrics(BaseModel):
    """User Feedback Metrics"""
    thumbs_up: int = Field(ge=0, default=0)
    thumbs_down: int = Field(ge=0, default=0)
    satisfaction_score: float = Field(ge=0, le=5, default=0)
    user_comments: List[str] = Field(default_factory=list)
    net_promoter_score: Optional[int] = Field(None, ge=0, le=10)


# Complete Evaluation Models

class AgentEvaluation(BaseModel):
    """Complete Agent Evaluation Result"""
    run_id: str
    agent_id: str
    objective: str
    overall_verdict: VerdictEnum
    overall_score: float = Field(ge=0, le=1)
    
    # Core metrics
    task_success: TaskSuccessMetrics
    tool_use: ToolUseMetrics
    trajectory_quality: TrajectoryQualityMetrics
    robustness: RobustnessMetrics
    performance: PerformanceMetrics
    qualitative: QualitativeMetrics
    
    # Summary
    executive_summary: str
    key_findings: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    evaluated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class WorkflowEvaluation(BaseModel):
    """Workflow-Level Evaluation (Multi-Agent)"""
    workflow_id: str
    workflow_type: str = Field(default="PEER")
    agent_evaluations: List[AgentEvaluation]
    overall_verdict: VerdictEnum
    overall_score: float = Field(ge=0, le=1)
    total_duration_ms: int = Field(ge=0)
    workflow_summary: str
    evaluated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class EvaluationSummary(BaseModel):
    """Lightweight Evaluation Summary for API responses"""
    run_id: str
    agent_id: str
    verdict: VerdictEnum
    overall_score: float
    key_metrics: Dict[str, float]
    timestamp: str
