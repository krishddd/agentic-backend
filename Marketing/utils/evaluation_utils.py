"""
Evaluation Utilities

Helper functions for agent evaluation metrics calculation and reporting.
"""
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime


def generate_run_id() -> str:
    """Generate unique evaluation run ID"""
    return str(uuid.uuid4())


def calculate_weighted_score(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate weighted average of metrics.
    
    Args:
        metrics: Dict of metric_name -> score (0-1)
        weights: Optional custom weights, defaults to equal weighting
        
    Returns:
        Weighted average score (0-1)
    """
    if not metrics:
        return 0.0
    
    if weights is None:
        # Equal weighting
        weights = {k: 1.0 / len(metrics) for k in metrics.keys()}
    
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    
    score = sum(metrics[k] * weights.get(k, 0) for k in metrics.keys())
    return score / total_weight


def format_evaluation_summary(evaluation: Dict[str, Any]) -> str:
    """
    Create executive summary from evaluation data.
    
    Args:
        evaluation: Complete evaluation dict
        
    Returns:
        Formatted executive summary string
    """
    verdict = evaluation.get('overall_verdict', 'UNKNOWN')
    score = evaluation.get('overall_score', 0) * 100
    objective = evaluation.get('objective', 'N/A')
    
    task_success = evaluation.get('task_success', {})
    total_steps = task_success.get('total_steps', 0)
    successful_steps = task_success.get('successful_steps', 0)
    success_pct = (successful_steps / total_steps * 100) if total_steps > 0 else 0
    
    tool_use = evaluation.get('tool_use', {})
    total_tools = tool_use.get('total_tool_calls', 0)
    tool_success_rate = tool_use.get('tool_execution_success_rate', 0) * 100
    
    summary = f"""**Evaluation Result: {verdict}** (Overall Score: {score:.1f}%)

The agent was tasked with: '{objective}'. It {'achieved' if evaluation.get('task_success', {}).get('objective_achieved') else 'failed to complete'} the objective. Out of {total_steps} steps, {success_pct:.0f}% were successful. The agent made {total_tools} tool calls with {tool_success_rate:.0f}% success rate."""
    
    return summary


def generate_key_findings(evaluation: Dict[str, Any]) -> List[str]:
    """
    Extract key findings from evaluation.
    
    Args:
        evaluation: Complete evaluation dict
        
    Returns:
        List of key finding strings
    """
    findings = []
    
    # Task success
    task_success = evaluation.get('task_success', {})
    if task_success.get('objective_achieved'):
        findings.append("✓ Successfully achieved stated objective")
    else:
        findings.append("✗ Failed to achieve stated objective")
    
    # Tool usage
    tool_use = evaluation.get('tool_use', {})
    if tool_use.get('failed_tool_calls', 0) > 0:
        findings.append(f"⚠ {tool_use['failed_tool_calls']} tool failures detected")
    else:
        findings.append("✓ All tool calls successful")
    
    # Trajectory efficiency
    trajectory = evaluation.get('trajectory_quality', {})
    efficiency = trajectory.get('trajectory_efficiency', 0)
    if efficiency >= 0.9:
        findings.append(f"✓ Highly efficient trajectory ({efficiency*100:.0f}%)")
    elif efficiency >= 0.7:
        findings.append(f"○ Moderate trajectory efficiency ({efficiency*100:.0f}%)")
    else:
        findings.append(f"⚠ Low trajectory efficiency ({efficiency*100:.0f}%)")
    
    # Qualitative
    qualitative = evaluation.get('qualitative', {})
    if qualitative.get('strengths'):
        for strength in qualitative['strengths'][:2]:  # First 2
            findings.append(f"✓ {strength}")
    
    if qualitative.get('hallucination_detected'):
        findings.append("⚠ Hallucination detected")
    
    return findings


def generate_recommendations(evaluation: Dict[str, Any]) -> List[str]:
    """
    Generate actionable recommendations from evaluation.
    
    Args:
        evaluation: Complete evaluation dict
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # From tool use
    tool_use = evaluation.get('tool_use', {})
    if tool_use.get('recommendations'):
        recommendations.extend(tool_use['recommendations'])
    
    # From qualitative
    qualitative = evaluation.get('qualitative', {})
    if qualitative.get('recommendations'):
        recommendations.extend(qualitative['recommendations'])
    
    # Performance-based recommendations
    performance = evaluation.get('performance', {})
    avg_latency = performance.get('avg_latency_ms', 0)
    if avg_latency > 10000:  # > 10 seconds
        recommendations.append("Optimize LLM calls to reduce high latency")
    
    # Trajectory-based recommendations
    trajectory = evaluation.get('trajectory_quality', {})
    if trajectory.get('redundant_actions_count', 0) > 0:
        recommendations.append("Reduce redundant actions to improve efficiency")
    
    return recommendations


def compare_evaluations(eval1: Dict[str, Any], eval2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two evaluations side by side.
    
    Args:
        eval1: First evaluation dict
        eval2: Second evaluation dict
        
    Returns:
        Comparison report dict
    """
    comparison = {
        "eval1_id": eval1.get('run_id'),
        "eval2_id": eval2.get('run_id'),
        "score_diff": eval2.get('overall_score', 0) - eval1.get('overall_score', 0),
        "verdict_changed": eval1.get('overall_verdict') != eval2.get('overall_verdict'),
        "improvements": [],
        "regressions": []
    }
    
    # Compare task success
    task1 = eval1.get('task_success', {}).get('task_success_rate', 0)
    task2 = eval2.get('task_success', {}).get('task_success_rate', 0)
    if task2 > task1:
        comparison['improvements'].append(f"Task success improved by {(task2-task1)*100:.1f}%")
    elif task2 < task1:
        comparison['regressions'].append(f"Task success decreased by {(task1-task2)*100:.1f}%")
    
    # Compare tool accuracy
    tool1 = eval1.get('tool_use', {}).get('tool_call_accuracy', 0)
    tool2 = eval2.get('tool_use', {}).get('tool_call_accuracy', 0)
    if tool2 > tool1:
        comparison['improvements'].append(f"Tool accuracy improved by {(tool2-tool1)*100:.1f}%")
    elif tool2 < tool1:
        comparison['regressions'].append(f"Tool accuracy decreased by {(tool1-tool2)*100:.1f}%")
    
    # Compare efficiency
    eff1 = eval1.get('trajectory_quality', {}).get('trajectory_efficiency', 0)
    eff2 = eval2.get('trajectory_quality', {}).get('trajectory_efficiency', 0)
    if eff2 > eff1:
        comparison['improvements'].append(f"Efficiency improved by {(eff2-eff1)*100:.1f}%")
    elif eff2 < eff1:
        comparison['regressions'].append(f"Efficiency decreased by {(eff1-eff2)*100:.1f}%")
    
    return comparison


def calculate_verdict(overall_score: float, threshold: float = 0.7) -> str:
    """
    Determine verdict based on overall score.
    
    Args:
        overall_score: Score from 0-1
        threshold: Pass threshold (default 0.7)
        
    Returns:
        'PASSED', 'FAILED', or 'PARTIAL'
    """
    if overall_score >= threshold:
        return "PASSED"
    elif overall_score >= threshold * 0.5:
        return "PARTIAL"
    else:
        return "FAILED"


def format_percentage(value: float) -> str:
    """Format fraction as percentage string"""
    return f"{value * 100:.1f}%"


def format_duration(milliseconds: float) -> str:
    """Format duration in human-readable format"""
    if milliseconds < 1000:
        return f"{milliseconds:.0f}ms"
    elif milliseconds < 60000:
        return f"{milliseconds/1000:.1f}s"
    else:
        minutes = int(milliseconds / 60000)
        seconds = (milliseconds % 60000) / 1000
        return f"{minutes}m {seconds:.0f}s"
