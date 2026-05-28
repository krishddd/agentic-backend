"""
Enhanced Evaluation Export with Comprehensive Data Tracking

Ensures all execution data is saved with proper timestamps and structure.
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from config.evaluation_config import (
    get_evaluation_path,
    TIMESTAMP_FORMAT,
    PRETTY_PRINT_JSON,
    JSON_INDENT,
    CAPTURE_DETAILED_STEPS,
    CAPTURE_TOOL_PARAMETERS,
    CAPTURE_TIMING_BREAKDOWN
)
from utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedEvaluationExporter:
    """
    Enhanced exporter for comprehensive evaluation data with timestamps.
    
    Ensures ALL execution data is captured and properly structured.
    """
    
    def __init__(self, agent_id: str, run_id: str):
        self.agent_id = agent_id
        self.run_id = run_id
        self.execution_log: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
    def log_execution_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        timestamp: datetime = None
    ):
        """
        Log every execution event with timestamp.
        
        Args:
            event_type: Type of event (step_start, step_complete, tool_call, llm_call, error)
            event_data: Event details
            timestamp: Event timestamp (auto-generated if not provided)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        event = {
            "timestamp": timestamp.strftime(TIMESTAMP_FORMAT),
            "event_type": event_type,
            "data": event_data,
            "elapsed_ms": (timestamp - self.start_time).total_seconds() * 1000
        }
        
        self.execution_log.append(event)
        logger.debug(f"[ExecutionLog] {event_type}: {event_data.get('name', 'N/A')}")
    
    def export_comprehensive_evaluation(
        self,
        evaluation: Dict[str, Any],
        raw_data: Dict[str, Any] = None
    ) -> str:
        """
        Export comprehensive evaluation with all execution data.
        
        Args:
            evaluation: Evaluation result dict
            raw_data: Optional raw execution data
            
        Returns:
            Path to saved file
        """
        # Get organized storage path
        filepath = get_evaluation_path(self.agent_id, self.run_id)
        
        # Build comprehensive export
        comprehensive_data = {
            # Core evaluation
            **evaluation,
            
            # Enhanced metadata
            "metadata": {
                "exported_at": datetime.now().strftime(TIMESTAMP_FORMAT),
                "evaluation_version": "1.0",
                "agent_version": "secretary_v1" if self.agent_id == "secretary" else "analyst_v1",
                "environment": "production"
            },
            
            # Detailed timestamps
            "timestamps": {
                "started_at": self.start_time.strftime(TIMESTAMP_FORMAT),
                "completed_at": datetime.now().strftime(TIMESTAMP_FORMAT),
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "timezone": "Asia/Kolkata"
            },
            
            # Execution log (if detailed capture enabled)
            "execution_log": self.execution_log if CAPTURE_DETAILED_STEPS else [],
            
            # Timing breakdown
            "timing_breakdown": self._get_timing_breakdown() if CAPTURE_TIMING_BREAKDOWN else {},
            
            # Raw data (if provided)
            "raw_data": raw_data if raw_data else {}
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save with proper formatting
        with open(filepath, 'w', encoding='utf-8') as f:
            if PRETTY_PRINT_JSON:
                json.dump(comprehensive_data, f, indent=JSON_INDENT, ensure_ascii=False)
            else:
                json.dump(comprehensive_data, f, ensure_ascii=False)
        
        logger.info(f"[Export] Saved comprehensive evaluation to {filepath}")
        
        # Also save a simple summary for quick access
        self._save_summary(comprehensive_data, filepath)
        
        return filepath
    
    def _get_timing_breakdown(self) -> Dict[str, Any]:
        """Calculate detailed timing breakdown from execution log"""
        breakdown = {
            "total_events": len(self.execution_log),
            "event_types": {},
            "slowest_operations": []
        }
        
        # Count event types
        for event in self.execution_log:
            event_type = event["event_type"]
            breakdown["event_types"][event_type] = breakdown["event_types"].get(event_type, 0) + 1
        
        # Find slowest operations (if we have timing data)
        events_with_duration = [e for e in self.execution_log if "duration_ms" in e.get("data", {})]
        if events_with_duration:
            sorted_events = sorted(
                events_with_duration,
                key=lambda x: x["data"].get("duration_ms", 0),
                reverse=True
            )[:5]  # Top 5 slowest
            
            breakdown["slowest_operations"] = [
                {
                    "operation": e["event_type"],
                    "duration_ms": e["data"].get("duration_ms"),
                    "timestamp": e["timestamp"]
                }
                for e in sorted_events
            ]
        
        return breakdown
    
    def _save_summary(self, full_data: Dict[str, Any], filepath: str):
        """Save a quick summary file alongside full evaluation"""
        summary = {
            "run_id": full_data["run_id"],
            "agent_id": full_data["agent_id"],
            "verdict": full_data["overall_verdict"],
            "score": full_data["overall_score"],
            "duration_seconds": full_data["timestamps"]["duration_seconds"],
            "evaluated_at": full_data["evaluated_at"],
            "key_findings": full_data.get("key_findings", []),
            "action_items": full_data.get("action_items", [])
        }
        
        summary_path = filepath.replace(".json", "_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"[Export] Saved quick summary to {summary_path}")


def export_with_comprehensive_tracking(
    agent_id: str,
    run_id: str,
    evaluation: Dict[str, Any],
    execution_events: List[Dict[str, Any]] = None,
    raw_data: Dict[str, Any] = None
) -> str:
    """
    Export evaluation with comprehensive tracking.
    
    Args:
        agent_id: Agent identifier
        run_id: Unique run ID
        evaluation: Evaluation result
        execution_events: List of execution events
        raw_data: Additional raw data
        
    Returns:
        Path to saved evaluation file
    """
    exporter = EnhancedEvaluationExporter(agent_id, run_id)
    
    # Add execution events if provided
    if execution_events:
        exporter.execution_log = execution_events
    
    return exporter.export_comprehensive_evaluation(evaluation, raw_data)
