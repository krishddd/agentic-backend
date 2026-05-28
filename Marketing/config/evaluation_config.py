"""
Enhanced Evaluation Configuration

Comprehensive settings for evaluation data capture, storage, and organization.
"""
import os
from datetime import datetime


# ========== Storage Configuration ==========

# Base directory for all evaluation data
EVALUATION_BASE_DIR = "logs/evaluations"

# Organized subdirectories by date and agent
def get_evaluation_path(agent_id: str, run_id: str) -> str:
    """
    Generate organized path for evaluation storage.
    
    Format: logs/evaluations/YYYY-MM-DD/agent_name/run_id.json
    Example: logs/evaluations/2026-01-01/secretary/abc-123-def.json
    """
    today = datetime.now().strftime("%Y-%m-%d")
    eval_dir = os.path.join(EVALUATION_BASE_DIR, today, agent_id)
    os.makedirs(eval_dir, exist_ok=True)
    return os.path.join(eval_dir, f"{run_id}.json")


# Retention policy
EVALUATION_RETENTION_DAYS = 90  # Keep evaluations for 90 days

# ========== Data Capture Configuration ==========

# What to capture in evaluations
CAPTURE_DETAILED_STEPS = True  # Capture every step with full context
CAPTURE_TOOL_PARAMETERS = True  # Save tool call parameters
CAPTURE_LLM_PROMPTS = True  # Save LLM prompts (preview only for privacy)
CAPTURE_ERROR_STACKTRACES = True  # Include full error details
CAPTURE_TIMING_BREAKDOWN = True  # Detailed timing for each operation

# ========== Metrics Configuration ==========

# Metric calculation settings
METRICS_CONFIG = {
    "task_success": {
        "weight": 0.25,
        "pass_threshold": 0.8,
        "track_per_step": True
    },
    "tool_use": {
        "weight": 0.20,
        "pass_threshold": 0.7,
        "track_failures": True
    },
    "trajectory_quality": {
        "weight": 0.20,
        "pass_threshold": 0.8,
        "detect_redundancy": True
    },
    "robustness": {
        "weight": 0.15,
        "pass_threshold": 0.7,
        "track_recovery": True
    },
    "performance": {
        "weight": 0.10,
        "latency_threshold_ms": 10000,
        "track_tokens": True
    },
    "qualitative": {
        "weight": 0.10,
        "check_hallucination": True,
        "check_safety": True
    }
}

# Overall pass threshold
OVERALL_PASS_THRESHOLD = 0.7

# ========== Timestamping Configuration ==========

# Timestamp format for all logs
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"  # e.g., 2026-01-01 10:34:22.123456
TIMESTAMP_FORMAT_SHORT = "%Y-%m-%d %H:%M:%S"  # e.g., 2026-01-01 10:34:22

# Include timestamps in all exports
INCLUDE_TIMESTAMPS = True
TIMEZONE = "Asia/Kolkata"  # IST timezone

# ========== Summary Reports Configuration ==========

# Generate daily summary reports
GENERATE_DAILY_SUMMARY = True
DAILY_SUMMARY_PATH = "logs/evaluations/summaries"

# Summary report format
SUMMARY_REPORT_TEMPLATE = """
# Agent Evaluation Summary - {date}

## Overview
- Total Runs: {total_runs}
- Passed: {passed_count} ({passed_pct}%)
- Failed: {failed_count} ({failed_pct}%)
- Average Score: {avg_score:.2%}

## By Agent
{agent_breakdown}

## Performance Insights
- Average Latency: {avg_latency_ms:.0f}ms
- Total Tokens Used: {total_tokens:,}
- Total Cost: ${total_cost:.4f}

## Key Issues
{key_issues}

## Recommendations
{recommendations}
"""

# ========== Export Configuration ==========

# Export formats
EXPORT_FORMATS = ["json"]  # Future: csv, excel
PRETTY_PRINT_JSON = True  # Indent JSON for readability
JSON_INDENT = 2

# Include in exports
EXPORT_INCLUDE_EXECUTIVE_SUMMARY = True
EXPORT_INCLUDE_KEY_FINDINGS = True
EXPORT_INCLUDE_ACTION_ITEMS = True
EXPORT_INCLUDE_RAW_METRICS = True

# ========== Logging Configuration ==========

# Evaluation-specific logging
EVALUATION_LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
EVALUATION_LOG_FILE = "logs/evaluation_audit.log"

# Log every evaluation event
LOG_EVALUATION_START = True
LOG_EVALUATION_STEPS = True
LOG_EVALUATION_COMPLETE = True
LOG_EVALUATION_ERRORS = True

# ========== Observability Integration ==========

# Enable observability service
ENABLE_OBSERVABILITY = True

# Trace configuration
TRACE_ALL_OPERATIONS = True
TRACE_RETENTION_HOURS = 24

# Structured logging
USE_STRUCTURED_LOGS = True
LOG_FORMAT_JSON = True

# ========== Alerting Configuration ==========

# Alert on evaluation failures
ALERT_ON_FAILED_EVALUATION = True
ALERT_THRESHOLD_SCORE = 0.5  # Alert if score below this

# Alert destinations
ALERT_EMAIL = "admin@example.com"
ALERT_SLACK = False  # Not configured yet

# ========== Validation Configuration ==========

# Validate all evaluations before saving
VALIDATE_BEFORE_SAVE = True
STRICT_VALIDATION = True  # Error if validation fails

# Required fields
REQUIRED_FIELDS = [
    "run_id",
    "agent_id",
    "objective",
    "overall_verdict",
    "overall_score",
    "evaluated_at"
]

# ========== Helper Functions ==========

def get_daily_summary_path(date: str = None) -> str:
    """Get path for daily summary report"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(DAILY_SUMMARY_PATH, exist_ok=True)
    return os.path.join(DAILY_SUMMARY_PATH, f"summary_{date}.md")


def get_timestamp() -> str:
    """Get current timestamp in configured format"""
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def get_timestamp_short() -> str:
    """Get current timestamp in short format"""
    return datetime.now().strftime(TIMESTAMP_FORMAT_SHORT)
