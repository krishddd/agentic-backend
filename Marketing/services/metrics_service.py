"""
Metrics Collection Service
Tracks performance metrics, success rates, and resource usage
"""
import time
from typing import Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
from threading import Lock

from utils.logger import get_logger

logger = get_logger(__name__)


class WorkflowMetrics:
    """Metrics for a single workflow execution"""
    
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.started_at = time.time()
        self.completed_at: Optional[float] = None
        self.status = "running"  # running, completed, failed
        self.current_step = "initializing"
        self.agents_completed = []
        self.errors = []
        self.latencies = {}
        self.success_counts = defaultdict(int)
        self.failure_counts = defaultdict(int)
        self.llm_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.metadata = {}
    
    def complete(self, status: str):
        """Mark workflow as completed"""
        self.completed_at = time.time()
        self.status = status
    
    def duration_ms(self) -> int:
        """Get duration in milliseconds"""
        end_time = self.completed_at or time.time()
        return int((end_time - self.started_at) * 1000)
    
    def progress_percent(self) -> int:
        """Calculate progress percentage (rough estimate)"""
        if self.status == "completed":
            return 100
        if self.status == "failed":
            return 0
        
        # Simple heuristic: 50% per agent
        return len(self.agents_completed) * 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export"""
        return {
            "thread_id": self.thread_id,
            "status": self.status,
            "current_step": self.current_step,
            "duration_ms": self.duration_ms(),
            "progress_percent": self.progress_percent(),
            "agents_completed": self.agents_completed,
            "error_count": len(self.errors),
            "latencies": self.latencies,
            "success_counts": dict(self.success_counts),
            "failure_counts": dict(self.failure_counts),
            "llm_tokens": self.llm_tokens,
            "metadata": self.metadata
        }


class MetricsCollector:
    """Centralized metrics collection"""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowMetrics] = {}
        self.global_counters = defaultdict(int)
        self.global_latencies = defaultdict(list)
        self.lock = Lock()
    
    def start_workflow(self, thread_id: str) -> WorkflowMetrics:
        """Start tracking a new workflow"""
        with self.lock:
            metrics = WorkflowMetrics(thread_id)
            self.workflows[thread_id] = metrics
            self.global_counters["workflows_started"] += 1
            logger.info(f"[Metrics] Started tracking workflow: {thread_id}")
            return metrics
    
    def get_workflow_metrics(self, thread_id: str) -> Optional[WorkflowMetrics]:
        """Get metrics for a specific workflow"""
        return self.workflows.get(thread_id)
    
    def complete_workflow(self, thread_id: str, status: str):
        """Mark workflow as completed"""
        with self.lock:
            if thread_id in self.workflows:
                self.workflows[thread_id].complete(status)
                self.global_counters[f"workflows_{status}"] += 1
                logger.info(f"[Metrics] Workflow {thread_id} completed with status: {status}")
    
    def record_latency(self, thread_id: str, operation: str, duration_ms: float):
        """Record operation latency"""
        with self.lock:
            if thread_id in self.workflows:
                self.workflows[thread_id].latencies[operation] = duration_ms
            
            self.global_latencies[operation].append(duration_ms)
            logger.debug(f"[Metrics] {operation} took {duration_ms:.2f}ms")
    
    def record_agent_completion(self, thread_id: str, agent_name: str):
        """Record agent completion"""
        with self.lock:
            if thread_id in self.workflows:
                self.workflows[thread_id].agents_completed.append(agent_name)
                self.workflows[thread_id].current_step = f"{agent_name}_completed"
    
    def record_success(self, thread_id: str, operation: str):
        """Record successful operation"""
        with self.lock:
            if thread_id in self.workflows:
                self.workflows[thread_id].success_counts[operation] += 1
            self.global_counters[f"{operation}_success"] += 1
    
    def record_failure(self, thread_id: str, operation: str, error: str):
        """Record failed operation"""
        with self.lock:
            if thread_id in self.workflows:
                self.workflows[thread_id].failure_counts[operation] += 1
                self.workflows[thread_id].errors.append({
                    "operation": operation,
                    "error": error,
                    "timestamp": datetime.now().isoformat()
                })
            self.global_counters[f"{operation}_failure"] += 1
    
    def record_llm_tokens(
        self,
        thread_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float = 0.0
    ):
        """Record LLM token usage"""
        with self.lock:
            if thread_id in self.workflows:
                metrics = self.workflows[thread_id]
                metrics.llm_tokens["prompt"] += prompt_tokens
                metrics.llm_tokens["completion"] += completion_tokens
                metrics.llm_tokens["total"] += (prompt_tokens + completion_tokens)
                metrics.metadata["llm_cost_usd"] = metrics.metadata.get("llm_cost_usd", 0.0) + cost_usd
    
    def update_current_step(self, thread_id: str, step: str):
        """Update current workflow step"""
        with self.lock:
            if thread_id in self.workflows:
                self.workflows[thread_id].current_step = step
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics"""
        with self.lock:
            avg_latencies = {}
            for operation, latencies in self.global_latencies.items():
                if latencies:
                    avg_latencies[operation] = sum(latencies) / len(latencies)
            
            return {
                "counters": dict(self.global_counters),
                "average_latencies_ms": avg_latencies,
                "active_workflows": sum(
                    1 for m in self.workflows.values() if m.status == "running"
                ),
                "total_workflows": len(self.workflows)
            }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        with self.lock:
            # Counters
            for name, value in self.global_counters.items():
                lines.append(f"# TYPE marketing_agent_{name} counter")
                lines.append(f"marketing_agent_{name} {value}")
            
            # Latencies (as histograms)
            for operation, latencies in self.global_latencies.items():
                if latencies:
                    avg = sum(latencies) / len(latencies)
                    lines.append(f"# TYPE marketing_agent_{operation}_latency_ms gauge")
                    lines.append(f"marketing_agent_{operation}_latency_ms {avg:.2f}")
            
            # Active workflows
            active = sum(1 for m in self.workflows.values() if m.status == "running")
            lines.append(f"# TYPE marketing_agent_active_workflows gauge")
            lines.append(f"marketing_agent_active_workflows {active}")
        
        return "\n".join(lines)
    
    def cleanup_old_workflows(self, max_age_hours: int = 24):
        """Remove old workflow metrics to prevent memory bloat"""
        with self.lock:
            cutoff_time = time.time() - (max_age_hours * 3600)
            to_remove = [
                tid for tid, metrics in self.workflows.items()
                if metrics.completed_at and metrics.completed_at < cutoff_time
            ]
            
            for tid in to_remove:
                del self.workflows[tid]
            
            if to_remove:
                logger.info(f"[Metrics] Cleaned up {len(to_remove)} old workflow metrics")


# Singleton instance
_metrics_collector = None

def get_metrics_collector() -> MetricsCollector:
    """Get or create metrics collector singleton"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
