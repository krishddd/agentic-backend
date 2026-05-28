"""Distributed tracing service with nested spans for multi-step agent workflows."""

import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
from dataclasses import dataclass, field, asdict

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Span:
    """Represents a single span in a trace (OpenTelemetry-style)."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    name: str
    kind: str  # "agent", "task", "tool", "llm", "rag"
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "in_progress"  # "in_progress", "success", "error"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    def end(self, status: str = "success", error: Optional[str] = None):
        """End the span."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        self.error = error
    
    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })
    
    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value


@dataclass
class Trace:
    """Represents a complete trace with multiple spans."""
    trace_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "spans": [asdict(span) for span in self.spans]
        }


class DistributedTracer:
    """Distributed tracing service with nested span support (LangSmith/Phoenix style)."""
    
    def __init__(self):
        """Initialize the tracer."""
        self.current_trace: Optional[Trace] = None
        self.active_spans: Dict[str, Span] = {}
        self._span_stack: List[str] = []  # Track parent-child relationships
    
    def start_trace(self, name: str, metadata: Dict[str, Any] = None) -> str:
        """Start a new trace.
        
        Args:
            name: Trace name (e.g., "financial-crew-analysis")
            metadata: Additional trace metadata
            
        Returns:
            trace_id
        """
        trace_id = str(uuid.uuid4())
        self.current_trace = Trace(
            trace_id=trace_id,
            name=name,
            start_time=time.time(),
            metadata=metadata or {}
        )
        logger.info(f"Started trace: {trace_id} - {name}")
        return trace_id
    
    def start_span(self, 
                   name: str, 
                   kind: str,
                   attributes: Dict[str, Any] = None,
                   parent_span_id: Optional[str] = None) -> str:
        """Start a new span (nested operation).
        
        Args:
            name: Span name (e.g., "research_task", "web_search_tool")
            kind: Span kind ("agent", "task", "tool", "llm", "rag")
            attributes: Initial span attributes
            parent_span_id: Parent span ID (auto-detected if None)
            
        Returns:
            span_id
        """
        if not self.current_trace:
            raise ValueError("No active trace. Call start_trace() first.")
        
        span_id = str(uuid.uuid4())
        
        # Auto-detect parent if not specified
        if parent_span_id is None and self._span_stack:
            parent_span_id = self._span_stack[-1]
        
        span = Span(
            span_id=span_id,
            trace_id=self.current_trace.trace_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            start_time=time.time(),
            attributes=attributes or {}
        )
        
        self.active_spans[span_id] = span
        self._span_stack.append(span_id)
        self.current_trace.spans.append(span)
        
        logger.debug(f"Started span: {span_id} - {name} (parent: {parent_span_id})")
        return span_id
    
    def end_span(self, span_id: str, status: str = "success", error: Optional[str] = None):
        """End a span.
        
        Args:
            span_id: Span ID to end
            status: "success" or "error"
            error: Error message if status is "error"
        """
        if span_id not in self.active_spans:
            logger.warning(f"Span {span_id} not found in active spans")
            return
        
        span = self.active_spans[span_id]
        span.end(status=status, error=error)
        
        # Remove from active spans and stack
        del self.active_spans[span_id]
        if span_id in self._span_stack:
            self._span_stack.remove(span_id)
        
        logger.debug(f"Ended span: {span_id} - {span.name} ({status})")
    
    def add_span_event(self, span_id: str, event_name: str, attributes: Dict[str, Any] = None):
        """Add an event to a span.
        
        Args:
            span_id: Span ID
            event_name: Event name
            attributes: Event attributes
        """
        if span_id in self.active_spans:
            self.active_spans[span_id].add_event(event_name, attributes)
    
    def set_span_attribute(self, span_id: str, key: str, value: Any):
        """Set a span attribute.
        
        Args:
            span_id: Span ID
            key: Attribute key
            value: Attribute value
        """
        if span_id in self.active_spans:
            self.active_spans[span_id].set_attribute(key, value)
    
    def end_trace(self) -> Optional[Trace]:
        """End the current trace and return it.
        
        Returns:
            Completed Trace object
        """
        if not self.current_trace:
            return None
        
        self.current_trace.end_time = time.time()
        self.current_trace.duration_ms = (
            (self.current_trace.end_time - self.current_trace.start_time) * 1000
        )
        
        # Close any remaining active spans
        for span_id in list(self.active_spans.keys()):
            self.end_span(span_id, status="error", error="Span not properly closed")
        
        trace = self.current_trace
        self.current_trace = None
        self._span_stack.clear()
        
        logger.info(f"Ended trace: {trace.trace_id} ({trace.duration_ms:.2f}ms, {len(trace.spans)} spans)")
        return trace
    
    def save_trace(self, trace: Trace, output_dir: Path):
        """Save trace to JSON file.
        
        Args:
            trace: Trace to save
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"trace_{trace.trace_id}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trace.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved trace to {filepath}")
        return filepath
    
    def visualize_trace_tree(self, trace: Trace) -> str:
        """Generate ASCII tree visualization of trace.
        
        Args:
            trace: Trace to visualize
            
        Returns:
            ASCII tree string
        """
        lines = [f"Trace: {trace.name} ({trace.duration_ms:.2f}ms)"]
        
        # Build span hierarchy
        span_by_id = {span.span_id: span for span in trace.spans}
        root_spans = [s for s in trace.spans if s.parent_span_id is None]
        
        def render_span(span: Span, indent: int = 0):
            """Recursively render span and children."""
            prefix = "  " * indent + ("└─ " if indent > 0 else "")
            status_icon = "✓" if span.status == "success" else "✗"
            duration = f"{span.duration_ms:.2f}ms" if span.duration_ms else "in progress"
            
            lines.append(f"{prefix}{status_icon} [{span.kind}] {span.name} ({duration})")
            
            # Add events if any
            for event in span.events:
                event_prefix = "  " * (indent + 1) + "• "
                lines.append(f"{event_prefix}{event['name']}")
            
            # Render children
            children = [s for s in trace.spans if s.parent_span_id == span.span_id]
            for child in children:
                render_span(child, indent + 1)
        
        # Render all root spans
        for root_span in root_spans:
            render_span(root_span)
        
        return "\n".join(lines)


# Global tracer instance
_global_tracer: Optional[DistributedTracer] = None


def get_tracer() -> DistributedTracer:
    """Get global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = DistributedTracer()
    return _global_tracer
