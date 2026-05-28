"""
Audit Logger - Immutable audit trail with PII masking
Provides compliance-ready logging for all agent actions
"""
import re
import json
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class PIIMasker:
    """Mask personally identifiable information"""
    
    # Regex patterns for PII detection
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    PHONE_PATTERN = r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
    SSN_PATTERN = r'\b\d{3}-\d{2}-\d{4}\b'
    CREDIT_CARD_PATTERN = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    
    @staticmethod
    def mask_email(text: str) -> str:
        """Mask email addresses"""
        return re.sub(
            PIIMasker.EMAIL_PATTERN,
            lambda m: f"{m.group(0)[:3]}***@{m.group(0).split('@')[1]}",
            text
        )
    
    @staticmethod
    def mask_phone(text: str) -> str:
        """Mask phone numbers"""
        return re.sub(PIIMasker.PHONE_PATTERN, 'XXX-XXX-XXXX', text)
    
    @staticmethod
    def mask_ssn(text: str) -> str:
        """Mask SSN"""
        return re.sub(PIIMasker.SSN_PATTERN, 'XXX-XX-XXXX', text)
    
    @staticmethod
    def mask_credit_card(text: str) -> str:
        """Mask credit card numbers"""
        return re.sub(PIIMasker.CREDIT_CARD_PATTERN, 'XXXX-XXXX-XXXX-XXXX', text)
    
    @classmethod
    def mask_all(cls, text: str) -> str:
        """Apply all PII masking"""
        if not isinstance(text, str):
            text = str(text)
        
        text = cls.mask_email(text)
        text = cls.mask_phone(text)
        text = cls.mask_ssn(text)
        text = cls.mask_credit_card(text)
        
        return text
    
    @classmethod
    def mask_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively mask PII in dictionary"""
        masked = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                masked[key] = cls.mask_all(value)
            elif isinstance(value, dict):
                masked[key] = cls.mask_dict(value)
            elif isinstance(value, list):
                masked[key] = [
                    cls.mask_dict(item) if isinstance(item, dict) else cls.mask_all(str(item))
                    for item in value
                ]
            else:
                masked[key] = value
        
        return masked


class AuditLogger:
    """Audit logger with immutable trail"""
    
    def __init__(self):
        self.audit_dir = Path(settings.LOGS_DIR) / "audit"
        self.audit_dir.mkdir(exist_ok=True)
        self.current_date = datetime.now().strftime("%Y%m%d")
        self.audit_file = self.audit_dir / f"audit_{self.current_date}.jsonl"
    
    def _create_audit_entry(
        self,
        event_type: str,
        action: str,
        resource: str,
        data: Dict[str, Any],
        success: bool,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create standardized audit entry"""
        # Mask PII in data
        masked_data = PIIMasker.mask_dict(data) if data else {}
        
        # Create data hash for integrity
        data_str = json.dumps(masked_data, sort_keys=True)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "action": action,
            "resource": resource,
            "success": success,
            "data_hash": data_hash,
            "metadata": masked_data
        }
        
        if error:
            entry["error"] = PIIMasker.mask_all(error)
        
        # Add entry hash for tamper detection
        entry_str = json.dumps(entry, sort_keys=True)
        entry["entry_hash"] = hashlib.sha256(entry_str.encode()).hexdigest()
        
        return entry
    
    def _write_audit_log(self, entry: Dict[str, Any]):
        """Write to append-only audit log"""
        try:
            # Check if date rolled over
            current_date = datetime.now().strftime("%Y%m%d")
            if current_date != self.current_date:
                self.current_date = current_date
                self.audit_file = self.audit_dir / f"audit_{current_date}.jsonl"
            
            # Append to file (JSONL format for easy parsing)
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        
        except Exception as e:
            logger.error(f"[AuditLogger] Failed to write audit log: {e}")
    
    def log_workflow_start(
        self,
        thread_id: str,
        spreadsheet_id: str,
        sheet_name: str,
        user_id: Optional[str] = None
    ):
        """Log workflow initiation"""
        entry = self._create_audit_entry(
            event_type="workflow",
            action="start",
            resource=f"workflow:{thread_id}",
            data={
                "thread_id": thread_id,
                "spreadsheet_id": spreadsheet_id,
                "sheet_name": sheet_name,
                "user_id": user_id or "system"
            },
            success=True
        )
        
        self._write_audit_log(entry)
        logger.info(f"[Audit] Workflow started: {thread_id}")
    
    def log_workflow_complete(
        self,
        thread_id: str,
        success: bool,
        duration_ms: int,
        stats: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Log workflow completion"""
        entry = self._create_audit_entry(
            event_type="workflow",
            action="complete",
            resource=f"workflow:{thread_id}",
            data={
                "thread_id": thread_id,
                "duration_ms": duration_ms,
                "stats": stats or {}
            },
            success=success,
            error=error
        )
        
        self._write_audit_log(entry)
        status = "succeeded" if success else "failed"
        logger.info(f"[Audit] Workflow {status}: {thread_id}")
    
    def log_agent_action(
        self,
        thread_id: str,
        agent_name: str,
        action: str,
        stats: Dict[str, Any],
        success: bool,
        error: Optional[str] = None
    ):
        """Log agent action"""
        entry = self._create_audit_entry(
            event_type="agent",
            action=action,
            resource=f"agent:{agent_name}",
            data={
                "thread_id": thread_id,
                "agent": agent_name,
                "stats": stats
            },
            success=success,
            error=error
        )
        
        self._write_audit_log(entry)
        logger.info(f"[Audit] Agent {agent_name} {action}: {thread_id}")
    
    def log_data_access(
        self,
        thread_id: str,
        operation: str,  # read, write, update
        resource: str,  # spreadsheet_id:sheet_name
        row_count: int,
        success: bool,
        error: Optional[str] = None
    ):
        """Log data access (sheet reads/writes)"""
        entry = self._create_audit_entry(
            event_type="data_access",
            action=operation,
            resource=resource,
            data={
                "thread_id": thread_id,
                "row_count": row_count
            },
            success=success,
            error=error
        )
        
        self._write_audit_log(entry)
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,  # low, medium, high, critical
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log security-related events"""
        entry = self._create_audit_entry(
            event_type="security",
            action=event_type,
            resource="system",
            data={
                "severity": severity,
                "description": PIIMasker.mask_all(description),
                "metadata": metadata or {}
            },
            success=True
        )
        
        self._write_audit_log(entry)
        logger.warning(f"[Audit] Security event ({severity}): {event_type}")
    
    def log_llm_usage(
        self,
        thread_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float
    ):
        """Log LLM API usage for cost tracking"""
        entry = self._create_audit_entry(
            event_type="llm_usage",
            action="api_call",
            resource=f"model:{model}",
            data={
                "thread_id": thread_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": cost_usd
            },
            success=True
        )
        
        self._write_audit_log(entry)


# Singleton instance
_audit_logger = None

def get_audit_logger() -> AuditLogger:
    """Get or create audit logger singleton"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
