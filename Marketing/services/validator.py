"""
Input/Output Validation with Pydantic Schemas
Provides type-safe validation and sanitization
"""
import re
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator, EmailStr
from enum import Enum


class WorkflowPriority(int, Enum):
    """Workflow priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class WorkflowRequest(BaseModel):
    """Validated request for workflow execution"""
    spreadsheet_id: str = Field(..., min_length=10, description="Google Spreadsheet ID")
    sheet_name: str = Field(default="Marketing Data", description="Sheet name")
    priority: WorkflowPriority = Field(default=WorkflowPriority.NORMAL, description="Execution priority")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('spreadsheet_id')
    def validate_spreadsheet_id(cls, v):
        """Validate spreadsheet ID format"""
        # Remove any potentially malicious characters
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Invalid spreadsheet ID format")
        return v
    
    @validator('sheet_name')
    def sanitize_sheet_name(cls, v):
        """Sanitize sheet name"""
        # Remove SQL injection patterns
        dangerous_patterns = ["';", "--", "/*", "*/", "drop", "delete", "update"]
        v_lower = v.lower()
        
        for pattern in dangerous_patterns:
            if pattern in v_lower:
                raise ValueError(f"Sheet name contains forbidden pattern: {pattern}")
        
        return v.strip()
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """Ensure metadata doesn't contain sensitive info"""
        if v:
            # Convert to string for checking
            metadata_str = str(v).lower()
            
            # Check for common patterns that shouldn't be in metadata
            forbidden = ["password", "secret", "api_key", "token"]
            for word in forbidden:
                if word in metadata_str:
                    raise ValueError(f"Metadata should not contain: {word}")
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "spreadsheet_id": "1b-KNdoqKl1gJs3BU4b3eiG_pHHA5jc5FQr7-Lg7faoA",
                "sheet_name": "Marketing Data",
                "priority": 5,
                "user_id": "user@example.com"
            }
        }


class AgentStats(BaseModel):
    """Validated agent execution statistics"""
    processed: int = Field(default=0, ge=0, description="Items processed")
    succeeded: int = Field(default=0, ge=0, description="Successful operations")
    failed: int = Field(default=0, ge=0, description="Failed operations")
    skipped: int = Field(default=0, ge=0, description="Skipped items")
    duration_ms: int = Field(default=0, ge=0, description="Execution duration in ms")
    
    @validator('succeeded', 'failed', 'skipped')
    def check_totals(cls, v, values):
        """Ensure totals add up"""
        if 'processed' in values:
            total = v + values.get('failed', 0) + values.get('skipped', 0)
            if total > values['processed']:
                raise ValueError("Sum of succeeded/failed/skipped cannot exceed processed")
        return v


class SecretaryOutput(BaseModel):
    """Validated secretary agent output"""
    emails_sent: int = Field(..., ge=0)
    emails_failed: int = Field(..., ge=0)
    total_processed: int = Field(..., ge=0)
    subject_lines: list[str] = Field(default_factory=list)
    
    @validator('subject_lines', each_item=True)
    def sanitize_subject(cls, v):
        """Ensure subject lines don't contain injection attempts"""
        # Remove potentially dangerous HTML/JS
        v = re.sub(r'<[^>]+>', '', v)
        return v[:200]  # Limit length


class AnalystOutput(BaseModel):
    """Validated analyst agent output"""
    rows_analyzed: int = Field(..., ge=0)
    high_score_count: int = Field(..., ge=0)
    average_score: float = Field(..., ge=0.0, le=10.0)
    risk_flags: list[str] = Field(default_factory=list)
    
    @validator('risk_flags', each_item=True)
    def sanitize_risk_flags(cls, v):
        """Sanitize risk flags"""
        # Remove HTML tags
        v = re.sub(r'<[^>]+>', '', v)
        return v[:500]


class WorkflowStatus(BaseModel):
    """Validated workflow status response"""
    thread_id: str
    status: str = Field(..., pattern="^(running|completed|failed|pending)$")
    progress_percent: int = Field(..., ge=0, le=100)
    current_step: str
    duration_ms: int = Field(..., ge=0)
    agents_completed: list[str] = Field(default_factory=list)
    error_count: int = Field(default=0, ge=0)
    last_updated: str


class InputSanitizer:
    """Utility class for input sanitization"""
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Remove HTML tags from text"""
        return re.sub(r'<[^>]+>', '', text)
    
    @staticmethod
    def remove_sql_injection_patterns(text: str) -> str:
        """Remove common SQL injection patterns"""
        dangerous = [
            r"';",
            r"--",
            r"/\*",
            r"\*/",
            r"\bDROP\b",
            r"\bDELETE\b",
            r"\bUPDATE\b",
            r"\bINSERT\b",
            r"\bEXEC\b"
        ]
        
        for pattern in dangerous:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def remove_script_tags(text: str) -> str:
        """Remove script tags and JavaScript"""
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)  # Remove event handlers
        return text
    
    @classmethod
    def sanitize_all(cls, text: str) -> str:
        """Apply all sanitization"""
        text = cls.remove_html_tags(text)
        text = cls.remove_sql_injection_patterns(text)
        text = cls.remove_script_tags(text)
        return text.strip()


def validate_workflow_request(data: Dict[str, Any]) -> WorkflowRequest:
    """
    Validate and sanitize workflow request
    
    Args:
        data: Raw request data
    
    Returns:
        Validated WorkflowRequest object
    
    Raises:
        ValidationError: If validation fails
    """
    return WorkflowRequest(**data)


def validate_and_sanitize_input(text: str) -> str:
    """
    Validate and sanitize text input
    
    Args:
        text: Raw text input
    
    Returns:
        Sanitized text
    """
    return InputSanitizer.sanitize_all(text)
