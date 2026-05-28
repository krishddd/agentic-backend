"""
Retry Handler with Exponential Backoff and Circuit Breaker
Provides resilient execution for external API calls
"""
import time
import random
from typing import Callable, TypeVar, Optional, Type, Tuple
from functools import wraps
from datetime import datetime, timedelta

from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
    
    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        if self.state == "half-open":
            self.state = "closed"
            logger.info("[CircuitBreaker] State: half-open -> closed")
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"[CircuitBreaker] OPENED after {self.failure_count} failures")
    
    def can_execute(self) -> bool:
        """Check if operation can proceed"""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.timeout_seconds:
                    self.state = "half-open"
                    logger.info("[CircuitBreaker] State: open -> half-open (attempting recovery)")
                    return True
            return False
        
        # Half-open: allow one attempt
        return True
    
    def is_open(self) -> bool:
        """Check if circuit is open"""
        return self.state == "open"


class RetryHandler:
    """Handles retry logic with exponential backoff"""
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        backoff_max: float = 30.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.jitter = jitter
        self.circuit_breaker = CircuitBreaker()
    
    def calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with optional jitter"""
        backoff = min(self.backoff_base ** attempt, self.backoff_max)
        
        if self.jitter:
            # Add jitter: random value between 0 and backoff
            backoff = random.uniform(0, backoff)
        
        return backoff
    
    def is_retryable_error(self, error: Exception) -> bool:
        """Determine if error is retryable"""
        # Retryable errors
        retryable_types = (
            ConnectionError,
            TimeoutError,
        )
        
        # Check by type
        if isinstance(error, retryable_types):
            return True
        
        # Check by error message patterns
        error_msg = str(error).lower()
        retryable_patterns = [
            "timeout",
            "connection",
            "rate limit",
            "429",
            "503",
            "502",
            "500"
        ]
        
        return any(pattern in error_msg for pattern in retryable_patterns)
    
    def execute_with_retry(
        self,
        func: Callable[[], T],
        operation_name: str = "operation",
        custom_exceptions: Tuple[Type[Exception], ...] = ()
    ) -> T:
        """
        Execute function with retry logic
        
        Args:
            func: Function to execute
            operation_name: Name for logging
            custom_exceptions: Additional exception types to retry
        
        Returns:
            Result from func()
        
        Raises:
            Exception: If all retries exhausted or circuit breaker open
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise CircuitBreakerOpen(f"Circuit breaker is open for {operation_name}")
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func()
                
                # Success - record and return
                self.circuit_breaker.record_success()
                
                if attempt > 0:
                    logger.info(f"[RetryHandler] {operation_name} succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if retryable
                is_retryable = (
                    self.is_retryable_error(e) or
                    (custom_exceptions and isinstance(e, custom_exceptions))
                )
                
                if not is_retryable or attempt >= self.max_retries:
                    # Not retryable or exhausted retries
                    logger.error(f"[RetryHandler] {operation_name} failed (attempt {attempt + 1}): {e}")
                    self.circuit_breaker.record_failure()
                    raise
                
                # Calculate backoff
                backoff = self.calculate_backoff(attempt)
                logger.warning(
                    f"[RetryHandler] {operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}). "
                    f"Retrying in {backoff:.2f}s... Error: {e}"
                )
                
                time.sleep(backoff)
        
        # Should never reach here, but just in case
        self.circuit_breaker.record_failure()
        raise last_exception or Exception(f"{operation_name} failed after all retries")
    
    def circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        return self.circuit_breaker.is_open()


# Decorator for easy retry
def with_retry(
    max_retries: int = 3,
    operation_name: Optional[str] = None
):
    """Decorator to add retry logic to functions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = RetryHandler(max_retries=max_retries)
            op_name = operation_name or func.__name__
            return handler.execute_with_retry(
                lambda: func(*args, **kwargs),
                operation_name=op_name
            )
        return wrapper
    return decorator


# Singleton instance
_retry_handler = None

def get_retry_handler() -> RetryHandler:
    """Get or create retry handler singleton"""
    global _retry_handler
    if _retry_handler is None:
        _retry_handler = RetryHandler()
    return _retry_handler
