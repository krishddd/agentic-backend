"""Input validation utilities."""

import re
from datetime import datetime
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)


def validate_ticker(ticker: str) -> str:
    """Validate and normalize stock ticker symbol.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Normalized ticker (uppercase, trimmed)
    
    Raises:
        ValueError: If ticker is invalid
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker must be a non-empty string")
    
    ticker = ticker.strip().upper()
    
    # Basic validation: 1-5 alphanumeric characters
    if not re.match(r'^[A-Z]{1,5}$', ticker):
        raise ValueError(
            f"Invalid ticker format: {ticker}. "
            "Ticker should be 1-5 uppercase letters (e.g., TSLA, AAPL)"
        )
    
    logger.debug(f"Validated ticker: {ticker}")
    return ticker


def validate_date(date_str: str, date_format: str = "%Y-%m-%d") -> datetime:
    """Validate and parse date string.
    
    Args:
        date_str: Date string to validate
        date_format: Expected date format (default: YYYY-MM-DD)
    
    Returns:
        Parsed datetime object
    
    Raises:
        ValueError: If date is invalid
    """
    try:
        date_obj = datetime.strptime(date_str, date_format)
        logger.debug(f"Validated date: {date_str}")
        return date_obj
    except ValueError as e:
        raise ValueError(
            f"Invalid date format: {date_str}. "
            f"Expected format: {date_format} (e.g., 2024-01-15)"
        ) from e


def validate_email(email: str) -> bool:
    """Validate email address format.
    
    Args:
        email: Email address to validate
    
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    is_valid = bool(re.match(pattern, email))
    
    if is_valid:
        logger.debug(f"Validated email: {email}")
    else:
        logger.warning(f"Invalid email format: {email}")
    
    return is_valid


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename safe for filesystem
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    # Limit length
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    
    logger.debug(f"Sanitized filename: {filename} -> {sanitized}")
    return sanitized
