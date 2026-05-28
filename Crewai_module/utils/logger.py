"""Production-grade logging configuration."""

import sys
from pathlib import Path
from loguru import logger
from config.settings import settings


def setup_logger():
    """Configure loguru logger with file rotation and formatting."""
    
    # Remove default handler
    logger.remove()
    
    # Console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )
    
    # File handler with rotation
    logger.add(
        settings.logs_dir / "financial_crew_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="500 MB",
        retention="30 days",
        compression="zip",
    )
    
    # Error file handler
    logger.add(
        settings.logs_dir / "errors_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="100 MB",
        retention="60 days",
        compression="zip",
    )
    
    return logger


# Initialize logger
_logger = setup_logger()


def get_logger(name: str = None):
    """Get logger instance with optional name binding."""
    if name:
        return _logger.bind(name=name)
    return _logger
