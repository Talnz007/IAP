"""
Logging utilities for the AQI Predictor.
"""
import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import get_config


def setup_logger(
    name: str = "aqi_predictor",
    level: str = None,
    log_file: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Whether to also log to file
        
    Returns:
        Configured logger
    """
    config = get_config()
    
    if level is None:
        level = config.app.log_level
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_dir = config.logs_dir
        log_dir.mkdir(exist_ok=True)
        
        log_filename = f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_dir / log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "aqi_predictor") -> logging.Logger:
    """
    Get or create a logger.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


class LoggerMixin:
    """Mixin class to add logging capability to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get the logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


# Create default logger
default_logger = setup_logger()


def log_info(message: str):
    """Log an info message."""
    default_logger.info(message)


def log_warning(message: str):
    """Log a warning message."""
    default_logger.warning(message)


def log_error(message: str):
    """Log an error message."""
    default_logger.error(message)


def log_debug(message: str):
    """Log a debug message."""
    default_logger.debug(message)
