"""
Utilities module for the Research Paper Answer Bot.

Provides logging setup, timing decorators, and string formatting helpers.
"""

import logging
import sys
import time
from functools import wraps
from typing import Any, Callable


def setup_logger(name: str = "research_paper_bot", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up and returns a configured logger.
    
    Args:
        name: Name of the logger.
        level: Logging level (e.g. logging.INFO).
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


logger = setup_logger()


def time_execution(func: Callable) -> Callable:
    """
    Decorator measuring execution time of a function in seconds.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logger.debug(f"Function '{func.__name__}' executed in {elapsed:.4f}s")
        return result
    return wrapper
