"""
Centralized logging configuration for the application.

This module follows the recommended pattern of configuring handlers only on the root logger
and letting module-specific loggers inherit this configuration through propagation.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config.config import Config

# Create logs directory if it doesn't exist
logs_dir: Path = Path(Config.BASE_DIR) / "logs"
logs_dir.mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.

    This function just returns a logger for the given name without configuring
    any handlers. The root logger is configured once at import time,
    and all other loggers inherit from it.

    Args:
        name: Usually __name__ of the calling module

    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)


def _configure_root_logger() -> None:
    """
    Configure the root logger with file and console handlers.
    This should be called only once during application startup.
    """
    # Get the root logger
    root_logger = logging.getLogger()

    # If handlers already exist, assume it's already configured
    if root_logger.handlers:
        return

    # Set level based on config
    root_logger.setLevel(Config.LOG_LEVEL)

    # Create a rotating file handler for all logs
    log_file = logs_dir / "discord_bot.log"
    file_handler = RotatingFileHandler(
        log_file,
        encoding="utf-8",
        maxBytes=Config.LOG_MAX_SIZE,
        backupCount=Config.LOG_BACKUP_COUNT,
    )

    # Use a formatter compatible with discord.py's style for consistency
    formatter = logging.Formatter(
        "[{asctime}] [{levelname:<8}] {name}: {message}", "%Y-%m-%d %H:%M:%S", style="{"
    )

    file_handler.setFormatter(formatter)
    file_handler.setLevel(Config.LOG_LEVEL)
    root_logger.addHandler(file_handler)

    # Note: We don't add a console handler here because discord.py's  # setup_logging() will handle that with nice colors


# Configure the root logger immediately on import
_configure_root_logger()
