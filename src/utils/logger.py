"""
Centralized logging configuration for the application.

This module follows the recommended pattern of configuring handlers only on the root logger
and letting module-specific loggers inherit this configuration through propagation.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config.config import Config

logs_dir: Path = Path(Config.BASE_DIR) / "logs"
logs_dir.mkdir(exist_ok=True)  # Ensure logs directory exists


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
    root_logger = logging.getLogger()  # Get the root logger

    if root_logger.handlers:  # Avoid adding handlers multiple times
        return

    root_logger.setLevel(Config.LOG_LEVEL)  # Set root logger level

    # File Handler for all logs (including dependencies)
    log_file = logs_dir / "discord_bot.log"
    file_handler = RotatingFileHandler(
        log_file,
        encoding="utf-8",
        maxBytes=Config.LOG_MAX_SIZE,
        backupCount=Config.LOG_BACKUP_COUNT,
    )
    # Formatter for file logs
    formatter = logging.Formatter(
        "[{asctime}] [{levelname:<8}] {name}: {message}", "%Y-%m-%d %H:%M:%S", style="{"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(
        Config.LOG_LEVEL
    )  # File handler logs at the general LOG_LEVEL
    root_logger.addHandler(file_handler)

    # Console logging is typically handled by discord.py's discord.utils.setup_logging(),
    # which provides colored output. If this app were not using discord.py's setup,
    # a StreamHandler for console output would be configured here.
    # Example:
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter) # Or a simpler one for console
    # console_handler.setLevel(Config.LOG_CONSOLE_LEVEL)
    # root_logger.addHandler(console_handler)
    logger = get_logger(__name__)  # Use our own get_logger for this internal message
    logger.info(
        "Root logger configured with file handler. Console logging managed by discord.py setup."
    )


_configure_root_logger()  # Configure root logger on import
