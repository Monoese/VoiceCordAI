"""
Centralized logging configuration for the application.

This module follows the recommended pattern of configuring handlers only on the root logger
and letting module-specific loggers inherit this configuration through propagation.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config.config import Config, ConfigError

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
    Configure the root logger with a file handler.
    Console logging is expected to be handled by discord.py.
    This should be called only once during application startup.
    """
    root_logger = logging.getLogger()

    if root_logger.handlers:  # Avoid adding handlers multiple times
        return

    try:
        # Resolve log level. Validation is now handled by the Config class.
        log_level_value = Config.LOG_LEVEL
        log_level = (
            logging.getLevelName(log_level_value.upper())
            if isinstance(log_level_value, str)
            else log_level_value
        )
    except Exception as e:
        # Fallback error handler for unexpected issues during logger setup.
        print(f"CRITICAL: Logger setup failed unexpectedly: {e}", file=sys.stderr)
        raise ConfigError(f"Logger setup failed: {e}") from e

    root_logger.setLevel(log_level)

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
    file_handler.setLevel(log_level)  # File handler logs at the effective log level
    root_logger.addHandler(file_handler)

    # Console logging is typically handled by discord.py's discord.utils.setup_logging(),
    # which provides colored output. If this app were not using discord.py's setup,
    # a StreamHandler for console output would be configured here.

    # Log the outcome of the successful configuration
    internal_logger = get_logger(__name__)
    internal_logger.info(
        "Root logger configured successfully with file handler. Console logging managed by discord.py setup."
    )


_configure_root_logger()  # Configure root logger on import
