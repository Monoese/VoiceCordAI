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
        # Attempt to retrieve and validate configuration values from Config
        conf_log_level_value = Config.LOG_LEVEL
        conf_max_bytes_value = Config.LOG_MAX_SIZE
        conf_backup_count_value = Config.LOG_BACKUP_COUNT

        # Validate types for max_bytes and backup_count
        if not isinstance(conf_max_bytes_value, int):
            raise TypeError(
                f"Config.LOG_MAX_SIZE must be an int, got {type(conf_max_bytes_value)}"
            )
        if not isinstance(conf_backup_count_value, int):
            raise TypeError(
                f"Config.LOG_BACKUP_COUNT must be an int, got {type(conf_backup_count_value)}"
            )

        # Resolve log level to an integer value
        log_level_to_use: int
        if isinstance(conf_log_level_value, str):
            resolved_level_int = logging.getLevelName(conf_log_level_value.upper())
            if not isinstance(
                resolved_level_int, int
            ):  # Check if getLevelName found a valid level
                raise ValueError(
                    f"Invalid log level string from Config: '{conf_log_level_value}'"
                )
            log_level_to_use = resolved_level_int
        elif isinstance(conf_log_level_value, int):
            log_level_to_use = conf_log_level_value
        else:
            raise TypeError(
                f"Config.LOG_LEVEL must be a str or int, got {type(conf_log_level_value)}"
            )

        log_max_size_to_use = conf_max_bytes_value
        log_backup_count_to_use = conf_backup_count_value

    except (AttributeError, ValueError, TypeError) as e:
        # Log critical error to stderr as logger might not be configured
        print(
            f"CRITICAL: Logger configuration failed due to invalid or missing "
            f"settings in Config: {e}. Application cannot start.",
            file=sys.stderr,
        )
        # Raise a ConfigError to halt application startup
        raise ConfigError(f"Logger configuration failed: {e}") from e

    root_logger.setLevel(log_level_to_use)

    # File Handler for all logs (including dependencies)
    log_file = logs_dir / "discord_bot.log"
    file_handler = RotatingFileHandler(
        log_file,
        encoding="utf-8",
        maxBytes=log_max_size_to_use,
        backupCount=log_backup_count_to_use,
    )
    # Formatter for file logs
    formatter = logging.Formatter(
        "[{asctime}] [{levelname:<8}] {name}: {message}", "%Y-%m-%d %H:%M:%S", style="{"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(
        log_level_to_use
    )  # File handler logs at the effective log level
    root_logger.addHandler(file_handler)

    # Console logging is typically handled by discord.py's discord.utils.setup_logging(),
    # which provides colored output. If this app were not using discord.py's setup,
    # a StreamHandler for console output would be configured here.

    # Log the outcome of the successful configuration
    internal_logger = get_logger(
        __name__
    )  # Using the module's get_logger for consistency
    internal_logger.info(
        "Root logger configured successfully with file handler. Console logging managed by discord.py setup."
    )


_configure_root_logger()  # Configure root logger on import
