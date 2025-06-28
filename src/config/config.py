"""
Application configuration module.

This module defines the Config class, which centralizes all
application-wide settings, such as API keys, server URLs,
audio processing parameters, and logging configurations.
It loads values from environment variables and provides
validation for required settings.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any

from dotenv import load_dotenv

load_dotenv()


class ConfigError(Exception):
    """Custom exception for configuration errors."""

    pass


class Config:
    """
    Centralized configuration settings for the application.

    This class holds all static configuration parameters, loaded primarily
    from environment variables. It includes settings for API keys,
    server connections, audio processing, bot behavior, and logging.
    The `validate` method ensures that critical configurations are present.
    """

    BASE_DIR: Path = (
        Path(__file__).resolve().parent.parent.parent
    )  # Base directory of the project

    # --- Core Bot Settings ---
    DISCORD_TOKEN: Optional[str] = os.getenv("DISCORD_TOKEN")
    COMMAND_PREFIX: str = os.getenv("COMMAND_PREFIX", "/")  # Prefix for bot commands
    # Determines which AI service manager to use ("openai" or "gemini")
    AI_SERVICE_PROVIDER: str = os.getenv("AI_SERVICE_PROVIDER", "openai").lower()

    # --- Voice & Connection Settings ---
    CONNECTION_TIMEOUT: int = (
        15 * 60
    )  # Timeout for voice channel connections in seconds
    AI_SERVICE_CONNECTION_TIMEOUT: float = 30.0  # Timeout for AI service connections
    CHUNK_DURATION_MS: int = 500  # Duration of audio chunks in milliseconds

    # --- Audio Processing Settings ---
    # General Audio
    SAMPLE_WIDTH: int = 2  # Sample width in bytes (e.g., 2 for 16-bit audio)
    FFMPEG_PCM_FORMAT: str = (
        "s16le"  # FFmpeg format string for PCM data (signed 16-bit little-endian)
    )

    # Discord Audio Format (for audio received from and sent to Discord)
    DISCORD_AUDIO_FRAME_RATE: int = 48000  # Samples per second, per channel
    DISCORD_AUDIO_CHANNELS: int = 2  # Number of audio channels (e.g., 2 for stereo)

    # --- Logging Configuration ---
    LOG_LEVEL: Union[int, str] = logging.INFO  # General log level for file logs
    LOG_CONSOLE_LEVEL: Union[int, str] = logging.INFO  # Log level for console output
    LOG_MAX_SIZE: int = (
        5 * 1024 * 1024
    )  # Max size of a log file before rotation (in bytes)
    LOG_BACKUP_COUNT: int = 3  # Number of backup log files to keep

    # --- API Keys ---
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration variables.

        This method checks that all critical configuration variables are set
        and have valid values. It is called automatically when the module
        is imported.

        Raises:
            ConfigError: If a required configuration is missing or invalid.
        """
        if not cls.DISCORD_TOKEN:
            raise ConfigError("DISCORD_TOKEN environment variable not set or empty.")

        # Validate that AI_SERVICE_PROVIDER has a recognized value
        if cls.AI_SERVICE_PROVIDER not in ("openai", "gemini"):
            raise ConfigError(
                f"Unsupported AI_SERVICE_PROVIDER: {cls.AI_SERVICE_PROVIDER}. Must be 'openai' or 'gemini'."
            )

        # Validate logging configuration
        if not isinstance(cls.LOG_MAX_SIZE, int):
            raise ConfigError(
                f"LOG_MAX_SIZE must be an int, but got {type(cls.LOG_MAX_SIZE).__name__}."
            )
        if not isinstance(cls.LOG_BACKUP_COUNT, int):
            raise ConfigError(
                f"LOG_BACKUP_COUNT must be an int, but got {type(cls.LOG_BACKUP_COUNT).__name__}."
            )

        if isinstance(cls.LOG_LEVEL, str):
            # Check if the string is a valid log level name
            if not isinstance(logging.getLevelName(cls.LOG_LEVEL.upper()), int):
                raise ConfigError(
                    f"Invalid log level string from Config: '{cls.LOG_LEVEL}'"
                )
        elif not isinstance(cls.LOG_LEVEL, int):
            raise ConfigError(
                f"LOG_LEVEL must be a str or int, but got {type(cls.LOG_LEVEL).__name__}."
            )


Config.validate()  # Validate configuration on import
