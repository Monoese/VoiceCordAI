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

    # Internal Processing Audio Format (for internal processing, e.g., by OpenAI)
    PROCESSING_AUDIO_FRAME_RATE: int = 24000  # Samples per second, per channel
    PROCESSING_AUDIO_CHANNELS: int = 1  # Number of audio channels (e.g., 1 for mono)

    # --- Logging Configuration ---
    LOG_LEVEL: Union[int, str] = logging.INFO  # General log level for file logs
    LOG_CONSOLE_LEVEL: Union[int, str] = logging.INFO  # Log level for console output
    LOG_MAX_SIZE: int = (
        5 * 1024 * 1024
    )  # Max size of a log file before rotation (in bytes)
    LOG_BACKUP_COUNT: int = 3  # Number of backup log files to keep

    # --- OpenAI Service Configuration ---
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    # Name of the OpenAI model for real-time services.
    OPENAI_REALTIME_MODEL_NAME: str = os.getenv(
        "OPENAI_REALTIME_MODEL_NAME", "gpt-4o-mini-realtime-preview"
    )
    # WebSocket URL for OpenAI real-time services (note: includes specific model and date)
    WS_SERVER_URL: str = (
        "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17"
    )
    OPENAI_SERVICE_INITIAL_SESSION_DATA: Dict[str, Any] = {
        "turn_detection": None
    }  # Example initial config for OpenAI
    OPENAI_SERVICE_CONNECTION_TIMEOUT: float = 30.0  # Example timeout for connection
    # OPENAI_SERVICE_RESPONSE_CREATION_DATA: Dict[str, Any] = {"modalities": ["text", "audio"]} # Example for response creation
    # OpenAI Service Configuration Dictionary
    # This is defined here, within the class, using other class attributes defined above.
    OPENAI_SERVICE_CONFIG: Dict[str, Any] = {
        "api_key": OPENAI_API_KEY,
        "initial_session_data": OPENAI_SERVICE_INITIAL_SESSION_DATA,
        "connection_timeout": OPENAI_SERVICE_CONNECTION_TIMEOUT,
        # "response_creation_data": OPENAI_SERVICE_RESPONSE_CREATION_DATA # Uncomment if using the above
    }

    # --- Gemini Service Configuration ---
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_REALTIME_MODEL_NAME: str = os.getenv(
        "GEMINI_REALTIME_MODEL_NAME",
        "models/gemini-2.5-flash-preview-native-audio-dialog",  # Default from example
    )
    # Default LiveConnectConfig parameters, can be overridden by environment or specific needs
    GEMINI_DEFAULT_LIVE_CONNECT_CONFIG: Dict[str, Any] = {
        "response_modalities": ["AUDIO"],  # Expect audio responses
        "media_resolution": "MEDIA_RESOLUTION_MEDIUM",  # Default from example
        "speech_config": {  # Default from example
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Zephyr"}}
        },
        "context_window_compression": {  # Default from example
            "trigger_tokens": 25600,
            "sliding_window": {"target_tokens": 12800},
        },
    }
    # GEMINI_SERVICE_CONFIG can be customized further if needed, e.g., via JSON string in env var
    GEMINI_SERVICE_CONFIG: Dict[str, Any] = {
        "api_key": GEMINI_API_KEY,
        "model_name": GEMINI_REALTIME_MODEL_NAME,
        "live_connect_config": GEMINI_DEFAULT_LIVE_CONNECT_CONFIG,
        "connection_timeout": 30.0,  # Example connection timeout for Gemini
    }

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration variables."""
        if not cls.DISCORD_TOKEN:
            raise ConfigError("DISCORD_TOKEN environment variable not set or empty.")

        # Validate that AI_SERVICE_PROVIDER has a recognized value
        if cls.AI_SERVICE_PROVIDER not in ("openai", "gemini"):
            raise ConfigError(
                f"Unsupported AI_SERVICE_PROVIDER: {cls.AI_SERVICE_PROVIDER}. Must be 'openai' or 'gemini'."
            )

        # Ensure that at least one of the AI service API keys is configured
        if not cls.OPENAI_API_KEY and not cls.GEMINI_API_KEY:
            raise ConfigError(
                "Neither OPENAI_API_KEY nor GEMINI_API_KEY environment variables are set. "
                "At least one AI service API key is required."
            )

        # Validate that the selected AI_SERVICE_PROVIDER has its API key configured
        if cls.AI_SERVICE_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ConfigError(
                "AI_SERVICE_PROVIDER is set to 'openai', but OPENAI_API_KEY is missing."
            )
        elif cls.AI_SERVICE_PROVIDER == "gemini" and not cls.GEMINI_API_KEY:
            raise ConfigError(
                "AI_SERVICE_PROVIDER is set to 'gemini', but GEMINI_API_KEY is missing."
            )


Config.validate()  # Validate configuration on import
