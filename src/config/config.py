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
from typing import Optional, Union

from dotenv import load_dotenv

from src.exceptions import ConfigurationError

load_dotenv()


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
    CONNECTION_CHECK_INTERVAL: float = float(
        os.getenv("CONNECTION_CHECK_INTERVAL", "10.0")
    )
    AI_SERVICE_CONNECTION_TIMEOUT: float = 30.0  # Timeout for AI service connections
    CHUNK_DURATION_MS: int = 500  # Duration of audio chunks in milliseconds

    # --- UI/UX Settings ---
    REACTION_GRANT_CONSENT: str = os.getenv("REACTION_GRANT_CONSENT", "🙏")
    REACTION_MODE_MANUAL: str = os.getenv("REACTION_MODE_MANUAL", "🙋")
    REACTION_MODE_REALTIME: str = os.getenv("REACTION_MODE_REALTIME", "🗣️")
    REACTION_TRIGGER_PTT: str = os.getenv("REACTION_TRIGGER_PTT", "🎙️")

    # --- Audio Cue Paths ---
    AUDIO_CUE_START_RECORDING: str = str(
        BASE_DIR / "assets/audio_cues/start_recording.mp3"
    )
    AUDIO_CUE_END_RECORDING: str = str(BASE_DIR / "assets/audio_cues/end_recording.mp3")

    # --- Audio Processing Settings ---
    # General Audio
    SAMPLE_WIDTH: int = 2  # Sample width in bytes (e.g., 2 for 16-bit audio)
    FFMPEG_PCM_FORMAT: str = (
        "s16le"  # FFmpeg format string for PCM data (signed 16-bit little-endian)
    )

    # Discord Audio Format (for audio received from and sent to Discord)
    DISCORD_AUDIO_FRAME_RATE: int = 48000  # Samples per second, per channel
    DISCORD_AUDIO_CHANNELS: int = 2  # Number of audio channels (e.g., 2 for stereo)

    # --- Wake Word & VAD Settings ---
    # openWakeWord settings
    WAKE_WORD_MODEL_PATH: str = str(BASE_DIR / "assets/wakeword_models/alexa_v0.1.onnx")
    WAKE_WORD_THRESHOLD: float = 0.5  # Confidence threshold for detection
    # VAD inside openWakeWord to improve ww accuracy.
    WAKE_WORD_VAD_THRESHOLD: float = 0.5
    WAKE_WORD_SAMPLE_RATE: int = 16000  # Sample rate for wake word model (Hz)

    # webrtcvad settings for end-of-speech detection
    VAD_SAMPLE_RATE: int = 16000  # Sample rate for VAD processing (Hz)
    VAD_FRAME_DURATION_MS: int = 30  # Frame duration for VAD (10, 20, or 30)
    VAD_AGGRESSIVENESS: int = 1  # VAD aggressiveness (0-3), 1 is a good balance
    VAD_GRACE_PERIOD_MS: int = (
        3000  # VAD will be ignored for this long after recording starts
    )
    VAD_MIN_SPEECH_DURATION_MS: int = 250  # Min speech to trigger recording stop
    VAD_SILENCE_TIMEOUT_MS: int = 1000  # Silence after speech to stop recording

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

        Returns:
            None.

        Raises:
            ConfigurationError: If a required configuration is missing or invalid.
        """
        if not cls.DISCORD_TOKEN:
            raise ConfigurationError(
                "DISCORD_TOKEN environment variable not set or empty."
            )

        # Validate that AI_SERVICE_PROVIDER has a recognized value
        if cls.AI_SERVICE_PROVIDER not in ("openai", "gemini"):
            raise ConfigurationError(
                f"Unsupported AI_SERVICE_PROVIDER: {cls.AI_SERVICE_PROVIDER}. Must be 'openai' or 'gemini'."
            )

        # Validate logging configuration
        if not isinstance(cls.LOG_MAX_SIZE, int):
            raise ConfigurationError(
                f"LOG_MAX_SIZE must be an int, but got {type(cls.LOG_MAX_SIZE).__name__}."
            )
        if not isinstance(cls.LOG_BACKUP_COUNT, int):
            raise ConfigurationError(
                f"LOG_BACKUP_COUNT must be an int, but got {type(cls.LOG_BACKUP_COUNT).__name__}."
            )

        if isinstance(cls.LOG_LEVEL, str):
            # Check if the string is a valid log level name
            if not isinstance(logging.getLevelName(cls.LOG_LEVEL.upper()), int):
                raise ConfigurationError(
                    f"Invalid log level string from Config: '{cls.LOG_LEVEL}'"
                )
        elif not isinstance(cls.LOG_LEVEL, int):
            raise ConfigurationError(
                f"LOG_LEVEL must be a str or int, but got {type(cls.LOG_LEVEL).__name__}."
            )

        # Validate VAD & Wake Word settings
        if cls.VAD_AGGRESSIVENESS not in (0, 1, 2, 3):
            raise ConfigurationError(
                "VAD_AGGRESSIVENESS must be an integer from 0 to 3."
            )
        if cls.VAD_FRAME_DURATION_MS not in (10, 20, 30):
            raise ConfigurationError("VAD_FRAME_DURATION_MS must be 10, 20, or 30.")
        if not (0.0 <= cls.WAKE_WORD_THRESHOLD <= 1.0):
            raise ConfigurationError("WAKE_WORD_THRESHOLD must be between 0.0 and 1.0.")
        if not (0.0 <= cls.WAKE_WORD_VAD_THRESHOLD <= 1.0):
            raise ConfigurationError(
                "WAKE_WORD_VAD_THRESHOLD must be between 0.0 and 1.0."
            )


Config.validate()  # Validate configuration on import
