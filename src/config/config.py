import logging
import os
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv

load_dotenv()


class ConfigError(Exception):
    """Custom exception for configuration errors."""

    pass


class Config:
    BASE_DIR: Path = (
        Path(__file__).resolve().parent.parent.parent
    )  # Base directory of the project

    # API and Server Settings
    DISCORD_TOKEN: Optional[str] = os.getenv("DISCORD_TOKEN")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    WS_SERVER_URL: str = (  # WebSocket server URL for real-time processing
        "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17"
    )

    # Bot Command Settings
    COMMAND_PREFIX: str = "/"  # Prefix for bot commands

    # Connection and Timing Settings
    CONNECTION_TIMEOUT: int = (
        15 * 60
    )  # Timeout for voice channel connections in seconds
    CHUNK_DURATION_MS: int = 500  # Duration of audio chunks in milliseconds

    # Audio Processing Settings
    SAMPLE_WIDTH: int = 2  # Sample width in bytes (e.g., 2 for 16-bit audio)
    FFMPEG_PCM_FORMAT: str = (
        "s16le"  # FFmpeg format string for PCM data (signed 16-bit little-endian)
    )

    # Discord Audio Format (for audio received from and sent to Discord)
    DISCORD_AUDIO_FRAME_RATE: int = 48000  # Samples per second, per channel
    DISCORD_AUDIO_CHANNELS: int = 2  # Number of audio channels (e.g., 2 for stereo)

    # Processing Audio Format (for internal processing, e.g., by OpenAI)
    PROCESSING_AUDIO_FRAME_RATE: int = 24000  # Samples per second, per channel
    PROCESSING_AUDIO_CHANNELS: int = 1  # Number of audio channels (e.g., 1 for mono)

    # Logging Configuration
    LOG_LEVEL: Union[int, str] = logging.INFO  # General log level for file logs
    LOG_CONSOLE_LEVEL: Union[int, str] = logging.INFO  # Log level for console output
    LOG_MAX_SIZE: int = (
        5 * 1024 * 1024
    )  # Max size of a log file before rotation (in bytes)
    LOG_BACKUP_COUNT: int = 3  # Number of backup log files to keep

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration variables."""
        if not cls.DISCORD_TOKEN:
            raise ConfigError("DISCORD_TOKEN environment variable not set or empty.")
        if not cls.OPENAI_API_KEY:
            raise ConfigError("OPENAI_API_KEY environment variable not set or empty.")


Config.validate()  # Validate configuration on import
