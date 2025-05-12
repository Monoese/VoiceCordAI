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
    # Base directory
    BASE_DIR: Path = Path(__file__).resolve().parent
    # Discord and OpenAI API settings
    DISCORD_TOKEN: Optional[str] = os.getenv("DISCORD_TOKEN")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    WS_SERVER_URL: str = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17"

    # Commands
    COMMAND_PREFIX: str = "/"

    # Connection settings
    CONNECTION_TIMEOUT: int = 15 * 60
    CHUNK_DURATION_MS: int = 500

    # Audio settings
    SAMPLE_WIDTH: int = 2
    DISCORD_FRAME_RATE: int = 96000
    TARGET_FRAME_RATE: int = 24000
    OUTPUT_FRAME_RATE: int = 48000
    CHANNELS: int = 1
    OUTPUT_CHANNELS: int = 2

    # Logging settings
    LOG_LEVEL: Union[int, str] = os.getenv("LOG_LEVEL", logging.INFO)
    LOG_CONSOLE_LEVEL: Union[int, str] = os.getenv("LOG_CONSOLE_LEVEL", logging.INFO)
    LOG_MAX_SIZE: int = int(os.getenv("LOG_MAX_SIZE", 5 * 1024 * 1024))
    LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", 3))

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration variables."""
        if not cls.DISCORD_TOKEN:
            raise ConfigError("DISCORD_TOKEN environment variable not set or empty.")
        if not cls.OPENAI_API_KEY:
            raise ConfigError("OPENAI_API_KEY environment variable not set or empty.")

# Validate configuration on import
Config.validate()
