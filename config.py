import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    # Base directory
    BASE_DIR = Path(__file__).resolve().parent
    # Discord and OpenAI API settings
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    WS_SERVER_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17"

    # Connection settings
    CONNECTION_TIMEOUT = 15 * 60
    CHUNK_DURATION_MS = 500

    # Audio settings
    SAMPLE_WIDTH = 2
    DISCORD_FRAME_RATE = 96000
    TARGET_FRAME_RATE = 24000
    OUTPUT_FRAME_RATE = 48000
    CHANNELS = 1
    OUTPUT_CHANNELS = 2

    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", logging.INFO)
    LOG_CONSOLE_LEVEL = os.getenv("LOG_CONSOLE_LEVEL", logging.INFO)
    LOG_MAX_SIZE = int(os.getenv("LOG_MAX_SIZE", 5 * 1024 * 1024))
    LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 3))
