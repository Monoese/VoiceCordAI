
import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent


load_dotenv()


class Config:

    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")


    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    WS_SERVER_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17"


    CONNECTION_TIMEOUT = 15 * 60
    CHUNK_DURATION_MS = 500


    SAMPLE_WIDTH = 2
    DISCORD_FRAME_RATE = 96000
    TARGET_FRAME_RATE = 24000
    OUTPUT_FRAME_RATE = 48000
    CHANNELS = 1
    OUTPUT_CHANNELS = 2
