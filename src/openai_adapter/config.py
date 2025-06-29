"""
Configuration for the OpenAI service adapter.
"""

import os
from typing import Dict, Any

from src.config.config import Config

# Name of the OpenAI model for real-time services.
OPENAI_REALTIME_MODEL_NAME: str = os.getenv(
    "OPENAI_REALTIME_MODEL_NAME", "gpt-4o-mini-realtime-preview"
)

# Default initial session data for OpenAI.
OPENAI_SERVICE_INITIAL_SESSION_DATA: Dict[str, Any] = {"turn_detection": None}

# Default data for creating a response.
OPENAI_SERVICE_RESPONSE_CREATION_DATA: Dict[str, Any] = {
    "modalities": ["text", "audio"]
}

# Assembles the complete service configuration dictionary for OpenAI.
# This dictionary is imported by the bot's main entry point to be used in the factory.
OPENAI_SERVICE_CONFIG: Dict[str, Any] = {
    "api_key": Config.OPENAI_API_KEY,
    "model_name": OPENAI_REALTIME_MODEL_NAME,
    "initial_session_data": OPENAI_SERVICE_INITIAL_SESSION_DATA,
    "connection_timeout": Config.AI_SERVICE_CONNECTION_TIMEOUT,
    "response_creation_data": OPENAI_SERVICE_RESPONSE_CREATION_DATA,
    "processing_audio_frame_rate": 24000,
    "processing_audio_channels": 1,
    "response_audio_frame_rate": 24000,
    "response_audio_channels": 1,
}
