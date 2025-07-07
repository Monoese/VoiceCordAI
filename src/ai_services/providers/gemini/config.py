"""
Configuration for the Gemini service adapter.
"""

import os
from typing import Dict, Any

from src.config.config import Config

# Name of the Gemini model for real-time services.
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

# Assembles the complete service configuration dictionary for Gemini.
# This dictionary is imported by the bot's main entry point to be used in the factory.
GEMINI_SERVICE_CONFIG: Dict[str, Any] = {
    "api_key": Config.GEMINI_API_KEY,
    "model_name": GEMINI_REALTIME_MODEL_NAME,
    "live_connect_config": GEMINI_DEFAULT_LIVE_CONNECT_CONFIG,
    "connection_timeout": Config.AI_SERVICE_CONNECTION_TIMEOUT,
    "processing_audio_frame_rate": 16000,  # As per Gemini docs
    "processing_audio_channels": 1,
    "response_audio_frame_rate": 24000,  # As per Gemini docs
    "response_audio_channels": 1,
}
