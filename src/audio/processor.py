import asyncio
import base64
import io
from typing import ByteString

from pydub import AudioSegment

from src.config.config import Config
from src.exceptions import AudioProcessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _blocking_process_audio(
    raw_audio_data: bytes, target_frame_rate: int, target_channels: int
) -> bytes:
    """Synchronous helper to perform CPU-bound audio processing."""
    try:
        audio_segment = AudioSegment(
            data=raw_audio_data,
            sample_width=Config.SAMPLE_WIDTH,
            frame_rate=Config.DISCORD_AUDIO_FRAME_RATE,
            channels=Config.DISCORD_AUDIO_CHANNELS,
        )
    except Exception as e:
        logger.error(f"Failed to load raw audio into pydub AudioSegment: {e}")
        raise AudioProcessingError(
            "Audio processing failed during data loading."
        ) from e

    processed_segment = audio_segment.set_channels(target_channels)
    processed_segment = processed_segment.set_frame_rate(target_frame_rate)

    buffer = io.BytesIO()
    processed_segment.export(buffer, format="raw")
    return buffer.getvalue()


async def process_recorded_audio(
    raw_audio_data: bytes, target_frame_rate: int, target_channels: int
) -> bytes:
    """
    Converts raw PCM audio from Discord's format into the format required by
    AI services, using efficient in-memory processing.

    The blocking `pydub` operations are run in a separate thread to avoid
    blocking the main asyncio event loop.

    Args:
        raw_audio_data: The raw S16LE PCM audio bytes from the Discord sink.
        target_frame_rate: The target sample rate (e.g., 16000 for Gemini, 24000 for OpenAI).
        target_channels: The target number of audio channels (e.g., 1 for mono).

    Returns:
        bytes: The processed S16LE PCM audio bytes ready for the AI service.

    Raises:
        AudioProcessingError: If the audio processing fails at any stage.
    """
    loop = asyncio.get_running_loop()
    processed_bytes = await loop.run_in_executor(
        None,
        _blocking_process_audio,
        raw_audio_data,
        target_frame_rate,
        target_channels,
    )
    logger.debug(
        f"Audio processed via pydub: {len(raw_audio_data)} bytes -> {len(processed_bytes)} bytes"
    )
    return processed_bytes


def encode_to_base64(pcm_data: ByteString) -> str:
    """
    Encode PCM audio data to a base64 string for transmission.

    This is used to prepare audio data for sending over WebSocket connections
    where binary data needs to be represented as text.

    Args:
        pcm_data: The PCM audio data to encode

    Returns:
        str: Base64-encoded string representation of the audio data
    """
    return base64.b64encode(pcm_data).decode("utf-8")
