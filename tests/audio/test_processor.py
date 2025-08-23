import base64

import pytest

from src.audio.processing import (
    UnifiedAudioProcessor,
    AudioFormat,
    ProcessingStrategy,
    DISCORD_FORMAT,
    encode_pcm_to_base64,
)
from src.config.config import Config


def test_encode_pcm_to_base64():
    """Tests the static method for base64 encoding."""
    pcm_data = b"some test audio data"
    expected_base64 = base64.b64encode(pcm_data).decode("utf-8")

    encoded_data = encode_pcm_to_base64(pcm_data)

    assert encoded_data == expected_base64


@pytest.mark.asyncio
async def test_unified_audio_processor_success():
    """
    Tests successful audio processing using the unified audio processor.
    """
    # Test data - valid PCM data
    raw_audio_data = b"\x00\x01" * 1000  # 2000 bytes of valid PCM data
    target_frame_rate = 16000
    target_channels = 1

    # Create target format
    target_format = AudioFormat(
        sample_rate=target_frame_rate,
        channels=target_channels,
        sample_width=Config.SAMPLE_WIDTH,
    )

    # Use unified processor directly
    processor = UnifiedAudioProcessor()
    result = await processor.convert(
        source_format=DISCORD_FORMAT,
        target_format=target_format,
        audio_data=raw_audio_data,
        strategy=ProcessingStrategy.QUALITY,
    )

    # Assertions
    assert isinstance(result, bytes)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_unified_audio_processor_empty_data():
    """
    Tests that the unified audio processor handles empty data gracefully.
    """
    # Test with empty audio data
    empty_audio_data = b""
    target_format = AudioFormat(
        sample_rate=16000, channels=1, sample_width=Config.SAMPLE_WIDTH
    )

    processor = UnifiedAudioProcessor()

    # Empty input should result in empty output
    result = await processor.convert(
        source_format=DISCORD_FORMAT,
        target_format=target_format,
        audio_data=empty_audio_data,
        strategy=ProcessingStrategy.QUALITY,
    )

    assert result == b""
