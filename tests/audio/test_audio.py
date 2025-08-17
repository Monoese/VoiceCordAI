"""
Integration tests for audio subsystem components working together.

This module tests the interaction between different audio components:
- AudioPlaybackManager for playback operations
- Audio processing functions for format conversion
- Audio sinks for recording
"""

import pytest
from unittest.mock import MagicMock
import discord

from src.audio.playback import AudioPlaybackManager
from src.audio.processor import encode_to_base64, process_recorded_audio


@pytest.fixture
def mock_guild() -> MagicMock:
    """Provides a mock Discord guild object."""
    guild = MagicMock(spec=discord.Guild)
    guild.id = 123456789
    guild.voice_client = None
    return guild


@pytest.mark.asyncio
async def test_audio_pipeline_integration(mock_guild: MagicMock):
    """
    Test the full audio pipeline from recording through processing to playback.
    """
    # Setup
    playback_manager = AudioPlaybackManager(mock_guild)
    raw_audio = b"\x01\x02\x03\x04" * 1000  # Simulate recorded audio

    # Process the recorded audio
    processed_audio = await process_recorded_audio(raw_audio, 24000, 1)
    assert len(processed_audio) > 0

    # Encode for transmission
    encoded_audio = encode_to_base64(processed_audio)
    assert isinstance(encoded_audio, str)

    # Setup playback stream
    stream_id = "test-stream"
    await playback_manager.start_new_audio_stream(stream_id, (24000, 1))

    # Add audio chunk to playback
    await playback_manager.add_audio_chunk(processed_audio)

    # Verify the audio is queued
    assert not playback_manager.audio_chunk_queue.empty()
    queued_item = await playback_manager.audio_chunk_queue.get()
    assert queued_item == (stream_id, processed_audio)

    # End the stream
    await playback_manager.end_audio_stream()

    # Verify EOS marker is queued
    eos_marker = await playback_manager.audio_chunk_queue.get()
    assert eos_marker == (stream_id, None)
