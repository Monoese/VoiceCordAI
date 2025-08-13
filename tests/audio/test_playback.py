import asyncio
from unittest.mock import MagicMock, patch

import pytest
import discord

from src.audio.playback import AudioPlaybackManager


@pytest.fixture
def mock_guild() -> MagicMock:
    """Provides a mock Discord guild object."""
    guild = MagicMock(spec=discord.Guild)
    guild.id = 123456789
    guild.voice_client = None
    return guild


@pytest.fixture
def audio_playback_manager(mock_guild: MagicMock) -> AudioPlaybackManager:
    """Provides a fresh AudioPlaybackManager instance for each test."""
    return AudioPlaybackManager(mock_guild)


def test_audio_playback_manager_initialization(audio_playback_manager: AudioPlaybackManager):
    """Tests that the AudioPlaybackManager initializes with the correct default state."""
    assert isinstance(audio_playback_manager.audio_chunk_queue, asyncio.Queue)
    assert audio_playback_manager._current_stream_id is None
    assert not audio_playback_manager._playback_control_event.is_set()
    assert audio_playback_manager._current_response_format is None
    assert len(audio_playback_manager._eos_queued_for_streams) == 0


def test_get_current_playing_response_id(audio_playback_manager: AudioPlaybackManager):
    """Tests that the correct response ID is extracted from the stream ID."""
    audio_playback_manager._current_stream_id = "response123-item456"

    response_id = audio_playback_manager.get_current_playing_response_id()

    assert response_id == "response123"

    # Test None case
    audio_playback_manager._current_stream_id = None

    response_id = audio_playback_manager.get_current_playing_response_id()

    assert response_id is None


@pytest.mark.asyncio
async def test_start_new_audio_stream(audio_playback_manager: AudioPlaybackManager):
    """Tests starting a new audio stream when none is active."""
    stream_id = "test-stream-1"
    response_format = (24000, 1)
    assert audio_playback_manager._current_stream_id is None
    assert not audio_playback_manager._playback_control_event.is_set()

    await audio_playback_manager.start_new_audio_stream(stream_id, response_format)

    assert audio_playback_manager._current_stream_id == stream_id
    assert audio_playback_manager._current_response_format == response_format
    assert audio_playback_manager._playback_control_event.is_set()


@pytest.mark.asyncio
async def test_start_new_audio_stream_replaces_old(audio_playback_manager: AudioPlaybackManager):
    """Tests that starting a new stream correctly handles the transition from an old one."""
    old_stream_id = "old-stream"
    new_stream_id = "new-stream"
    response_format = (24000, 1)
    
    await audio_playback_manager.start_new_audio_stream(old_stream_id, response_format)
    audio_playback_manager._playback_control_event.clear()  # Reset for the next action

    await audio_playback_manager.start_new_audio_stream(new_stream_id, response_format)

    assert audio_playback_manager._current_stream_id == new_stream_id
    assert audio_playback_manager._playback_control_event.is_set()


@pytest.mark.asyncio
async def test_add_audio_chunk(audio_playback_manager: AudioPlaybackManager):
    """Tests that audio chunks are added to the queue for the current stream."""
    stream_id = "test-stream-1"
    audio_chunk = b"\xde\xad\xbe\xef"
    audio_playback_manager._current_stream_id = stream_id

    await audio_playback_manager.add_audio_chunk(audio_chunk)

    assert not audio_playback_manager.audio_chunk_queue.empty()
    queued_item = await audio_playback_manager.audio_chunk_queue.get()
    assert queued_item == (stream_id, audio_chunk)


@pytest.mark.asyncio
async def test_add_audio_chunk_no_stream(audio_playback_manager: AudioPlaybackManager):
    """Tests that audio chunks are ignored if no stream is active."""
    audio_chunk = b"\xde\xad\xbe\xef"
    assert audio_playback_manager._current_stream_id is None

    await audio_playback_manager.add_audio_chunk(audio_chunk)

    assert audio_playback_manager.audio_chunk_queue.empty()


@pytest.mark.asyncio
async def test_end_audio_stream(audio_playback_manager: AudioPlaybackManager):
    """Tests signaling the end of the current audio stream."""
    stream_id = "test-stream-1"
    audio_playback_manager._current_stream_id = stream_id
    audio_playback_manager._playback_control_event.clear()

    await audio_playback_manager.end_audio_stream()

    # Check that an EOS marker was queued
    assert not audio_playback_manager.audio_chunk_queue.empty()
    queued_eos = await audio_playback_manager.audio_chunk_queue.get()
    assert queued_eos == (stream_id, None)
    # Check that the stream is marked as having EOS queued
    assert stream_id in audio_playback_manager._eos_queued_for_streams
    # Playback loop signal is not set for end_audio_stream (different from original test)


@pytest.mark.asyncio
async def test_end_audio_stream_with_override(audio_playback_manager: AudioPlaybackManager):
    """Tests signaling the end of a specific stream using override."""
    audio_playback_manager._current_stream_id = "active-stream"
    override_stream_id = "override-stream"

    await audio_playback_manager.end_audio_stream(stream_id_override=override_stream_id)

    assert not audio_playback_manager.audio_chunk_queue.empty()
    queued_eos = await audio_playback_manager.audio_chunk_queue.get()
    assert queued_eos == (override_stream_id, None)
    assert override_stream_id in audio_playback_manager._eos_queued_for_streams


@pytest.mark.asyncio
async def test_end_audio_stream_idempotent(audio_playback_manager: AudioPlaybackManager):
    """Tests that signaling EOS for a stream is idempotent."""
    stream_id = "test-stream-1"
    audio_playback_manager._current_stream_id = stream_id

    await audio_playback_manager.end_audio_stream()
    await audio_playback_manager.end_audio_stream()  # Second call

    # There should only be one EOS marker in the queue
    assert audio_playback_manager.audio_chunk_queue.qsize() == 1
    await audio_playback_manager.audio_chunk_queue.get()
    assert audio_playback_manager.audio_chunk_queue.empty()
    assert stream_id in audio_playback_manager._eos_queued_for_streams