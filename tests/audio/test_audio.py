import asyncio
import base64
from unittest.mock import MagicMock, patch

import pytest

from src.audio.audio import AudioManager
from src.config.config import Config


@pytest.fixture
def audio_manager() -> AudioManager:
    """Provides a fresh AudioManager instance for each test."""
    return AudioManager()


@pytest.fixture
def mock_user() -> MagicMock:
    """Provides a mock Discord user object."""
    user = MagicMock()
    user.id = 12345
    return user


def test_audio_manager_initialization(audio_manager: AudioManager):
    """Tests that the AudioManager initializes with the correct default state."""
    assert isinstance(audio_manager.audio_chunk_queue, asyncio.Queue)
    assert audio_manager._current_stream_id is None
    assert not audio_manager._playback_control_event.is_set()


def test_pcm16sink_write(audio_manager: AudioManager, mock_user: MagicMock):
    """Tests that the PCM16Sink correctly accumulates audio data."""
    # Arrange
    sink = audio_manager.create_sink()
    test_data1 = b"\x01\x02\x03\x04"
    test_data2 = b"\x05\x06"
    mock_packet1 = MagicMock()
    mock_packet1.pcm = test_data1
    mock_packet2 = MagicMock()
    mock_packet2.pcm = test_data2

    # Act
    sink.write(mock_user, mock_packet1)
    sink.write(mock_user, mock_packet2)

    # Assert
    assert sink.audio_data == bytearray(test_data1 + test_data2)
    assert sink.total_bytes == len(test_data1) + len(test_data2)


def test_encode_to_base64():
    """Tests the static method for base64 encoding."""
    # Arrange
    pcm_data = b"some test audio data"
    expected_base64 = base64.b64encode(pcm_data).decode("utf-8")

    # Act
    encoded_data = AudioManager.encode_to_base64(pcm_data)

    # Assert
    assert encoded_data == expected_base64


@pytest.mark.asyncio
@patch("src.audio.audio.AudioSegment")
async def test_process_recorded_audio_success(
    mock_audio_segment_cls: MagicMock, audio_manager: AudioManager
):
    """
    Tests successful audio processing using the pydub-based method.
    """
    # Arrange
    raw_audio_data = b"\x01\x02\x03\x04"
    processed_audio_data = b"\x05\x06\x07\x08"
    target_frame_rate = 16000
    target_channels = 1

    # Mock the chain of pydub calls
    mock_segment_instance = MagicMock()
    mock_audio_segment_cls.return_value = mock_segment_instance
    mock_segment_instance.set_channels.return_value = mock_segment_instance
    mock_segment_instance.set_frame_rate.return_value = mock_segment_instance

    # Mock the export method to write to the buffer
    def mock_export(buffer, format):
        buffer.write(processed_audio_data)

    mock_segment_instance.export.side_effect = mock_export

    # Act
    result = await audio_manager.process_recorded_audio(
        raw_audio_data, target_frame_rate, target_channels
    )

    # Assert
    assert result == processed_audio_data
    mock_audio_segment_cls.assert_called_once_with(
        data=raw_audio_data,
        sample_width=Config.SAMPLE_WIDTH,
        frame_rate=Config.DISCORD_AUDIO_FRAME_RATE,
        channels=Config.DISCORD_AUDIO_CHANNELS,
    )
    mock_segment_instance.set_channels.assert_called_once_with(target_channels)
    mock_segment_instance.set_frame_rate.assert_called_once_with(target_frame_rate)
    assert mock_segment_instance.export.call_args.kwargs["format"] == "raw"


@pytest.mark.asyncio
@patch("src.audio.audio.AudioSegment")
async def test_process_recorded_audio_failure(
    mock_audio_segment_cls: MagicMock, audio_manager: AudioManager
):
    """
    Tests the failure path of audio processing when pydub fails.
    """
    # Arrange
    raw_audio_data = b"\x01\x02\x03\x04"
    error_message = "pydub failed"
    mock_audio_segment_cls.side_effect = Exception(error_message)

    # Act & Assert
    with pytest.raises(RuntimeError) as excinfo:
        await audio_manager.process_recorded_audio(raw_audio_data, 16000, 1)

    assert "Audio processing failed" in str(excinfo.value)
    assert error_message in str(excinfo.value.__cause__)


def test_get_current_playing_response_id(audio_manager: AudioManager):
    """Tests that the correct response ID is extracted from the stream ID."""
    # Arrange
    audio_manager._current_stream_id = "response123-item456"

    # Act
    response_id = audio_manager.get_current_playing_response_id()

    # Assert
    assert response_id == "response123"

    # Arrange for None case
    audio_manager._current_stream_id = None

    # Act
    response_id = audio_manager.get_current_playing_response_id()

    # Assert
    assert response_id is None


@pytest.mark.asyncio
async def test_start_new_audio_stream(audio_manager: AudioManager):
    """Tests starting a new audio stream when none is active."""
    # Arrange
    stream_id = "test-stream-1"
    response_format = (24000, 1)
    assert audio_manager._current_stream_id is None
    assert not audio_manager._playback_control_event.is_set()

    # Act
    await audio_manager.start_new_audio_stream(stream_id, response_format)

    # Assert
    assert audio_manager._current_stream_id == stream_id
    assert audio_manager._current_response_format == response_format
    assert audio_manager._playback_control_event.is_set()


@pytest.mark.asyncio
async def test_start_new_audio_stream_replaces_old(audio_manager: AudioManager):
    """Tests that starting a new stream correctly signals the end of the old one."""
    # Arrange
    old_stream_id = "old-stream"
    new_stream_id = "new-stream"
    response_format = (24000, 1)
    await audio_manager.start_new_audio_stream(old_stream_id, response_format)
    audio_manager._playback_control_event.clear()  # Reset for the next action

    # Act
    await audio_manager.start_new_audio_stream(new_stream_id, response_format)

    # Assert
    assert audio_manager._current_stream_id == new_stream_id
    assert audio_manager._playback_control_event.is_set()
    # Check that an EOS marker for the *old* stream was queued
    assert not audio_manager.audio_chunk_queue.empty()
    queued_eos = await audio_manager.audio_chunk_queue.get()
    assert queued_eos == (old_stream_id, None)
    assert old_stream_id in audio_manager._eos_queued_for_streams


@pytest.mark.asyncio
async def test_add_audio_chunk(audio_manager: AudioManager):
    """Tests that audio chunks are added to the queue for the current stream."""
    # Arrange
    stream_id = "test-stream-1"
    audio_chunk = b"\xde\xad\xbe\xef"
    audio_manager._current_stream_id = stream_id

    # Act
    await audio_manager.add_audio_chunk(audio_chunk)

    # Assert
    assert not audio_manager.audio_chunk_queue.empty()
    queued_item = await audio_manager.audio_chunk_queue.get()
    assert queued_item == (stream_id, audio_chunk)


@pytest.mark.asyncio
async def test_add_audio_chunk_no_stream(audio_manager: AudioManager):
    """Tests that audio chunks are ignored if no stream is active."""
    # Arrange
    audio_chunk = b"\xde\xad\xbe\xef"
    assert audio_manager._current_stream_id is None

    # Act
    await audio_manager.add_audio_chunk(audio_chunk)

    # Assert
    assert audio_manager.audio_chunk_queue.empty()


@pytest.mark.asyncio
async def test_end_audio_stream(audio_manager: AudioManager):
    """Tests signaling the end of the current audio stream."""
    # Arrange
    stream_id = "test-stream-1"
    audio_manager._current_stream_id = stream_id
    audio_manager._playback_control_event.clear()

    # Act
    await audio_manager.end_audio_stream()

    # Assert
    # Check that an EOS marker was queued
    assert not audio_manager.audio_chunk_queue.empty()
    queued_eos = await audio_manager.audio_chunk_queue.get()
    assert queued_eos == (stream_id, None)
    # Check that the stream is marked as having EOS queued
    assert stream_id in audio_manager._eos_queued_for_streams
    # Check that the playback loop was signaled
    assert audio_manager._playback_control_event.is_set()


@pytest.mark.asyncio
async def test_end_audio_stream_with_override(audio_manager: AudioManager):
    """Tests signaling the end of a specific stream using override."""
    # Arrange
    audio_manager._current_stream_id = "active-stream"
    override_stream_id = "override-stream"

    # Act
    await audio_manager.end_audio_stream(stream_id_override=override_stream_id)

    # Assert
    assert not audio_manager.audio_chunk_queue.empty()
    queued_eos = await audio_manager.audio_chunk_queue.get()
    assert queued_eos == (override_stream_id, None)
    assert override_stream_id in audio_manager._eos_queued_for_streams


@pytest.mark.asyncio
async def test_end_audio_stream_idempotent(audio_manager: AudioManager):
    """Tests that signaling EOS for a stream is idempotent."""
    # Arrange
    stream_id = "test-stream-1"
    audio_manager._current_stream_id = stream_id

    # Act
    await audio_manager.end_audio_stream()
    await audio_manager.end_audio_stream()  # Second call

    # Assert
    # There should only be one EOS marker in the queue
    assert audio_manager.audio_chunk_queue.qsize() == 1
    await audio_manager.audio_chunk_queue.get()
    assert audio_manager.audio_chunk_queue.empty()
    assert stream_id in audio_manager._eos_queued_for_streams
