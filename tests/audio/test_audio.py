import asyncio
import base64
import subprocess
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
@patch("subprocess.run")
async def test_resample_and_convert_audio_success(
    mock_subprocess_run: MagicMock, audio_manager: AudioManager
):
    """
    Tests successful audio resampling by mocking the ffmpeg subprocess.
    """
    # Arrange
    raw_audio_data = b"\x01\x02\x03\x04"
    resampled_audio_data = b"\x05\x06\x07\x08"
    mock_subprocess_run.return_value = MagicMock(
        stdout=resampled_audio_data, stderr=b""
    )

    # Act
    result = await audio_manager.resample_and_convert_audio(raw_audio_data)

    # Assert
    assert result == resampled_audio_data
    mock_subprocess_run.assert_called_once()
    args, kwargs = mock_subprocess_run.call_args
    # Check that ffmpeg was called with the correct parameters
    assert "ffmpeg" in args[0]
    assert str(Config.DISCORD_AUDIO_FRAME_RATE) in args[0]
    assert str(Config.PROCESSING_AUDIO_FRAME_RATE) in args[0]
    assert kwargs["input"] == raw_audio_data


@pytest.mark.asyncio
@patch("subprocess.run")
async def test_resample_and_convert_audio_failure(
    mock_subprocess_run: MagicMock, audio_manager: AudioManager
):
    """
    Tests the audio resampling failure path.
    """
    # Arrange
    raw_audio_data = b"\x01\x02\x03\x04"
    error_output = b"ffmpeg error message"
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        1, "ffmpeg", stderr=error_output
    )

    # Act & Assert
    with pytest.raises(RuntimeError) as excinfo:
        await audio_manager.resample_and_convert_audio(raw_audio_data)

    assert error_output.decode() in str(excinfo.value)
