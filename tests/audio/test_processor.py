import asyncio
import base64
from unittest.mock import MagicMock, patch

import pytest

from src.audio.processor import encode_to_base64, process_recorded_audio
from src.config.config import Config
from src.exceptions import AudioProcessingError


def test_encode_to_base64():
    """Tests the static method for base64 encoding."""
    pcm_data = b"some test audio data"
    expected_base64 = base64.b64encode(pcm_data).decode("utf-8")

    encoded_data = encode_to_base64(pcm_data)

    assert encoded_data == expected_base64


@pytest.mark.asyncio
@patch("src.audio.processor.AudioSegment")
async def test_process_recorded_audio_success(mock_audio_segment_cls: MagicMock):
    """
    Tests successful audio processing using the pydub-based method.
    """
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

    result = await process_recorded_audio(
        raw_audio_data, target_frame_rate, target_channels
    )

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
@patch("src.audio.processor.AudioSegment")
async def test_process_recorded_audio_failure(mock_audio_segment_cls: MagicMock):
    """
    Tests the failure path of audio processing when pydub fails.
    """
    raw_audio_data = b"\x01\x02\x03\x04"
    error_message = "pydub failed"
    mock_audio_segment_cls.side_effect = Exception(error_message)

    with pytest.raises(AudioProcessingError) as excinfo:
        await process_recorded_audio(raw_audio_data, 16000, 1)

    assert "Audio processing failed" in str(excinfo.value)
    assert error_message in str(excinfo.value.__cause__)