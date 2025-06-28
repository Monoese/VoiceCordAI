from typing import Any

from discord import User
from discord.ext import voice_recv

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioSink(voice_recv.AudioSink):
    """
    Custom audio sink that captures PCM audio data from Discord voice channels.

    This sink:
    - Captures raw PCM audio data from users in voice channels
    - Accumulates the data in a bytearray for later processing
    - Tracks the total amount of data captured for debugging
    """

    def __init__(self) -> None:
        """Initialize the audio sink with empty buffer and counters."""
        super().__init__()
        self.audio_data: bytearray = bytearray()
        self.total_bytes: int = 0
        logger.debug("New audio sink initialized")

    def wants_opus(self) -> bool:
        """
        Indicate whether this sink wants Opus-encoded audio.

        Returns:
            bool: False to receive PCM audio data instead of Opus-encoded data
        """
        return False

    def write(self, _user: User, data: Any) -> None:
        """
        Process incoming audio data from a user.

        This method is called by discord.py for each audio packet received.
        It accumulates PCM audio data in the audio_data buffer.

        Args:
            _user: The Discord user who sent the audio
            data: The audio data packet containing PCM data (discord.VoiceReceivePacket)
        """
        # data.pcm might be None or empty if silence is received from Discord
        # or during specific discord.py internal states.
        if data.pcm:
            self.audio_data.extend(data.pcm)
            self.total_bytes += len(data.pcm)
            logger.debug(
                f"AudioSink.write: Adding {len(data.pcm)} bytes. Total accumulated: {self.total_bytes} bytes"
            )
        else:
            logger.warning("Write called with no PCM data")

    def cleanup(self) -> None:
        """
        Clean up resources when the sink is no longer needed.

        This method is called by discord.py when the bot stops listening.
        It logs the final state and clears the audio buffer.
        """
        logger.debug(f"Cleanup called. Final total bytes: {self.total_bytes}")
        data_len = len(self.audio_data)
        logger.debug(f"Length of audio_data before clear: {data_len}")
        self.audio_data.clear()


def create_sink() -> AudioSink:
    """
    Create a new audio sink instance for capturing voice data.

    This method instantiates a fresh AudioSink that can be attached to a
    Discord voice client to capture audio from users in a voice channel.

    Returns:
        AudioSink: A new audio sink instance ready to capture audio
    """
    return AudioSink()
