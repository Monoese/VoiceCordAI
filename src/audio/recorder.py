from typing import Any

from discord import User
from discord.ext import voice_recv

from src.bot.state import BotState
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConsentFilterSink(voice_recv.AudioSink):
    """
    A sink that filters audio packets based on user consent and authority.

    This sink acts as a filter in a sink chain. It inspects each audio packet
    and passes it to its destination sink only if the originating user is either
    the session authority or has explicitly granted consent.
    """

    def __init__(self, destination: "BufferingSink", bot_state: BotState):
        """
        Initialize the consent filter.

        Args:
            destination: The next sink in the chain to receive filtered data.
            bot_state: The BotState object providing consent information.
        """
        super().__init__(destination)
        self.destination: "BufferingSink" = destination
        self._bot_state = bot_state

    @property
    def audio_data(self) -> bytearray:
        """Provides access to the buffered audio data from the destination sink."""
        return self.destination.audio_data

    def wants_opus(self) -> bool:
        """Delegates the audio format preference to the destination sink."""
        return self.destination.wants_opus()

    def write(self, user: User, data: Any) -> None:
        """
        Processes and filters an incoming audio packet.

        The packet is passed to the destination sink if the user is authorized.
        """
        if not user:
            return

        authority_user_id = self._bot_state.authority_user_id
        is_authority = user.id == authority_user_id

        consented_user_ids = self._bot_state.get_consented_user_ids()
        is_consented = user.id in consented_user_ids

        if is_authority or is_consented:
            self.destination.write(user, data)

    def cleanup(self) -> None:
        """Clean up resources. This sink owns no resources, so it does nothing."""
        pass


class BufferingSink(voice_recv.AudioSink):
    """
    A sink that buffers incoming PCM audio data into a bytearray.

    This sink's sole responsibility is to accumulate audio data that it receives.
    It does not perform any filtering.
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
                f"BufferingSink.write: Adding {len(data.pcm)} bytes. Total accumulated: {self.total_bytes} bytes"
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


def create_sink(bot_state: BotState) -> ConsentFilterSink:
    """
    Create a new audio sink chain for consent-based recording.

    This factory function constructs a sink chain consisting of:
    1. A ConsentFilterSink that performs the filtering logic.
    2. A BufferingSink that stores the audio data that passes the filter.

    Args:
        bot_state: The BotState object providing access to consent information.

    Returns:
        ConsentFilterSink: The head of the sink chain, ready to be used by the voice client.
    """
    buffering_sink = BufferingSink()
    return ConsentFilterSink(destination=buffering_sink, bot_state=bot_state)
