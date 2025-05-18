"""
Audio processing module for handling Discord voice data.

This module provides functionality for:
- Recording audio from Discord voice channels
- Processing and converting audio data between different formats
- Encoding audio to base64 for transmission
- Playing back audio responses in Discord voice channels

The module uses pydub for audio processing and the discord.py voice_recv
extension for capturing audio from Discord voice channels.
"""

import asyncio
import base64
from io import BytesIO
from typing import Any, Optional, ByteString

from discord import FFmpegPCMAudio, User
from discord.ext import voice_recv
from pydub import AudioSegment

from src.config.config import Config
from src.utils.logger import get_logger

# Configure logger for this module
logger = get_logger(__name__)


class AudioManager:
    """
    Manages audio processing, recording, and playback for the Discord bot.

    This class handles:
    - Creating audio sinks for recording from Discord voice channels
    - Processing raw PCM audio data (resampling, format conversion)
    - Encoding audio to base64 for transmission to external services
    - Managing audio playback in Discord voice channels
    - Buffering and queuing audio for smooth playback
    """
    def __init__(self) -> None:
        """
        Initialize the AudioManager with empty buffers and queues.

        Sets up:
        - response_buffer: Temporary storage for audio data received from external services
        - output_queue: Queue for audio files waiting to be played back
        """
        self.response_buffer: bytearray = bytearray()  # Buffer for incoming audio responses
        self.output_queue: asyncio.Queue[BytesIO] = asyncio.Queue()  # Queue for audio playback

    class PCM16Sink(voice_recv.AudioSink):
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
            self.audio_data: bytearray = bytearray()  # Buffer to store captured audio
            self.total_bytes: int = 0  # Counter for total bytes captured
            logger.debug("New audio sink initialized")

        def wants_opus(self) -> bool:
            """
            Indicate whether this sink wants Opus-encoded audio.

            Returns:
                bool: False to receive PCM audio data instead of Opus-encoded data
            """
            return False

        def write(self, user: User, data: Any) -> None:
            """
            Process incoming audio data from a user.

            This method is called by discord.py for each audio packet received.
            It accumulates PCM audio data in the audio_data buffer.

            Args:
                user: The Discord user who sent the audio
                data: The audio data packet containing PCM data
            """
            if data.pcm:
                # Add the PCM data to our buffer
                self.audio_data.extend(data.pcm)
                self.total_bytes += len(data.pcm)
                logger.debug(f"Write called: Adding {len(data.pcm)} bytes. Total accumulated: {self.total_bytes} bytes")
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

    def create_sink(self) -> PCM16Sink:
        """
        Create a new audio sink instance for capturing voice data.

        This method instantiates a fresh PCM16Sink that can be attached to a
        Discord voice client to capture audio from users in a voice channel.

        Returns:
            PCM16Sink: A new audio sink instance ready to capture audio
        """
        return self.PCM16Sink()

    def extend_response_buffer(self, audio_data: ByteString) -> None:
        """
        Add new audio data to the response buffer.

        This method is used to accumulate audio data received from external services
        before it's processed and queued for playback.

        Args:
            audio_data: The audio data to add to the buffer
        """
        self.response_buffer.extend(audio_data)
        logger.debug(f"Added {len(audio_data)} bytes to response buffer")

    def clear_response_buffer(self) -> None:
        """
        Clear the response buffer after processing.

        This method is called after the accumulated audio data has been
        processed and queued for playback to free up memory.
        """
        self.response_buffer.clear()

    @staticmethod
    def process_audio(raw_pcm_data: ByteString) -> bytes:
        """
        Convert and process raw PCM data for transmission.

        This method:
        1. Creates an AudioSegment from the raw PCM data with Discord's frame rate
        2. Resamples the audio to the target frame rate for external services

        Args:
            raw_pcm_data: The raw PCM audio data to process

        Returns:
            bytes: Processed PCM audio data at the target frame rate
        """
        # Create AudioSegment from raw PCM data with Discord's frame rate
        audio_segment = AudioSegment(data=raw_pcm_data, sample_width=Config.SAMPLE_WIDTH,
                                     frame_rate=Config.DISCORD_FRAME_RATE, channels=Config.CHANNELS)

        # Resample to target frame rate for external services
        audio_segment = audio_segment.set_frame_rate(Config.TARGET_FRAME_RATE)
        logger.debug(f"Processed audio sample rate: {audio_segment.frame_rate}")

        return audio_segment.raw_data

    @staticmethod
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

    async def enqueue_audio(self, audio_buffer: ByteString) -> None:
        """
        Enqueue raw PCM audio data for playback in Discord.

        This method puts the raw PCM audio in a buffer for FFmpegPCMAudio
        to handle resampling and encoding on the fly.

        Args:
            audio_buffer: Raw PCM data (e.g. 24kHz mono s16le) to enqueue
        """
        # Wrap raw PCM data in a buffer for FFmpegPCMAudio to read and resample
        pcm_buffer = BytesIO(bytes(audio_buffer))
        await self.output_queue.put(pcm_buffer)

    async def playback_loop(self, voice_client: voice_recv.VoiceRecvClient) -> None:
        """
        Continuous loop that handles audio playback in the voice channel.

        This method:
        1. Waits for audio buffers to be added to the output queue
        2. Creates an FFmpeg audio source from each buffer
        3. Plays the audio in the voice channel
        4. Waits for playback to complete before processing the next item

        The loop runs indefinitely until the task is cancelled externally.

        Args:
            voice_client: The Discord voice client to use for playback
        """
        try:
            while True:
                # Wait for an audio buffer to be available in the queue
                audio_buffer = await self.output_queue.get()
                # Create an event that notifies the end of a playback
                playback_done = asyncio.Event()
                # Create an audio source from the buffer
                audio_source = FFmpegPCMAudio(
                    audio_buffer,
                    pipe=True,
                    before_options="-f s16le -ar 24000 -ac 1",
                    options="-ar 48000 -ac 2"  # resample to 48 kHz stereo for Discord
                )

                # Define callback for when playback finishes
                def log_playback_finished(error: Optional[Exception]) -> None:
                    """Log the result of audio playback."""
                    if error:
                        logger.error(f"Error during audio playback: {error}")
                    else:
                        logger.debug("Finished playing audio successfully")
                    playback_done.set()

                # Start playback
                voice_client.play(audio_source, after=log_playback_finished)

                # Wait for playback to complete
                await playback_done.wait()

        except Exception as e:
            logger.error(f"Error during audio playback: {e}")
        finally:
            # Mark this item as processed
            self.output_queue.task_done()
