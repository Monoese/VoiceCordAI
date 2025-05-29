import asyncio
import os
import base64
from typing import Any, Optional, ByteString, Tuple

from discord import FFmpegPCMAudio, User
from discord.ext import voice_recv

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioManager:
    """
    Manages audio processing, recording, and streaming playback for the Discord bot.

    This class handles:
    - Creating audio sinks for recording from Discord voice channels
    - Processing raw PCM audio data (resampling, format conversion)
    - Encoding audio to base64 for transmission to external services
    - Managing streaming audio playback in Discord voice channels
    - Queuing audio chunks for real-time playback
    """

    def __init__(self) -> None:
        """
        Initialize the AudioManager for streaming playback.

        Sets up:
        - audio_chunk_queue: Queue for audio chunks to be streamed for playback
        - _playback_control_event: Event to signal the playback loop
        - _current_stream_id: Identifier for the current playback stream
        """
        self.audio_chunk_queue: asyncio.Queue[Tuple[str, Optional[bytes]]] = asyncio.Queue()
        self._playback_control_event = asyncio.Event() # Used to signal the playback_loop
        self._current_stream_id: Optional[str] = None # Stores "response_id-item_id" for the current playback
        logger.debug("AudioManager initialized for streaming playback.")

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
                logger.debug(
                    f"Write called: Adding {len(data.pcm)} bytes. Total accumulated: {self.total_bytes} bytes"
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

    def create_sink(self) -> PCM16Sink:
        """
        Create a new audio sink instance for capturing voice data.

        This method instantiates a fresh PCM16Sink that can be attached to a
        Discord voice client to capture audio from users in a voice channel.

        Returns:
            PCM16Sink: A new audio sink instance ready to capture audio
        """
        return self.PCM16Sink()

    async def ffmpeg_to_24k_mono(self, raw: bytes) -> bytes:
        """
        Executes FFmpeg to resample and convert raw audio input from 48kHz stereo
        to 24kHz mono. Utilizes an asynchronous subprocess execution to call FFmpeg
        with specified parameters. The input audio is received as raw bytes, processed
        through FFmpeg, and the resulting audio output is returned as raw bytes.

        Parameters:
        raw: bytes
            Raw audio data to be processed by FFmpeg.

        Returns:
        bytes
            The processed audio data converted to 24kHz mono.

        Raises:
        RuntimeError
            If FFmpeg fails or returns a non-zero exit code.
        """
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ar",
            "24000",
            "-ac",
            "1",
            "pipe:1",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate(raw)
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg failed: " + err.decode().strip())
        return out

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

    def get_current_playing_response_id(self) -> Optional[str]:
        """
        Returns the response_id part of the current playing stream ID.
        Assumes _current_stream_id is in the format "response_id-item_id".
        """
        if self._current_stream_id:
            parts = self._current_stream_id.split('-', 1)
            if parts:
                return parts[0]
        return None

    async def start_new_audio_stream(self, stream_id: str):
        """Signals the playback_loop to prepare for a new audio stream."""
        if self._current_stream_id is not None and self._current_stream_id != stream_id:
            logger.warning(
                f"AudioManager: Request to start new stream {stream_id} while {self._current_stream_id} is active. Ending previous stream first."
            )
            await self.end_audio_stream()

        self._current_stream_id = stream_id
        logger.info(
            f"AudioManager: Preparing for new audio stream {self._current_stream_id}"
        )

        self._playback_control_event.set()  # Signal playback_loop to start/check for the new stream

    async def add_audio_chunk(self, audio_chunk: bytes):
        """Adds an audio chunk to the current stream's queue."""
        current_stream_id_for_chunk = self._current_stream_id
        if current_stream_id_for_chunk is None:
            logger.warning(
                "AudioManager: No active stream to add audio chunk. Ignoring."
            )
            return
        await self.audio_chunk_queue.put((current_stream_id_for_chunk, audio_chunk)) # Tuple: (stream_id, chunk_data)
        logger.debug(
            f"AudioManager: Added chunk of {len(audio_chunk)} bytes to stream {current_stream_id_for_chunk}"
        )

    async def end_audio_stream(self):
        """Signals the end of the current audio stream."""
        stream_id_to_end = self._current_stream_id # Capture before any potential concurrent modification

        if stream_id_to_end is None:
            logger.info("AudioManager: No active stream to end or already ended.")
            return

        logger.info(
            f"AudioManager: Signaling end of audio stream {stream_id_to_end}"
        )
        await self.audio_chunk_queue.put((stream_id_to_end, None)) # Tuple: (stream_id, None) for EOS

    async def playback_loop(self, voice_client: voice_recv.VoiceRecvClient) -> None:
        """
        Continuous loop that handles audio playback in the voice channel.
        Streams audio chunks from `audio_chunk_queue` through FFmpeg.
        """
        active_ffmpeg_process = None
        w_pipe_local = None
        current_processing_stream_id: Optional[str] = None # Initialize for robustness

        try:
            while True:
                await self._playback_control_event.wait()
                self._playback_control_event.clear()

                if self._current_stream_id is None and self.audio_chunk_queue.empty():
                    continue

                # Capture the stream ID for this specific playback session
                current_processing_stream_id = self._current_stream_id
                if current_processing_stream_id is None:
                    logger.warning(
                        "PlaybackLoop: Woke up but _current_stream_id is None. Skipping."
                    )
                    continue

                logger.info(
                    f"PlaybackLoop: Starting playback for stream {current_processing_stream_id}"
                )

                r_pipe, w_pipe_local = os.pipe()
                playback_done = asyncio.Event()

                # FFmpegPCMAudio is created per stream to handle its audio pipe
                audio_source = FFmpegPCMAudio(
                    os.fdopen(r_pipe, "rb"), # Pass the read end of the OS pipe
                    pipe=True,
                    before_options="-f s16le -ar 24000 -ac 1", # Input format for FFmpeg
                    options="-ar 48000 -ac 2", # Output format for Discord
                )
                active_ffmpeg_process = audio_source

                def log_playback_finished(error: Optional[Exception]) -> None:
                    # This callback is invoked by discord.py when voice_client.play() finishes or is stopped.
                    stream_id_in_callback = current_processing_stream_id # Capture stream ID for this callback context
                    if error:
                        logger.error(
                            f"Error during audio playback for stream {stream_id_in_callback}: {error}" # Includes errors from voice_client.stop()
                        )
                    else:
                        logger.info(
                            f"PlaybackLoop: Finished playing audio for stream {stream_id_in_callback} successfully"
                        )
                    playback_done.set()

                voice_client.play(audio_source, after=log_playback_finished)

                # Pass current_processing_stream_id explicitly to feed_ffmpeg
                async def feed_ffmpeg(pipe_to_write, feeder_stream_id: str):
                    writer_fp = None
                    loop = asyncio.get_running_loop()
                    
                    try:
                        writer_fp = os.fdopen(pipe_to_write, "wb")
                        while True:
                            item_stream_id, item_data = await self.audio_chunk_queue.get()

                            if item_stream_id != feeder_stream_id: # Item not for this feeder
                                logger.warning(
                                    f"PlaybackLoop (Feeder for {feeder_stream_id}): Discarding item for different stream {item_stream_id}."
                                )
                                self.audio_chunk_queue.task_done()
                                # If AudioManager's current stream has changed and it's not this feeder's stream,
                                # this feeder can stop to avoid pointlessly draining unrelated items from the queue.
                                if self._current_stream_id and self._current_stream_id != feeder_stream_id:
                                    logger.info(f"PlaybackLoop (Feeder for {feeder_stream_id}): Global stream changed to {self._current_stream_id}, stopping this feeder early.")
                                    break 
                                continue

                            if item_data is None:  # EOS marker for this stream
                                logger.debug(
                                    f"PlaybackLoop: EOS for stream {feeder_stream_id} received by its feeder."
                                )
                                self.audio_chunk_queue.task_done()
                                break # End of this stream

                            # Audio chunk for the current stream
                            if not self._current_stream_id: # Global context check
                                logger.warning(
                                    f"PlaybackLoop (Feeder for {feeder_stream_id}): Global stream ID became None while processing. Stopping feed."
                                )
                                self.audio_chunk_queue.task_done()
                                break

                            try:
                                await loop.run_in_executor(
                                    None, writer_fp.write, item_data
                                )
                                await loop.run_in_executor(None, writer_fp.flush)
                                logger.debug(
                                    f"PlaybackLoop: Fed {len(item_data)} bytes to FFmpeg for stream {feeder_stream_id}"
                                )
                            except Exception as e: # Typically BrokenPipeError if FFmpeg/discord.py closes pipe (e.g., on voice_client.stop())
                                logger.error(
                                    f"PlaybackLoop: Error writing/flushing to FFmpeg pipe for stream {feeder_stream_id}: {e}"
                                )
                                self.audio_chunk_queue.task_done()
                                break # Exit on write error

                            self.audio_chunk_queue.task_done()
                    
                    except asyncio.CancelledError:
                        logger.info(f"PlaybackLoop (Feeder for {feeder_stream_id}): Cancelled.")
                        # If cancelled after a get() but before task_done(), queue might be off
                        # However, typical cancellation should handle this.
                        # If self.audio_chunk_queue.get() was the cancellation point, no task_done needed.
                        # If it was after, the specific item might not be task_done'd.
                        # This is complex; relying on explicit task_done() in normal flow is primary.
                    except Exception as e:
                        logger.error(
                            f"PlaybackLoop: Unhandled error in feed_ffmpeg task for stream {feeder_stream_id}: {e}"
                        )
                        # If an item was fetched by get() and then an unexpected error occurred before task_done(),
                        # it needs to be marked. However, Queue.get() itself is the main point where _unfinished_tasks increments.
                        # If an error happens after get() but before task_done(), the count is off.
                        # The current structure aims to call task_done() in all logical exits.
                    finally:
                        if writer_fp:
                            try:
                                logger.debug(
                                    f"PlaybackLoop: Closing FFmpeg pipe writer for stream {feeder_stream_id}."
                                )
                                await loop.run_in_executor(None, writer_fp.close)
                                logger.debug(
                                    f"PlaybackLoop: FFmpeg pipe writer closed for stream {feeder_stream_id}."
                                )
                            except Exception as e:
                                logger.error(
                                    f"PlaybackLoop: Error closing FFmpeg pipe writer for stream {feeder_stream_id}: {e}"
                                )
                        elif pipe_to_write is not None: # If writer_fp was never opened but pipe exists
                            try:
                                os.close(pipe_to_write)
                                logger.debug(
                                    f"PlaybackLoop: Raw write pipe {pipe_to_write} closed directly for stream {feeder_stream_id}."
                                )
                            except OSError as e:
                                logger.error(
                                    f"PlaybackLoop: Error closing raw pipe {pipe_to_write} for stream {feeder_stream_id}: {e}"
                                )
                
                feeder_task = asyncio.create_task(feed_ffmpeg(w_pipe_local, current_processing_stream_id))

                await playback_done.wait() # Waits for log_playback_finished to be called
                await feeder_task

                logger.info(
                    f"PlaybackLoop: Fully completed stream {current_processing_stream_id}"
                )
                if active_ffmpeg_process:
                    active_ffmpeg_process.cleanup()
                    active_ffmpeg_process = None

                # Reset _current_stream_id only if it's still the one we just processed.
                # This avoids a race condition if a new stream started (and set _current_stream_id)
                # while this one was in its final cleanup stages.
                if self._current_stream_id == current_processing_stream_id:
                    self._current_stream_id = None
                
                w_pipe_local = None # Pipe is closed, reset local var

        except asyncio.CancelledError:
            logger.info("PlaybackLoop: Cancelled.")
        except Exception as e:
            logger.error(f"PlaybackLoop: Unhandled error: {e}", exc_info=True)
        finally:
            logger.info("PlaybackLoop: Exiting.")
            if active_ffmpeg_process:
                active_ffmpeg_process.cleanup()
            if w_pipe_local is not None:  # If loop exited abruptly before feed_ffmpeg closed its end
                try:
                    os.close(w_pipe_local)
                except OSError: # Pipe might already be closed
                    pass
            self._playback_control_event.clear()
            # Final check to ensure _current_stream_id is cleared if it matches the stream
            # that was being processed when an unhandled exception or cancellation occurred.
            if current_processing_stream_id is not None and \
               self._current_stream_id == current_processing_stream_id:
                self._current_stream_id = None
