import asyncio
import os
import base64
from typing import Any, Optional, ByteString, Tuple
from dataclasses import dataclass, field

from discord import FFmpegPCMAudio, User
from discord.ext import voice_recv

from src.config.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _PlaybackStream:
    """Encapsulates resources and state for a single audio playback instance."""

    stream_id: str
    ffmpeg_audio_source: FFmpegPCMAudio
    pipe_read_fd: int
    pipe_write_fd: int
    playback_done_event: asyncio.Event
    feeder_task: Optional[asyncio.Task] = field(default=None, repr=False)


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
        self.audio_chunk_queue: asyncio.Queue[Tuple[str, Optional[bytes]]] = (
            asyncio.Queue()
        )
        self._playback_control_event = (
            asyncio.Event()
        )  # Signals playback_loop to check for new work
        # _current_stream_id stores the ID of the stream that the AudioManager is currently
        # *expecting* to receive chunks for or has been told to start.
        # This is set by start_new_audio_stream().
        self._current_stream_id: Optional[str] = None
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

        def write(self, user: User, data: Any) -> None:
            """
            Process incoming audio data from a user.

            This method is called by discord.py for each audio packet received.
            It accumulates PCM audio data in the audio_data buffer.

            Args:
                user: The Discord user who sent the audio
                data: The audio data packet containing PCM data (discord.VoiceReceivePacket)
            """
            # data.pcm might be None or empty if silence is received from Discord
            # or during specific discord.py internal states.
            if data.pcm:
                self.audio_data.extend(data.pcm)  # Append new PCM data
                self.total_bytes += len(data.pcm)
                logger.debug(
                    f"PCM16Sink.write: Adding {len(data.pcm)} bytes. Total accumulated: {self.total_bytes} bytes"
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
            Config.FFMPEG_PCM_FORMAT,  # Input format
            "-ar",
            str(Config.DISCORD_AUDIO_FRAME_RATE),  # Input sample rate from Discord
            "-ac",
            str(Config.DISCORD_AUDIO_CHANNELS),  # Input channels from Discord
            "-i",
            "pipe:0",  # Input from stdin pipe
            "-f",
            Config.FFMPEG_PCM_FORMAT,  # Output format
            "-ar",
            str(
                Config.PROCESSING_AUDIO_FRAME_RATE
            ),  # Output sample rate for processing
            "-ac",
            str(
                Config.PROCESSING_AUDIO_CHANNELS
            ),  # Output channels for processing (mono)
            "pipe:1",  # Output to stdout pipe
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
        Returns the response_id part of the current target/playing stream ID.
        Assumes _current_stream_id is in the format "response_id-item_id".
        """
        if self._current_stream_id:
            parts = self._current_stream_id.split("-", 1)
            if parts:
                return parts[0]
        return None

    async def start_new_audio_stream(self, stream_id: str):
        """
        Signals the playback_loop to prepare for a new audio stream.
        If another stream is active, an EOS marker is queued for it.
        """
        if self._current_stream_id is not None and self._current_stream_id != stream_id:
            logger.warning(
                f"AudioManager: Request to start new stream '{stream_id}' while '{self._current_stream_id}' is active. "
                f"Signaling EOS for previous stream '{self._current_stream_id}'."
            )
            # Queue EOS for the old stream. The playback_loop and its feeder task
            # for the old stream will handle this to terminate gracefully.
            await self.audio_chunk_queue.put((self._current_stream_id, None))
            # The playback_loop, when it awakens, will see the new _current_stream_id
            # and transition by cleaning up the old stream processor and starting a new one.

        self._current_stream_id = stream_id
        logger.info(
            f"AudioManager: New target audio stream set to '{self._current_stream_id}'"
        )
        self._playback_control_event.set()  # Wake up playback_loop to handle the new stream

    async def add_audio_chunk(self, audio_chunk: bytes):
        """Adds an audio chunk to the queue for the stream identified by `_current_stream_id`."""
        current_target_stream_id = self._current_stream_id
        if current_target_stream_id is None:
            logger.warning(
                "AudioManager: No active target stream (_current_stream_id is None) to add audio chunk. Ignoring."
            )
            return
        await self.audio_chunk_queue.put((current_target_stream_id, audio_chunk))
        logger.debug(
            f"AudioManager: Added chunk of {len(audio_chunk)} bytes to queue for stream {current_target_stream_id}"
        )

    async def end_audio_stream(self):
        """
        Signals the end of the audio stream identified by `_current_stream_id`
        by placing an EOS (None) marker into the `audio_chunk_queue`.
        """
        stream_id_to_end = self._current_stream_id

        if stream_id_to_end is None:
            logger.info(
                "AudioManager.end_audio_stream: No target stream (_current_stream_id is None) to end."
            )
            return

        logger.info(
            f"AudioManager: Signaling end of audio stream '{stream_id_to_end}' by queueing EOS."
        )
        await self.audio_chunk_queue.put(
            (stream_id_to_end, None)
        )  # (stream_id, None) is the EOS marker
        self._playback_control_event.set()  # Wake up playback_loop to process the EOS if it's idle

    def _prepare_new_playback_stream(self, stream_id: str) -> _PlaybackStream:
        """Prepares OS pipes and FFmpegPCMAudio source for a new playback stream."""
        logger.debug(f"Preparing playback resources for stream '{stream_id}'")
        r_pipe, w_pipe = os.pipe()

        # FFmpegPCMAudio will read from the read-end of the pipe (r_pipe).
        # The _feed_audio_to_pipe task will write to the write-end (w_pipe).
        audio_source = FFmpegPCMAudio(
            os.fdopen(r_pipe, "rb"),  # Source for FFmpeg is the read-end of the pipe
            pipe=True,
            # FFmpeg input options: raw PCM, from PROCESSING format
            before_options=f"-f {Config.FFMPEG_PCM_FORMAT} -ar {Config.PROCESSING_AUDIO_FRAME_RATE} -ac {Config.PROCESSING_AUDIO_CHANNELS}",
            # FFmpeg output options (for Discord): to DISCORD format
            options=f"-ar {Config.DISCORD_AUDIO_FRAME_RATE} -ac {Config.DISCORD_AUDIO_CHANNELS}",
        )
        return _PlaybackStream(
            stream_id=stream_id,
            ffmpeg_audio_source=audio_source,
            pipe_read_fd=r_pipe,
            pipe_write_fd=w_pipe,
            playback_done_event=asyncio.Event(),
        )

    async def _feed_audio_to_pipe(self, stream: _PlaybackStream):
        """Task to feed audio chunks from the queue to the FFmpeg pipe for a given stream."""
        writer_fp = None
        loop = asyncio.get_running_loop()
        feeder_stream_id = stream.stream_id
        logger.info(f"Feeder task started for stream '{feeder_stream_id}'.")

        try:
            # Open the write-end of the pipe. FFmpeg reads from the other end.
            writer_fp = os.fdopen(stream.pipe_write_fd, "wb")
            while True:
                try:
                    item_stream_id, item_data = await self.audio_chunk_queue.get()
                except asyncio.CancelledError:
                    logger.info(
                        f"Feeder for '{feeder_stream_id}': Cancelled while waiting for queue item."
                    )
                    raise  # Propagate cancellation

                if item_stream_id != feeder_stream_id:
                    logger.warning(
                        f"Feeder for '{feeder_stream_id}': Got item for different stream '{item_stream_id}'. Discarding."
                    )
                    self.audio_chunk_queue.task_done()
                    # If the global target stream (_current_stream_id) has changed and is no longer this feeder's stream,
                    # this feeder should stop to allow a new feeder for the new target to take over.
                    if (
                        self._current_stream_id != feeder_stream_id
                        and self._current_stream_id is not None
                    ):
                        logger.info(
                            f"Feeder for '{feeder_stream_id}': Global target stream changed to '{self._current_stream_id}'. Stopping this feeder."
                        )
                        break  # Exit loop, leading to cleanup
                    continue  # Wait for the next relevant item

                if (
                    item_data is None
                ):  # EOS (End Of Stream) marker for this specific stream
                    logger.info(
                        f"Feeder for '{feeder_stream_id}': EOS marker received."
                    )
                    self.audio_chunk_queue.task_done()
                    break  # Exit loop, signaling end of this stream to FFmpeg via pipe closure

                # Defensive check: If the global target stream ID has changed *while this feeder was processing*,
                # it should stop to prevent feeding data to a potentially obsolete FFmpeg process.
                if self._current_stream_id != feeder_stream_id:
                    logger.info(
                        f"Feeder for '{feeder_stream_id}': Global target stream '{self._current_stream_id}' no longer matches. Stopping feeder."
                    )
                    self.audio_chunk_queue.task_done()  # Mark current item as "processed" before exiting
                    break

                try:
                    # Write audio data to the pipe in a separate thread to avoid blocking asyncio loop
                    await loop.run_in_executor(None, writer_fp.write, item_data)
                    await loop.run_in_executor(
                        None, writer_fp.flush
                    )  # Ensure data is sent to FFmpeg
                    logger.debug(
                        f"Feeder for '{feeder_stream_id}': Fed {len(item_data)} bytes."
                    )
                except BrokenPipeError:  # pragma: no cover
                    # This typically means FFmpeg has closed the read end of the pipe.
                    logger.warning(
                        f"Feeder for '{feeder_stream_id}': Broken pipe. FFmpeg likely closed. Stopping feeder."
                    )
                    self.audio_chunk_queue.task_done()
                    break
                except Exception as e:  # pragma: no cover
                    logger.error(
                        f"Feeder for '{feeder_stream_id}': Error writing/flushing: {e}",
                        exc_info=True,
                    )
                    self.audio_chunk_queue.task_done()
                    break

                self.audio_chunk_queue.task_done()

        except asyncio.CancelledError:
            logger.info(f"Feeder for '{feeder_stream_id}': Cancelled.")
            # Do not mark task_done here if cancelled during queue.get(), as item remains.
            # If cancelled elsewhere, task_done should have been called.
            raise  # Propagate cancellation
        except Exception as e:  # pragma: no cover
            logger.error(
                f"Feeder for '{feeder_stream_id}': Unhandled error: {e}", exc_info=True
            )
        finally:
            logger.info(f"Feeder for '{feeder_stream_id}': Exiting.")
            if writer_fp:
                try:
                    logger.debug(
                        f"Feeder for '{feeder_stream_id}': Closing pipe writer (fd: {stream.pipe_write_fd})."
                    )
                    # Closing the writer_fp also closes the underlying stream.pipe_write_fd.
                    # This signals EOF to FFmpeg if it hasn't already exited.
                    await loop.run_in_executor(None, writer_fp.close)
                except Exception as e:  # pragma: no cover
                    logger.error(
                        f"Feeder for '{feeder_stream_id}': Error closing pipe writer: {e}"
                    )

    async def _cleanup_playback_stream(self, stream_to_cleanup: _PlaybackStream):
        """Safely cleans up all resources for a given _PlaybackStream instance."""
        stream_id = stream_to_cleanup.stream_id
        logger.info(f"Cleaning up playback resources for stream '{stream_id}'")

        # 1. Cancel and await the feeder task
        if stream_to_cleanup.feeder_task and not stream_to_cleanup.feeder_task.done():
            logger.debug(f"Cancelling feeder task for stream '{stream_id}'")
            stream_to_cleanup.feeder_task.cancel()
            try:
                await (
                    stream_to_cleanup.feeder_task
                )  # Wait for feeder to finish cleanup (e.g., close pipe)
            except asyncio.CancelledError:
                logger.debug(
                    f"Feeder task for stream '{stream_id}' was cancelled successfully."
                )
            except Exception as e:  # pragma: no cover
                logger.error(
                    f"Feeder task for stream '{stream_id}' raised an error during cancellation/await: {e}",
                    exc_info=True,
                )

        # 2. Cleanup FFmpegPCMAudio source
        # This should close the read-end of the pipe (stream_to_cleanup.pipe_read_fd).
        logger.debug(
            f"Cleaning up FFmpegPCMAudio for stream '{stream_id}' (read pipe fd: {stream_to_cleanup.pipe_read_fd})"
        )
        stream_to_cleanup.ffmpeg_audio_source.cleanup()

        # Note on pipe FDs:
        # - stream_to_cleanup.pipe_write_fd is closed by the feeder_task's writer_fp.close() in its finally block.
        # - stream_to_cleanup.pipe_read_fd is closed by ffmpeg_audio_source.cleanup().
        # Explicit os.close() calls here would be redundant and could cause errors if FDs are already closed.

        if self._current_stream_id == stream_id:
            logger.info(
                f"AudioManager: Cleared _current_stream_id '{self._current_stream_id}' as its playback and cleanup finished."
            )
            self._current_stream_id = None

        logger.info(f"Finished cleaning up stream '{stream_id}'")

    async def playback_loop(self, voice_client: voice_recv.VoiceRecvClient) -> None:
        """
        Main loop for managing audio playback.

        This loop waits for signals (via `_playback_control_event`) and manages the
        lifecycle of audio streams:
        - Stops and cleans up old streams if a new `_current_stream_id` is set or if the current one ends.
        - Prepares and starts new streams based on `_current_stream_id`.
        - Handles one `_PlaybackStream` (via `current_stream_processor`) at a time.
        """
        current_stream_processor: Optional[_PlaybackStream] = None
        try:
            while True:
                await self._playback_control_event.wait()
                self._playback_control_event.clear()

                logger.debug(
                    f"PlaybackLoop: Awakened. Target stream: '{self._current_stream_id}'. "
                    f"Current processor: '{current_stream_processor.stream_id if current_stream_processor else 'None'}'"
                )

                # --- Phase 1: Stop/Transition Current Stream ---
                # This phase handles the graceful shutdown of the currently active audio stream processor.
                # Stopping the voice_client (voice_client.stop()) is key, as it triggers the 'after' callback
                # associated with the FFmpegPCMAudioSource. This callback, in turn, signals
                # current_stream_processor.playback_done_event, facilitating an orderly resource cleanup.
                if current_stream_processor:
                    processor_id = current_stream_processor.stream_id
                    global_target_id = self._current_stream_id

                    if global_target_id is None or global_target_id != processor_id:
                        logger.info(
                            f"PlaybackLoop: Global target is now '{global_target_id}'. "
                            f"Current processor for '{processor_id}' needs to stop."
                        )
                        if voice_client.is_playing() or voice_client.is_paused():
                            logger.debug(
                                f"PlaybackLoop: Calling voice_client.stop() for stream '{processor_id}'."
                            )
                            voice_client.stop()  # This will trigger the 'after' callback for the FFmpegPCMAudio source.
                            # The 'after' callback is responsible for setting current_stream_processor.playback_done_event.

                        # Wait for the 'after' callback to signal playback completion or for a timeout.
                        # This ensures that resources aren't cleaned up prematurely if discord.py is still using them.
                        if not current_stream_processor.playback_done_event.is_set():
                            logger.debug(
                                f"PlaybackLoop: Waiting for playback_done_event for '{processor_id}' after signaling stop."
                            )
                            try:
                                await asyncio.wait_for(
                                    current_stream_processor.playback_done_event.wait(),
                                    timeout=5.0,
                                )
                            except asyncio.TimeoutError:  # pragma: no cover
                                logger.warning(
                                    f"PlaybackLoop: Timeout waiting for playback_done_event for '{processor_id}'. Proceeding with cleanup."
                                )

                        # Now, perform full cleanup of the stream processor.
                        await self._cleanup_playback_stream(current_stream_processor)
                        current_stream_processor = None
                        logger.info(
                            f"PlaybackLoop: Finished stopping and cleaning up processor for '{processor_id}'."
                        )

                # --- Phase 2: Start New Stream ---
                if self._current_stream_id and not current_stream_processor:
                    target_id_for_new_stream = self._current_stream_id
                    logger.info(
                        f"PlaybackLoop: Preparing to start new stream processor for target '{target_id_for_new_stream}'."
                    )

                    current_stream_processor = self._prepare_new_playback_stream(
                        target_id_for_new_stream
                    )

                    current_stream_processor.feeder_task = asyncio.create_task(
                        self._feed_audio_to_pipe(current_stream_processor)
                    )

                    # Define the 'after' callback for voice_client.play()
                    # This callback is crucial for signaling when playback of the current source has finished or errored.
                    def after_playback_callback(
                        error: Optional[Exception], played_source_stream_id: str
                    ):
                        log_msg = f"PlaybackLoop: 'after' callback triggered for played stream '{played_source_stream_id}'."
                        if error:  # pragma: no cover
                            log_msg += f" Error: {error}"
                        logger.info(log_msg)

                        # Ensure this callback is for the currently active stream processor.
                        if (
                            current_stream_processor
                            and current_stream_processor.stream_id
                            == played_source_stream_id
                        ):
                            current_stream_processor.playback_done_event.set()
                        else:  # pragma: no cover
                            # This might happen if a new stream started very quickly after an old one stopped,
                            # and a late 'after' callback from the old stream arrives.
                            logger.warning(
                                f"PlaybackLoop: 'after' callback for '{played_source_stream_id}' but current processor is for "
                                f"'{current_stream_processor.stream_id if current_stream_processor else 'None'}'. Event might be stale."
                            )

                    logger.info(
                        f"PlaybackLoop: Starting voice_client.play for stream '{current_stream_processor.stream_id}'"
                    )
                    voice_client.play(
                        current_stream_processor.ffmpeg_audio_source,
                        # Pass the stream_id to the lambda to capture its current value for the callback.
                        after=lambda e,
                        sid=current_stream_processor.stream_id: after_playback_callback(
                            e, sid
                        ),
                    )

                    # Wait until the playback_done_event is set (by the 'after' callback).
                    # This means the current audio source has finished playing or an error occurred.
                    await current_stream_processor.playback_done_event.wait()
                    logger.info(
                        f"PlaybackLoop: Playback done event received for stream '{current_stream_processor.stream_id}'."
                    )

                    # Playback is done for this source, so clean it up.
                    await self._cleanup_playback_stream(current_stream_processor)
                    current_stream_processor = None
                    logger.info(
                        f"PlaybackLoop: Finished processing and cleaning up stream '{target_id_for_new_stream}'."
                    )

                if not self._current_stream_id and not current_stream_processor:
                    logger.debug(
                        "PlaybackLoop: No target stream and no active processor. Idling."
                    )

        except asyncio.CancelledError:  # pragma: no cover
            logger.info("PlaybackLoop: Cancelled.")
        except Exception as e:  # pragma: no cover
            logger.error(f"PlaybackLoop: Unhandled error: {e}", exc_info=True)
        finally:
            logger.info("PlaybackLoop: Exiting final cleanup.")
            # Ensure voice client is stopped if it was playing/paused.
            if voice_client and (
                voice_client.is_playing() or voice_client.is_paused()
            ):  # pragma: no cover
                logger.info(
                    "PlaybackLoop: Stopping voice_client playback during final exit."
                )
                voice_client.stop()
            # Ensure any active stream processor is cleaned up.
            if current_stream_processor:  # pragma: no cover
                logger.info(
                    f"PlaybackLoop: Cleaning up active stream processor for '{current_stream_processor.stream_id}' during final exit."
                )
                # We must ensure its playback_done_event is set if voice_client.stop() didn't trigger 'after' or it timed out.
                if not current_stream_processor.playback_done_event.is_set():
                    # Force-set playback_done_event if not already set (e.g., if the 'after' callback
                    # didn't run due to abrupt termination or a timeout while waiting for it).
                    # This ensures _cleanup_playback_stream, which might await this event,
                    # can proceed without deadlocking during final cleanup.
                    current_stream_processor.playback_done_event.set()
                await self._cleanup_playback_stream(current_stream_processor)
                current_stream_processor = None
            self._playback_control_event.clear()
