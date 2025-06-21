import asyncio
import os
import base64
import functools
import io
import subprocess
from typing import Any, Optional, ByteString, Tuple
from dataclasses import dataclass, field

from discord import FFmpegPCMAudio, User
from discord.ext import voice_recv
from pydub import AudioSegment

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
        self._current_response_format: Optional[Tuple[int, int]] = None
        self._eos_queued_for_streams: set[str] = (
            set()
        )  # Tracks streams for which EOS has been queued
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
                self.audio_data.extend(data.pcm)
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

    async def process_recorded_audio(
        self, raw_audio_data: bytes, target_frame_rate: int, target_channels: int
    ) -> bytes:
        """
        Converts raw PCM audio from Discord's format into the format required by
        AI services, using efficient in-memory processing.

        This method uses `pydub` for all transcoding (resampling and channel mixing),
        which happens within the Python process, reducing latency and CPU overhead.

        Args:
            raw_audio_data: The raw S16LE PCM audio bytes from the Discord sink.
            target_frame_rate: The target sample rate (e.g., 16000 for Gemini, 24000 for OpenAI).
            target_channels: The target number of audio channels (e.g., 1 for mono).

        Returns:
            bytes: The processed S16LE PCM audio bytes ready for the AI service.

        Raises:
            RuntimeError: If the audio processing fails at any stage.
        """
        # 1. Load the raw PCM data into a pydub AudioSegment.
        #    We must explicitly define the format of the incoming raw data using
        #    the application's configuration settings.
        try:
            audio_segment = AudioSegment(
                data=raw_audio_data,
                sample_width=Config.SAMPLE_WIDTH,
                frame_rate=Config.DISCORD_AUDIO_FRAME_RATE,
                channels=Config.DISCORD_AUDIO_CHANNELS,
            )
        except Exception as e:
            logger.error(f"Failed to load raw audio into pydub AudioSegment: {e}")
            raise RuntimeError("Audio processing failed during data loading.") from e

        # 2. Downmix to the number of channels required by the AI service.
        processed_segment = audio_segment.set_channels(target_channels)

        # 3. Resample to the sample rate required by the AI service.
        processed_segment = processed_segment.set_frame_rate(target_frame_rate)

        # 4. Export the processed audio back to raw PCM bytes.
        buffer = io.BytesIO()
        processed_segment.export(buffer, format="raw")  # "raw" corresponds to pcm_s16le

        processed_bytes = buffer.getvalue()
        logger.debug(
            f"Audio processed via pydub: {len(raw_audio_data)} bytes -> {len(processed_bytes)} bytes"
        )
        return processed_bytes

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

    async def start_new_audio_stream(
        self, stream_id: str, response_format: Tuple[int, int]
    ):
        """
        Signals the playback_loop to prepare for a new audio stream.
        If another stream is active, an EOS marker is queued for it.

        Args:
            stream_id: A unique identifier for the new audio stream.
            response_format: A tuple (frame_rate, channels) specifying the format
                             of the incoming audio from the AI service.
        """
        previous_stream_id = self._current_stream_id
        if previous_stream_id is not None and previous_stream_id != stream_id:
            logger.warning(
                f"AudioManager: Request to start new stream '{stream_id}' while '{previous_stream_id}' is active. "
                f"Signaling EOS for previous stream '{previous_stream_id}'."
            )
            # Use the idempotent end_audio_stream to signal the end of the previous stream.
            await self.end_audio_stream(stream_id_override=previous_stream_id)
            # The playback_loop, when it awakens, will see the new _current_stream_id
            # and transition by cleaning up the old stream processor and starting a new one.

        self._current_stream_id = stream_id
        self._current_response_format = response_format
        logger.info(
            f"AudioManager: New target audio stream set to '{self._current_stream_id}' with format {response_format}"
        )
        self._playback_control_event.set()

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

    async def end_audio_stream(self, stream_id_override: Optional[str] = None):
        """
        Signals the end of an audio stream by placing an EOS (None) marker into the queue.
        If stream_id_override is provided, it attempts to end that specific stream.
        Otherwise, it attempts to end the stream identified by `_current_stream_id`.
        This method is idempotent: it will only queue an EOS once for a given stream ID
        during its active lifecycle (until it's fully cleaned up).
        """
        target_stream_id = (
            stream_id_override
            if stream_id_override is not None
            else self._current_stream_id
        )

        if target_stream_id is None:
            logger.info(
                f"AudioManager.end_audio_stream: No target stream to end (target_stream_id is None). "
                f"Provided override: {stream_id_override}, _current_stream_id: {self._current_stream_id}"
            )
            return

        if target_stream_id in self._eos_queued_for_streams:
            logger.info(
                f"AudioManager.end_audio_stream: EOS already signaled for stream '{target_stream_id}'. Ignoring duplicate request."
            )
            return

        logger.info(
            f"AudioManager: Signaling end of audio stream '{target_stream_id}' by queueing EOS."
        )
        await self.audio_chunk_queue.put(
            (target_stream_id, None)
        )  # (stream_id, None) is the EOS marker
        self._eos_queued_for_streams.add(target_stream_id)
        self._playback_control_event.set()  # Signal playback loop to check state

    def _prepare_new_playback_stream(self, stream_id: str) -> _PlaybackStream:
        """Prepares OS pipes and FFmpegPCMAudio source for a new playback stream."""
        logger.debug(f"Preparing playback resources for stream '{stream_id}'")
        r_pipe, w_pipe = os.pipe()

        # FFmpegPCMAudio will read from the read-end of the pipe (r_pipe).
        # The _feed_audio_to_pipe task will write to the write-end (w_pipe).
        # The audio coming from the AI service is in the format specified by
        # self._current_response_format when the stream was started.
        if not self._current_response_format:
            # This is a fallback for safety, but should not be reached in normal operation
            # as start_new_audio_stream now requires the format.
            logger.error(
                "Cannot prepare playback stream: response format not set. Falling back to 24kHz mono."
            )
            self._current_response_format = (24000, 1)

        playback_input_frame_rate, playback_input_channels = (
            self._current_response_format
        )

        audio_source = FFmpegPCMAudio(
            os.fdopen(r_pipe, "rb"),  # Source for FFmpeg is the read-end of the pipe
            pipe=True,
            # FFmpeg input options: raw PCM, from AI service's output format
            before_options=f"-f {Config.FFMPEG_PCM_FORMAT} -ar {playback_input_frame_rate} -ac {playback_input_channels}",
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
                    raise

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
                        break
                    continue

                if (
                    item_data is None
                ):  # EOS (End Of Stream) marker for this specific stream
                    logger.info(
                        f"Feeder for '{feeder_stream_id}': EOS marker received."
                    )
                    self.audio_chunk_queue.task_done()
                    break

                # Defensive check: If the global target stream ID has changed *while this feeder was processing*,
                # it should stop to prevent feeding data to a potentially obsolete FFmpeg process.
                if self._current_stream_id != feeder_stream_id:
                    logger.info(
                        f"Feeder for '{feeder_stream_id}': Global target stream '{self._current_stream_id}' no longer matches. Stopping feeder."
                    )
                    self.audio_chunk_queue.task_done()
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

        # Only clear _current_stream_id if it still refers to the stream being cleaned up.
        # This prevents a race condition where a new stream might have been started
        # and _current_stream_id updated, then this cleanup for an old stream
        # incorrectly nullifies it.
        if self._current_stream_id == stream_id:
            logger.info(
                f"AudioManager: Cleared _current_stream_id '{self._current_stream_id}' as its playback and cleanup finished."
            )
            self._current_stream_id = None
            self._current_response_format = None
        else:
            logger.info(
                f"AudioManager: Stream '{stream_id}' cleanup finished, but _current_stream_id is now '{self._current_stream_id}'. Not clearing _current_stream_id."
            )

        # Remove from the set tracking EOS-queued streams as this stream is now fully processed.
        self._eos_queued_for_streams.discard(stream_id)
        logger.info(f"Finished cleaning up stream '{stream_id}'")

    async def playback_loop(self, voice_client: voice_recv.VoiceRecvClient) -> None:
        """
        Main loop for managing audio playback.

        This loop operates as a state machine driven by `_current_stream_id`, which acts
        as the "target" stream to be played. It handles one audio stream at a time,
        ensuring smooth transitions and proper resource cleanup.

        The loop's cycle is triggered by `_playback_control_event` and has two main phases:
        1.  **Phase 1: Stop/Transition:** If a stream is currently being processed
            (`current_stream_processor` exists) but the global target
            (`_current_stream_id`) has changed or been cleared, this phase stops the
            current playback and cleans up all associated resources (tasks, pipes, etc.).
            It relies on the `playback_done_event`, which is set by the `after=`
            callback of `voice_client.play()`, to know when it's safe to clean up.

        2.  **Phase 2: Start New Stream:** If a new target `_current_stream_id` is set
            and no stream is currently being processed, this phase prepares all necessary
            resources for the new stream (creates pipes, FFmpeg source, feeder task)
            and starts playback.

        This design ensures that only one audio stream is active at any given time and
        that resources from a previous stream are fully released before a new one begins.
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
                            # Wake up the main loop to process the end of this stream.
                            self._playback_control_event.set()
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

                    # Playback has been started. The loop will now wait for the next control event.
                    # The 'after' callback will trigger the next state change (cleanup).

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
