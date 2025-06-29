import asyncio
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Callable

from discord import FFmpegPCMAudio
from discord.ext import voice_recv

from src.config.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PlaybackState(Enum):
    """Represents the operational state of the playback loop."""

    IDLE = auto()  # Not playing anything.
    PLAYING = auto()  # Actively playing an audio stream.
    STOPPING = auto()  # In the process of stopping a stream and cleaning up resources.


@dataclass
class _PlaybackStream:
    """Encapsulates resources and state for a single audio playback instance."""

    stream_id: str
    ffmpeg_audio_source: FFmpegPCMAudio
    pipe_read_fd: int
    pipe_write_fd: int
    playback_done_event: asyncio.Event
    feeder_task: Optional[asyncio.Task] = field(default=None, repr=False)


class AudioPlaybackManager:
    """
    Manages streaming audio playback for the Discord bot.

    This class handles:
    - Managing streaming audio playback in Discord voice channels
    - Queuing audio chunks for real-time playback
    - A state machine for handling playback lifecycle (playing, stopping, idle)
    """

    def __init__(self) -> None:
        """
        Initialize the AudioPlaybackManager for streaming playback.

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
        # _current_stream_id stores the ID of the stream that the AudioPlaybackManager is currently
        # *expecting* to receive chunks for or has been told to start.
        # This is set by start_new_audio_stream().
        self._current_stream_id: Optional[str] = None
        self._playback_state: PlaybackState = PlaybackState.IDLE
        self._current_response_format: Optional[Tuple[int, int]] = None
        self._eos_queued_for_streams: set[str] = (
            set()
        )  # Tracks streams for which EOS has been queued
        logger.debug("AudioPlaybackManager initialized for streaming playback.")

    def get_current_playing_response_id(self) -> Optional[str]:
        """
        Returns the base identifier of the current target/playing stream ID.

        For stream IDs with a composite format (e.g., "response_id-item_id"),
        this returns the part before the first hyphen. For simple IDs, it returns
        the entire ID.

        Returns:
            Optional[str]: The base identifier of the current stream, or None if no
                           stream is active.
        """
        if self._current_stream_id:
            parts = self._current_stream_id.split("-", 1)
            if parts:
                return parts[0]
        return None

    async def start_new_audio_stream(
        self, stream_id: str, response_format: Tuple[int, int]
    ) -> None:
        """
        Signals the playback_loop to prepare for a new audio stream.
        If another stream is active, it will be gracefully stopped and cleaned up
        by the playback_loop during the transition.

        Args:
            stream_id: A unique identifier for the new audio stream.
            response_format: A tuple (frame_rate, channels) specifying the format
                             of the incoming audio from the AI service.
        """
        if self._current_stream_id is not None and self._current_stream_id != stream_id:
            logger.warning(
                f"AudioPlaybackManager: Request to start new stream '{stream_id}' while '{self._current_stream_id}' is active. "
                "The playback loop will handle the transition."
            )

        self._current_stream_id = stream_id
        self._current_response_format = response_format
        logger.info(
            f"AudioPlaybackManager: New target audio stream set to '{self._current_stream_id}' with format {response_format}"
        )
        self._playback_control_event.set()

    async def add_audio_chunk(self, audio_chunk: bytes) -> None:
        """
        Adds an audio chunk to the queue for the stream identified by `_current_stream_id`.

        Args:
            audio_chunk: The audio data bytes to be added to the queue.
        """
        current_target_stream_id = self._current_stream_id
        if current_target_stream_id is None:
            logger.warning(
                "AudioPlaybackManager: No active target stream (_current_stream_id is None) to add audio chunk. Ignoring."
            )
            return
        await self.audio_chunk_queue.put((current_target_stream_id, audio_chunk))
        logger.debug(
            f"AudioPlaybackManager: Added chunk of {len(audio_chunk)} bytes to queue for stream {current_target_stream_id}"
        )

    async def end_audio_stream(self, stream_id_override: Optional[str] = None) -> None:
        """
        Signals the end of an audio stream by placing an EOS (None) marker into the queue.

        This method is idempotent: it will only queue an EOS once for a given stream ID
        during its active lifecycle (until it's fully cleaned up).

        Args:
            stream_id_override: If provided, attempts to end this specific stream ID.
                                Otherwise, it ends the stream identified by `_current_stream_id`.
        """
        target_stream_id = (
            stream_id_override
            if stream_id_override is not None
            else self._current_stream_id
        )

        if target_stream_id is None:
            logger.info(
                f"AudioPlaybackManager.end_audio_stream: No target stream to end (target_stream_id is None). "
                f"Provided override: {stream_id_override}, _current_stream_id: {self._current_stream_id}"
            )
            return

        if target_stream_id in self._eos_queued_for_streams:
            logger.info(
                f"AudioPlaybackManager.end_audio_stream: EOS already signaled for stream '{target_stream_id}'. Ignoring duplicate request."
            )
            return

        logger.info(
            f"AudioPlaybackManager: Signaling end of audio stream '{target_stream_id}' by queueing EOS."
        )
        await self.audio_chunk_queue.put(
            (target_stream_id, None)
        )  # (stream_id, None) is the EOS marker
        self._eos_queued_for_streams.add(target_stream_id)
        self._playback_control_event.set()

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

    async def _feed_audio_to_pipe(self, stream: _PlaybackStream) -> None:
        """
        Task to feed audio chunks from the queue to the FFmpeg pipe for a given stream.

        Args:
            stream: The _PlaybackStream instance whose pipe will receive the audio data.
        """
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
                        None, lambda *args: writer_fp.flush(), None
                    )
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
                    await loop.run_in_executor(
                        None, lambda *args: writer_fp.close(), None
                    )
                except Exception as e:  # pragma: no cover
                    logger.error(
                        f"Feeder for '{feeder_stream_id}': Error closing pipe writer: {e}"
                    )

    async def _cleanup_playback_stream(
        self, stream_to_cleanup: _PlaybackStream
    ) -> None:
        """
        Safely cleans up all resources for a given _PlaybackStream instance.

        Args:
            stream_to_cleanup: The _PlaybackStream instance to clean up.
        """
        stream_id = stream_to_cleanup.stream_id
        logger.info(f"Cleaning up playback resources for stream '{stream_id}'")

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
                f"AudioPlaybackManager: Cleared _current_stream_id '{self._current_stream_id}' as its playback and cleanup finished."
            )
            self._current_stream_id = None
            self._current_response_format = None
        else:
            logger.info(
                f"AudioPlaybackManager: Stream '{stream_id}' cleanup finished, but _current_stream_id is now '{self._current_stream_id}'. Not clearing _current_stream_id."
            )

        # Remove from the set tracking EOS-queued streams as this stream is now fully processed.
        self._eos_queued_for_streams.discard(stream_id)
        logger.info(f"Finished cleaning up stream '{stream_id}'")

    async def _stop_and_cleanup_processor(
        self, processor: _PlaybackStream, voice_client: voice_recv.VoiceRecvClient
    ) -> None:
        """Stops voice client playback and cleans up all resources for a processor.

        Args:
            processor: The playback stream processor to clean up.
            voice_client: The Discord voice client used for playback.
        """
        processor_id = processor.stream_id
        logger.info(
            f"PlaybackLoop: Stopping and cleaning up processor for '{processor_id}'."
        )
        if voice_client.is_playing() or voice_client.is_paused():
            logger.debug(f"Calling voice_client.stop() for stream '{processor_id}'.")
            voice_client.stop()

        if not processor.playback_done_event.is_set():
            logger.debug(
                f"Waiting for playback_done_event for '{processor_id}' after signaling stop."
            )
            try:
                await asyncio.wait_for(
                    processor.playback_done_event.wait(),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:  # pragma: no cover
                logger.warning(
                    f"Timeout waiting for playback_done_event for '{processor_id}'. Proceeding with cleanup."
                )
        await self._cleanup_playback_stream(processor)
        logger.info(
            f"Finished stopping and cleaning up processor for '{processor_id}'."
        )

    def _start_new_processor(
        self,
        target_id: str,
        voice_client: voice_recv.VoiceRecvClient,
        after_callback: Callable,
    ) -> _PlaybackStream:
        """Prepares resources for a new stream and starts playback.

        Args:
            target_id: The ID for the new stream.
            voice_client: The Discord voice client to play audio on.
            after_callback: The callback function to execute when playback finishes.

        Returns:
            The newly created and started playback stream processor.
        """
        logger.info(
            f"PlaybackLoop: Preparing to start new processor for '{target_id}'."
        )
        processor = self._prepare_new_playback_stream(target_id)
        processor.feeder_task = asyncio.create_task(self._feed_audio_to_pipe(processor))

        logger.info(
            f"PlaybackLoop: Starting voice_client.play for stream '{target_id}'"
        )
        voice_client.play(
            processor.ffmpeg_audio_source,
            after=lambda err: after_callback(err, processor.stream_id),
        )
        return processor

    async def playback_loop(self, voice_client: voice_recv.VoiceRecvClient) -> None:
        """
        Main loop for managing audio playback via an explicit state machine.
        This loop checks the current playback state and performs actions accordingly.

        Args:
            voice_client: The Discord voice client used for playing audio.
        """
        current_stream_processor: Optional[_PlaybackStream] = None

        def after_playback_callback(
            error: Optional[Exception], played_source_stream_id: str
        ):
            """
            Callback passed to voice_client.play(), triggered on playback end/error.
            This callback is the primary trigger for transitioning to the STOPPING state.
            Args:
                error: The exception that occurred, if any.
                played_source_stream_id: The ID of the stream that finished playing.
            """
            log_msg = f"PlaybackLoop: 'after' callback for stream '{played_source_stream_id}'."
            if error:  # pragma: no cover
                log_msg += f" Error: {error}"
            logger.info(log_msg)

            if (
                current_stream_processor
                and current_stream_processor.stream_id == played_source_stream_id
            ):
                current_stream_processor.playback_done_event.set()
                self._playback_state = PlaybackState.STOPPING
                self._playback_control_event.set()
            else:  # pragma: no cover
                logger.warning(
                    f"PlaybackLoop: 'after' callback for stale stream '{played_source_stream_id}'."
                )

        try:
            while True:
                await self._playback_control_event.wait()
                self._playback_control_event.clear()

                logger.debug(
                    f"Playback state machine: Awakened in state {self._playback_state.name}"
                )

                # State: PLAYING -> Check if we need to transition to STOPPING
                if self._playback_state == PlaybackState.PLAYING:
                    if current_stream_processor:
                        global_target_id = self._current_stream_id
                        if (
                            global_target_id is None
                            or global_target_id != current_stream_processor.stream_id
                        ):
                            logger.info(
                                "Playback state machine: Target changed. Transitioning from PLAYING to STOPPING."
                            )
                            self._playback_state = PlaybackState.STOPPING
                    else:  # pragma: no cover
                        # This should not happen in normal operation, but acts as a safeguard.
                        logger.error(
                            "In PLAYING state but no processor exists. Correcting to IDLE."
                        )
                        self._playback_state = PlaybackState.IDLE

                # State: STOPPING -> Clean up and transition to IDLE
                if self._playback_state == PlaybackState.STOPPING:
                    logger.info("Playback state machine: Handling STOPPING state.")
                    if current_stream_processor:
                        await self._stop_and_cleanup_processor(
                            current_stream_processor, voice_client
                        )
                        current_stream_processor = None
                    self._playback_state = PlaybackState.IDLE
                    logger.info(
                        "Playback state machine: Transitioned from STOPPING to IDLE."
                    )
                    # We might be able to start a new stream immediately, so we re-set the event.
                    self._playback_control_event.set()

                # State: IDLE -> Check if we can start PLAYING
                if self._playback_state == PlaybackState.IDLE:
                    if self._current_stream_id and not current_stream_processor:
                        logger.info(
                            "Playback state machine: New target detected. Transitioning from IDLE to PLAYING."
                        )
                        current_stream_processor = self._start_new_processor(
                            self._current_stream_id,
                            voice_client,
                            after_playback_callback,
                        )
                        self._playback_state = PlaybackState.PLAYING
                    else:
                        logger.debug("Playback state machine: Idling.")

        except asyncio.CancelledError:  # pragma: no cover
            logger.info("PlaybackLoop: Cancelled.")
        except Exception as e:  # pragma: no cover
            logger.error(f"PlaybackLoop: Unhandled error: {e}", exc_info=True)
        finally:
            logger.info("PlaybackLoop: Exiting final cleanup.")
            if voice_client and (
                voice_client.is_playing() or voice_client.is_paused()
            ):  # pragma: no cover
                voice_client.stop()
            if current_stream_processor:  # pragma: no cover
                if not current_stream_processor.playback_done_event.is_set():
                    current_stream_processor.playback_done_event.set()
                await self._cleanup_playback_stream(current_stream_processor)
            self._playback_control_event.clear()
