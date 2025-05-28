import asyncio
import os
import base64
from typing import Any, Optional, ByteString

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
        self.audio_chunk_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._playback_control_event = asyncio.Event()  # Used to signal playback loop
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
        if self._current_stream_id is None:
            logger.warning(
                "AudioManager: No active stream to add audio chunk. Ignoring."
            )
            return
        await self.audio_chunk_queue.put(audio_chunk)
        logger.debug(
            f"AudioManager: Added chunk of {len(audio_chunk)} bytes to stream {self._current_stream_id}"
        )

    async def end_audio_stream(self):
        """Signals the end of the current audio stream."""
        if self._current_stream_id is None:
            logger.info("AudioManager: No active stream to end or already ended.")
            return

        logger.info(
            f"AudioManager: Signaling end of audio stream {self._current_stream_id}"
        )
        await self.audio_chunk_queue.put(None)  # None is the sentinel for end of stream

    async def playback_loop(self, voice_client: voice_recv.VoiceRecvClient) -> None:
        """
        Continuous loop that handles audio playback in the voice channel.
        Streams audio chunks from `audio_chunk_queue` through FFmpeg.
        """
        active_ffmpeg_process = None
        current_processing_stream_id = None
        w_pipe_local = (
            None  # Keep track of w_pipe locally for cleanup in this loop's scope
        )

        try:
            while True:
                await self._playback_control_event.wait()
                self._playback_control_event.clear()

                if self._current_stream_id is None and self.audio_chunk_queue.empty():
                    continue

                current_processing_stream_id = self._current_stream_id
                if current_processing_stream_id is None:
                    logger.warning(
                        "PlaybackLoop: Woke up but _current_stream_id is None. Skipping."
                    )
                    continue

                logger.info(
                    f"PlaybackLoop: Starting playback for stream {current_processing_stream_id}"
                )

                r_pipe, w_pipe_local = os.pipe()  # Assign to w_pipe_local
                playback_done = asyncio.Event()

                # Ensure FFmpegPCMAudio is created within the loop for each new stream
                audio_source = FFmpegPCMAudio(
                    os.fdopen(r_pipe, "rb"),  # Pass the read end of the pipe
                    pipe=True,
                    before_options="-f s16le -ar 24000 -ac 1",
                    options="-ar 48000 -ac 2",
                )
                active_ffmpeg_process = (
                    audio_source  # Keep track of the current FFmpeg process
                )

                def log_playback_finished(error: Optional[Exception]) -> None:
                    # Use a different variable for stream_id_in_callback to capture its value at definition time
                    stream_id_in_callback = current_processing_stream_id
                    if error:
                        logger.error(
                            f"Error during audio playback for stream {stream_id_in_callback}: {error}"
                        )
                    else:
                        logger.info(
                            f"PlaybackLoop: Finished playing audio for stream {stream_id_in_callback} successfully"
                        )
                    playback_done.set()

                voice_client.play(audio_source, after=log_playback_finished)

                async def feed_ffmpeg(
                    pipe_to_write,
                ):  # Pass w_pipe_local as an argument
                    # Use different stream_id_in_feeder for clarity within this task
                    stream_id_in_feeder = current_processing_stream_id
                    writer_fp = None
                    loop = asyncio.get_running_loop()
                    chunk_being_processed = None
                    try:
                        writer_fp = os.fdopen(pipe_to_write, "wb")
                        while True:
                            chunk_being_processed = await self.audio_chunk_queue.get()
                            if chunk_being_processed is None:
                                logger.debug(
                                    f"PlaybackLoop: EOS for stream {stream_id_in_feeder}, signaling FFmpeg."
                                )
                                self.audio_chunk_queue.task_done()
                                break

                            if (
                                not self._current_stream_id
                                or self._current_stream_id != stream_id_in_feeder
                            ):
                                logger.warning(
                                    f"PlaybackLoop: Stream changed during feed. Expected {stream_id_in_feeder}, now {self._current_stream_id}. Stopping feed for old stream."
                                )
                                self.audio_chunk_queue.task_done()
                                break

                            try:
                                await loop.run_in_executor(
                                    None, writer_fp.write, chunk_being_processed
                                )
                                await loop.run_in_executor(None, writer_fp.flush)
                                logger.debug(
                                    f"PlaybackLoop: Fed {len(chunk_being_processed)} bytes to FFmpeg for stream {stream_id_in_feeder}"
                                )
                            except Exception as e:
                                logger.error(
                                    f"PlaybackLoop: Error writing/flushing to FFmpeg pipe for stream {stream_id_in_feeder}: {e}"
                                )
                                self.audio_chunk_queue.task_done()
                                break

                            self.audio_chunk_queue.task_done()
                            chunk_being_processed = None
                    except Exception as e:
                        logger.error(
                            f"PlaybackLoop: Unhandled error in feed_ffmpeg task for stream {stream_id_in_feeder}: {e}"
                        )
                        if (
                            chunk_being_processed is not None
                            and hasattr(self.audio_chunk_queue, "_unfinished_tasks")
                            and self.audio_chunk_queue._unfinished_tasks > 0
                        ):
                            self.audio_chunk_queue.task_done()
                    finally:
                        if writer_fp:
                            try:
                                logger.debug(
                                    f"PlaybackLoop: Closing FFmpeg pipe writer for stream {stream_id_in_feeder}."
                                )
                                await loop.run_in_executor(
                                    None, writer_fp.close
                                )  # writer_fp.close() also closes pipe_to_write
                                logger.debug(
                                    f"PlaybackLoop: FFmpeg pipe writer closed for stream {stream_id_in_feeder}."
                                )
                            except Exception as e:
                                logger.error(
                                    f"PlaybackLoop: Error closing FFmpeg pipe writer for stream {stream_id_in_feeder}: {e}"
                                )
                        else:  # if os.fdopen(pipe_to_write, 'wb') failed
                            try:
                                if pipe_to_write is not None:
                                    os.close(
                                        pipe_to_write
                                    )  # Close raw descriptor if fdopen failed
                                    logger.debug(
                                        f"PlaybackLoop: Raw write pipe {pipe_to_write} closed directly for stream {stream_id_in_feeder}."
                                    )
                            except OSError as e:
                                logger.error(
                                    f"PlaybackLoop: Error closing raw pipe {pipe_to_write} for stream {stream_id_in_feeder}: {e}"
                                )

                # Pass w_pipe_local to feed_ffmpeg
                feeder_task = asyncio.create_task(feed_ffmpeg(w_pipe_local))

                await playback_done.wait()
                await feeder_task  # Ensure feeder task is complete before proceeding

                logger.info(
                    f"PlaybackLoop: Fully completed stream {current_processing_stream_id}"
                )
                if active_ffmpeg_process:  # Cleanup the FFmpeg process after it's done
                    active_ffmpeg_process.cleanup()
                    active_ffmpeg_process = None

                if self._current_stream_id == current_processing_stream_id:
                    self._current_stream_id = None

                # w_pipe_local is closed by feed_ffmpeg's writer_fp.close()
                # r_pipe is closed by FFmpegPCMAudio's cleanup or when the process ends
                w_pipe_local = None  # Reset for next iteration
                current_processing_stream_id = None

        except asyncio.CancelledError:
            logger.info("PlaybackLoop: Cancelled.")
        except Exception as e:
            logger.error(f"PlaybackLoop: Unhandled error: {e}", exc_info=True)
        finally:
            logger.info("PlaybackLoop: Exiting.")
            if active_ffmpeg_process:
                active_ffmpeg_process.cleanup()
            if (
                w_pipe_local is not None
            ):  # If loop exited abruptly before feed_ffmpeg closed it
                try:
                    os.close(w_pipe_local)
                except OSError:
                    pass
            self._playback_control_event.clear()
            if (
                self._current_stream_id == current_processing_stream_id
            ):  # ensure reset if loop exits abnormally
                self._current_stream_id = None
