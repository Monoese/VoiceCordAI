import asyncio
import io
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import discord
from discord import FFmpegPCMAudio
from discord.ext import voice_recv
from pydub import AudioSegment

from src.config.config import Config
from src.exceptions import AudioProcessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _PlaybackStream:
    """Encapsulates resources for a single audio playback instance."""

    stream_id: str
    ffmpeg_audio_source: FFmpegPCMAudio
    pipe_read_fd: int
    pipe_write_fd: int
    feeder_task: Optional[asyncio.Task] = field(default=None, repr=False)


class AudioPlaybackManager:
    """
    Manages streaming audio playback for a specific guild using a self-contained,
    task-based approach for each playback instance to ensure isolation.
    """

    def __init__(self, guild: discord.Guild) -> None:
        """
        Initialize the AudioPlaybackManager for streaming playback.
        """
        self.guild = guild
        self.audio_chunk_queue: asyncio.Queue[Tuple[str, Optional[bytes]]] = (
            asyncio.Queue()
        )
        self._playback_control_event = asyncio.Event()
        self._current_stream_id: Optional[str] = None
        self._current_response_format: Optional[Tuple[int, int]] = None
        self._eos_queued_for_streams: set[str] = set()
        self._manager_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        logger.debug(f"AudioPlaybackManager initialized for guild {self.guild.id}.")

    def start(self) -> None:
        """Starts the main manager loop as a background task."""
        if self._manager_task is None or self._manager_task.done():
            logger.info(f"Starting manager loop task for guild {self.guild.id}.")
            self._manager_task = asyncio.create_task(self._manager_loop())

    async def stop(self) -> None:
        """Stops all background tasks gracefully."""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
        if self._manager_task and not self._manager_task.done():
            self._manager_task.cancel()
            try:
                await self._manager_task
            except asyncio.CancelledError:
                pass  # Expected
        logger.info(f"AudioPlaybackManager stopped for guild {self.guild.id}.")

    def get_current_playing_response_id(self) -> Optional[str]:
        """Returns the base identifier of the current target/playing stream ID."""
        if self._current_stream_id:
            return self._current_stream_id.split("-", 1)[0]
        return None

    async def play_cue(self, cue_name: str) -> None:
        """
        Plays a short audio cue file by treating it as a high-priority stream.

        This method leverages the existing stream management infrastructure to
        play a cue, ensuring it doesn't conflict with ongoing AI responses.

        Args:
            cue_name: The name of the cue ("start_recording" or "end_recording").
        """
        if not self.guild.voice_client:
            return

        cue_path = None
        if cue_name == "start_recording":
            cue_path = Config.AUDIO_CUE_START_RECORDING
        elif cue_name == "end_recording":
            cue_path = Config.AUDIO_CUE_END_RECORDING

        if not cue_path or not cue_path.exists():
            logger.error(f"Audio cue '{cue_name}' not found at path: {cue_path}")
            return

        cue_stream_id = f"cue_{cue_name}_{int(asyncio.get_running_loop().time())}"
        stream_started = False
        try:
            # Decode the mp3 file into raw PCM audio using pydub
            cue_audio = AudioSegment.from_mp3(str(cue_path))
            cue_format = (cue_audio.frame_rate, cue_audio.channels)

            # The manager loop will see this new stream and interrupt any current playback
            await self.start_new_audio_stream(cue_stream_id, cue_format)
            stream_started = True

            buffer = io.BytesIO()
            cue_audio.export(buffer, format="raw")
            await self.add_audio_chunk(buffer.getvalue())

        except AudioProcessingError as e:
            logger.error(
                f"Audio processing error with cue file {cue_name}: {e}", exc_info=True
            )
        except Exception as e:
            logger.error(
                f"Unexpected error processing audio cue file {cue_name}: {e}",
                exc_info=True,
            )
        finally:
            # Signal the end of this specific cue stream so the manager can clean it up
            if stream_started:
                await self.end_audio_stream(stream_id_override=cue_stream_id)

    async def start_new_audio_stream(
        self, stream_id: str, response_format: Tuple[int, int]
    ) -> None:
        """Signals the manager loop to prepare for a new audio stream."""
        if self._current_stream_id is not None and self._current_stream_id != stream_id:
            logger.warning(
                f"Guild {self.guild.id}: Request to start new stream '{stream_id}' while '{self._current_stream_id}' is active."
            )
        self._current_stream_id = stream_id
        self._current_response_format = response_format
        self._playback_control_event.set()

    async def add_audio_chunk(self, audio_chunk: bytes) -> None:
        """Adds an audio chunk to the queue for the current target stream."""
        if self._current_stream_id:
            await self.audio_chunk_queue.put((self._current_stream_id, audio_chunk))

    async def end_audio_stream(self, stream_id_override: Optional[str] = None) -> None:
        """Signals the end of an audio stream by placing an EOS marker into the queue."""
        target_stream_id = stream_id_override or self._current_stream_id
        if target_stream_id and target_stream_id not in self._eos_queued_for_streams:
            logger.info(
                f"Guild {self.guild.id}: Signaling end of audio stream '{target_stream_id}'."
            )
            await self.audio_chunk_queue.put((target_stream_id, None))
            self._eos_queued_for_streams.add(target_stream_id)

    def _prepare_new_playback_stream(self, stream_id: str) -> _PlaybackStream:
        """Prepares OS pipes and FFmpegPCMAudio source for a new playback stream."""
        r_pipe, w_pipe = os.pipe()
        rate, channels = self._current_response_format or (24000, 1)
        audio_source = FFmpegPCMAudio(
            os.fdopen(r_pipe, "rb"),
            pipe=True,
            before_options=f"-f {Config.FFMPEG_PCM_FORMAT} -ar {rate} -ac {channels}",
            options=f"-ar {Config.DISCORD_AUDIO_FRAME_RATE} -ac {Config.DISCORD_AUDIO_CHANNELS}",
        )
        return _PlaybackStream(
            stream_id=stream_id,
            ffmpeg_audio_source=audio_source,
            pipe_read_fd=r_pipe,
            pipe_write_fd=w_pipe,
        )

    async def _feed_audio_to_pipe(self, stream: _PlaybackStream) -> None:
        """Task to feed audio chunks from the queue to the FFmpeg pipe."""
        writer_fp = None
        loop = asyncio.get_running_loop()
        try:
            writer_fp = os.fdopen(stream.pipe_write_fd, "wb")
            while True:
                item_stream_id, item_data = await self.audio_chunk_queue.get()
                try:
                    if item_stream_id != stream.stream_id:
                        continue
                    if item_data is None:
                        break
                    await loop.run_in_executor(None, writer_fp.write, item_data)
                    await loop.run_in_executor(None, writer_fp.flush)
                finally:
                    self.audio_chunk_queue.task_done()
        except asyncio.CancelledError:
            logger.debug(f"Feeder task for stream '{stream.stream_id}' cancelled.")
        finally:
            if writer_fp:
                try:
                    await loop.run_in_executor(None, writer_fp.close)
                except Exception as e:
                    logger.error(
                        f"Error closing pipe writer for guild {self.guild.id}: {e}"
                    )

    async def _cleanup_playback_stream(self, stream: _PlaybackStream) -> None:
        """Safely cleans up all resources for a given _PlaybackStream instance."""
        if stream.feeder_task and not stream.feeder_task.done():
            stream.feeder_task.cancel()
            try:
                await stream.feeder_task
            except asyncio.CancelledError:
                pass
        stream.ffmpeg_audio_source.cleanup()
        if self._current_stream_id == stream.stream_id:
            self._current_stream_id = None
            self._current_response_format = None
        self._eos_queued_for_streams.discard(stream.stream_id)
        logger.debug(f"Cleaned up resources for stream '{stream.stream_id}'.")

    async def _monitor_playback(
        self, stream: _PlaybackStream, voice_client: voice_recv.VoiceRecvClient
    ):
        """A self-contained task to play one audio stream and clean up."""
        try:
            stream.feeder_task = asyncio.create_task(self._feed_audio_to_pipe(stream))
            voice_client.play(stream.ffmpeg_audio_source)
            while voice_client.is_playing() or voice_client.is_paused():
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info(
                f"Playback monitor for stream '{stream.stream_id}' in guild {self.guild.id} cancelled."
            )
            if voice_client.is_playing() or voice_client.is_paused():
                # Use stop_playing() instead of stop() to preserve audio reception
                voice_client.stop_playing()
        except Exception as e:
            logger.error(
                f"Error in playback monitor for guild {self.guild.id}: {e}",
                exc_info=True,
            )
        finally:
            await self._cleanup_playback_stream(stream)
            logger.info(
                f"Finished playback for stream '{stream.stream_id}' in guild {self.guild.id}."
            )

    async def _manager_loop(self) -> None:
        """Main loop that manages starting and stopping playback monitors."""
        try:
            while True:
                await self._playback_control_event.wait()
                self._playback_control_event.clear()

                if self._monitor_task and not self._monitor_task.done():
                    self._monitor_task.cancel()
                    await asyncio.sleep(0.1)

                voice_client = self.guild.voice_client
                if (
                    self._current_stream_id
                    and isinstance(voice_client, voice_recv.VoiceRecvClient)
                    and voice_client.is_connected()
                ):
                    logger.info(
                        f"Manager loop for guild {self.guild.id}: Starting new playback monitor for stream '{self._current_stream_id}'."
                    )
                    new_stream = self._prepare_new_playback_stream(
                        self._current_stream_id
                    )
                    self._monitor_task = asyncio.create_task(
                        self._monitor_playback(new_stream, voice_client)
                    )
                elif not voice_client or not voice_client.is_connected():
                    logger.warning(
                        f"Manager loop for guild {self.guild.id}: No valid voice client, cannot start playback."
                    )

        except asyncio.CancelledError:
            logger.info(f"Manager loop for guild {self.guild.id} cancelled.")
        finally:
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
            logger.info(f"Manager loop for guild {self.guild.id} exited.")
