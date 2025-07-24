"""
Audio Sink Implementations for different operational modes.

This module provides the concrete AudioSink classes for the bot's different
operational modes, such as ManualControl and RealtimeTalk. Each sink is
responsible for processing raw audio from users according to its mode's logic.
"""

import asyncio
import audioop
import os
import threading
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Dict, List, Optional, Set

import discord
import numpy as np
import webrtcvad
from discord.ext import voice_recv
from openwakeword.model import Model

from src.bot.state import BotState, BotStateEnum, RecordingMethod
from src.config.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioSink(voice_recv.AudioSink, ABC):
    """
    Abstract base class for all custom audio sinks.

    Defines a common interface for dynamic user management and resource cleanup
    that all operational-mode-specific sinks must implement.
    """

    @abstractmethod
    def add_user(self, user_id: int) -> None:
        """
        Dynamically starts processing audio for a new user.

        Args:
            user_id: The ID of the user to start processing.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_user(self, user_id: int) -> None:
        """
        Dynamically stops processing audio for a user.

        Args:
            user_id: The ID of the user to stop processing.
        """
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        """
        Release all underlying resources held by the sink.

        This must be called to ensure no memory leaks or orphaned tasks.
        """
        raise NotImplementedError


class VADAnalyzer:
    """
    A stateful Voice Activity Detection (VAD) analyzer for detecting speech boundaries.

    This class implements a robust VAD algorithm that tracks speech and silence periods
    to reliably detect when a user has finished speaking. It uses configurable thresholds
    to avoid false positives from short pauses or background noise.

    Threading Model:
    - VAD processing is called from audio sink write() methods (Discord's audio thread)
    - Callbacks are scheduled on the main event loop using call_soon_threadsafe()
    - This ensures thread-safe communication between audio processing and async logic

    State Machine:
    - SILENT: Initial state, waiting for sustained speech
    - SPEAKING: Speech detected, monitoring for end-of-speech silence
    - TRIGGERED: Speech end detected, callback scheduled (terminal state)
    """

    def __init__(
        self,
        on_speech_end: Callable[[], Awaitable[None]],
        sample_rate: int,
        frame_duration_ms: int,
        min_speech_duration_ms: int,
        silence_timeout_ms: int,
        grace_period_ms: int,
        loop: asyncio.AbstractEventLoop,
    ):
        self._on_speech_end = on_speech_end
        self._loop = loop
        self._sample_rate = sample_rate
        self._frame_duration_ms = frame_duration_ms
        self._min_speech_frames = min_speech_duration_ms // frame_duration_ms
        self._silence_frames_timeout = silence_timeout_ms // frame_duration_ms
        self._grace_period_frames = grace_period_ms // frame_duration_ms
        self._frame_size = (sample_rate * frame_duration_ms) // 1000
        self._frame_bytes = self._frame_size * Config.SAMPLE_WIDTH
        self._vad = webrtcvad.Vad(Config.VAD_AGGRESSIVENESS)
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._frames_processed = 0
        self._is_speech = False
        self._triggered = False

    def process(self, frame: bytes):
        if len(frame) != self._frame_bytes:
            return
        self._frames_processed += 1
        is_speech = self._vad.is_speech(frame, self._sample_rate)
        if not self._is_speech:
            if is_speech:
                self._speech_frame_count += 1
                if self._speech_frame_count >= self._min_speech_frames:
                    self._is_speech = True
                    self._speech_frame_count = 0
                    self._silence_frame_count = 0
            else:
                self._speech_frame_count = 0
        else:
            if not is_speech:
                self._silence_frame_count += 1
                if self._silence_frame_count >= self._silence_frames_timeout:
                    if (
                        not self._triggered
                        and self._frames_processed > self._grace_period_frames
                    ):
                        self._triggered = True
                        self._loop.call_soon_threadsafe(
                            asyncio.create_task, self._on_speech_end()
                        )
            else:
                self._silence_frame_count = 0

    def reset(self):
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._frames_processed = 0
        self._is_speech = False
        self._triggered = False


class RealtimeMixingSink(AudioSink):
    """
    An audio sink for the 'RealtimeTalk' mode - continuous multi-user conversation.

    OVERVIEW:
    This sink creates a real-time mixed audio stream from multiple consented users,
    suitable for streaming to AI services that support multi-party conversations.

    AUDIO PROCESSING PIPELINE:
    1. Raw PCM audio arrives from Discord (48kHz, 16-bit, stereo, 20ms chunks)
    2. Audio is buffered per-user in _user_buffers
    3. Mixer task runs every 20ms, synchronized with Discord's audio timing
    4. For each user: extract 20ms chunk OR pad with silence if no audio available
    5. Mix all chunks into single audio stream using audioop.add()
    6. Place mixed audio on output_queue for consumption by GuildSession

    THREADING MODEL:
    - write() method: Called from Discord's audio thread, thread-safe buffer updates
    - _mixer_loop(): Async task on main event loop, uses run_in_executor for CPU-intensive mixing
    - This design prevents blocking the event loop during audio mixing operations

    BUFFER MANAGEMENT:
    - Each user has independent audio buffer (bytearray for efficient append/slice)
    - Silence padding ensures continuous streams even when users aren't speaking
    - Dynamic user management allows real-time consent changes during conversation
    """

    def __init__(self, bot_state: BotState, initial_consented_users: Set[int]):
        super().__init__()
        self._bot_state = bot_state
        self.output_queue: asyncio.Queue[bytes] = asyncio.Queue()

        self._consented_users: Set[int] = set()
        self._user_buffers: Dict[int, bytearray] = {}

        # Audio timing and format constants
        # 20ms of 48kHz, 16-bit, stereo audio = 3840 bytes
        self._chunk_size = (
            48000 // 50 * Config.DISCORD_AUDIO_CHANNELS * Config.SAMPLE_WIDTH
        )
        self._silence_chunk = (
            b"\x00" * self._chunk_size
        )  # Pre-computed silence for padding

        self._mixer_task: Optional[asyncio.Task] = None
        self._loop = asyncio.get_running_loop()

        # Initialize with users who have already consented
        for user_id in initial_consented_users:
            self.add_user(user_id)

        self.start()

    def add_user(self, user_id: int) -> None:
        if user_id in self._consented_users:
            return
        logger.info(f"Adding user {user_id} to RealtimeMixingSink.")
        self._consented_users.add(user_id)
        self._user_buffers[user_id] = bytearray()

    def remove_user(self, user_id: int) -> None:
        if user_id not in self._consented_users:
            return
        logger.info(f"Removing user {user_id} from RealtimeMixingSink.")
        self._consented_users.discard(user_id)
        self._user_buffers.pop(user_id, None)

    def cleanup(self) -> None:
        logger.info("Cleaning up RealtimeMixingSink.")
        if self._mixer_task and not self._mixer_task.done():
            self._mixer_task.cancel()
        self._consented_users.clear()
        self._user_buffers.clear()

    def wants_opus(self) -> bool:
        return False

    def write(self, user: discord.User, data: voice_recv.VoiceData) -> None:
        if user and user.id in self._consented_users:
            if user.id not in self._user_buffers:
                self.add_user(user.id)  # Should not happen, but defensive
            self._user_buffers[user.id].extend(data.pcm)

    def start(self):
        """Starts the mixer background task."""
        if self._mixer_task is None or self._mixer_task.done():
            self._mixer_task = asyncio.create_task(self._mixer_loop())
            logger.info("RealtimeMixingSink mixer task started.")

    def _mix_audio_chunks(self, chunks: List[bytes]) -> bytes:
        """Synchronous function to mix multiple audio chunks."""
        if not chunks:
            return self._silence_chunk

        mixed_chunk = chunks[0]
        for chunk in chunks[1:]:
            # audioop.add requires both chunks to be the same length
            if len(chunk) == len(mixed_chunk):
                mixed_chunk = audioop.add(mixed_chunk, chunk, Config.SAMPLE_WIDTH)
        return mixed_chunk

    async def _mixer_loop(self):
        """The background loop that mixes audio at regular intervals."""
        try:
            while True:
                # Sleep for 20ms, the duration of one Discord audio frame
                await asyncio.sleep(0.02)

                chunks_to_mix = []
                for user_id in self._consented_users:
                    buffer = self._user_buffers.get(user_id)
                    if buffer and len(buffer) >= self._chunk_size:
                        # User has audio data, slice it
                        chunks_to_mix.append(buffer[: self._chunk_size])
                        del buffer[: self._chunk_size]
                    else:
                        # User is silent, pad with silence
                        chunks_to_mix.append(self._silence_chunk)

                if chunks_to_mix:
                    mixed_audio = await self._loop.run_in_executor(
                        None, self._mix_audio_chunks, chunks_to_mix
                    )
                    await self.output_queue.put(mixed_audio)
        except asyncio.CancelledError:
            logger.info("RealtimeMixingSink mixer task cancelled.")
        except Exception as e:
            logger.error(f"Error in RealtimeMixingSink mixer loop: {e}", exc_info=True)


class ManualControlSink(AudioSink):
    """
    An audio sink for the 'ManualControl' mode - request/response interaction model.

    OVERVIEW:
    This sink implements a sophisticated dual-mode audio processor that alternates
    between listening for wake words from all users and recording commands from
    a single authority user. It coordinates wake word detection, voice activity
    detection, and audio buffering for command processing.

    OPERATIONAL MODES:

    1. STANDBY MODE (bot_state.current_state == BotStateEnum.STANDBY):
       - Continuous parallel wake word detection for all consented users
       - Each user gets dedicated openWakeWord model instance
       - Audio processing: 48kHz stereo -> 16kHz mono -> 80ms chunks for wake word models
       - When wake word detected: triggers callback to GuildSession, clears buffers

    2. RECORDING MODE (bot_state.current_state == BotStateEnum.RECORDING):
       - Single-user audio capture from authority_user
       - Optional VAD (Voice Activity Detection) for wake word triggered recordings
       - Audio buffering: Raw PCM accumulated in _authority_buffer
       - VAD processing: 48kHz stereo -> 16kHz mono -> configurable frame analysis
       - When speech ends: triggers callback with captured audio data

    AUDIO PROCESSING PIPELINE:

    Wake Word Path (STANDBY):
    1. Raw PCM (48kHz, 16-bit, stereo) -> _user_audio_buffers[user_id]
    2. Resample to 16kHz mono using audioop.ratecv (maintains state per user)
    3. Buffer resampled audio in _ww_resampled_buffers[user_id]
    4. Process in 1280-byte chunks (80ms of 16kHz mono) through openWakeWord models
    5. On detection: reset model state, clear buffers, schedule callback

    VAD Path (RECORDING with wake word trigger):
    1. Raw PCM -> _authority_buffer (for final output) + _vad_raw_buffer (for VAD)
    2. Resample VAD stream to 16kHz mono for webrtcvad compatibility
    3. Process in configurable frame sizes (10ms, 20ms, or 30ms)
    4. VADAnalyzer tracks speech/silence patterns with configurable thresholds
    5. On speech end: callback with buffered audio, reset all state

    THREADING MODEL:
    - write() method: Discord audio thread, must be non-blocking and thread-safe
    - Wake word processing: Synchronous in write() method (fast, <1ms per chunk)
    - VAD processing: Scheduled on event loop via run_coroutine_threadsafe()
    - _vad_monitor_loop(): Async task that injects silence when no audio received
    - Callbacks: Scheduled on main event loop for integration with GuildSession

    CONCURRENCY & SAFETY:
    - _vad_lock: Protects VAD processing from race conditions between real audio and silence injection
    - Atomic state capture in write() method prevents race conditions during state transitions
    - Thread-safe callback scheduling using run_coroutine_threadsafe()
    - Graceful cleanup with proper task cancellation and resource deallocation

    BUFFER MANAGEMENT:
    - _user_audio_buffers: Raw 48kHz audio per user for wake word detection
    - _ww_resampled_buffers: 16kHz mono audio per user, ready for wake word models
    - _authority_buffer: Final output buffer containing command audio
    - _vad_raw_buffer + _vad_resampled_buffer: VAD processing pipeline buffers
    - Automatic buffer clearing prevents memory leaks and audio contamination

    ERROR HANDLING:
    - Wake word model initialization failures are logged but don't crash the sink
    - VAD processing errors are isolated and logged
    - Resource cleanup is comprehensive to prevent memory/task leaks
    """

    def __init__(
        self,
        bot_state: BotState,
        initial_consented_users: Set[int],
        on_wake_word_detected: Callable[[discord.User], Awaitable[None]],
        on_vad_speech_end: Callable[[bytes], Awaitable[None]],
    ):
        super().__init__()
        self._bot_state = bot_state
        self._on_wake_word_detected = on_wake_word_detected
        self._on_vad_speech_end = on_vad_speech_end
        self._loop = asyncio.get_running_loop()
        self._detectors: Dict[int, Model] = {}
        self._user_audio_buffers: Dict[int, bytearray] = {}
        self._authority_buffer = bytearray()
        self._vad_raw_buffer = bytearray()
        self._vad_resampled_buffer = bytearray()
        self._ww_resample_state: Dict[int, Optional[any]] = {}
        self._ww_resampled_buffers: Dict[int, bytearray] = {}
        self._vad_resample_state = None

        # Thread-safe synchronization primitives for TOCTOU fix
        self._authority_buffer_lock = (
            threading.Lock()
        )  # Protects authority audio buffer operations
        self._ww_buffer_locks: Dict[
            int, threading.Lock
        ] = {}  # Per-user wake word buffer protection
        self._vad_flag_lock = threading.Lock()  # Protects VAD flag updates
        self._user_data_lock = (
            threading.Lock()
        )  # CONCURRENCY FIX: Protects all user data dictionaries
        self._is_vad_enabled = False
        self._vad_grace_period_frames = (
            Config.VAD_GRACE_PERIOD_MS // 20
        )  # 20ms per raw discord frame
        self._vad_frames_processed = 0
        self._ww_chunk_size = 1280  # 80ms of 16kHz, 16-bit, mono audio
        self._vad_analyzer: Optional[VADAnalyzer] = None

        # VAD silence injection system - coordinates between real audio and synthetic silence
        self._has_received_audio_for_vad: bool = (
            False  # Flag: real audio received this cycle
        )
        self._vad_monitor_task: Optional[asyncio.Task] = (
            None  # Background silence injection task
        )
        self._silence_chunk_vad = (
            b"\x00" * 3840
        )  # 20ms of 48kHz stereo 16-bit PCM silence
        self._vad_lock = (
            asyncio.Lock()
        )  # Prevents race conditions between real audio and silence

        for user_id in initial_consented_users:
            self.add_user(user_id)

        self.start()

    def add_user(self, user_id: int) -> None:
        if user_id in self._detectors:
            return
        logger.info(f"Adding user {user_id} to ManualControlSink detectors.")
        try:
            # CONCURRENCY FIX: Use dedicated lock for all user data operations
            with self._user_data_lock:
                self._detectors[user_id] = Model(
                    wakeword_models=[Config.WAKE_WORD_MODEL_PATH],
                    vad_threshold=Config.WAKE_WORD_VAD_THRESHOLD,
                )
                self._user_audio_buffers[user_id] = bytearray()
                self._ww_resampled_buffers[user_id] = bytearray()
                self._ww_resample_state[user_id] = None
                self._ww_buffer_locks[user_id] = (
                    threading.Lock()
                )  # Create per-user lock
        except Exception as e:
            logger.error(
                f"Failed to initialize wake word model for user {user_id}: {e}",
                exc_info=True,
            )

    def remove_user(self, user_id: int) -> None:
        if user_id not in self._detectors:
            return
        logger.info(f"Removing user {user_id} from ManualControlSink detectors.")

        # CONCURRENCY FIX: Use dedicated lock for all user data operations
        with self._user_data_lock:
            # Clean up all user data atomically
            self._detectors.pop(user_id, None)
            self._user_audio_buffers.pop(user_id, None)
            self._ww_resample_state.pop(user_id, None)
            self._ww_resampled_buffers.pop(user_id, None)
            self._ww_buffer_locks.pop(
                user_id, None
            )  # Safe to delete after data cleanup

    def cleanup(self) -> None:
        logger.info("Cleaning up ManualControlSink.")
        if self._vad_monitor_task and not self._vad_monitor_task.done():
            self._vad_monitor_task.cancel()
        self._detectors.clear()
        self._user_audio_buffers.clear()
        self._authority_buffer.clear()

    def start(self):
        """
        Initializes and starts the VAD monitor background task.

        The VAD monitor is essential for proper speech end detection as it
        ensures continuous VAD processing even during audio gaps from Discord.
        """
        if self._vad_monitor_task is None or self._vad_monitor_task.done():
            self._vad_monitor_task = asyncio.create_task(self._vad_monitor_loop())
            logger.info("ManualControlSink VAD monitor task started.")

    def enable_vad(self, enabled: bool):
        """
        Controls VAD processing for the current recording session.

        VAD is only used for wake word triggered recordings to detect natural
        speech end. Push-to-talk recordings don't use VAD since the user
        explicitly controls the recording duration.

        When disabled, clears the received audio flag and resets VAD state
        to ensure clean state for the next recording session.
        """
        self._is_vad_enabled = enabled
        self._has_received_audio_for_vad = False  # Reset on state change
        if self._vad_analyzer:
            self._vad_analyzer.reset()

    def stop_and_get_audio(self) -> bytes:
        """
        Immediately stops recording and returns captured audio data.

        Used for push-to-talk interactions where the user explicitly controls
        recording duration by releasing the reaction. This method ensures clean
        state reset and prevents wake word re-triggering from buffered audio.

        CLEANUP SEQUENCE:
        1. Disable VAD (not used for PTT anyway)
        2. Capture current authority buffer contents
        3. Clear wake word detection state for authority user
        4. Reset all recording-related buffers and state

        Returns:
            bytes: The captured audio data in Discord's native PCM format
        """
        self.enable_vad(False)

        # TOCTOU Fix: Atomic authority buffer capture and clear
        with self._authority_buffer_lock:
            audio_data = bytes(self._authority_buffer)
            self._authority_buffer.clear()
            self._vad_raw_buffer.clear()
            self._vad_resampled_buffer.clear()
            self._vad_resample_state = None
            if self._vad_analyzer:
                self._vad_analyzer.reset()

        # TOCTOU Fix: Thread-safe wake word buffer cleanup
        authority_user_id = self._bot_state.authority_user_id
        if (
            isinstance(authority_user_id, int)
            and authority_user_id in self._ww_buffer_locks
        ):
            with self._ww_buffer_locks[authority_user_id]:
                if authority_user_id in self._detectors:
                    self._detectors[authority_user_id].reset()
                if authority_user_id in self._user_audio_buffers:
                    self._user_audio_buffers[authority_user_id].clear()
                if authority_user_id in self._ww_resampled_buffers:
                    self._ww_resampled_buffers[authority_user_id].clear()
                if authority_user_id in self._ww_resample_state:
                    self._ww_resample_state[authority_user_id] = None

        logger.info(f"PTT recording stopped, returning {len(audio_data)} bytes.")
        return audio_data

    def wants_opus(self) -> bool:
        """
        Discord audio format preference.

        Returns False to request raw PCM data instead of Opus-encoded audio.
        PCM is required for real-time audio processing like wake word detection
        and VAD analysis.
        """
        return False

    async def _vad_monitor_loop(self):
        """
        Background task that ensures VAD continues processing during audio gaps.

        PURPOSE:
        VAD requires continuous audio input to detect silence periods that indicate
        end-of-speech. Discord only sends audio when users are actively speaking,
        so we must inject synthetic silence during gaps to maintain VAD timing.

        OPERATION:
        - Runs every 20ms (synchronized with Discord's audio frame timing)
        - Only active when VAD is enabled (_is_vad_enabled = True)
        - Injects silence only when no real audio was received in the last interval
        - Uses _has_received_audio_for_vad flag to coordinate with write() method

        This approach ensures VAD can detect the transition from speech to silence
        that indicates the user has finished their command.
        """
        try:
            while True:
                # Check every 20ms, the duration of one Discord audio frame
                await asyncio.sleep(0.02)

                if not self._is_vad_enabled:
                    continue

                # VAD flag race fix: Thread-safe flag check and reset
                with self._vad_flag_lock:
                    has_received_audio = self._has_received_audio_for_vad
                    self._has_received_audio_for_vad = False  # Reset atomically

                if not has_received_audio:
                    # No audio received, inject silence into the VAD process.
                    await self._process_vad_async(self._silence_chunk_vad)
        except asyncio.CancelledError:
            logger.info("ManualControlSink VAD monitor task cancelled.")
        except Exception as e:
            logger.error(
                f"Error in ManualControlSink VAD monitor loop: {e}", exc_info=True
            )

    def _resample_and_convert(self, raw_chunk: bytes, user_id: int) -> bytes:
        """
        Convert Discord audio format to wake word model requirements.

        Discord provides: 48kHz, 16-bit, stereo PCM
        Wake word models need: 16kHz, 16-bit, mono PCM

        Uses audioop.ratecv() which maintains per-user conversion state
        for smooth resampling across chunk boundaries.
        """
        mono_audio = audioop.tomono(raw_chunk, Config.SAMPLE_WIDTH, 1, 1)
        resampled_audio, state = audioop.ratecv(
            mono_audio,
            Config.SAMPLE_WIDTH,
            1,
            Config.DISCORD_AUDIO_FRAME_RATE,
            Config.WAKE_WORD_SAMPLE_RATE,
            self._ww_resample_state.get(user_id),
        )
        self._ww_resample_state[user_id] = state
        return resampled_audio

    async def _handle_vad_speech_end(self):
        """
        Handles the completion of a VAD-detected speech segment.

        CRITICAL TIMING:
        This method is called by VADAnalyzer when speech end is detected.
        It must immediately reset all recording state to prevent audio loss
        or contamination for the next interaction.

        SEQUENCE:
        1. Disable VAD to stop further processing
        2. Capture current audio buffer contents
        3. Clear wake word buffers for authority user (prevents re-triggering)
        4. Reset all VAD-related state and buffers
        5. Schedule background processing task for captured audio

        The callback is scheduled as a background task rather than awaited
        to ensure the sink becomes available immediately for the next interaction.
        This prevents blocking and maintains responsiveness.
        """
        self.enable_vad(False)

        # TOCTOU Fix: Atomic buffer capture and cleanup
        with self._authority_buffer_lock:
            if not self._authority_buffer:  # Already processed by another callback
                return
            audio_data = bytes(self._authority_buffer)
            self._authority_buffer.clear()
            self._vad_raw_buffer.clear()
            self._vad_resampled_buffer.clear()
            self._vad_resample_state = None
            if self._vad_analyzer:
                self._vad_analyzer.reset()

        # TOCTOU Fix: Thread-safe wake word buffer cleanup
        authority_user_id = self._bot_state.authority_user_id
        if (
            isinstance(authority_user_id, int)
            and authority_user_id in self._ww_buffer_locks
        ):
            with self._ww_buffer_locks[authority_user_id]:
                if authority_user_id in self._detectors:
                    self._detectors[authority_user_id].reset()
                if authority_user_id in self._user_audio_buffers:
                    self._user_audio_buffers[authority_user_id].clear()
                if authority_user_id in self._ww_resampled_buffers:
                    self._ww_resampled_buffers[authority_user_id].clear()
                if authority_user_id in self._ww_resample_state:
                    self._ww_resample_state[authority_user_id] = None

        # Schedule the callback to run in the background instead of awaiting it.
        # This prevents blocking the sink and avoids state inconsistencies.
        if audio_data:
            asyncio.create_task(self._on_vad_speech_end(audio_data))

    async def _process_vad_async(self, pcm_data: bytes):
        """
        Thread-safe async wrapper for VAD processing.

        This method is called via run_coroutine_threadsafe() from the write() method
        to move VAD processing from Discord's audio thread to the main event loop.
        The _vad_lock prevents race conditions between real audio and silence injection.
        """
        async with self._vad_lock:
            self._process_vad(pcm_data)

    def _process_vad(self, pcm_data: bytes):
        """
        Core VAD processing logic for detecting end-of-speech.

        PROCESSING PIPELINE:
        1. Buffer incoming PCM data in _vad_raw_buffer
        2. Process in 20ms chunks (3840 bytes of 48kHz stereo)
        3. Convert each chunk: stereo -> mono, 48kHz -> 16kHz (VAD sample rate)
        4. Buffer resampled audio for frame-based VAD analysis
        5. Feed VAD-compatible frames to VADAnalyzer

        The VADAnalyzer handles the state machine for detecting sustained
        speech and meaningful silence periods that indicate end-of-command.
        """
        if not self._vad_analyzer:
            self._vad_analyzer = VADAnalyzer(
                on_speech_end=self._handle_vad_speech_end,
                sample_rate=Config.VAD_SAMPLE_RATE,
                frame_duration_ms=Config.VAD_FRAME_DURATION_MS,
                min_speech_duration_ms=Config.VAD_MIN_SPEECH_DURATION_MS,
                silence_timeout_ms=Config.VAD_SILENCE_TIMEOUT_MS,
                grace_period_ms=Config.VAD_GRACE_PERIOD_MS,
                loop=self._loop,
            )

        # Buffer raw audio before resampling, similar to the wake word path
        self._vad_raw_buffer.extend(pcm_data)
        min_vad_raw_bytes = 3840  # Process in 20ms chunks of raw 48kHz stereo

        while len(self._vad_raw_buffer) >= min_vad_raw_bytes:
            raw_chunk = self._vad_raw_buffer[:min_vad_raw_bytes]
            del self._vad_raw_buffer[:min_vad_raw_bytes]

            mono_audio = audioop.tomono(raw_chunk, Config.SAMPLE_WIDTH, 1, 1)
            resampled_audio, self._vad_resample_state = audioop.ratecv(
                mono_audio,
                Config.SAMPLE_WIDTH,
                1,
                Config.DISCORD_AUDIO_FRAME_RATE,
                Config.VAD_SAMPLE_RATE,
                self._vad_resample_state,
            )
            self._vad_resampled_buffer.extend(resampled_audio)

        # Process the resampled buffer in VAD-compatible frames
        frame_bytes = (
            (Config.VAD_SAMPLE_RATE * Config.VAD_FRAME_DURATION_MS) // 1000
        ) * Config.SAMPLE_WIDTH
        while len(self._vad_resampled_buffer) >= frame_bytes:
            frame = self._vad_resampled_buffer[:frame_bytes]
            del self._vad_resampled_buffer[:frame_bytes]
            self._vad_analyzer.process(frame)

    def write(self, user: discord.User, data: voice_recv.VoiceData):
        """
        Main entry point for all audio data from Discord.

        This method is called from Discord's audio thread for every 20ms audio frame
        from each user in the voice channel. It must be fast, non-blocking, and
        thread-safe since it's not running on the main event loop.

        ROUTING LOGIC:
        - RECORDING state: Route authority user's audio to recording pipeline
        - STANDBY state: Route all consented users' audio to wake word detection
        - Invalid users or states: Ignore audio data

        PERFORMANCE NOTES:
        - Wake word processing is synchronous (typically <1ms per chunk)
        - VAD processing is scheduled asynchronously to avoid blocking
        - State capture is atomic to prevent race conditions
        - Extensive debug logging available for troubleshooting
        """
        if not user:
            return

        logger.debug(
            f"Sink Write: user={user.id}, size={len(data.pcm)}, state={self._bot_state.current_state.value}, "
            f"method={self._bot_state.recording_method}, auth_user={self._bot_state.authority_user_id}"
        )

        if self._bot_state.current_state == BotStateEnum.RECORDING:
            if self._bot_state.is_authorized(user):
                # TOCTOU Fix: Atomic check-and-write operation
                with self._authority_buffer_lock:
                    # Re-check state inside the lock to ensure atomicity
                    if (
                        self._bot_state.current_state == BotStateEnum.RECORDING
                        and self._bot_state.is_authorized(user)
                    ):
                        self._authority_buffer.extend(data.pcm)
                        logger.debug(
                            f"Authority buffer size: {len(self._authority_buffer)}"
                        )

                        # Conditionally process VAD only for wake word recordings
                        if (
                            self._bot_state.recording_method == RecordingMethod.WakeWord
                            and self._is_vad_enabled
                        ):
                            # VAD flag race fix: Thread-safe flag update
                            with self._vad_flag_lock:
                                self._has_received_audio_for_vad = True
                            # Schedule VAD processing on the event loop to avoid race conditions
                            asyncio.run_coroutine_threadsafe(
                                self._process_vad_async(data.pcm), self._loop
                            )
        elif (
            self._bot_state.current_state == BotStateEnum.STANDBY
            and user.id in self._detectors
        ):
            self._process_standby_audio(user, data)

    def _process_standby_audio(self, user: discord.User, data: voice_recv.VoiceData):
        """
        Processes audio data when the bot is in STANDBY state for wake word detection.

        This method handles the complex wake word detection pipeline including:
        - Audio buffering and resampling from Discord format to wake word model format
        - Processing audio through openWakeWord models for detection
        - Handling wake word detection events and state cleanup
        """
        # CONCURRENCY FIX: Use dedicated lock for all user data access
        with self._user_data_lock:
            # Check if user was removed during processing
            if user.id not in self._detectors:
                return  # User was removed, skip processing
            self._user_audio_buffers[user.id].extend(data.pcm)
            buffer = self._user_audio_buffers[user.id]
            logger.debug(f"User {user.id} buffer size: {len(buffer)}")

            # Process in chunks large enough for at least one resample operation
            # 7680 bytes of 48kHz stereo -> 1920 frames -> 640 frames @ 16kHz mono -> 1280 bytes
            min_raw_bytes = 7680
            processed_bytes = 0
            resampled_buffer = self._ww_resampled_buffers[user.id]
            while len(buffer) - processed_bytes >= min_raw_bytes:
                raw_chunk = buffer[processed_bytes : processed_bytes + min_raw_bytes]
                resampled = self._resample_and_convert(raw_chunk, user.id)
                resampled_buffer.extend(resampled)
                processed_bytes += min_raw_bytes

            # Remove the processed raw data from the beginning of the buffer
            del buffer[:processed_bytes]

            # Process the resampled buffer for wake words
            while len(resampled_buffer) >= self._ww_chunk_size:
                ww_chunk_bytes = resampled_buffer[: self._ww_chunk_size]
                del resampled_buffer[: self._ww_chunk_size]

                # Convert bytes to numpy array for the model
                ww_chunk_np = np.frombuffer(ww_chunk_bytes, dtype=np.int16)

                model = self._detectors[user.id]
                prediction = model.predict(ww_chunk_np)
                logger.debug(f"Wake word prediction for user {user.id}: {prediction}")

                model_name = os.path.splitext(
                    os.path.basename(Config.WAKE_WORD_MODEL_PATH)
                )[0]

                if (
                    model_name in prediction
                    and prediction[model_name] > Config.WAKE_WORD_THRESHOLD
                ):
                    logger.info(f"Wake word detected for user {user.id}")
                    model.reset()
                    self._loop.call_soon_threadsafe(
                        asyncio.create_task, self._on_wake_word_detected(user)
                    )
                    self._user_audio_buffers[user.id].clear()
                    self._ww_resampled_buffers[user.id].clear()
                    self._ww_resample_state[user.id] = None
                    return
