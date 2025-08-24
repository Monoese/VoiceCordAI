"""
Audio Sink Implementations for different operational modes.

This module provides the concrete AudioSink classes for the bot's different
operational modes, such as ManualControl and RealtimeTalk. Each sink is
responsible for processing raw audio from users according to its mode's logic.
"""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

import audioop
import discord
import numpy as np
import openwakeword
import webrtcvad
from discord.ext import voice_recv
from openwakeword.model import Model

from src.audio.processing import (
    UnifiedAudioProcessor,
    AudioFormat,
    DISCORD_FORMAT,
    WAKE_WORD_FORMAT,
    VAD_FORMAT,
    ProcessingStrategy,
)
from src.bot.state import BotState, BotStateEnum, RecordingMethod
from src.config.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Detect OpenWakeWord version for API compatibility
OWW_VERSION = getattr(openwakeword, "__version__", "0.4.0")
if "0.4" in OWW_VERSION:
    OWW_PARAM_NAME = "wakeword_model_paths"
elif "0.6" in OWW_VERSION:
    OWW_PARAM_NAME = "wakeword_models"
else:
    # Default to newer API for future versions
    OWW_PARAM_NAME = "wakeword_models"

logger.info(
    f"OpenWakeWord version {OWW_VERSION} detected, using parameter '{OWW_PARAM_NAME}'"
)


class CleanupMetrics:
    """Tracks cleanup operations and race condition prevention."""

    def __init__(self):
        self.comprehensive_cleanups = 0
        self.race_conditions_prevented = 0
        self.cleanup_errors = 0
        self.zero_byte_recordings_prevented = 0
        self.successful_interactions = 0
        self.last_successful_cleanup = time.time()

    def record_comprehensive_cleanup(self, stats: Dict[str, int]):
        """Record results of comprehensive cleanup operation."""
        self.comprehensive_cleanups += 1
        self.cleanup_errors += stats.get("errors", 0)
        self.last_successful_cleanup = time.time()

        if stats.get("buffers_cleared", 0) > 0:
            logger.info(f"Comprehensive cleanup cleared stale data: {stats}")

    def record_race_condition_prevention(self, captured_id, current_id):
        """Track when race condition would have occurred but was prevented."""
        if captured_id != current_id:
            self.race_conditions_prevented += 1
            logger.warning(
                f"Race condition prevented: captured={captured_id}, current={current_id}"
            )

    def record_session_id_mismatch(self, expected_session_id, actual_session_id):
        """Track when session ID mismatch prevents cross-session contamination."""
        if expected_session_id != actual_session_id:
            self.race_conditions_prevented += 1
            logger.warning(
                f"Session ID mismatch prevented contamination: expected={expected_session_id}, actual={actual_session_id}"
            )

    def record_successful_interaction(self, audio_size: int):
        """Track successful audio interactions."""
        self.successful_interactions += 1
        if audio_size > 0:
            logger.debug(f"Successful interaction recorded: {audio_size} bytes")

    def export_health_report(self) -> Dict[str, Any]:
        """Export comprehensive health metrics for monitoring."""
        return {
            "total_cleanups": self.comprehensive_cleanups,
            "race_conditions_prevented": self.race_conditions_prevented,
            "cleanup_error_rate": self.cleanup_errors
            / max(1, self.comprehensive_cleanups),
            "successful_interactions": self.successful_interactions,
            "time_since_last_cleanup": time.time() - self.last_successful_cleanup,
            "health_status": "healthy" if self.cleanup_errors == 0 else "degraded",
        }


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
            logger.debug(
                f"Invalid VAD frame size: expected {self._frame_bytes}, got {len(frame)}"
            )
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
        # 20ms of 48kHz, 16-bit, stereo audio
        self._chunk_size = Config.DISCORD_CHUNK_SIZE
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
        action_lock: asyncio.Lock,
    ):
        super().__init__()
        self._bot_state = bot_state
        self._on_wake_word_detected = on_wake_word_detected
        self._on_vad_speech_end = on_vad_speech_end
        self._loop = asyncio.get_running_loop()

        # Initialize unified audio processor for real-time processing
        self._audio_processor = UnifiedAudioProcessor()

        self._detectors: Dict[int, Model] = {}
        self._user_audio_buffers: Dict[int, bytearray] = {}
        self._authority_buffer = bytearray()
        self._vad_raw_buffer = bytearray()
        self._vad_resampled_buffer = bytearray()
        self._ww_resampled_buffers: Dict[int, bytearray] = {}

        # Thread-safe synchronization primitives for TOCTOU fix
        self._action_lock = action_lock
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
        self._ww_chunk_size = (
            Config.WAKE_WORD_CHUNK_SIZE
        )  # 80ms of 16kHz, 16-bit, mono audio
        self._vad_analyzer: Optional[VADAnalyzer] = None

        # VAD silence injection system - coordinates between real audio and synthetic silence
        self._has_received_audio_for_vad: bool = (
            False  # Flag: real audio received this cycle
        )
        self._vad_monitor_task: Optional[asyncio.Task] = (
            None  # Background silence injection task
        )
        self._silence_chunk_vad = (
            b"\x00" * Config.DISCORD_CHUNK_SIZE
        )  # 20ms of 48kHz stereo 16-bit PCM silence
        self._vad_lock = (
            asyncio.Lock()
        )  # Prevents race conditions between real audio and silence

        # Add cleanup metrics tracking
        self._cleanup_metrics = CleanupMetrics()

        # Session ID tracking for cross-session contamination prevention
        self._session_id_at_creation = self._bot_state.current_session_id
        self._active_session_id = self._session_id_at_creation
        logger.debug(
            f"ManualControlSink created for session {self._session_id_at_creation}"
        )

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
                # Validate wake word model file exists
                model_path = str(Config.WAKE_WORD_MODEL_PATH)
                if not Config.WAKE_WORD_MODEL_PATH.exists():
                    raise FileNotFoundError(
                        f"Wake word model file not found: {model_path}"
                    )

                # Use the appropriate parameter name based on OpenWakeWord version
                model_kwargs = {OWW_PARAM_NAME: [model_path]}
                # Only add inference_framework and vad_threshold for 0.6+ (0.4.x doesn't support them)
                if OWW_PARAM_NAME == "wakeword_models":
                    model_kwargs["inference_framework"] = "onnx"
                    model_kwargs["vad_threshold"] = Config.WAKE_WORD_VAD_THRESHOLD

                self._detectors[user_id] = Model(**model_kwargs)
                self._user_audio_buffers[user_id] = bytearray()
                self._ww_resampled_buffers[user_id] = bytearray()
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
            self._ww_resampled_buffers.pop(user_id, None)
            self._ww_buffer_locks.pop(
                user_id, None
            )  # Safe to delete after data cleanup

            # Reset unified processor state for this user
            wake_word_state_key = f"wake_word_user_{user_id}"
            self._audio_processor.reset_state(wake_word_state_key)

    def cleanup(self) -> None:
        logger.info("Cleaning up ManualControlSink.")
        if self._vad_monitor_task and not self._vad_monitor_task.done():
            self._vad_monitor_task.cancel()
        # Destroy VAD analyzer to ensure clean shutdown
        self._vad_analyzer = None
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

    def _clear_all_wake_word_buffers(self) -> None:
        """
        Comprehensive cleanup method to clear all wake word buffers for all users.

        This method ensures a complete "clean slate" state between recording interactions,
        eliminating any possibility of stale audio data corruption. It's more robust than
        individual user cleanup as it handles edge cases like non-authority user state
        corruption.

        Thread Safety: Uses _user_data_lock to ensure atomic cleanup operations.
        """
        logger.debug("Starting comprehensive wake word buffer cleanup for all users")

        cleanup_stats = {
            "models_reset": 0,
            "buffers_cleared": 0,
            "states_reset": 0,
            "errors": 0,
        }

        with self._user_data_lock:
            # Reset all wake word models
            for user_id, model in self._detectors.items():
                try:
                    model.reset()
                    cleanup_stats["models_reset"] += 1
                    logger.debug(f"Reset wake word model for user {user_id}")
                except Exception as e:
                    cleanup_stats["errors"] += 1
                    logger.error(
                        f"Failed to reset wake word model for user {user_id}: {e}"
                    )

            # Clear all user audio buffers
            for user_id in list(self._user_audio_buffers.keys()):
                buffer_size = len(self._user_audio_buffers[user_id])
                self._user_audio_buffers[user_id].clear()
                cleanup_stats["buffers_cleared"] += 1
                if buffer_size > 0:
                    logger.debug(
                        f"Cleared {buffer_size} bytes from user {user_id} audio buffer"
                    )

            # Clear all resampled buffers
            for user_id in list(self._ww_resampled_buffers.keys()):
                buffer_size = len(self._ww_resampled_buffers[user_id])
                self._ww_resampled_buffers[user_id].clear()
                if buffer_size > 0:
                    logger.debug(
                        f"Cleared {buffer_size} bytes from user {user_id} resampled buffer"
                    )

            # Reset all resample states using unified processor
            for user_id in list(self._ww_resampled_buffers.keys()):
                # Reset state in unified processor
                wake_word_state_key = f"wake_word_user_{user_id}"
                self._audio_processor.reset_state(wake_word_state_key)
                cleanup_stats["states_reset"] += 1

        logger.info(f"Comprehensive wake word cleanup completed: {cleanup_stats}")

        # Track cleanup success for monitoring
        self._cleanup_metrics.record_comprehensive_cleanup(cleanup_stats)

    def enable_vad(self, enabled: bool):
        """
        Controls VAD processing for the current recording session.

        VAD is only used for wake word triggered recordings to detect natural
        speech end. Push-to-talk recordings don't use VAD since the user
        explicitly controls the recording duration.

        Creates a fresh VAD analyzer for each recording session to ensure
        completely clean state and eliminate any timing or state corruption issues.
        """
        self._is_vad_enabled = enabled
        self._has_received_audio_for_vad = False  # Reset on state change

        if enabled:
            # CREATE fresh VAD analyzer for this recording session
            self._vad_analyzer = VADAnalyzer(
                on_speech_end=self._handle_vad_speech_end,
                sample_rate=Config.VAD_SAMPLE_RATE,
                frame_duration_ms=Config.VAD_FRAME_DURATION_MS,
                min_speech_duration_ms=Config.VAD_MIN_SPEECH_DURATION_MS,
                silence_timeout_ms=Config.VAD_SILENCE_TIMEOUT_MS,
                grace_period_ms=Config.VAD_GRACE_PERIOD_MS,
                loop=self._loop,
            )
        else:
            # DESTROY VAD analyzer when disabled
            self._vad_analyzer = None

        # Clear VAD buffers for fresh start
        self._vad_raw_buffer.clear()
        self._vad_resampled_buffer.clear()

    def update_session_id(self) -> None:
        """
        Updates the active session ID to match the current bot state session.

        This should be called when a new recording starts to ensure the sink
        is synchronized with the current session and prevent cross-session contamination.
        """
        new_session_id = self._bot_state.current_session_id
        if new_session_id != self._active_session_id:
            logger.info(
                f"Updating ManualControlSink session ID: {self._active_session_id} -> {new_session_id}"
            )
            self._active_session_id = new_session_id

    def stop_and_get_audio(self) -> bytes:
        """
        Stop PTT recording and return captured audio with race condition protection.

        Uses comprehensive cleanup to ensure complete clean state for next interaction.

        Returns:
            bytes: The captured audio data in Discord's native PCM format
        """
        # SESSION ID VALIDATION: Prevent cross-session contamination
        current_session_id = self._bot_state.current_session_id
        if current_session_id != self._active_session_id:
            self._cleanup_metrics.record_session_id_mismatch(
                self._active_session_id, current_session_id
            )
            logger.warning(
                f"Session ID mismatch in stop_and_get_audio: sink={self._active_session_id}, state={current_session_id}. "
                f"Returning empty audio to prevent contamination."
            )
            return b""

        # Capture authority user ID for logging purposes
        captured_authority_id = self._bot_state.authority_user_id

        self.enable_vad(False)

        audio_data = bytes(self._authority_buffer)
        self._authority_buffer.clear()

        # Clear VAD-specific state
        self._vad_raw_buffer.clear()
        self._vad_resampled_buffer.clear()
        # Destroy VAD analyzer to ensure fresh state for next session
        self._vad_analyzer = None

        # SIMPLIFIED CLEANUP: Comprehensive cleanup for all users
        # Ensures complete clean state for next interaction
        self._clear_all_wake_word_buffers()

        # Enhanced logging for debugging
        logger.debug(
            f"PTT recording stopped: user_id={captured_authority_id}, "
            f"audio_size={len(audio_data)}, buffers_cleared=all_users"
        )

        # Track successful interaction
        self._cleanup_metrics.record_successful_interaction(len(audio_data))

        return audio_data

    async def validate_sink_health(self) -> Dict[str, Any]:
        """
        Comprehensive health check for the audio sink.

        Returns:
            Dict containing health status and metrics
        """
        health_report = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "issues": [],
        }

        try:
            # Check buffer states
            authority_buffer_size = len(self._authority_buffer)
            if authority_buffer_size > 1024 * 1024:  # 1MB threshold
                health_report["issues"].append(
                    f"Authority buffer unusually large: {authority_buffer_size} bytes"
                )
                health_report["overall_status"] = "warning"

            # Check wake word model states
            corrupted_models = []
            for user_id, model in self._detectors.items():
                try:
                    # Basic sanity check - ensure model has required methods
                    if not hasattr(model, "predict") or not hasattr(model, "reset"):
                        corrupted_models.append(user_id)
                except Exception as e:
                    corrupted_models.append(user_id)
                    logger.error(
                        f"Wake word model health check failed for user {user_id}: {e}"
                    )

            if corrupted_models:
                health_report["issues"].append(
                    f"Corrupted wake word models: {corrupted_models}"
                )
                health_report["overall_status"] = "critical"

            # Check background task status
            if self._vad_monitor_task and self._vad_monitor_task.done():
                exception = self._vad_monitor_task.exception()
                if exception:
                    health_report["issues"].append(
                        f"VAD monitor task failed: {exception}"
                    )
                    health_report["overall_status"] = "critical"

            # Add cleanup metrics
            health_report["metrics"] = self._cleanup_metrics.export_health_report()

            return health_report

        except Exception as e:
            logger.error(f"Sink health check failed: {e}")
            health_report["overall_status"] = "critical"
            health_report["issues"].append(f"Health check exception: {str(e)}")
            return health_report

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

    def _resample_audio(
        self,
        raw_chunk: bytes,
        resample_state: Optional[Any],
        target_sample_rate: int = 16000,
    ) -> tuple[bytes, Optional[Any]]:
        """
        Generic helper method for resampling Discord audio to target format.

        Now uses the unified audio processing framework for consistent resampling
        across the codebase while maintaining backward compatibility.

        Args:
            raw_chunk: Raw PCM audio data from Discord (48kHz, 16-bit, stereo)
            target_sample_rate: Target sample rate in Hz (default: 16000)

        Returns:
            bytes: Resampled audio data
        """
        # Map target sample rate to appropriate format
        if target_sample_rate == Config.VAD_SAMPLE_RATE:
            target_format = VAD_FORMAT
            state_key = f"vad_session_{self._active_session_id}"
        elif target_sample_rate == Config.WAKE_WORD_SAMPLE_RATE:
            target_format = WAKE_WORD_FORMAT
            state_key = f"wake_word_session_{self._active_session_id}"
        else:
            # Custom target rate
            target_format = AudioFormat(target_sample_rate, 1, Config.SAMPLE_WIDTH)
            state_key = f"custom_{target_sample_rate}_session_{self._active_session_id}"

        # Use unified processor for consistent resampling
        resampled_audio = self._audio_processor.convert_sync(
            DISCORD_FORMAT,
            target_format,
            raw_chunk,
            strategy=ProcessingStrategy.REALTIME,
            state_key=state_key,
        )

        return resampled_audio

    def _resample_and_convert(self, raw_chunk: bytes, user_id: int) -> bytes:
        """
        Convert Discord audio format to wake word model requirements.

        Now uses the unified audio processing framework for consistent wake word
        audio conversion across the codebase.

        Discord provides: 48kHz, 16-bit, stereo PCM
        Wake word models need: 16kHz, 16-bit, mono PCM

        Args:
            raw_chunk: Raw PCM audio data from Discord
            user_id: User ID for per-user state management

        Returns:
            bytes: Resampled mono PCM audio for wake word detection
        """
        # Create per-user state key for wake word processing
        state_key = f"wake_word_user_{user_id}"

        # Use unified processor for consistent wake word audio conversion
        return self._audio_processor.convert_sync(
            DISCORD_FORMAT,
            WAKE_WORD_FORMAT,
            raw_chunk,
            strategy=ProcessingStrategy.REALTIME,
            state_key=state_key,
        )

    async def _handle_vad_speech_end(self):
        """
        Handle VAD-detected speech end with race condition protection.

        RACE CONDITION FIX: Captures authority_user_id immediately to prevent
        race condition where concurrent state changes cause cleanup to fail.

        SIMPLIFIED CLEANUP: Uses comprehensive cleanup instead of individual
        user cleanup for better robustness and maintainability.
        """
        # SESSION ID VALIDATION: Prevent cross-session contamination
        current_session_id = self._bot_state.current_session_id
        if current_session_id != self._active_session_id:
            self._cleanup_metrics.record_session_id_mismatch(
                self._active_session_id, current_session_id
            )
            logger.warning(
                f"Session ID mismatch in VAD speech end: sink={self._active_session_id}, state={current_session_id}. "
                f"Skipping callback to prevent contamination."
            )
            return

        # ATOMIC CAPTURE: Prevent race condition by capturing state for logging
        authority_user_id_at_speech_end = self._bot_state.authority_user_id

        self.enable_vad(False)

        if not self._authority_buffer:
            # If there's no audio, still perform cleanup to ensure clean state
            self._clear_all_wake_word_buffers()
            logger.debug(
                f"VAD cleanup with no audio: user_id={authority_user_id_at_speech_end}"
            )
            return

        audio_data = bytes(self._authority_buffer)
        self._authority_buffer.clear()

        # Clear VAD-specific buffers
        self._vad_raw_buffer.clear()
        self._vad_resampled_buffer.clear()
        # Destroy VAD analyzer to ensure fresh state for next session
        self._vad_analyzer = None

        # SIMPLIFIED CLEANUP: Comprehensive cleanup instead of individual + comprehensive
        # This is more robust and eliminates redundant logic
        self._clear_all_wake_word_buffers()

        # Enhanced logging for race condition debugging
        logger.debug(
            f"VAD speech end cleanup completed: user_id={authority_user_id_at_speech_end}, "
            f"audio_size={len(audio_data)}, buffers_cleared=all_users"
        )

        # Track successful interaction
        self._cleanup_metrics.record_successful_interaction(len(audio_data))

        # Schedule the callback with captured audio
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
        # VAD analyzer is created by enable_vad() method for each recording session
        if not self._vad_analyzer:
            logger.warning(
                "VAD processing called but no analyzer exists - VAD not enabled"
            )
            return

        # Buffer raw audio before resampling, similar to the wake word path
        self._vad_raw_buffer.extend(pcm_data)
        min_vad_raw_bytes = (
            Config.DISCORD_CHUNK_SIZE
        )  # Process in 20ms chunks of raw 48kHz stereo

        while len(self._vad_raw_buffer) >= min_vad_raw_bytes:
            raw_chunk = self._vad_raw_buffer[:min_vad_raw_bytes]
            del self._vad_raw_buffer[:min_vad_raw_bytes]

            resampled_audio = self._resample_audio(
                raw_chunk, None, Config.VAD_SAMPLE_RATE
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
        Enhanced write method with comprehensive race condition diagnostics.

        This method is called from Discord's audio thread for every 20ms audio frame
        from each user in the voice channel. It must be fast, non-blocking, and
        thread-safe since it's not running on the main event loop.
        """
        if not user:
            return

        # SESSION ID VALIDATION: Early exit if session has changed
        current_session_id = self._bot_state.current_session_id
        if current_session_id != self._active_session_id:
            # Immediately update session ID to prevent race conditions
            # This eliminates the window where valid frames might be discarded
            old_session_id = self._active_session_id
            self._active_session_id = current_session_id

            # Log the session change (not every frame to avoid spam)
            if (
                hash(data.pcm) % Config.AUDIO_LOG_SAMPLING_RATE == 0
            ):  # Log ~1% of frames
                logger.info(
                    f"Auto-updated ManualControlSink session ID: {old_session_id} -> {current_session_id}"
                )
                self._cleanup_metrics.record_session_id_mismatch(
                    old_session_id, current_session_id
                )

            # Continue processing with updated session ID instead of returning
            # This ensures valid frames aren't lost during session transitions

        # Capture state atomically for consistent logging
        current_state = self._bot_state.current_state
        authority_id = self._bot_state.authority_user_id
        recording_method = self._bot_state.recording_method
        is_authorized = self._bot_state.is_authorized(user)

        # Enhanced logging for race condition debugging
        logger.debug(
            f"Audio write: user={user.id}, size={len(data.pcm)}, "
            f"state={current_state.value}, method={recording_method}, "
            f"auth_user={authority_id}, is_authorized={is_authorized}, "
            f"buffer_size={len(self._authority_buffer)}"
        )

        # Use captured state for consistent behavior
        if current_state == BotStateEnum.RECORDING:
            if is_authorized:
                # TOCTOU Fix: Schedule atomic operation on event loop since write() is called from Discord thread
                asyncio.run_coroutine_threadsafe(
                    self._atomic_authority_buffer_update(
                        user, data.pcm, current_state, is_authorized
                    ),
                    self._loop,
                )
                # Don't wait to avoid blocking Discord's audio thread

                # Conditionally process VAD only for wake word recordings
                if (
                    recording_method == RecordingMethod.WakeWord
                    and self._is_vad_enabled
                ):
                    # VAD flag race fix: Thread-safe flag update
                    with self._vad_flag_lock:
                        self._has_received_audio_for_vad = True
                    # Schedule VAD processing on the event loop
                    asyncio.run_coroutine_threadsafe(
                        self._process_vad_async(data.pcm), self._loop
                    )
        elif current_state == BotStateEnum.STANDBY and user.id in self._detectors:
            self._process_standby_audio(user, data)

    async def _atomic_authority_buffer_update(
        self,
        user: discord.User,
        pcm_data: bytes,
        captured_state: BotStateEnum,
        captured_authorized: bool,
    ) -> None:
        """Atomically check state and update buffer under shared lock."""
        async with self._action_lock:
            # Re-validate state under lock
            if (
                self._bot_state.current_state
                == captured_state
                == BotStateEnum.RECORDING
                and self._bot_state.is_authorized(user)
                and captured_authorized
            ):
                self._authority_buffer.extend(pcm_data)
                logger.debug(
                    f"Authority buffer updated: size={len(self._authority_buffer)}"
                )

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
            min_raw_bytes = Config.VAD_PROCESSING_CHUNK
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

                model_name = Config.WAKE_WORD_MODEL_PATH.stem

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
                    return
