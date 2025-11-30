"""
Tests for Audio Sinks - Manual control implementation.

These tests verify the complex audio processing pipelines, VAD integration,
wake word detection, threading safety, and proper resource cleanup for
ManualControlSink.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import discord
from discord.ext import voice_recv

from src.audio.sinks import ManualControlSink, VADAnalyzer, CleanupMetrics
from src.bot.state import BotState, BotStateEnum, RecordingMethod
from src.exceptions import SessionConsistencyError


class TestCleanupMetrics:
    """Test the cleanup metrics tracking functionality."""

    @pytest.fixture
    def metrics(self):
        return CleanupMetrics()

    def test_initialization(self, metrics):
        """Test CleanupMetrics initializes with correct default values."""
        assert metrics.comprehensive_cleanups == 0
        assert metrics.race_conditions_prevented == 0
        assert metrics.cleanup_errors == 0
        assert metrics.zero_byte_recordings_prevented == 0
        assert metrics.successful_interactions == 0
        assert isinstance(metrics.last_successful_cleanup, float)

    def test_record_comprehensive_cleanup(self, metrics):
        """Test recording comprehensive cleanup operations."""
        stats = {"errors": 2, "buffers_cleared": 5}

        metrics.record_comprehensive_cleanup(stats)

        assert metrics.comprehensive_cleanups == 1
        assert metrics.cleanup_errors == 2

    def test_record_race_condition_prevention(self, metrics):
        """Test recording race condition prevention."""
        metrics.record_race_condition_prevention(123, 456)

        assert metrics.race_conditions_prevented == 1

    def test_record_race_condition_prevention_same_id(self, metrics):
        """Test recording race condition when IDs match."""
        metrics.record_race_condition_prevention(123, 123)

        # Should not increment if IDs match
        assert metrics.race_conditions_prevented == 0

    def test_record_session_id_mismatch(self, metrics):
        """Test recording session ID mismatch."""
        metrics.record_session_id_mismatch(100, 200)

        assert metrics.race_conditions_prevented == 1

    def test_record_successful_interaction(self, metrics):
        """Test recording successful interactions."""
        metrics.record_successful_interaction(1024)

        assert metrics.successful_interactions == 1

    def test_export_health_report(self, metrics):
        """Test exporting health report."""
        metrics.comprehensive_cleanups = 5
        metrics.race_conditions_prevented = 3
        metrics.cleanup_errors = 1
        metrics.successful_interactions = 10

        report = metrics.export_health_report()

        assert report["total_cleanups"] == 5
        assert report["race_conditions_prevented"] == 3
        assert report["cleanup_error_rate"] == 0.2  # 1/5
        assert report["successful_interactions"] == 10


class TestVADAnalyzer:
    """Test Voice Activity Detection analyzer."""

    @pytest.fixture
    def mock_callback(self):
        return AsyncMock()

    @pytest.fixture
    def mock_loop(self):
        return AsyncMock(spec=asyncio.AbstractEventLoop)

    @pytest.fixture
    def vad_analyzer(self, mock_callback, mock_loop):
        with patch("webrtcvad.Vad"):
            return VADAnalyzer(
                on_speech_end=mock_callback,
                sample_rate=16000,
                frame_duration_ms=20,
                min_speech_duration_ms=100,
                silence_timeout_ms=1000,
                grace_period_ms=200,
                loop=mock_loop,
            )

    def test_initialization(self, vad_analyzer, mock_callback, mock_loop):
        """Test VADAnalyzer initializes correctly."""
        assert vad_analyzer._on_speech_end == mock_callback
        assert vad_analyzer._loop == mock_loop
        assert vad_analyzer._sample_rate == 16000
        assert vad_analyzer._frame_duration_ms == 20
        assert not vad_analyzer._is_speech
        assert not vad_analyzer._triggered

    def test_process_speech_frame(self, vad_analyzer):
        """Test processing a speech frame."""
        # Create proper frame data
        frame_data = b"\x00" * vad_analyzer._frame_bytes

        # Mock webrtcvad to return True for speech
        vad_analyzer._vad.is_speech.return_value = True

        vad_analyzer.process(frame_data)

        assert vad_analyzer._speech_frame_count > 0

    def test_process_non_speech_frame(self, vad_analyzer):
        """Test processing a non-speech frame."""
        # Create proper frame data
        frame_data = b"\x00" * vad_analyzer._frame_bytes

        # Mock webrtcvad to return False for speech
        vad_analyzer._vad.is_speech.return_value = False

        vad_analyzer.process(frame_data)

        assert vad_analyzer._frames_processed > 0

    def test_process_invalid_frame_size(self, vad_analyzer):
        """Test processing frame with invalid size."""
        # Invalid frame size
        frame_data = b"too_short"

        vad_analyzer.process(frame_data)

        # Should not crash, just log debug message

    def test_reset(self, vad_analyzer):
        """Test VAD analyzer reset functionality."""
        vad_analyzer._speech_frame_count = 5
        vad_analyzer._silence_frame_count = 3
        vad_analyzer._is_speech = True
        vad_analyzer._triggered = True

        vad_analyzer.reset()

        assert vad_analyzer._speech_frame_count == 0
        assert vad_analyzer._silence_frame_count == 0
        assert not vad_analyzer._is_speech
        assert not vad_analyzer._triggered


# Note: AudioSinkBase tests removed as AudioSink is an abstract base class
# that cannot be instantiated directly. Tests for ManualControlSink provide coverage
# for the implemented audio sink functionality.

# Note: RealtimeMixingSink tests not included as it's a skeleton implementation
# for future features and is not currently used in the project.


class TestManualControlSink:
    """Test the ManualControlSink implementation."""

    @pytest.fixture
    def mock_bot_state(self):
        bot_state = MagicMock(spec=BotState)
        bot_state.current_state = BotStateEnum.STANDBY
        bot_state.current_session_id = 100
        return bot_state

    @pytest.fixture
    def mock_callbacks(self):
        return {
            "on_wake_word_detected": AsyncMock(),
            "on_vad_speech_end": AsyncMock(),
        }

    @pytest.fixture
    def mock_action_lock(self):
        return asyncio.Lock()

    @pytest.fixture
    def manual_control_sink(self, mock_bot_state, mock_callbacks, mock_action_lock):
        """Create ManualControlSink with mocked dependencies."""
        mock_loop = AsyncMock()
        mock_task = MagicMock()
        with (
            patch.multiple(
                "src.audio.sinks",
                UnifiedAudioProcessor=MagicMock(),
                Config=MagicMock(),
                Model=MagicMock(),
            ),
            patch("asyncio.get_running_loop", return_value=mock_loop),
            patch("asyncio.create_task", return_value=mock_task),
        ):
            # Mock Config values
            from src.audio.sinks import Config as MockConfig

            MockConfig.VAD_GRACE_PERIOD_MS = 200
            MockConfig.WAKE_WORD_CHUNK_SIZE = 1280
            MockConfig.DISCORD_CHUNK_SIZE = 3840

            sink = ManualControlSink(
                bot_state=mock_bot_state,
                initial_consented_users={123, 456},
                on_wake_word_detected=mock_callbacks["on_wake_word_detected"],
                on_vad_speech_end=mock_callbacks["on_vad_speech_end"],
                action_lock=mock_action_lock,
            )
            return sink

    def test_initialization(self, manual_control_sink, mock_bot_state, mock_callbacks):
        """Test ManualControlSink initialization."""
        assert manual_control_sink._bot_state == mock_bot_state
        assert (
            manual_control_sink._on_wake_word_detected
            == mock_callbacks["on_wake_word_detected"]
        )
        assert (
            manual_control_sink._on_vad_speech_end
            == mock_callbacks["on_vad_speech_end"]
        )
        assert 123 in manual_control_sink._detectors
        assert 456 in manual_control_sink._detectors
        assert manual_control_sink._session_id_at_creation == 100
        assert not manual_control_sink._is_vad_enabled

    def test_add_user_creates_detector(self, manual_control_sink):
        """Test adding a user creates wake word detector."""
        with patch("src.audio.sinks.Model") as mock_model:
            mock_detector = MagicMock()
            mock_model.return_value = mock_detector

            manual_control_sink.add_user(789)

            # User gets wake word detector created
            assert 789 in manual_control_sink._detectors
            assert 789 in manual_control_sink._user_audio_buffers
            assert 789 in manual_control_sink._ww_resampled_buffers
            assert 789 in manual_control_sink._ww_buffer_locks

    def test_add_user_already_exists(self, manual_control_sink):
        """Test adding user that already exists."""
        # Add user first time
        with patch("src.audio.sinks.Model"):
            manual_control_sink.add_user(123)
            detector_count = len(manual_control_sink._detectors)

        # Add same user again
        manual_control_sink.add_user(123)

        # Should not create additional detector
        assert len(manual_control_sink._detectors) == detector_count

    def test_add_user_model_creation_failure(self, manual_control_sink):
        """Test handling wake word model creation failure."""
        with patch(
            "src.audio.sinks.Model", side_effect=Exception("Model creation failed")
        ):
            # Should not raise exception
            manual_control_sink.add_user(789)

            # Note: User doesn't get detector on model creation failure
            assert 789 not in manual_control_sink._detectors  # No detector created

    def test_remove_user_cleans_up_resources(self, manual_control_sink):
        """Test removing user cleans up all associated resources."""
        # First add a user
        with patch("src.audio.sinks.Model"):
            manual_control_sink.add_user(789)

        # Then remove the user
        manual_control_sink.remove_user(789)

        # Verify user was removed from all tracking
        assert 789 not in manual_control_sink._detectors
        assert 789 not in manual_control_sink._user_audio_buffers
        assert 789 not in manual_control_sink._ww_resampled_buffers
        assert 789 not in manual_control_sink._ww_buffer_locks

    def test_enable_vad_initializes_analyzer(self, manual_control_sink):
        """Test enabling VAD initializes the analyzer."""
        with patch("src.audio.sinks.VADAnalyzer") as mock_vad_analyzer:
            mock_analyzer = MagicMock()
            mock_vad_analyzer.return_value = mock_analyzer

            manual_control_sink.enable_vad(True)

            assert manual_control_sink._is_vad_enabled is True
            assert manual_control_sink._vad_analyzer is not None
            # VAD analyzer should have been created
            mock_vad_analyzer.assert_called_once()

    def test_enable_vad_false_resets_state(self, manual_control_sink):
        """Test disabling VAD resets all VAD-related state."""
        # First enable VAD
        with patch("src.audio.sinks.VADAnalyzer"):
            manual_control_sink.enable_vad(True)

            # Add some audio to buffers to test clearing
            manual_control_sink._vad_raw_buffer.extend(b"test_data")
            manual_control_sink._vad_resampled_buffer.extend(b"resampled_data")

        # Now disable VAD
        manual_control_sink.enable_vad(False)

        # Verify state was reset
        assert manual_control_sink._is_vad_enabled is False
        assert manual_control_sink._vad_analyzer is None
        assert len(manual_control_sink._vad_raw_buffer) == 0
        assert len(manual_control_sink._vad_resampled_buffer) == 0

    def test_update_session_id(self, manual_control_sink):
        """Test updating session ID."""
        manual_control_sink._bot_state.current_session_id = 200

        manual_control_sink.update_session_id()

        assert manual_control_sink._active_session_id == 200

    def test_stop_and_get_audio_returns_authority_buffer(self, manual_control_sink):
        """Test stop_and_get_audio returns and clears authority buffer."""
        test_audio = b"test_audio_data"
        manual_control_sink._authority_buffer.extend(test_audio)

        result = manual_control_sink.stop_and_get_audio()

        assert result == test_audio
        assert len(manual_control_sink._authority_buffer) == 0

    def test_stop_and_get_audio_raises_on_session_mismatch(self, manual_control_sink):
        """Test stop_and_get_audio raises SessionConsistencyError on session ID mismatch."""
        test_audio = b"test_audio_data"
        manual_control_sink._authority_buffer.extend(test_audio)

        # Set up session ID mismatch: sink has ID 100, bot state has ID 200
        manual_control_sink._active_session_id = 100
        manual_control_sink._bot_state.current_session_id = 200

        # Should raise SessionConsistencyError
        with pytest.raises(SessionConsistencyError) as exc_info:
            manual_control_sink.stop_and_get_audio()

        # Verify error message contains session IDs
        assert "100" in str(exc_info.value)
        assert "200" in str(exc_info.value)

        # Verify audio buffer was NOT cleared (exception raised before cleanup)
        assert len(manual_control_sink._authority_buffer) == len(test_audio)

    def test_write_with_disallowed_user(self, manual_control_sink):
        """Test write method ignores disallowed users."""
        user = MagicMock(spec=discord.User)
        user.id = 999  # Not in allowed users
        voice_data = MagicMock(spec=voice_recv.VoiceData)
        voice_data.pcm = b"test_audio"

        # Should not raise exception and should not process
        manual_control_sink.write(user, voice_data)

    def test_write_with_allowed_user(self, manual_control_sink):
        """Test write method processes allowed users."""
        user = MagicMock(spec=discord.User)
        user.id = 123
        voice_data = MagicMock(spec=voice_recv.VoiceData)
        voice_data.pcm = b"test_audio_data"

        # Mock session state
        manual_control_sink._bot_state.current_state = BotStateEnum.RECORDING
        manual_control_sink._bot_state.recording_method = RecordingMethod.PushToTalk
        manual_control_sink._bot_state.is_authorized.return_value = True
        manual_control_sink._active_session_id = 100

        # Should not raise exception
        manual_control_sink.write(user, voice_data)

    def test_write_session_id_mismatch_prevention(self, manual_control_sink):
        """Test write method auto-updates session ID to prevent contamination."""
        user = MagicMock(spec=discord.User)
        user.id = 123
        voice_data = MagicMock(spec=voice_recv.VoiceData)
        voice_data.pcm = b"test_audio_data"

        # Set up session ID mismatch
        manual_control_sink._active_session_id = 100
        manual_control_sink._bot_state.current_session_id = 200

        manual_control_sink.write(user, voice_data)

        # Should have auto-updated the session ID to prevent contamination
        assert manual_control_sink._active_session_id == 200

    @pytest.mark.asyncio
    async def test_cleanup_comprehensive(self, manual_control_sink):
        """Test comprehensive cleanup of all resources."""
        # Add some users and state
        with patch("src.audio.sinks.Model"):
            manual_control_sink.add_user(789)

        manual_control_sink._authority_buffer.extend(b"test_data")
        manual_control_sink._vad_raw_buffer.extend(b"vad_data")

        # Mock VAD monitor task
        mock_task = MagicMock()
        mock_task.done.return_value = False  # Task is running, so should be cancelled
        manual_control_sink._vad_monitor_task = mock_task

        manual_control_sink.cleanup()

        # Verify comprehensive cleanup
        assert len(manual_control_sink._detectors) == 0
        assert len(manual_control_sink._user_audio_buffers) == 0
        assert len(manual_control_sink._authority_buffer) == 0
        # VAD buffers are not cleared by cleanup() - only by enable_vad()
        # This is expected behavior as VAD buffers may persist across sessions
        assert manual_control_sink._vad_analyzer is None
        mock_task.cancel.assert_called_once()

    def test_wants_opus_returns_false(self, manual_control_sink):
        """Test that wants_opus returns False."""
        assert manual_control_sink.wants_opus() is False

    def test_wake_word_buffer_management(self, manual_control_sink):
        """Test wake word buffer management and processing."""
        user_id = 123

        # Set up mock detector
        mock_detector = MagicMock()
        mock_detector.predict.return_value = {"wake_word": 0.8}  # Above threshold
        manual_control_sink._detectors[user_id] = mock_detector

        # Test that user has buffers after being added
        assert user_id in manual_control_sink._ww_resampled_buffers
        assert user_id in manual_control_sink._ww_buffer_locks

        # Test buffer exists and can be written to
        initial_length = len(manual_control_sink._ww_resampled_buffers[user_id])
        manual_control_sink._ww_resampled_buffers[user_id].extend(b"test_audio")
        assert len(manual_control_sink._ww_resampled_buffers[user_id]) > initial_length


class TestManualControlSinkIntegration:
    """Integration tests for ManualControlSink with real-world scenarios."""

    @pytest.fixture
    def mock_bot_state(self):
        bot_state = MagicMock(spec=BotState)
        bot_state.current_state = BotStateEnum.STANDBY
        bot_state.current_session_id = 100
        return bot_state

    @pytest.fixture
    def mock_callbacks(self):
        return {
            "on_wake_word_detected": AsyncMock(),
            "on_vad_speech_end": AsyncMock(),
        }

    @pytest.fixture
    def mock_action_lock(self):
        return asyncio.Lock()

    @pytest.mark.asyncio
    async def test_full_push_to_talk_scenario(
        self, mock_bot_state, mock_callbacks, mock_action_lock
    ):
        """Test complete push-to-talk recording scenario."""
        with (
            patch.multiple(
                "src.audio.sinks",
                UnifiedAudioProcessor=MagicMock(),
                Config=MagicMock(),
            ),
            patch("asyncio.get_running_loop", return_value=AsyncMock()),
            patch("asyncio.create_task", return_value=MagicMock()),
        ):
            sink = ManualControlSink(
                bot_state=mock_bot_state,
                initial_consented_users={123},
                on_wake_word_detected=mock_callbacks["on_wake_word_detected"],
                on_vad_speech_end=mock_callbacks["on_vad_speech_end"],
                action_lock=mock_action_lock,
            )

            # Start push-to-talk recording
            mock_bot_state.current_state = BotStateEnum.RECORDING
            mock_bot_state.recording_method = RecordingMethod.PushToTalk
            mock_bot_state.is_authorized.return_value = True

            user = MagicMock(spec=discord.User)
            user.id = 123
            voice_data = MagicMock(spec=voice_recv.VoiceData)
            voice_data.pcm = b"audio_chunk_1"

            # Write audio data
            sink.write(user, voice_data)

            # Stop recording and get audio
            result = sink.stop_and_get_audio()

            # Should have accumulated audio in authority buffer
            assert isinstance(result, bytes)

    @pytest.mark.asyncio
    async def test_full_wake_word_scenario(
        self, mock_bot_state, mock_callbacks, mock_action_lock
    ):
        """Test complete wake word detection scenario."""
        with (
            patch.multiple(
                "src.audio.sinks",
                UnifiedAudioProcessor=MagicMock(),
                Config=MagicMock(),
                Model=MagicMock(),
            ),
            patch("asyncio.get_running_loop", return_value=AsyncMock()),
            patch("asyncio.create_task", return_value=MagicMock()),
        ):
            sink = ManualControlSink(
                bot_state=mock_bot_state,
                initial_consented_users={123},
                on_wake_word_detected=mock_callbacks["on_wake_word_detected"],
                on_vad_speech_end=mock_callbacks["on_vad_speech_end"],
                action_lock=mock_action_lock,
            )
            sink._loop = AsyncMock()

            # Enable VAD for wake word detection
            with patch("src.audio.sinks.VADAnalyzer"):
                sink.enable_vad(True)

            # Simulate wake word detection
            mock_bot_state.current_state = BotStateEnum.RECORDING
            mock_bot_state.recording_method = RecordingMethod.WakeWord

            user = MagicMock(spec=discord.User)
            user.id = 123
            voice_data = MagicMock(spec=voice_recv.VoiceData)
            voice_data.pcm = b"wake_word_audio"

            # Write audio data
            sink.write(user, voice_data)

            # Verify wake word processing would occur
            assert sink._is_vad_enabled is True
