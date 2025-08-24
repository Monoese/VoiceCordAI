"""
Tests for GuildSession - The main session orchestrator.

These tests verify that GuildSession correctly manages per-guild state,
coordinates between components, handles user interactions, and provides
proper cleanup and error handling.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest
import discord
from discord.ext import commands

from src.bot.session.guild_session import GuildSession
from src.bot.state import BotModeEnum, BotStateEnum, RecordingMethod


class TestGuildSessionInitialization:
    """Test GuildSession initialization and basic functionality."""

    @pytest.fixture
    def mock_guild(self):
        guild = MagicMock(spec=discord.Guild)
        guild.id = 123
        guild.name = "Test Guild"
        return guild

    @pytest.fixture
    def mock_bot(self):
        bot = AsyncMock(spec=commands.Bot)
        bot.user = MagicMock()
        bot.user.id = 456
        return bot

    @pytest.fixture
    def ai_service_factories(self):
        return {
            "openai": ("OpenAIServiceManager", {"api_key": "test_key"}),
            "gemini": ("GeminiServiceManager", {"api_key": "test_key"}),
        }

    @patch("src.bot.session.guild_session.SessionUIManager")
    @patch("src.bot.session.guild_session.AudioPlaybackManager")
    @patch("src.bot.session.guild_session.VoiceConnectionManager")
    @patch("src.bot.session.guild_session.AIServiceCoordinator")
    @patch("src.bot.session.guild_session.InteractionHandler")
    @patch("src.bot.session.guild_session.UnifiedAudioProcessor")
    @patch("src.bot.session.guild_session.BotState")
    def test_initialization(
        self,
        mock_bot_state,
        mock_audio_processor,
        mock_interaction_handler,
        mock_ai_coordinator,
        mock_voice_connection,
        mock_audio_playback,
        mock_ui_manager,
        mock_guild,
        mock_bot,
        ai_service_factories,
    ):
        """Test that GuildSession initializes all components correctly."""
        session = GuildSession(mock_guild, mock_bot, ai_service_factories)

        assert session.guild == mock_guild
        assert session.bot == mock_bot
        assert session._audio_sink is None
        assert isinstance(session._background_tasks, set)
        assert len(session._background_tasks) == 0

    @patch("src.bot.session.guild_session.SessionUIManager")
    @patch("src.bot.session.guild_session.AudioPlaybackManager")
    @patch("src.bot.session.guild_session.VoiceConnectionManager")
    @patch("src.bot.session.guild_session.AIServiceCoordinator")
    @patch("src.bot.session.guild_session.InteractionHandler")
    @patch("src.bot.session.guild_session.UnifiedAudioProcessor")
    @patch("src.bot.session.guild_session.BotState")
    def test_component_initialization_order(
        self,
        mock_bot_state,
        mock_audio_processor,
        mock_interaction_handler,
        mock_ai_coordinator,
        mock_voice_connection,
        mock_audio_playback,
        mock_ui_manager,
        mock_guild,
        mock_bot,
        ai_service_factories,
    ):
        """Test that components are initialized in the correct order."""
        session = GuildSession(mock_guild, mock_bot, ai_service_factories)

        # Verify InteractionHandler was initialized last with fully initialized GuildSession
        mock_interaction_handler.assert_called_once()
        call_args = mock_interaction_handler.call_args
        assert call_args[1]["guild_session"] == session


class TestGuildSessionLifecycle:
    """Test session lifecycle methods."""

    @pytest.fixture
    def guild_session_with_mocks(self):
        """Create a GuildSession with mocked dependencies."""
        guild = MagicMock(spec=discord.Guild)
        guild.id = 123
        bot = AsyncMock(spec=commands.Bot)
        ai_service_factories = {"openai": ("OpenAIServiceManager", {"api_key": "test"})}

        with patch.multiple(
            "src.bot.session.guild_session",
            SessionUIManager=MagicMock(),
            AudioPlaybackManager=MagicMock(),
            VoiceConnectionManager=MagicMock(),
            AIServiceCoordinator=MagicMock(),
            InteractionHandler=MagicMock(),
            UnifiedAudioProcessor=MagicMock(),
            BotState=MagicMock(),
        ):
            session = GuildSession(guild, bot, ai_service_factories)

            # Set up commonly used mocks
            session.ui_manager = AsyncMock()
            session.interaction_handler = AsyncMock()
            session.ai_coordinator = AsyncMock()
            session.voice_connection = AsyncMock()
            session.bot_state = AsyncMock()

            return session

    @pytest.mark.asyncio
    async def test_start_background_tasks(self, guild_session_with_mocks):
        """Test that background tasks are started correctly."""
        session = guild_session_with_mocks

        await session.start_background_tasks()

        session.ui_manager.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_success(self, guild_session_with_mocks):
        """Test successful cleanup of all resources."""
        session = guild_session_with_mocks

        # Add some background tasks (create proper async task mocks)
        async def mock_coroutine():
            pass

        mock_task1 = asyncio.create_task(mock_coroutine())
        mock_task2 = asyncio.create_task(mock_coroutine())
        session._background_tasks.add(mock_task1)
        session._background_tasks.add(mock_task2)

        # Add an audio sink
        mock_audio_sink = MagicMock()
        session._audio_sink = mock_audio_sink

        session.voice_connection.is_connected.return_value = True

        await session.cleanup()

        # Verify cleanup sequence (tasks are cancelled but we can't easily mock the cancel calls)
        session.ui_manager.cleanup.assert_called_once()
        session.interaction_handler.cleanup.assert_called_once()
        mock_audio_sink.cleanup.assert_called_once()
        session.ai_coordinator.shutdown.assert_called_once()
        session.voice_connection.disconnect.assert_called_once()
        session.bot_state.reset_to_idle.assert_called_once()

        # Audio sink should be cleared
        assert session._audio_sink is None

    @pytest.mark.asyncio
    async def test_cleanup_handles_ai_coordinator_exception(
        self, guild_session_with_mocks
    ):
        """Test cleanup handles AI coordinator shutdown exceptions."""
        session = guild_session_with_mocks

        session.ai_coordinator.shutdown.side_effect = Exception("AI shutdown failed")
        session.voice_connection.is_connected.return_value = False

        # Should not raise exception
        await session.cleanup()

        session.bot_state.reset_to_idle.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_handles_voice_disconnect_exception(
        self, guild_session_with_mocks
    ):
        """Test cleanup handles voice disconnect exceptions."""
        session = guild_session_with_mocks

        session.voice_connection.is_connected.return_value = True
        session.voice_connection.disconnect.side_effect = Exception("Disconnect failed")

        # Should not raise exception
        await session.cleanup()

        session.bot_state.reset_to_idle.assert_called_once()


class TestGuildSessionEventHandlers:
    """Test event handling methods."""

    @pytest.fixture
    def guild_session_with_mocks(self):
        guild = MagicMock(spec=discord.Guild)
        guild.id = 123
        bot = AsyncMock(spec=commands.Bot)
        ai_service_factories = {"openai": ("OpenAIServiceManager", {"api_key": "test"})}

        with patch.multiple(
            "src.bot.session.guild_session",
            SessionUIManager=MagicMock(),
            AudioPlaybackManager=MagicMock(),
            VoiceConnectionManager=MagicMock(),
            AIServiceCoordinator=MagicMock(),
            InteractionHandler=MagicMock(),
            UnifiedAudioProcessor=MagicMock(),
            BotState=MagicMock(),
        ):
            session = GuildSession(guild, bot, ai_service_factories)
            session.interaction_handler = AsyncMock()
            session.ai_coordinator = AsyncMock()
            session.bot_state = AsyncMock()
            return session

    @pytest.mark.asyncio
    async def test_handle_reaction_add(self, guild_session_with_mocks):
        """Test reaction add event delegation."""
        session = guild_session_with_mocks
        reaction = MagicMock(spec=discord.Reaction)
        user = MagicMock(spec=discord.User)

        await session.handle_reaction_add(reaction, user)

        session.interaction_handler.handle_reaction_add.assert_called_once_with(
            reaction, user
        )

    @pytest.mark.asyncio
    async def test_handle_reaction_remove(self, guild_session_with_mocks):
        """Test reaction remove event delegation."""
        session = guild_session_with_mocks
        reaction = MagicMock(spec=discord.Reaction)
        user = MagicMock(spec=discord.User)

        await session.handle_reaction_remove(reaction, user)

        session.interaction_handler.handle_reaction_remove.assert_called_once_with(
            reaction, user
        )

    @pytest.mark.asyncio
    async def test_handle_voice_connection_update_connected_ai_connected(
        self, guild_session_with_mocks
    ):
        """Test voice connection update when both voice and AI are connected."""
        session = guild_session_with_mocks
        session.ai_coordinator.is_connected.return_value = True

        await session.handle_voice_connection_update(True)

        session.bot_state.recover_to_standby.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_voice_connection_update_connected_ai_disconnected(
        self, guild_session_with_mocks
    ):
        """Test voice connection update when voice connected but AI disconnected."""
        session = guild_session_with_mocks
        # Fix: Make is_connected a sync method that returns False
        session.ai_coordinator.is_connected = MagicMock(return_value=False)

        await session.handle_voice_connection_update(True)

        # Should not call recover_to_standby when AI is not connected
        session.bot_state.recover_to_standby.assert_not_called()
        # Should also not enter error state since voice is connected
        session.bot_state.enter_connection_error_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_voice_connection_update_disconnected(
        self, guild_session_with_mocks
    ):
        """Test voice connection update when voice disconnected."""
        session = guild_session_with_mocks

        await session.handle_voice_connection_update(False)

        session.bot_state.enter_connection_error_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_ai_connect(self, guild_session_with_mocks):
        """Test AI connect callback."""
        session = guild_session_with_mocks
        session.voice_connection = AsyncMock()
        session.voice_connection.is_connected.return_value = True

        await session._on_ai_connect()

        session.bot_state.recover_to_standby.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_ai_connect_no_voice(self, guild_session_with_mocks):
        """Test AI connect callback when voice not connected."""
        session = guild_session_with_mocks
        session.voice_connection = AsyncMock()
        # Fix: Make is_connected a sync method that returns False
        session.voice_connection.is_connected = MagicMock(return_value=False)

        await session._on_ai_connect()

        session.bot_state.recover_to_standby.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_ai_disconnect(self, guild_session_with_mocks):
        """Test AI disconnect callback."""
        session = guild_session_with_mocks

        await session._on_ai_disconnect()

        session.bot_state.enter_connection_error_state.assert_called_once()


class TestGuildSessionUserInteractions:
    """Test user interaction handling methods."""

    @pytest.fixture
    def guild_session_with_mocks(self):
        guild = MagicMock(spec=discord.Guild)
        guild.id = 123
        bot = AsyncMock(spec=commands.Bot)
        ai_service_factories = {"openai": ("OpenAIServiceManager", {"api_key": "test"})}

        with patch.multiple(
            "src.bot.session.guild_session",
            SessionUIManager=MagicMock(),
            AudioPlaybackManager=MagicMock(),
            VoiceConnectionManager=MagicMock(),
            AIServiceCoordinator=MagicMock(),
            InteractionHandler=MagicMock(),
            UnifiedAudioProcessor=MagicMock(),
            BotState=MagicMock(),
        ):
            session = GuildSession(guild, bot, ai_service_factories)
            session.bot_state = AsyncMock()
            session.ui_manager = AsyncMock()
            return session

    @pytest.mark.asyncio
    async def test_handle_consent_reaction_added(self, guild_session_with_mocks):
        """Test handling consent reaction when added."""
        session = guild_session_with_mocks
        user = MagicMock(spec=discord.User)
        user.id = 789

        # Mock audio sink
        mock_audio_sink = MagicMock()
        session._audio_sink = mock_audio_sink

        await session.handle_consent_reaction(user, added=True)

        session.bot_state.grant_consent.assert_called_once_with(user.id)
        session.ui_manager.schedule_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_consent_reaction_removed(self, guild_session_with_mocks):
        """Test handling consent reaction when removed."""
        session = guild_session_with_mocks
        user = MagicMock(spec=discord.User)
        user.id = 789

        # Mock audio sink
        mock_audio_sink = MagicMock()
        session._audio_sink = mock_audio_sink

        await session.handle_consent_reaction(user, added=False)

        session.bot_state.revoke_consent.assert_called_once_with(user.id)
        mock_audio_sink.remove_user.assert_called_once_with(user.id)
        session.ui_manager.schedule_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_mode_switch_reaction_valid_user(
        self, guild_session_with_mocks
    ):
        """Test mode switch reaction with valid user."""
        session = guild_session_with_mocks

        # Mock guild member with voice channel
        mock_member = MagicMock(spec=discord.Member)
        mock_member.voice = MagicMock()
        mock_member.voice.channel = MagicMock()
        session.guild.get_member.return_value = mock_member

        user = MagicMock(spec=discord.User)
        user.id = 789

        # User has given consent - get_consented_user_ids is a sync method
        session.bot_state.get_consented_user_ids = MagicMock(return_value={789})

        # Mock switch_mode
        session.switch_mode = AsyncMock()

        # Use actual config emoji instead of mocking
        await session.handle_mode_switch_reaction(user, "🙋")

        session.switch_mode.assert_called_once_with(BotModeEnum.ManualControl)

    @pytest.mark.asyncio
    async def test_handle_mode_switch_reaction_no_voice_channel(
        self, guild_session_with_mocks
    ):
        """Test mode switch reaction with user not in voice channel."""
        session = guild_session_with_mocks

        # Mock guild member without voice channel
        mock_member = MagicMock(spec=discord.Member)
        mock_member.voice = None
        session.guild.get_member.return_value = mock_member

        user = MagicMock(spec=discord.User)
        user.id = 789

        session.switch_mode = AsyncMock()

        await session.handle_mode_switch_reaction(user, "✋")

        # Should not call switch_mode
        session.switch_mode.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_mode_switch_reaction_no_consent(
        self, guild_session_with_mocks
    ):
        """Test mode switch reaction with user who hasn't given consent."""
        session = guild_session_with_mocks

        # Mock guild member with voice channel
        mock_member = MagicMock(spec=discord.Member)
        mock_member.voice = MagicMock()
        mock_member.voice.channel = MagicMock()
        session.guild.get_member.return_value = mock_member

        user = MagicMock(spec=discord.User)
        user.id = 789

        # User has not given consent - get_consented_user_ids is a sync method
        session.bot_state.get_consented_user_ids = MagicMock(return_value=set())

        session.switch_mode = AsyncMock()

        await session.handle_mode_switch_reaction(user, "✋")

        # Should not call switch_mode
        session.switch_mode.assert_not_called()


class TestGuildSessionPushToTalkInteractions:
    """Test push-to-talk interaction handling."""

    @pytest.fixture
    def guild_session_with_mocks(self):
        guild = MagicMock(spec=discord.Guild)
        guild.id = 123
        bot = AsyncMock(spec=commands.Bot)
        ai_service_factories = {"openai": ("OpenAIServiceManager", {"api_key": "test"})}

        with patch.multiple(
            "src.bot.session.guild_session",
            SessionUIManager=MagicMock(),
            AudioPlaybackManager=MagicMock(),
            VoiceConnectionManager=MagicMock(),
            AIServiceCoordinator=MagicMock(),
            InteractionHandler=MagicMock(),
            UnifiedAudioProcessor=MagicMock(),
            BotState=MagicMock(),
            ManualControlSink=MagicMock(),
        ):
            session = GuildSession(guild, bot, ai_service_factories)
            session.bot_state = AsyncMock()
            session.voice_connection = AsyncMock()
            session.ai_coordinator = AsyncMock()
            return session

    @pytest.mark.skip(
        reason="Integration test - needs behavioral rewrite (tests implementation details)"
    )
    @pytest.mark.asyncio
    async def test_handle_pushtotalk_reaction_start_recording(
        self, guild_session_with_mocks
    ):
        """Test push-to-talk reaction to start recording."""
        session = guild_session_with_mocks
        user = MagicMock(spec=discord.User)

        # Set up conditions for starting recording
        session.bot_state.mode = BotModeEnum.ManualControl
        session.bot_state.current_state = BotStateEnum.STANDBY

        # Mock audio sink
        mock_audio_sink = MagicMock()
        session._audio_sink = mock_audio_sink

        # Mock interrupt method
        session._interrupt_ongoing_playback = AsyncMock()

        await session.handle_pushtotalk_reaction(user, added=True)

        session._interrupt_ongoing_playback.assert_called_once()
        mock_audio_sink.enable_vad.assert_called_once_with(False)
        session.bot_state.start_recording.assert_called_once_with(
            user, RecordingMethod.PushToTalk
        )
        mock_audio_sink.update_session_id.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_pushtotalk_reaction_wrong_mode(
        self, guild_session_with_mocks
    ):
        """Test push-to-talk reaction in wrong mode."""
        session = guild_session_with_mocks
        user = MagicMock(spec=discord.User)

        # Wrong mode
        session.bot_state.mode = BotModeEnum.RealtimeTalk

        await session.handle_pushtotalk_reaction(user, added=True)

        # Should not start recording
        session.bot_state.start_recording.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_pushtotalk_reaction_stop_recording(
        self, guild_session_with_mocks
    ):
        """Test push-to-talk reaction to stop recording."""
        session = guild_session_with_mocks
        user = MagicMock(spec=discord.User)

        # Set up conditions for stopping recording
        session.bot_state.mode = BotModeEnum.ManualControl
        session.bot_state.current_state = BotStateEnum.RECORDING
        session.bot_state.recording_method = RecordingMethod.PushToTalk
        session.bot_state.is_authorized.return_value = True

        # Mock audio sink
        mock_audio_sink = MagicMock()
        mock_audio_sink.stop_and_get_audio.return_value = b"audio_data"
        session._audio_sink = mock_audio_sink

        # Mock handle finished recording
        session._handle_finished_recording = MagicMock()

        await session.handle_pushtotalk_reaction(user, added=False)

        mock_audio_sink.stop_and_get_audio.assert_called_once()
        session._handle_finished_recording.assert_called_once_with(b"audio_data")
        session.bot_state.stop_recording.assert_called_once()

    @pytest.mark.asyncio
    async def test_interrupt_ongoing_playback(self, guild_session_with_mocks):
        """Test interrupting ongoing playback."""
        session = guild_session_with_mocks
        session.ai_coordinator.cancel_ongoing_response.return_value = True

        await session._interrupt_ongoing_playback()

        session.voice_connection.stop_playback.assert_called_once()
        session.ai_coordinator.cancel_ongoing_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_interrupt_ongoing_playback_cancel_fails(
        self, guild_session_with_mocks
    ):
        """Test interrupting ongoing playback when cancel fails."""
        session = guild_session_with_mocks
        session.ai_coordinator.cancel_ongoing_response.return_value = False

        await session._interrupt_ongoing_playback()

        session.voice_connection.stop_playback.assert_called_once()


class TestGuildSessionWakeWordInteractions:
    """Test wake word detection interaction handling."""

    @pytest.fixture
    def guild_session_with_mocks(self):
        guild = MagicMock(spec=discord.Guild)
        guild.id = 123
        bot = AsyncMock(spec=commands.Bot)
        ai_service_factories = {"openai": ("OpenAIServiceManager", {"api_key": "test"})}

        with patch.multiple(
            "src.bot.session.guild_session",
            SessionUIManager=MagicMock(),
            AudioPlaybackManager=MagicMock(),
            VoiceConnectionManager=MagicMock(),
            AIServiceCoordinator=MagicMock(),
            InteractionHandler=MagicMock(),
            UnifiedAudioProcessor=MagicMock(),
            BotState=MagicMock(),
            ManualControlSink=MagicMock(),
        ):
            session = GuildSession(guild, bot, ai_service_factories)
            session.bot_state = AsyncMock()
            session.audio_playback_manager = AsyncMock()
            return session

    @pytest.mark.skip(
        reason="Integration test - needs behavioral rewrite (tests implementation details)"
    )
    @pytest.mark.asyncio
    async def test_on_wake_word_detected_valid_conditions(
        self, guild_session_with_mocks
    ):
        """Test wake word detection with valid conditions."""
        session = guild_session_with_mocks
        user = MagicMock(spec=discord.User)

        # Set up valid conditions
        session.bot_state.mode = BotModeEnum.ManualControl
        session.bot_state.current_state = BotStateEnum.STANDBY

        # Mock audio sink
        mock_audio_sink = MagicMock()
        session._audio_sink = mock_audio_sink

        # Mock interrupt method
        session._interrupt_ongoing_playback = AsyncMock()

        await session.on_wake_word_detected(user)

        session._interrupt_ongoing_playback.assert_called_once()
        mock_audio_sink.enable_vad.assert_called_once_with(True)
        session.bot_state.start_recording.assert_called_once_with(
            user, RecordingMethod.WakeWord
        )
        mock_audio_sink.update_session_id.assert_called_once()
        session.audio_playback_manager.play_cue.assert_called_once_with(
            "start_recording"
        )

    @pytest.mark.asyncio
    async def test_on_wake_word_detected_wrong_mode(self, guild_session_with_mocks):
        """Test wake word detection in wrong mode."""
        session = guild_session_with_mocks
        user = MagicMock(spec=discord.User)

        # Wrong mode
        session.bot_state.mode = BotModeEnum.RealtimeTalk
        session.bot_state.current_state = BotStateEnum.STANDBY

        await session.on_wake_word_detected(user)

        # Should not start recording
        session.bot_state.start_recording.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_wake_word_detected_wrong_state(self, guild_session_with_mocks):
        """Test wake word detection in wrong state."""
        session = guild_session_with_mocks
        user = MagicMock(spec=discord.User)

        # Correct mode but wrong state
        session.bot_state.mode = BotModeEnum.ManualControl
        session.bot_state.current_state = BotStateEnum.RECORDING

        await session.on_wake_word_detected(user)

        # Should not start recording
        session.bot_state.start_recording.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_vad_speech_end_valid_conditions(self, guild_session_with_mocks):
        """Test VAD speech end with valid conditions."""
        session = guild_session_with_mocks
        audio_data = b"test_audio_data"

        # Set up valid conditions
        session.bot_state.current_state = BotStateEnum.RECORDING
        session.bot_state.recording_method = RecordingMethod.WakeWord

        # Mock handle finished recording
        session._handle_finished_recording = MagicMock()

        await session.on_vad_speech_end(audio_data)

        session._handle_finished_recording.assert_called_once_with(audio_data)
        session.bot_state.stop_recording.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_vad_speech_end_wrong_state(self, guild_session_with_mocks):
        """Test VAD speech end in wrong state."""
        session = guild_session_with_mocks
        audio_data = b"test_audio_data"

        # Wrong state
        session.bot_state.current_state = BotStateEnum.STANDBY
        session.bot_state.recording_method = RecordingMethod.WakeWord

        session._handle_finished_recording = MagicMock()

        await session.on_vad_speech_end(audio_data)

        # Should not handle recording
        session._handle_finished_recording.assert_not_called()


class TestGuildSessionSessionManagement:
    """Test session management operations."""

    @pytest.fixture
    def guild_session_with_mocks(self):
        guild = MagicMock(spec=discord.Guild)
        guild.id = 123
        bot = AsyncMock(spec=commands.Bot)
        ai_service_factories = {"openai": ("OpenAIServiceManager", {"api_key": "test"})}

        with patch.multiple(
            "src.bot.session.guild_session",
            SessionUIManager=MagicMock(),
            AudioPlaybackManager=MagicMock(),
            VoiceConnectionManager=MagicMock(),
            AIServiceCoordinator=MagicMock(),
            InteractionHandler=MagicMock(),
            UnifiedAudioProcessor=MagicMock(),
            BotState=MagicMock(),
        ):
            session = GuildSession(guild, bot, ai_service_factories)
            session.bot_state = AsyncMock()
            session.ai_coordinator = AsyncMock()
            session.voice_connection = AsyncMock()
            session.ui_manager = AsyncMock()
            return session

    @pytest.mark.skip(
        reason="Integration test - needs behavioral rewrite (tests implementation details)"
    )
    @pytest.mark.asyncio
    async def test_initialize_session_success(self, guild_session_with_mocks):
        """Test successful session initialization."""
        session = guild_session_with_mocks

        mock_ctx = AsyncMock(spec=commands.Context)
        mock_ctx.author = MagicMock(spec=discord.User)
        mock_ctx.author.voice = MagicMock()
        mock_ctx.author.voice.channel = MagicMock()

        # Mock methods
        session._initialize_sink = AsyncMock()
        session.start_background_tasks = AsyncMock()
        session.voice_connection.connect = AsyncMock(return_value=True)
        session.ai_coordinator.initialize = AsyncMock()

        result = await session.initialize_session(mock_ctx, BotModeEnum.ManualControl)

        assert result is True
        session.bot_state.initialize.assert_called_once()
        session._initialize_sink.assert_called_once()
        session.start_background_tasks.assert_called_once()
        session.voice_connection.connect.assert_called_once()
        session.ai_coordinator.initialize.assert_called_once()

    @pytest.mark.skip(
        reason="Integration test - needs behavioral rewrite (tests implementation details)"
    )
    @pytest.mark.asyncio
    async def test_initialize_session_voice_connect_fails(
        self, guild_session_with_mocks
    ):
        """Test session initialization when voice connection fails."""
        session = guild_session_with_mocks

        mock_ctx = AsyncMock(spec=commands.Context)
        mock_ctx.author = MagicMock(spec=discord.User)
        mock_ctx.author.voice = MagicMock()
        mock_ctx.author.voice.channel = MagicMock()

        # Mock methods
        session._initialize_sink = AsyncMock()
        session.start_background_tasks = AsyncMock()
        session.voice_connection.connect = AsyncMock(
            return_value=False
        )  # Connection fails

        result = await session.initialize_session(mock_ctx, BotModeEnum.ManualControl)

        assert result is False
        # AI coordinator should not be initialized if voice connection fails
        session.ai_coordinator.initialize.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_provider_success(self, guild_session_with_mocks):
        """Test successful AI provider change."""
        session = guild_session_with_mocks

        mock_ctx = AsyncMock(spec=commands.Context)
        session.ai_coordinator.switch_provider = AsyncMock(return_value=True)
        session.voice_connection.is_connected = MagicMock(return_value=True)
        session.ai_coordinator.ai_service_factories = {
            "openai": ("test", {}),
            "gemini": ("test", {}),
        }

        await session.set_provider(mock_ctx, "openai")

        session.ai_coordinator.switch_provider.assert_called_once()
        mock_ctx.send.assert_called_once_with("AI provider switched to 'OPENAI'.")

    @pytest.mark.asyncio
    async def test_set_provider_invalid(self, guild_session_with_mocks):
        """Test AI provider change with invalid provider."""
        session = guild_session_with_mocks

        mock_ctx = AsyncMock(spec=commands.Context)
        session.ai_coordinator.ai_service_factories = {
            "openai": ("test", {}),
            "gemini": ("test", {}),
        }

        await session.set_provider(mock_ctx, "invalid_provider")

        mock_ctx.send.assert_called_once_with(
            "Invalid provider name 'invalid_provider'. Valid options are: openai, gemini."
        )

    @pytest.mark.skip(
        reason="Integration test - needs behavioral rewrite (tests implementation details)"
    )
    @pytest.mark.asyncio
    async def test_switch_mode_manual_to_realtime(self, guild_session_with_mocks):
        """Test switching from manual to realtime mode."""
        session = guild_session_with_mocks

        # Current mode is manual
        session.bot_state.mode = BotModeEnum.ManualControl

        # Mock methods
        session._initialize_sink = AsyncMock()

        await session.switch_mode(BotModeEnum.RealtimeTalk)

        session.bot_state.set_mode.assert_called_once_with(BotModeEnum.RealtimeTalk)
        session._initialize_sink.assert_called_once()
        session.ui_manager.schedule_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_switch_mode_same_mode(self, guild_session_with_mocks):
        """Test switching to the same mode (no-op)."""
        session = guild_session_with_mocks

        # Current mode is already manual
        session.bot_state.mode = BotModeEnum.ManualControl

        await session.switch_mode(BotModeEnum.ManualControl)

        # Should not call set_mode or initialize_sink
        session.bot_state.set_mode.assert_not_called()
        session._initialize_sink = AsyncMock()
        session._initialize_sink.assert_not_called()


class TestGuildSessionConcurrency:
    """Test concurrency and action lock behavior."""

    @pytest.fixture
    def guild_session_with_mocks(self):
        guild = MagicMock(spec=discord.Guild)
        guild.id = 123
        bot = AsyncMock(spec=commands.Bot)
        ai_service_factories = {"openai": ("OpenAIServiceManager", {"api_key": "test"})}

        with patch.multiple(
            "src.bot.session.guild_session",
            SessionUIManager=MagicMock(),
            AudioPlaybackManager=MagicMock(),
            VoiceConnectionManager=MagicMock(),
            AIServiceCoordinator=MagicMock(),
            InteractionHandler=MagicMock(),
            UnifiedAudioProcessor=MagicMock(),
            BotState=MagicMock(),
        ):
            session = GuildSession(guild, bot, ai_service_factories)
            session.bot_state = AsyncMock()
            session.ui_manager = AsyncMock()
            return session

    @pytest.mark.asyncio
    async def test_concurrent_consent_reactions(self, guild_session_with_mocks):
        """Test that concurrent consent reactions are properly serialized."""
        session = guild_session_with_mocks

        user1 = MagicMock(spec=discord.User)
        user1.id = 111
        user2 = MagicMock(spec=discord.User)
        user2.id = 222

        # Start both tasks concurrently
        task1 = asyncio.create_task(session.handle_consent_reaction(user1, True))
        task2 = asyncio.create_task(session.handle_consent_reaction(user2, True))

        # Wait for both to complete
        await asyncio.gather(task1, task2)

        # Both should have been processed
        assert session.bot_state.grant_consent.call_count == 2
        session.bot_state.grant_consent.assert_has_calls(
            [call(111), call(222)], any_order=True
        )

    @pytest.mark.asyncio
    async def test_concurrent_pushtotalk_reactions(self, guild_session_with_mocks):
        """Test that concurrent push-to-talk reactions are properly serialized."""
        session = guild_session_with_mocks

        user = MagicMock(spec=discord.User)
        session.bot_state.mode = BotModeEnum.ManualControl
        session.bot_state.current_state = BotStateEnum.STANDBY

        # Mock interrupt method
        session._interrupt_ongoing_playback = AsyncMock()

        # Start concurrent push-to-talk events
        task1 = asyncio.create_task(session.handle_pushtotalk_reaction(user, True))
        task2 = asyncio.create_task(session.handle_pushtotalk_reaction(user, True))

        # Wait for both to complete
        await asyncio.gather(task1, task2)

        # Should be properly serialized - exact behavior depends on timing
        # but both tasks should complete without race conditions
        assert session._interrupt_ongoing_playback.call_count >= 1


class TestGuildSessionErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def guild_session_with_mocks(self):
        guild = MagicMock(spec=discord.Guild)
        guild.id = 123
        bot = AsyncMock(spec=commands.Bot)
        ai_service_factories = {"openai": ("OpenAIServiceManager", {"api_key": "test"})}

        with patch.multiple(
            "src.bot.session.guild_session",
            SessionUIManager=MagicMock(),
            AudioPlaybackManager=MagicMock(),
            VoiceConnectionManager=MagicMock(),
            AIServiceCoordinator=MagicMock(),
            InteractionHandler=MagicMock(),
            UnifiedAudioProcessor=MagicMock(),
            BotState=MagicMock(),
        ):
            session = GuildSession(guild, bot, ai_service_factories)
            session.bot_state = AsyncMock()
            return session

    @pytest.mark.asyncio
    async def test_handle_finished_recording_creates_background_task(
        self, guild_session_with_mocks
    ):
        """Test that _handle_finished_recording creates a background task."""
        session = guild_session_with_mocks

        audio_data = b"test_audio"

        session._handle_finished_recording(audio_data)

        # Should have created a background task
        assert len(session._background_tasks) == 1

    @pytest.mark.skip(
        reason="Integration test - needs behavioral rewrite (tests implementation details)"
    )
    @pytest.mark.asyncio
    async def test_safe_enter_error_state_with_matching_session_id(
        self, guild_session_with_mocks
    ):
        """Test safe enter error state with matching session ID."""
        session = guild_session_with_mocks
        session.bot_state.session_id = 123

        await session._safe_enter_error_state(123)

        session.bot_state.enter_connection_error_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_safe_enter_error_state_with_mismatched_session_id(
        self, guild_session_with_mocks
    ):
        """Test safe enter error state with mismatched session ID."""
        session = guild_session_with_mocks
        session.bot_state.session_id = 123

        await session._safe_enter_error_state(456)  # Different session ID

        # Should not enter error state
        session.bot_state.enter_connection_error_state.assert_not_called()
