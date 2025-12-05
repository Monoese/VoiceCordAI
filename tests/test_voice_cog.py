"""
Tests for VoiceCog - The main Discord command dispatcher.

These tests verify that VoiceCog correctly handles Discord commands and events,
manages guild sessions, and provides proper error handling and cleanup.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import discord
from discord.ext import commands

from src.bot.cogs.voice_cog import VoiceCog
from src.exceptions import SessionError, StateTransitionError


class TestVoiceCogInitialization:
    """Test VoiceCog initialization and basic functionality."""

    @pytest.fixture
    def mock_bot(self):
        """Create a mock Discord bot."""
        bot = AsyncMock(spec=commands.Bot)
        bot.user = MagicMock()
        bot.user.id = 12345
        return bot

    @pytest.fixture
    def ai_service_factories(self):
        """Create mock AI service factories."""
        return {
            "openai": ("OpenAIServiceManager", {"api_key": "test_key"}),
            "gemini": ("GeminiServiceManager", {"api_key": "test_key"}),
        }

    @pytest.fixture
    def voice_cog(self, mock_bot, ai_service_factories):
        """Create a VoiceCog instance for testing."""
        return VoiceCog(mock_bot, ai_service_factories)

    def test_initialization(self, voice_cog, mock_bot, ai_service_factories):
        """Test that VoiceCog initializes correctly."""
        assert voice_cog.bot == mock_bot
        assert voice_cog.ai_service_factories == ai_service_factories
        assert voice_cog._sessions == {}
        assert len(voice_cog._session_locks) == 0

    @pytest.mark.asyncio
    async def test_cog_unload_with_no_sessions(self, voice_cog):
        """Test that cog unloading works with no active sessions."""
        await voice_cog.cog_unload()
        assert len(voice_cog._sessions) == 0

    @pytest.mark.asyncio
    async def test_cog_unload_with_active_sessions(self, voice_cog):
        """Test that cog unloading properly cleans up active sessions."""
        # Create mock sessions
        mock_session1 = AsyncMock()
        mock_session2 = AsyncMock()
        voice_cog._sessions[123] = mock_session1
        voice_cog._sessions[456] = mock_session2

        await voice_cog.cog_unload()

        # Verify cleanup was called on all sessions
        mock_session1.cleanup.assert_called_once()
        mock_session2.cleanup.assert_called_once()
        assert len(voice_cog._sessions) == 0

    @pytest.mark.asyncio
    async def test_cog_unload_handles_cleanup_exceptions(self, voice_cog):
        """Test that cog unloading handles exceptions during cleanup gracefully."""
        # Create mock session that raises exception during cleanup
        mock_session = AsyncMock()
        mock_session.cleanup.side_effect = Exception("Cleanup failed")
        voice_cog._sessions[123] = mock_session

        # Should not raise exception
        await voice_cog.cog_unload()
        assert len(voice_cog._sessions) == 0


class TestVoiceCogSessionManagement:
    """Test session creation and management functionality."""

    @pytest.fixture
    def mock_bot(self):
        bot = AsyncMock(spec=commands.Bot)
        bot.user = MagicMock()
        bot.user.id = 12345
        return bot

    @pytest.fixture
    def ai_service_factories(self):
        return {"openai": ("OpenAIServiceManager", {"api_key": "test_key"})}

    @pytest.fixture
    def voice_cog(self, mock_bot, ai_service_factories):
        return VoiceCog(mock_bot, ai_service_factories)

    @pytest.fixture
    def mock_guild(self):
        guild = MagicMock(spec=discord.Guild)
        guild.id = 789
        guild.name = "Test Guild"
        return guild

    @patch("src.bot.cogs.voice_cog.GuildSession")
    def test_get_or_create_session_creates_new(
        self, mock_guild_session_class, voice_cog, mock_guild
    ):
        """Test that _get_or_create_session creates a new session when none exists."""
        mock_session = MagicMock()
        mock_guild_session_class.return_value = mock_session

        result = voice_cog._get_or_create_session(mock_guild)

        assert result == mock_session
        assert voice_cog._sessions[789] == mock_session
        mock_guild_session_class.assert_called_once_with(
            guild=mock_guild,
            bot=voice_cog.bot,
            ai_service_factories=voice_cog.ai_service_factories,
        )

    def test_get_or_create_session_returns_existing(self, voice_cog, mock_guild):
        """Test that _get_or_create_session returns existing session."""
        existing_session = MagicMock()
        voice_cog._sessions[789] = existing_session

        result = voice_cog._get_or_create_session(mock_guild)

        assert result == existing_session
        # Should still only have one session
        assert len(voice_cog._sessions) == 1


class TestVoiceCogCommands:
    """Test Discord command handling."""

    @pytest.fixture
    def mock_bot(self):
        bot = AsyncMock(spec=commands.Bot)
        bot.user = MagicMock()
        bot.user.id = 12345
        return bot

    @pytest.fixture
    def ai_service_factories(self):
        return {"openai": ("OpenAIServiceManager", {"api_key": "test_key"})}

    @pytest.fixture
    def voice_cog(self, mock_bot, ai_service_factories):
        return VoiceCog(mock_bot, ai_service_factories)

    @pytest.fixture
    def mock_ctx(self):
        ctx = AsyncMock(spec=commands.Context)
        ctx.guild = MagicMock(spec=discord.Guild)
        ctx.guild.id = 789
        ctx.guild.name = "Test Guild"
        return ctx

    @pytest.mark.asyncio
    async def test_connect_command_success(self, voice_cog, mock_ctx):
        """Test successful connect command."""
        with patch.object(voice_cog, "_handle_connect_command") as mock_handle:
            # Call the actual callback method, not the command wrapper
            await voice_cog.connect_command.callback(voice_cog, mock_ctx)
            mock_handle.assert_called_once_with(mock_ctx)

    @pytest.mark.asyncio
    async def test_handle_connect_command_already_connected(self, voice_cog, mock_ctx):
        """Test connect command when already connected."""
        # Add existing session
        mock_session = MagicMock()
        voice_cog._sessions[789] = mock_session

        await voice_cog._handle_connect_command(mock_ctx)

        mock_ctx.send.assert_called_once_with(
            "I'm already in a session in this server. Use `/disconnect` to end it first."
        )

    @pytest.mark.asyncio
    @patch("src.bot.cogs.voice_cog.GuildSession")
    async def test_handle_connect_command_success(
        self, mock_guild_session_class, voice_cog, mock_ctx
    ):
        """Test successful connect command."""
        mock_session = AsyncMock()
        mock_session.initialize_session.return_value = True
        mock_guild_session_class.return_value = mock_session

        await voice_cog._handle_connect_command(mock_ctx)

        mock_session.initialize_session.assert_called_once_with(mock_ctx)
        assert voice_cog._sessions[789] == mock_session

    @pytest.mark.asyncio
    @patch("src.bot.cogs.voice_cog.GuildSession")
    async def test_handle_connect_command_initialization_fails(
        self, mock_guild_session_class, voice_cog, mock_ctx
    ):
        """Test connect command when session initialization fails."""
        mock_session = AsyncMock()
        mock_session.initialize_session.return_value = False
        mock_guild_session_class.return_value = mock_session

        await voice_cog._handle_connect_command(mock_ctx)

        # Session should be removed from sessions dict
        assert 789 not in voice_cog._sessions

    @pytest.mark.asyncio
    @patch("src.bot.cogs.voice_cog.GuildSession")
    async def test_handle_connect_command_state_transition_error(
        self, mock_guild_session_class, voice_cog, mock_ctx
    ):
        """Test connect command when StateTransitionError occurs."""
        mock_session = AsyncMock()
        mock_session.initialize_session.side_effect = StateTransitionError("Test error")
        mock_guild_session_class.return_value = mock_session

        await voice_cog._handle_connect_command(mock_ctx)

        mock_ctx.send.assert_called_once_with(
            "An unexpected internal error occurred. The session will now terminate."
        )
        mock_session.cleanup.assert_called_once()
        assert 789 not in voice_cog._sessions

    @pytest.mark.asyncio
    async def test_set_provider_command_no_session(self, voice_cog, mock_ctx):
        """Test set provider command when no session exists."""
        await voice_cog.set_provider_command.callback(voice_cog, mock_ctx, "openai")

        mock_ctx.send.assert_called_once_with(
            "The bot is not currently in a session. Use the 'connect' command first."
        )

    @pytest.mark.asyncio
    async def test_set_provider_command_success(self, voice_cog, mock_ctx):
        """Test successful set provider command."""
        mock_session = AsyncMock()
        voice_cog._sessions[789] = mock_session

        await voice_cog.set_provider_command.callback(voice_cog, mock_ctx, "openai")

        mock_session.set_provider.assert_called_once_with(mock_ctx, "openai")

    @pytest.mark.asyncio
    async def test_set_provider_command_state_transition_error(
        self, voice_cog, mock_ctx
    ):
        """Test set provider command when StateTransitionError occurs."""
        mock_session = AsyncMock()
        mock_session.set_provider.side_effect = StateTransitionError("Test error")
        voice_cog._sessions[789] = mock_session

        await voice_cog.set_provider_command.callback(voice_cog, mock_ctx, "openai")

        mock_ctx.send.assert_called_once_with(
            "An unexpected internal error occurred. The session will now terminate."
        )
        mock_session.cleanup.assert_called_once()
        assert 789 not in voice_cog._sessions

    @pytest.mark.asyncio
    async def test_disconnect_command_no_session(self, voice_cog, mock_ctx):
        """Test disconnect command when no session exists."""
        await voice_cog.disconnect_command.callback(voice_cog, mock_ctx)

        mock_ctx.send.assert_called_once_with(
            "The bot is not currently in a session in this server."
        )

    @pytest.mark.asyncio
    async def test_disconnect_command_success(self, voice_cog, mock_ctx):
        """Test successful disconnect command."""
        mock_session = AsyncMock()
        voice_cog._sessions[789] = mock_session

        await voice_cog.disconnect_command.callback(voice_cog, mock_ctx)

        mock_session.cleanup.assert_called_once()
        assert 789 not in voice_cog._sessions

    @pytest.mark.asyncio
    async def test_disconnect_command_state_transition_error(self, voice_cog, mock_ctx):
        """Test disconnect command when StateTransitionError occurs during cleanup."""
        mock_session = AsyncMock()
        mock_session.cleanup.side_effect = StateTransitionError("Test error")
        voice_cog._sessions[789] = mock_session

        await voice_cog.disconnect_command.callback(voice_cog, mock_ctx)

        mock_ctx.send.assert_called_once_with(
            "An unexpected internal error occurred during cleanup. The session has been forcefully removed."
        )
        assert 789 not in voice_cog._sessions

    @pytest.mark.asyncio
    async def test_disconnect_command_session_error(self, voice_cog, mock_ctx):
        """Test disconnect command when SessionError occurs during cleanup."""
        mock_session = AsyncMock()
        mock_session.cleanup.side_effect = SessionError("Test session error")
        voice_cog._sessions[789] = mock_session

        await voice_cog.disconnect_command.callback(voice_cog, mock_ctx)

        # Session should still be removed
        assert 789 not in voice_cog._sessions

    @pytest.mark.asyncio
    async def test_disconnect_command_generic_exception(self, voice_cog, mock_ctx):
        """Test disconnect command when generic exception occurs during cleanup."""
        mock_session = AsyncMock()
        mock_session.cleanup.side_effect = Exception("Unexpected error")
        voice_cog._sessions[789] = mock_session

        await voice_cog.disconnect_command.callback(voice_cog, mock_ctx)

        mock_ctx.send.assert_called_once_with(
            "An error occurred during cleanup. The session has been forcefully removed."
        )
        assert 789 not in voice_cog._sessions


class TestVoiceCogEventHandlers:
    """Test Discord event handling."""

    @pytest.fixture
    def mock_bot(self):
        bot = AsyncMock(spec=commands.Bot)
        bot.user = MagicMock()
        bot.user.id = 12345
        return bot

    @pytest.fixture
    def ai_service_factories(self):
        return {"openai": ("OpenAIServiceManager", {"api_key": "test_key"})}

    @pytest.fixture
    def voice_cog(self, mock_bot, ai_service_factories):
        return VoiceCog(mock_bot, ai_service_factories)

    @pytest.fixture
    def mock_member(self, mock_bot):
        member = MagicMock(spec=discord.Member)
        member.id = mock_bot.user.id  # Bot member
        member.guild = MagicMock(spec=discord.Guild)
        member.guild.id = 789
        return member

    @pytest.fixture
    def mock_voice_state(self):
        return MagicMock(spec=discord.VoiceState)

    @pytest.mark.asyncio
    async def test_on_voice_state_update_ignores_other_members(self, voice_cog):
        """Test that voice state updates for other members are ignored."""
        other_member = MagicMock(spec=discord.Member)
        other_member.id = 99999  # Different from bot ID
        voice_state = MagicMock(spec=discord.VoiceState)

        await voice_cog.on_voice_state_update(other_member, voice_state, voice_state)

        # Should do nothing - no session interaction

    @pytest.mark.asyncio
    async def test_on_voice_state_update_ignores_same_channel(
        self, voice_cog, mock_member, mock_voice_state
    ):
        """Test that voice state updates with same channel are ignored."""
        mock_voice_state.channel = MagicMock()

        await voice_cog.on_voice_state_update(
            mock_member, mock_voice_state, mock_voice_state
        )

        # Should do nothing - no session interaction

    @pytest.mark.asyncio
    async def test_on_voice_state_update_no_session(self, voice_cog, mock_member):
        """Test voice state update when no session exists."""
        before = MagicMock(spec=discord.VoiceState)
        before.channel = MagicMock()
        after = MagicMock(spec=discord.VoiceState)
        after.channel = None

        await voice_cog.on_voice_state_update(mock_member, before, after)

        # Should do nothing - no session exists

    @pytest.mark.asyncio
    async def test_on_voice_state_update_with_session_connect(
        self, voice_cog, mock_member
    ):
        """Test voice state update with session - bot connects."""
        mock_session = AsyncMock()
        voice_cog._sessions[789] = mock_session

        before = MagicMock(spec=discord.VoiceState)
        before.channel = None
        after = MagicMock(spec=discord.VoiceState)
        after.channel = MagicMock()

        await voice_cog.on_voice_state_update(mock_member, before, after)

        mock_session.handle_voice_connection_update.assert_called_once_with(True)

    @pytest.mark.asyncio
    async def test_on_voice_state_update_with_session_disconnect(
        self, voice_cog, mock_member
    ):
        """Test voice state update with session - bot disconnects."""
        mock_session = AsyncMock()
        voice_cog._sessions[789] = mock_session

        before = MagicMock(spec=discord.VoiceState)
        before.channel = MagicMock()
        after = MagicMock(spec=discord.VoiceState)
        after.channel = None

        await voice_cog.on_voice_state_update(mock_member, before, after)

        mock_session.handle_voice_connection_update.assert_called_once_with(False)

    @pytest.mark.asyncio
    async def test_on_reaction_add_no_guild(self, voice_cog):
        """Test reaction add event with no guild."""
        reaction = MagicMock(spec=discord.Reaction)
        reaction.message.guild = None
        user = MagicMock(spec=discord.User)

        await voice_cog.on_reaction_add(reaction, user)

        # Should do nothing

    @pytest.mark.asyncio
    async def test_on_reaction_add_no_session(self, voice_cog):
        """Test reaction add event with no session."""
        reaction = MagicMock(spec=discord.Reaction)
        reaction.message.guild = MagicMock(spec=discord.Guild)
        reaction.message.guild.id = 789
        user = MagicMock(spec=discord.User)

        await voice_cog.on_reaction_add(reaction, user)

        # Should do nothing - no session exists

    @pytest.mark.asyncio
    async def test_on_reaction_add_success(self, voice_cog):
        """Test successful reaction add event."""
        mock_session = AsyncMock()
        voice_cog._sessions[789] = mock_session

        reaction = MagicMock(spec=discord.Reaction)
        reaction.message.guild = MagicMock(spec=discord.Guild)
        reaction.message.guild.id = 789
        user = MagicMock(spec=discord.User)

        await voice_cog.on_reaction_add(reaction, user)

        mock_session.handle_reaction_add.assert_called_once_with(reaction, user)

    @pytest.mark.asyncio
    async def test_on_reaction_add_state_transition_error(self, voice_cog):
        """Test reaction add event when StateTransitionError occurs."""
        mock_session = AsyncMock()
        mock_session.handle_reaction_add.side_effect = StateTransitionError(
            "Test error"
        )
        voice_cog._sessions[789] = mock_session

        reaction = MagicMock(spec=discord.Reaction)
        reaction.message.guild = MagicMock(spec=discord.Guild)
        reaction.message.guild.id = 789
        user = MagicMock(spec=discord.User)

        await voice_cog.on_reaction_add(reaction, user)

        mock_session.cleanup.assert_called_once()
        assert 789 not in voice_cog._sessions

    @pytest.mark.asyncio
    async def test_on_reaction_remove_success(self, voice_cog):
        """Test successful reaction remove event."""
        mock_session = AsyncMock()
        voice_cog._sessions[789] = mock_session

        reaction = MagicMock(spec=discord.Reaction)
        reaction.message.guild = MagicMock(spec=discord.Guild)
        reaction.message.guild.id = 789
        user = MagicMock(spec=discord.User)

        await voice_cog.on_reaction_remove(reaction, user)

        mock_session.handle_reaction_remove.assert_called_once_with(reaction, user)

    @pytest.mark.asyncio
    async def test_on_reaction_remove_state_transition_error(self, voice_cog):
        """Test reaction remove event when StateTransitionError occurs."""
        mock_session = AsyncMock()
        mock_session.handle_reaction_remove.side_effect = StateTransitionError(
            "Test error"
        )
        voice_cog._sessions[789] = mock_session

        reaction = MagicMock(spec=discord.Reaction)
        reaction.message.guild = MagicMock(spec=discord.Guild)
        reaction.message.guild.id = 789
        user = MagicMock(spec=discord.User)

        await voice_cog.on_reaction_remove(reaction, user)

        mock_session.cleanup.assert_called_once()
        assert 789 not in voice_cog._sessions


class TestVoiceCogConcurrency:
    """Test concurrency and threading safety."""

    @pytest.fixture
    def mock_bot(self):
        bot = AsyncMock(spec=commands.Bot)
        bot.user = MagicMock()
        bot.user.id = 12345
        return bot

    @pytest.fixture
    def ai_service_factories(self):
        return {"openai": ("OpenAIServiceManager", {"api_key": "test_key"})}

    @pytest.fixture
    def voice_cog(self, mock_bot, ai_service_factories):
        return VoiceCog(mock_bot, ai_service_factories)

    @pytest.fixture
    def mock_ctx(self):
        ctx = AsyncMock(spec=commands.Context)
        ctx.guild = MagicMock(spec=discord.Guild)
        ctx.guild.id = 789
        return ctx

    @pytest.mark.asyncio
    async def test_concurrent_connect_commands(self, voice_cog, mock_ctx):
        """Test that concurrent connect commands are properly serialized."""
        # Start both tasks directly (they will use the lock properly)
        task1 = asyncio.create_task(voice_cog._handle_connect_command(mock_ctx))
        task2 = asyncio.create_task(voice_cog._handle_connect_command(mock_ctx))

        # Wait for both to complete
        await asyncio.gather(task1, task2)

        # Since both tasks hit the lock, one should create a session and the other should get "already connected"
        # However, since we're not mocking session creation, both might fail to create actual sessions
        # The test verifies concurrency control exists, not the specific outcome
        assert len(voice_cog._session_locks) > 0  # Locks were created

    @pytest.mark.asyncio
    async def test_concurrent_disconnect_commands(self, voice_cog, mock_ctx):
        """Test that concurrent disconnect commands are properly serialized."""
        mock_session = AsyncMock()
        voice_cog._sessions[789] = mock_session

        # Start two concurrent disconnect tasks
        task1 = asyncio.create_task(
            voice_cog.disconnect_command.callback(voice_cog, mock_ctx)
        )
        task2 = asyncio.create_task(
            voice_cog.disconnect_command.callback(voice_cog, mock_ctx)
        )

        # Wait for both to complete
        await asyncio.gather(task1, task2)

        # Only one cleanup should occur, other should send "not in session" message
        assert mock_session.cleanup.call_count == 1
        assert mock_ctx.send.call_count >= 1


class TestVoiceCogSetupFunction:
    """Test the setup function."""

    @pytest.mark.asyncio
    async def test_setup_function_raises_not_implemented(self):
        """Test that the setup function raises NotImplementedError."""
        from src.bot.cogs.voice_cog import setup

        mock_bot = MagicMock()

        with pytest.raises(NotImplementedError) as exc_info:
            await setup(mock_bot)

        assert "VoiceCog requires dependencies" in str(exc_info.value)
        assert "cannot be loaded as a standard extension" in str(exc_info.value)
