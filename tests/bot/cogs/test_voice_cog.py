"""
Unit tests for the VoiceCog class.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from discord.ext import commands

from src.bot.cogs.voice_cog import VoiceCog
from src.config.config import Config


# Mocks and Fixtures
@pytest.fixture
def mock_bot() -> MagicMock:
    """Fixture for a mocked discord.ext.commands.Bot."""
    bot = MagicMock(spec=commands.Bot)
    bot.user = MagicMock()
    bot.user.name = "TestBot"
    return bot


@pytest.fixture
def mock_audio_manager() -> MagicMock:
    """Fixture for a mocked AudioManager."""
    return MagicMock()


@pytest.fixture
def mock_bot_state_manager() -> MagicMock:
    """Fixture for a mocked BotState."""
    mock = MagicMock()
    # Use AsyncMock for async methods
    mock.initialize_standby = AsyncMock(return_value=True)
    mock.enter_connection_error_state = AsyncMock()
    # Default state
    mock.active_ai_provider_name = "openai"
    return mock


@pytest.fixture
def mock_openai_service_manager() -> MagicMock:
    """Fixture for a mocked OpenAI IRealtimeAIServiceManager."""
    manager = MagicMock()
    manager.is_connected = MagicMock(return_value=False)
    manager.connect = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def mock_gemini_service_manager() -> MagicMock:
    """Fixture for a mocked Gemini IRealtimeAIServiceManager."""
    manager = MagicMock()
    manager.is_connected = MagicMock(return_value=False)
    manager.connect = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def mock_ai_service_managers(
    mock_openai_service_manager, mock_gemini_service_manager
) -> dict:
    """Fixture for the dictionary of AI service managers."""
    return {
        "openai": mock_openai_service_manager,
        "gemini": mock_gemini_service_manager,
    }


@pytest.fixture
def mock_voice_connection_manager() -> MagicMock:
    """Fixture for a mocked VoiceConnectionManager."""
    mock = MagicMock()
    mock.connect_to_channel = AsyncMock(return_value=True)
    return mock


@pytest.fixture
@patch("discord.ext.tasks.Loop.start")
@patch("src.bot.cogs.voice_cog.VoiceConnectionManager")
def voice_cog(
    mock_vcm_class: MagicMock,
    mock_loop_start: MagicMock,
    mock_bot: MagicMock,
    mock_audio_manager: MagicMock,
    mock_bot_state_manager: MagicMock,
    mock_ai_service_managers: dict,
    mock_voice_connection_manager: MagicMock,
) -> VoiceCog:
    """Fixture for an initialized VoiceCog instance with mocked dependencies."""
    # Patch the config to use a known default provider
    with patch.object(Config, "AI_SERVICE_PROVIDER", "openai"):
        # Mock the instance returned by the VoiceConnectionManager constructor
        mock_vcm_class.return_value = mock_voice_connection_manager

        cog = VoiceCog(
            bot=mock_bot,
            audio_manager=mock_audio_manager,
            bot_state_manager=mock_bot_state_manager,
            ai_service_managers=mock_ai_service_managers,
        )
        return cog


@pytest.fixture
def mock_ctx() -> MagicMock:
    """Fixture for a mocked command context."""
    ctx = MagicMock()
    ctx.author = MagicMock()
    ctx.author.voice = MagicMock()
    ctx.author.voice.channel = MagicMock()
    ctx.send = AsyncMock()
    return ctx


# Test Cases
def test_voice_cog_initialization(
    voice_cog: VoiceCog,
    mock_bot: MagicMock,
    mock_audio_manager: MagicMock,
    mock_bot_state_manager: MagicMock,
    mock_ai_service_managers: dict,
):
    """Tests that the VoiceCog initializes correctly."""
    assert voice_cog.bot is mock_bot
    assert voice_cog.audio_manager is mock_audio_manager
    assert voice_cog.bot_state_manager is mock_bot_state_manager
    assert voice_cog.all_ai_service_managers is mock_ai_service_managers
    # Check if the active manager is set to the default from the patched config
    assert (
        voice_cog.active_ai_service_manager is mock_ai_service_managers["openai"]
    ), "Default AI provider was not set correctly on initialization."
    assert voice_cog.voice_connection is not None


def test_voice_cog_initialization_invalid_provider():
    """Tests that VoiceCog raises ValueError for an invalid default AI provider."""
    with patch.object(Config, "AI_SERVICE_PROVIDER", "invalid_provider"):
        with pytest.raises(ValueError, match="Default AI provider 'invalid_provider' not found"):
            VoiceCog(
                bot=MagicMock(),
                audio_manager=MagicMock(),
                bot_state_manager=MagicMock(),
                ai_service_managers={"openai": MagicMock()},
            )


@pytest.mark.asyncio
async def test_connect_command_success(voice_cog: VoiceCog, mock_ctx: MagicMock):
    """Tests the successful execution of the connect command."""
    # Act
    await voice_cog.connect_command.callback(voice_cog, mock_ctx)

    # Assert
    # 1. Connect to voice channel
    voice_cog.voice_connection.connect_to_channel.assert_awaited_once_with(
        mock_ctx.author.voice.channel
    )
    # 2. Connect to AI service
    voice_cog.active_ai_service_manager.connect.assert_awaited_once()
    # 3. Initialize standby state
    voice_cog.bot_state_manager.initialize_standby.assert_awaited_once_with(mock_ctx)


@pytest.mark.asyncio
async def test_connect_command_user_not_in_voice(voice_cog: VoiceCog, mock_ctx: MagicMock):
    """Tests connect command failure when the user is not in a voice channel."""
    # Arrange
    mock_ctx.author.voice = None

    # Act
    await voice_cog.connect_command.callback(voice_cog, mock_ctx)

    # Assert
    mock_ctx.send.assert_awaited_once_with("You are not connected to a voice channel.")
    voice_cog.voice_connection.connect_to_channel.assert_not_called()
    voice_cog.active_ai_service_manager.connect.assert_not_called()


@pytest.mark.asyncio
async def test_connect_command_voice_connection_fails(
    voice_cog: VoiceCog, mock_ctx: MagicMock
):
    """Tests connect command failure when the voice connection fails."""
    # Arrange
    voice_cog.voice_connection.connect_to_channel.return_value = False

    # Act
    await voice_cog.connect_command.callback(voice_cog, mock_ctx)

    # Assert
    mock_ctx.send.assert_awaited_once_with("Failed to connect to the voice channel.")
    voice_cog.bot_state_manager.enter_connection_error_state.assert_awaited_once()
    voice_cog.active_ai_service_manager.connect.assert_not_called()


@pytest.mark.asyncio
async def test_connect_command_ai_connection_fails(
    voice_cog: VoiceCog, mock_ctx: MagicMock
):
    """Tests connect command failure when the AI service connection fails."""
    # Arrange
    voice_cog.active_ai_service_manager.connect.return_value = False

    # Act
    await voice_cog.connect_command.callback(voice_cog, mock_ctx)

    # Assert
    mock_ctx.send.assert_awaited_once_with("Failed to establish AI service connection.")
    voice_cog.bot_state_manager.enter_connection_error_state.assert_awaited_once()
    voice_cog.bot_state_manager.initialize_standby.assert_not_called()
