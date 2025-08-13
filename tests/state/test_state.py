import pytest
from unittest.mock import AsyncMock, MagicMock

from src.bot.state import BotState, BotStateEnum, RecordingMethod


@pytest.fixture
def bot_state() -> BotState:
    """Pytest fixture to provide a fresh BotState instance for each test."""
    return BotState()


@pytest.fixture
def mock_user() -> MagicMock:
    """Pytest fixture for a mock Discord user."""
    user = MagicMock()
    user.id = 123456789
    user.name = "TestUser"
    return user


def test_botstate_initialization(bot_state: BotState):
    """
    Tests that a new BotState object initializes in the correct default state.
    """
    assert bot_state.current_state == BotStateEnum.IDLE
    assert bot_state.authority_user_id == "anyone"
    assert bot_state.active_ai_provider_name is not None  # Check default is set
    assert bot_state.mode is not None  # Check mode is set
    assert bot_state.recording_method is None  # No recording method initially


@pytest.mark.asyncio
async def test_set_state_from_idle_to_standby(bot_state: BotState):
    """
    Tests the valid transition from IDLE to STANDBY using set_state.
    """
    # Act
    success = await bot_state.set_state(BotStateEnum.STANDBY)

    # Assert
    assert success is True
    assert bot_state.current_state == BotStateEnum.STANDBY


@pytest.mark.asyncio
async def test_set_state_when_already_in_state(bot_state: BotState):
    """
    Tests that setting the same state returns False.
    """
    # Arrange - ensure we're in IDLE
    assert bot_state.current_state == BotStateEnum.IDLE
    
    # Act
    success = await bot_state.set_state(BotStateEnum.IDLE)

    # Assert
    assert success is False
    assert bot_state.current_state == BotStateEnum.IDLE


@pytest.mark.asyncio
async def test_start_recording_from_standby(bot_state: BotState, mock_user: MagicMock):
    """
    Tests the transition from STANDBY to RECORDING when starting a recording.
    """
    # Arrange
    await bot_state.set_state(BotStateEnum.STANDBY)
    
    # Act
    await bot_state.start_recording(mock_user, RecordingMethod.PushToTalk)

    # Assert
    assert bot_state.current_state == BotStateEnum.RECORDING
    assert bot_state.authority_user_id == mock_user.id
    assert bot_state.recording_method == RecordingMethod.PushToTalk


@pytest.mark.asyncio
async def test_start_recording_when_not_in_standby(bot_state: BotState, mock_user: MagicMock):
    """
    Tests that start_recording does nothing when not in STANDBY state.
    """
    # Arrange - ensure we're in IDLE
    assert bot_state.current_state == BotStateEnum.IDLE
    
    # Act
    await bot_state.start_recording(mock_user, RecordingMethod.PushToTalk)

    # Assert - state should remain unchanged
    assert bot_state.current_state == BotStateEnum.IDLE
    assert bot_state.authority_user_id == "anyone"
    assert bot_state.recording_method is None


@pytest.mark.asyncio
async def test_stop_recording_from_recording(bot_state: BotState, mock_user: MagicMock):
    """
    Tests the transition from RECORDING to STANDBY when stopping a recording.
    """
    # Arrange
    await bot_state.set_state(BotStateEnum.STANDBY)
    await bot_state.start_recording(mock_user, RecordingMethod.WakeWord)
    assert bot_state.current_state == BotStateEnum.RECORDING
    
    # Act
    await bot_state.stop_recording()

    # Assert
    assert bot_state.current_state == BotStateEnum.STANDBY
    assert bot_state.authority_user_id == "anyone"
    assert bot_state.recording_method is None


@pytest.mark.asyncio
async def test_stop_recording_when_not_recording(bot_state: BotState):
    """
    Tests that stop_recording does nothing when not in RECORDING state.
    """
    # Arrange
    await bot_state.set_state(BotStateEnum.STANDBY)
    
    # Act
    await bot_state.stop_recording()

    # Assert - state should remain unchanged
    assert bot_state.current_state == BotStateEnum.STANDBY


@pytest.mark.asyncio
async def test_reset_to_idle(bot_state: BotState, mock_user: MagicMock):
    """
    Tests resetting the bot state to IDLE from various states.
    """
    # Arrange - put bot in STANDBY with some state
    await bot_state.set_state(BotStateEnum.STANDBY)
    await bot_state.grant_consent(mock_user.id)
    
    # Act
    await bot_state.reset_to_idle()

    # Assert
    assert bot_state.current_state == BotStateEnum.IDLE
    assert bot_state.authority_user_id == "anyone"
    assert len(bot_state.get_consented_user_ids()) == 0


def test_is_authorized(bot_state: BotState, mock_user: MagicMock):
    """
    Tests the authorization check for users.
    """
    # When authority is "anyone"
    assert bot_state.is_authorized(mock_user) is True

    # Set specific authority
    bot_state._set_authority(mock_user)
    assert bot_state.is_authorized(mock_user) is True

    # Different user
    other_user = MagicMock()
    other_user.id = 987654321
    assert bot_state.is_authorized(other_user) is False


@pytest.mark.asyncio
async def test_enter_connection_error_state(bot_state: BotState):
    """
    Tests entering the CONNECTION_ERROR state.
    """
    # Arrange
    await bot_state.set_state(BotStateEnum.STANDBY)
    
    # Act
    result = await bot_state.enter_connection_error_state()

    # Assert
    assert result is True
    assert bot_state.current_state == BotStateEnum.CONNECTION_ERROR
    assert bot_state.authority_user_id == "anyone"


@pytest.mark.asyncio
async def test_enter_connection_error_state_when_already_in_error(bot_state: BotState):
    """
    Tests that entering error state when already in error returns False.
    """

    # Arrange - first get to a valid state that can transition to CONNECTION_ERROR
    await bot_state.set_state(BotStateEnum.STANDBY)
    await bot_state.enter_connection_error_state()
    assert bot_state.current_state == BotStateEnum.CONNECTION_ERROR
    
    # Act
    result = await bot_state.enter_connection_error_state()

    # Assert
    assert result is False
    assert bot_state.current_state == BotStateEnum.CONNECTION_ERROR


@pytest.mark.asyncio
async def test_recover_to_standby(bot_state: BotState):
    """
    Tests recovering from CONNECTION_ERROR to STANDBY.
    """
    # Arrange - first get to a valid state that can transition to CONNECTION_ERROR
    await bot_state.set_state(BotStateEnum.STANDBY)
    await bot_state.enter_connection_error_state()
    assert bot_state.current_state == BotStateEnum.CONNECTION_ERROR
    
    # Act
    result = await bot_state.recover_to_standby()

    # Assert
    assert result is True
    assert bot_state.current_state == BotStateEnum.STANDBY
    assert bot_state.authority_user_id == "anyone"


@pytest.mark.asyncio
async def test_recover_to_standby_when_not_in_error(bot_state: BotState):
    """
    Tests that recovery only works from CONNECTION_ERROR state.
    """
    # Arrange - ensure we're in IDLE
    assert bot_state.current_state == BotStateEnum.IDLE
    
    # Act
    result = await bot_state.recover_to_standby()

    # Assert
    assert result is False
    assert bot_state.current_state == BotStateEnum.IDLE


@pytest.mark.asyncio
async def test_set_active_ai_provider_name(bot_state: BotState):
    """
    Tests setting the active AI provider name.
    """
    # Arrange
    initial_provider = bot_state.active_ai_provider_name
    new_provider = "gemini"
    
    # Act
    await bot_state.set_active_ai_provider_name(new_provider)

    # Assert
    assert bot_state.active_ai_provider_name == new_provider
    assert bot_state.active_ai_provider_name != initial_provider


@pytest.mark.asyncio
async def test_consent_management(bot_state: BotState):
    """
    Tests granting and revoking consent for users.
    """
    user_id = 123456789
    
    # Initially no consent
    assert user_id not in bot_state.get_consented_user_ids()
    
    # Grant consent
    await bot_state.grant_consent(user_id)
    assert user_id in bot_state.get_consented_user_ids()
    
    # Revoke consent
    await bot_state.revoke_consent(user_id)
    assert user_id not in bot_state.get_consented_user_ids()