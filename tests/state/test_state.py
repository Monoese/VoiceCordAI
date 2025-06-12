import pytest
from unittest.mock import AsyncMock, MagicMock

from src.state.state import BotState, BotStateEnum


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


@pytest.fixture
def mock_ctx() -> MagicMock:
    """Pytest fixture for a mock Discord context with an async `send` method."""
    ctx = MagicMock()
    # The message returned by send() needs an add_reaction method
    mock_message = MagicMock()
    mock_message.add_reaction = AsyncMock()
    ctx.send = AsyncMock(return_value=mock_message)
    return ctx


def test_botstate_initialization(bot_state: BotState):
    """
    Tests that a new BotState object initializes in the correct default state.
    """
    # Assert
    assert bot_state.current_state == BotStateEnum.IDLE
    assert bot_state.authority_user_id == "anyone"
    assert bot_state.standby_message is None
    assert bot_state.active_ai_provider_name is not None  # Check default is set


@pytest.mark.asyncio
async def test_initialize_standby_from_idle(bot_state: BotState, mock_ctx: MagicMock):
    """
    Tests the valid transition from IDLE to STANDBY.
    """
    # Act
    success = await bot_state.initialize_standby(mock_ctx)

    # Assert
    assert success is True
    assert bot_state.current_state == BotStateEnum.STANDBY
    mock_ctx.send.assert_called_once()
    assert bot_state.standby_message is not None
    # Check that the reaction was added to the message that ctx.send returned
    mock_ctx.send.return_value.add_reaction.assert_called_once_with("🎙")


@pytest.mark.asyncio
async def test_initialize_standby_when_not_idle(
    bot_state: BotState, mock_ctx: MagicMock
):
    """
    Tests that standby cannot be initialized from a state other than IDLE.
    """
    # Arrange
    bot_state._current_state = BotStateEnum.RECORDING

    # Act
    success = await bot_state.initialize_standby(mock_ctx)

    # Assert
    assert success is False
    assert bot_state.current_state == BotStateEnum.RECORDING
    mock_ctx.send.assert_not_called()


@pytest.mark.asyncio
async def test_start_recording_from_standby(bot_state: BotState, mock_user: MagicMock):
    """
    Tests the valid transition from STANDBY to RECORDING.
    """
    # Arrange: Manually put the bot into STANDBY state with a mock message
    mock_message = MagicMock()
    mock_message.edit = AsyncMock()
    bot_state._standby_message = mock_message
    bot_state._current_state = BotStateEnum.STANDBY

    # Act
    success = await bot_state.start_recording(mock_user)

    # Assert
    assert success is True
    assert bot_state.current_state == BotStateEnum.RECORDING
    assert bot_state.authority_user_id == mock_user.id
    assert bot_state._authority_user_name == mock_user.name
    mock_message.edit.assert_called_once()


@pytest.mark.asyncio
async def test_start_recording_when_not_in_standby(
    bot_state: BotState, mock_user: MagicMock
):
    """
    Tests that recording cannot be started from a state other than STANDBY.
    """
    # Arrange
    bot_state._current_state = BotStateEnum.IDLE

    # Act
    success = await bot_state.start_recording(mock_user)

    # Assert
    assert success is False
    assert bot_state.current_state == BotStateEnum.IDLE
    assert bot_state.authority_user_id == "anyone"


@pytest.mark.asyncio
async def test_stop_recording_from_recording(bot_state: BotState):
    """
    Tests the valid transition from RECORDING back to STANDBY.
    """
    # Arrange: Manually put the bot into RECORDING state
    mock_message = MagicMock()
    mock_message.edit = AsyncMock()
    bot_state._standby_message = mock_message
    bot_state._current_state = BotStateEnum.RECORDING
    bot_state.authority_user_id = 12345
    bot_state._authority_user_name = "TestUser"

    # Act
    success = await bot_state.stop_recording()

    # Assert
    assert success is True
    assert bot_state.current_state == BotStateEnum.STANDBY
    assert bot_state.authority_user_id == "anyone"
    mock_message.edit.assert_called_once()


@pytest.mark.asyncio
async def test_stop_recording_when_not_recording(bot_state: BotState):
    """
    Tests that recording cannot be stopped if not in RECORDING state.
    """
    # Arrange
    bot_state._current_state = BotStateEnum.STANDBY

    # Act
    success = await bot_state.stop_recording()

    # Assert
    assert success is False
    assert bot_state.current_state == BotStateEnum.STANDBY


@pytest.mark.asyncio
async def test_reset_to_idle(bot_state: BotState):
    """
    Tests resetting the state to IDLE from another state.
    """
    # Arrange: Manually put into a non-idle state with a message
    mock_message = MagicMock()
    mock_message.delete = AsyncMock()
    bot_state._standby_message = mock_message
    bot_state._current_state = BotStateEnum.RECORDING
    bot_state.authority_user_id = 12345

    # Act
    success = await bot_state.reset_to_idle()

    # Assert
    assert success is True
    assert bot_state.current_state == BotStateEnum.IDLE
    assert bot_state.authority_user_id == "anyone"
    assert bot_state.standby_message is None
    mock_message.delete.assert_called_once()


def test_is_authorized(bot_state: BotState, mock_user: MagicMock):
    """
    Tests the authorization logic.
    """
    # 1. Test when anyone is authorized
    bot_state._authority_user_id = "anyone"
    assert bot_state.is_authorized(mock_user) is True

    # 2. Test when a specific user is authorized and is the one asking
    bot_state.authority_user_id = mock_user.id
    assert bot_state.is_authorized(mock_user) is True

    # 3. Test when a specific user is authorized and a different user is asking
    another_user = MagicMock()
    another_user.id = 987654321
    assert bot_state.is_authorized(another_user) is False


@pytest.mark.asyncio
async def test_enter_connection_error_state(bot_state: BotState):
    """
    Tests transitioning to the CONNECTION_ERROR state.
    """
    # Arrange
    mock_message = MagicMock()
    mock_message.edit = AsyncMock()
    bot_state._standby_message = mock_message
    bot_state._current_state = BotStateEnum.STANDBY
    bot_state.authority_user_id = 12345

    # Act
    success = await bot_state.enter_connection_error_state()

    # Assert
    assert success is True
    assert bot_state.current_state == BotStateEnum.CONNECTION_ERROR
    assert bot_state.authority_user_id == "anyone"
    mock_message.edit.assert_called_once()


@pytest.mark.asyncio
async def test_recover_to_standby(bot_state: BotState):
    """
    Tests recovering from CONNECTION_ERROR back to STANDBY.
    """
    # Arrange
    mock_message = MagicMock()
    mock_message.edit = AsyncMock()
    bot_state._standby_message = mock_message
    bot_state._current_state = BotStateEnum.CONNECTION_ERROR

    # Act
    success = await bot_state.recover_to_standby()

    # Assert
    assert success is True
    assert bot_state.current_state == BotStateEnum.STANDBY
    assert bot_state.authority_user_id == "anyone"
    mock_message.edit.assert_called_once()


@pytest.mark.asyncio
async def test_set_active_ai_provider_name(bot_state: BotState):
    """
    Tests that setting the AI provider name updates the state and the UI message.
    """
    # Arrange: Manually put the bot into STANDBY state with a mock message
    mock_message = MagicMock()
    mock_message.edit = AsyncMock()
    bot_state._standby_message = mock_message
    bot_state._current_state = BotStateEnum.STANDBY
    new_provider_name = "new_provider"

    # Act
    await bot_state.set_active_ai_provider_name(new_provider_name)

    # Assert
    assert bot_state.active_ai_provider_name == new_provider_name
    # Verify that the UI was updated to reflect the change
    mock_message.edit.assert_called_once()
