"""
Bot state management module for tracking and controlling the Discord bot's state.

This module provides:
- A state enumeration (BotStateEnum) defining the possible states of the bot
- A state manager (BotState) that handles state transitions and permissions
- Methods for initializing, updating, and resetting the bot's state
- User authority tracking to control who can interact with the bot

The state system ensures the bot operates in a predictable manner and
prevents conflicting operations from occurring simultaneously.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Callable, Awaitable, Union, Set, Optional

import discord
from src.config.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StateChangedEvent:
    """Event fired when the bot's state changes."""

    old_state: "BotStateEnum"
    new_state: "BotStateEnum"


@dataclass
class ProviderChangedEvent:
    """Event fired when the AI provider changes."""

    new_provider_name: str


class BotModeEnum(Enum):
    """Enumeration of the bot's high-level operational modes."""

    RealtimeTalk = "realtime_talk"
    ManualControl = "manual_control"


@dataclass
class ModeChangedEvent:
    """Event fired when the bot's mode changes."""

    new_mode: "BotModeEnum"


class RecordingMethod(Enum):
    """Enumeration of methods for starting a recording."""

    PushToTalk = auto()
    WakeWord = auto()


StateEvent = Union[StateChangedEvent, ProviderChangedEvent, ModeChangedEvent]


class BotStateEnum(Enum):
    """
    Enumeration of possible operational states for the Discord bot.

    Each state represents a distinct mode of operation, influencing
    how the bot responds to commands and events.
    """

    IDLE = "idle"
    # --- ManualControl States ---
    STANDBY = "standby"
    RECORDING = "recording"
    # --- RealtimeTalk States ---
    LISTENING = "listening"
    SPEAKING = "speaking"
    # --- Shared State ---
    CONNECTION_ERROR = "connection_error"


class BotState:
    """
    Manages the state and permissions of the Discord bot for a single guild session.

    This class acts as the single source of truth for all session-related status,
    including the operational mode, the current state, and user consent. It handles
    state transitions and notifies subscribed listeners of any changes. This class
    is not thread-safe and relies on an external lock (e.g., in GuildSession)
    for safe concurrent access.
    """

    def __init__(self) -> None:
        """
        Initialize the BotState with default values.

        The bot starts in IDLE state with no specific authority user.
        """
        # The high-level operational mode (e.g., ManualControl, RealtimeTalk).
        self._mode: BotModeEnum = BotModeEnum.ManualControl
        # The specific state within the current mode (e.g., STANDBY, RECORDING).
        self._current_state: BotStateEnum = BotStateEnum.IDLE
        # How the current recording was initiated (e.g., WakeWord, PushToTalk).
        self._recording_method: Optional[RecordingMethod] = None
        # The user who is allowed to provide audio or stop the recording.
        self._authority_user_id: Union[str, int] = "anyone"
        self._authority_user_name: str = "anyone"
        # The name of the AI service provider currently in use (e.g., "openai").
        self._active_ai_provider_name: str = Config.AI_SERVICE_PROVIDER
        # A list of async callbacks to be notified of state changes.
        self._listeners: List[Callable[[StateEvent], Awaitable[None]]] = []
        # A set of user IDs who have granted consent to be recorded.
        self._consented_user_ids: Set[int] = set()
        # Session tracking to prevent cross-session state corruption
        self._current_session_id: int = 0

    @property
    def current_state(self) -> BotStateEnum:
        """
        Get the current state of the bot.

        Returns:
            BotStateEnum: The current state (IDLE, STANDBY, RECORDING, or CONNECTION_ERROR)
        """
        return self._current_state

    @property
    def authority_user_id(self) -> Union[str, int]:
        """
        Get the ID of the user who currently has authority to control the bot.

        Returns:
            Union[str, int]: The user ID (int) or "anyone" (str).
        """
        return self._authority_user_id

    @property
    def recording_method(self) -> Optional[RecordingMethod]:
        """Get the method used to start the current recording, if any."""
        return self._recording_method

    @property
    def active_ai_provider_name(self) -> str:
        """Get the name of the currently active AI service provider.

        Returns:
            str: The name of the active AI provider.
        """
        return self._active_ai_provider_name

    def get_authority_user_name(self) -> str:
        """Get the display name of the user who currently has authority.

        Returns:
            str: The display name of the authority user.
        """
        return self._authority_user_name

    @property
    def mode(self) -> BotModeEnum:
        """Get the current operational mode of the bot."""
        return self._mode

    def get_consented_user_ids(self) -> Set[int]:
        """Get a copy of the set of user IDs who have consented to be recorded."""
        return self._consented_user_ids.copy()

    @property
    def current_session_id(self) -> int:
        """Get the current session ID for tracking cross-session state corruption."""
        return self._current_session_id

    def subscribe_to_state_changes(
        self, callback: Callable[[StateEvent], Awaitable[None]]
    ) -> None:
        """Subscribe a listener to be called on state changes."""
        self._listeners.append(callback)

    async def _notify_listeners(self, event: StateEvent) -> None:
        """Notifies all listeners of a given event without blocking."""
        # Listeners are called as background tasks to prevent a slow listener
        # from blocking the state transition.
        for listener in self._listeners:
            asyncio.create_task(listener(event))

    async def _set_state(self, new_state: BotStateEnum) -> bool:
        """Atomically sets a new state and notifies listeners."""
        # Import here to avoid circular imports
        from .state_validator import StateTransitionValidator

        # Enforce state transition validation
        StateTransitionValidator.validate(self._current_state, new_state)

        if self._current_state == new_state:
            return False

        old_state = self._current_state
        self._current_state = new_state

        # Log successful state transition for debugging
        logger.debug(f"State transitioned from {old_state.value} to {new_state.value}")

        event = StateChangedEvent(old_state=old_state, new_state=new_state)
        await self._notify_listeners(event)

        return True

    async def grant_consent(self, user_id: int) -> None:
        """Adds a user's ID to the consent list."""
        self._consented_user_ids.add(user_id)

    async def revoke_consent(self, user_id: int) -> None:
        """Removes a user's ID from the consent list."""
        self._consented_user_ids.discard(user_id)

    async def set_active_ai_provider_name(self, provider_name: str) -> None:
        """
        Set the name of the active AI service provider.

        Args:
            provider_name: The name of the provider (e.g., "openai", "gemini").
        """
        if self._active_ai_provider_name == provider_name:
            return
        self._active_ai_provider_name = provider_name
        event = ProviderChangedEvent(new_provider_name=provider_name)
        await self._notify_listeners(event)

    async def set_mode(self, new_mode: BotModeEnum) -> None:
        """
        Set the operational mode of the bot.

        This should be called by GuildSession during initialization or mode switching.
        Note: This does not change the bot's state (e.g., STANDBY, RECORDING),
        which must be managed separately.

        Args:
            new_mode: The new operational mode to set.
        """
        if self._mode == new_mode:
            return
        self._mode = new_mode
        event = ModeChangedEvent(new_mode=new_mode)
        await self._notify_listeners(event)

    async def set_state(self, new_state: BotStateEnum) -> bool:
        """
        Sets a new state directly.

        This is used by GuildSession for controlled state transitions
        that do not have a more specific method (e.g., 'start_recording').

        Args:
            new_state: The state to transition to.

        Returns:
            True if the state was changed, False otherwise.
        """
        return await self._set_state(new_state)

    def _reset_authority(self) -> None:
        """
        Resets the authority user to 'anyone' and clears the recording method.

        This is an internal helper method used when a recording session ends or
        the bot state is reset.
        """
        self._authority_user_id = "anyone"
        self._authority_user_name = "anyone"
        self._recording_method = None

    def _set_authority(self, user: discord.User) -> None:
        """
        Sets the authority to a specific user.

        This is an internal helper method used when a recording session begins.
        """
        self._authority_user_id = user.id
        self._authority_user_name = user.name

    async def start_recording(
        self, user: discord.User, method: RecordingMethod
    ) -> None:
        """
        Transition from standby to recording state.

        This method:
        1. Transitions the bot from STANDBY to RECORDING state
        2. Sets the authority user to the one who started recording
        3. Records the method used to start the recording (PTT or Wake Word)
        4. Increments session ID to track recording sessions

        Args:
            user: The Discord user who is starting the recording.
            method: The method used to start the recording.
        """
        # Guard clause: only allow starting a recording from the STANDBY state.
        if self._current_state != BotStateEnum.STANDBY:
            return

        await self._set_state(BotStateEnum.RECORDING)
        self._set_authority(user)  # Assign control to the user who started recording
        self._recording_method = method
        self._current_session_id += 1  # Increment session ID for cross-session tracking

    async def stop_recording(self) -> None:
        """
        Stop recording and return to standby state.

        This method:
        1. Transitions the bot from RECORDING back to STANDBY state
        2. Resets the authority user to "anyone"
        """
        if self._current_state != BotStateEnum.RECORDING:
            # Only allow transition from RECORDING state.
            return

        await self._set_state(BotStateEnum.STANDBY)
        self._reset_authority()  # Release specific user control

    async def reset_to_idle(self) -> None:
        """
        Reset the bot state to idle.

        This method:
        1. Transitions the bot from any state back to IDLE state
        2. Resets the authority user to "anyone"
        """
        if self._current_state == BotStateEnum.IDLE:
            # Already idle, no action needed.
            return

        self._consented_user_ids.clear()
        await self._set_state(BotStateEnum.IDLE)
        self._reset_authority()  # Reset authority

    def is_authorized(self, user: discord.User) -> bool:
        """
        Check if the user is authorized to control the bot.

        This method determines whether a user has permission to control the bot,
        based on the current authority settings. A user is authorized if:
        - The authority is set to "anyone" (default in STANDBY state)
        - The user's ID matches the current authority user ID (in RECORDING or DEBUG_RECORDING state)

        Args:
            user: The Discord user to check for authorization

        Returns:
            bool: True if the user is authorized, False otherwise
        """
        return self.authority_user_id == "anyone" or user.id == self.authority_user_id

    async def enter_connection_error_state(self) -> bool:
        """
        Transition the bot to the CONNECTION_ERROR state.

        This method:
        1. Transitions the bot to CONNECTION_ERROR state.
        2. Resets authority user.

        Returns:
            bool: True if the state was changed, False if already in error state.
        """
        if self._current_state == BotStateEnum.CONNECTION_ERROR:
            return False  # Already in the error state

        await self._set_state(BotStateEnum.CONNECTION_ERROR)
        self._reset_authority()  # Reset authority
        return True

    async def recover_to_standby(self) -> bool:
        """
        Attempt to recover from CONNECTION_ERROR state back to STANDBY.

        This method should be called when external connections are confirmed to be restored.
        It transitions the state to STANDBY.

        Returns:
            bool: True if recovery was successful, False if not in error state.
        """
        if self._current_state != BotStateEnum.CONNECTION_ERROR:
            return False  # Can only recover from connection error state

        await self._set_state(BotStateEnum.STANDBY)
        self._reset_authority()
        return True
