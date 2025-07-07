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
from enum import Enum
from typing import List, Callable, Awaitable, Union, Set

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


StateEvent = Union[StateChangedEvent, ProviderChangedEvent]


class BotStateEnum(Enum):
    """
    Enumeration of possible operational states for the Discord bot.

    Each state represents a distinct mode of operation, influencing
    how the bot responds to commands and events.

    - IDLE: Bot is not actively listening or processing voice.
    - STANDBY: Bot is ready to start recording when triggered.
    - RECORDING: Bot is actively recording audio from a user.
    - CONNECTION_ERROR: Bot has encountered a connection issue and may not be fully functional.
    """

    IDLE = "idle"
    STANDBY = "standby"
    RECORDING = "recording"
    CONNECTION_ERROR = "connection_error"


class BotState:
    """
    Manages the state and permissions of the Discord bot.

    This class:
    - Tracks the current state of the bot (idle, standby, recording, connection_error)
    - Manages state transitions with appropriate validation
    - Tracks which user has authority to control the bot
    """

    def __init__(self) -> None:
        """
        Initialize the BotState with default values.

        The bot starts in IDLE state with no specific authority user.
        """
        self._lock = asyncio.Lock()
        self._current_state: BotStateEnum = BotStateEnum.IDLE
        self._authority_user_id: Union[str, int] = "anyone"
        self._authority_user_name: str = "anyone"
        self._active_ai_provider_name: str = Config.AI_SERVICE_PROVIDER
        self._listeners: List[Callable[[StateEvent], Awaitable[None]]] = []
        self._consented_user_ids: Set[int] = set()

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

    def get_consented_user_ids(self) -> Set[int]:
        """Get a copy of the set of user IDs who have consented to be recorded."""
        return self._consented_user_ids.copy()

    def subscribe_to_state_changes(
        self, callback: Callable[[StateEvent], Awaitable[None]]
    ) -> None:
        """Subscribe a listener to be called on state changes."""
        self._listeners.append(callback)

    async def _notify_listeners(self, event: StateEvent) -> None:
        """Notifies all listeners of a given event."""
        for listener in self._listeners:
            asyncio.create_task(listener(event))

    async def _set_state(self, new_state: BotStateEnum) -> bool:
        """Atomically sets a new state and notifies listeners."""
        if self._current_state == new_state:
            return False

        old_state = self._current_state
        self._current_state = new_state

        event = StateChangedEvent(old_state=old_state, new_state=new_state)
        await self._notify_listeners(event)

        return True

    async def grant_consent(self, user_id: int) -> None:
        """Adds a user's ID to the consent list."""
        async with self._lock:
            self._consented_user_ids.add(user_id)

    async def revoke_consent(self, user_id: int) -> None:
        """Removes a user's ID from the consent list."""
        async with self._lock:
            self._consented_user_ids.discard(user_id)

    async def set_active_ai_provider_name(self, provider_name: str) -> None:
        """
        Set the name of the active AI service provider.

        Args:
            provider_name: The name of the provider (e.g., "openai", "gemini").
        """
        async with self._lock:
            if self._active_ai_provider_name == provider_name:
                return
            self._active_ai_provider_name = provider_name
            event = ProviderChangedEvent(new_provider_name=provider_name)
            await self._notify_listeners(event)

    def _reset_authority(self) -> None:
        """
        Resets the authority user to 'anyone'.

        This is a helper method and must be called from within a locked context.
        """
        self._authority_user_id = "anyone"
        self._authority_user_name = "anyone"

    def _set_authority(self, user: discord.User) -> None:
        """
        Sets the authority user's ID and name.

        This is a helper method and must be called from within a locked context.
        """
        self._authority_user_id = user.id
        self._authority_user_name = user.name

    async def transition_to_standby(self) -> bool:
        """
        Transition from IDLE to STANDBY state.

        Returns:
            bool: True if transition was successful, False if bot was not in IDLE state.
        """
        async with self._lock:
            if self._current_state != BotStateEnum.IDLE:
                # Only allow transition from IDLE state to prevent unexpected state changes.
                return False
            self._consented_user_ids.clear()
            return await self._set_state(BotStateEnum.STANDBY)

    async def start_recording(self, user: discord.User) -> bool:
        """
        Transition from standby to recording state.

        This method:
        1. Transitions the bot from STANDBY to RECORDING state
        2. Sets the authority user to the one who started recording

        Args:
            user: The Discord user who is starting the recording

        Returns:
            bool: True if transition was successful, False if bot was not in STANDBY state
        """
        async with self._lock:
            if self._current_state != BotStateEnum.STANDBY:
                # Only allow transition from STANDBY state.
                return False

            if await self._set_state(BotStateEnum.RECORDING):
                self._set_authority(
                    user
                )  # Assign control to the user who started recording
                return True
            return False

    async def stop_recording(self) -> bool:
        """
        Stop recording and return to standby state.

        This method:
        1. Transitions the bot from RECORDING back to STANDBY state
        2. Resets the authority user to "anyone"

        Returns:
            bool: True if transition was successful, False if bot was not in RECORDING state
        """
        async with self._lock:
            if self._current_state != BotStateEnum.RECORDING:
                # Only allow transition from RECORDING state.
                return False

            if await self._set_state(BotStateEnum.STANDBY):
                self._reset_authority()  # Release specific user control
                return True
            return False

    async def reset_to_idle(self) -> bool:
        """
        Reset the bot state to idle.

        This method:
        1. Transitions the bot from any state back to IDLE state
        2. Resets the authority user to "anyone"

        Returns:
            bool: True if transition was successful, False if bot was already in IDLE state
        """
        async with self._lock:
            if self._current_state == BotStateEnum.IDLE:
                # Already idle, no action needed.
                return False

            self._consented_user_ids.clear()
            if await self._set_state(BotStateEnum.IDLE):
                self._reset_authority()  # Reset authority
                return True
            return False

    def is_authorized(self, user: discord.User) -> bool:
        """
        Check if the user is authorized to control the bot.

        This method determines whether a user has permission to control the bot,
        based on the current authority settings. A user is authorized if:
        - The authority is set to "anyone" (default in STANDBY state)
        - The user's ID matches the current authority user ID (in RECORDING state)

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
            bool: True if transition was successful, False if bot was already in CONNECTION_ERROR state.
        """
        async with self._lock:
            if self._current_state == BotStateEnum.CONNECTION_ERROR:
                return False  # Already in the error state

            if await self._set_state(BotStateEnum.CONNECTION_ERROR):
                self._reset_authority()  # Reset authority
                return True
            return False

    async def recover_to_standby(self) -> bool:
        """
        Attempt to recover from CONNECTION_ERROR state back to STANDBY.

        This method should be called when external connections are confirmed to be restored.
        It transitions the state to STANDBY.

        Returns:
            bool: True if recovery to STANDBY was successful, False if not in CONNECTION_ERROR state.
        """
        async with self._lock:
            if self._current_state != BotStateEnum.CONNECTION_ERROR:
                return False  # Can only recover from connection error state

            if await self._set_state(BotStateEnum.STANDBY):
                self._reset_authority()
                return True
            return False
