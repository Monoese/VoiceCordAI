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
from enum import Enum
from typing import Optional

import discord
from src.config.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
        self._authority_user_id: Optional[str] = "anyone"  # User ID or "anyone"
        self._authority_user_name: Optional[str] = "anyone"  # User display name
        self._active_ai_provider_name: str = (
            Config.AI_SERVICE_PROVIDER
        )  # Initialize with default

    @property
    def current_state(self) -> BotStateEnum:
        """
        Get the current state of the bot.

        Returns:
            BotStateEnum: The current state (IDLE, STANDBY, RECORDING, or CONNECTION_ERROR)
        """
        return self._current_state

    @property
    def authority_user_id(self) -> str:
        """
        Get the ID of the user who currently has authority to control the bot.

        Returns:
            str: The user ID or "anyone" if no specific user has authority
        """
        return self._authority_user_id

    @property
    def active_ai_provider_name(self) -> str:
        """
        Get the name of the currently active AI service provider.
        """
        return self._active_ai_provider_name

    def get_authority_user_name(self) -> str:
        """
        Get the display name of the user who currently has authority.
        """
        return self._authority_user_name

    async def set_active_ai_provider_name(self, provider_name: str) -> None:
        """
        Set the name of the active AI service provider.

        Args:
            provider_name: The name of the provider (e.g., "openai", "gemini").
        """
        async with self._lock:
            self._active_ai_provider_name = provider_name

    def _reset_authority(self) -> None:
        """Atomically resets the authority user to 'anyone'."""
        self._authority_user_id = "anyone"
        self._authority_user_name = "anyone"

    def _set_authority(self, user: discord.User) -> None:
        """Atomically sets the authority user's ID and name."""
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

            self._current_state = BotStateEnum.STANDBY
            return True

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

            self._current_state = BotStateEnum.RECORDING
            self._set_authority(
                user
            )  # Assign control to the user who started recording

            return True

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

            self._current_state = BotStateEnum.STANDBY
            self._reset_authority()  # Release specific user control

            return True

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

            self._current_state = BotStateEnum.IDLE
            self._reset_authority()  # Reset authority
            # No UI message to update in IDLE state.
        return True

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

            self._current_state = BotStateEnum.CONNECTION_ERROR
            self._reset_authority()  # Reset authority

            return True

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

            self._current_state = BotStateEnum.STANDBY
            self._reset_authority()
            return True
