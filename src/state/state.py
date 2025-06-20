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
from discord import Message
from src.config.config import Config


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
    - Maintains the standby message for reaction-based controls
    - Provides methods to update the UI based on state changes
    """

    def __init__(self):
        """
        Initialize the BotState with default values.

        The bot starts in IDLE state with no specific authority user
        and no standby message.
        """
        self._lock = asyncio.Lock()
        self._current_state: BotStateEnum = BotStateEnum.IDLE
        self._authority_user_id: Optional[str] = "anyone"  # User ID or "anyone"
        self._authority_user_name: Optional[str] = "anyone"  # User display name
        self._standby_message: Optional[Message] = (
            None  # The Discord message used for UI
        )
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
    def standby_message(self) -> Optional[Message]:
        """
        Get the standby message that displays the bot's status and controls.

        Returns:
            Optional[Message]: The Discord message object or None if not in standby
        """
        return self._standby_message

    @property
    def active_ai_provider_name(self) -> str:
        """
        Get the name of the currently active AI service provider.
        """
        return self._active_ai_provider_name

    async def set_active_ai_provider_name(self, provider_name: str) -> None:
        """
        Set the name of the active AI service provider and update the standby message.
        Args:
            provider_name: The name of the provider (e.g., "openai", "gemini").
        """
        async with self._lock:
            self._active_ai_provider_name = provider_name
            await self._update_message()  # Update UI to reflect the change

    def _reset_authority(self):
        """Atomically resets the authority user to 'anyone'."""
        self._authority_user_id = "anyone"
        self._authority_user_name = "anyone"

    def _set_authority(self, user: discord.User):
        """Atomically sets the authority user's ID and name."""
        self._authority_user_id = user.id
        self._authority_user_name = user.name

    def get_message_content(self) -> str:
        """
        Generate the standby message content based on the current state.

        This method creates a formatted message that displays:
        - The current bot mode (IDLE, STANDBY, RECORDING, or CONNECTION_ERROR)
        - Instructions for using the bot's reaction controls
        - The current recording status
        - Which user has authority to control the bot

        Returns:
            str: Formatted message content for the standby message
        """
        if self._current_state == BotStateEnum.CONNECTION_ERROR:
            return (
                f"**⚠️ Voice Chat Session - CONNECTION ERROR **\n\n"
                f"---\n"
                f"### 🛠 Current State:\n"
                f"- **State**: `{self._current_state.value}`\n"
                f"- **Details**: The bot has encountered a connection issue (voice or services) and may not be fully functional.\n"
                f"- **Action**: Please try `{Config.COMMAND_PREFIX}disconnect` and then `{Config.COMMAND_PREFIX}connect` again.\n"
                f"If the issue persists, contact an administrator.\n"
                f"---\n"
                f"### 🤖 AI Provider:\n"
                f"> Active Service: `{self.active_ai_provider_name.upper()}`\n"
                f"---\n"
                f"### 🧑 Authority User:\n"
                f"> `{self._authority_user_name}` can control the recording actions (if applicable)."
            )

        return (
            f"**🎙 Voice Chat Session - **\n\n"
            f"---\n"
            f"### 🔄 How to control the bot:\n"
            f"1. **Start Recording**: React to this message with 🎙 to start recording.\n"
            f"2. **Finish Recording**: Remove your 🎙 reaction to finish recording.\n"
            f"3. **End Session**: Use `{Config.COMMAND_PREFIX}disconnect` to end the session.\n"
            f"4. **Switch AI**: Use `{Config.COMMAND_PREFIX}set_provider <name>` (e.g., openai, gemini).\n"
            f"---\n"
            f"### 🛠 Current State:\n"
            f"- **State**: `{self._current_state.value}`\n"
            f"---\n"
            f"### 🤖 AI Provider:\n"
            f"> Active Service: `{self.active_ai_provider_name.upper()}`\n"
            f"---\n"
            f"### 🧑 Authority User:\n"
            f"> `{self._authority_user_name}` can control the recording actions."
        )

    async def initialize_standby(self, ctx) -> bool:
        """
        Initialize standby state from idle state.

        This method:
        1. Transitions the bot from IDLE to STANDBY state
        2. Creates and sends the standby message with controls
        3. Adds the 🎙 reaction for users to start recording

        Args:
            ctx: The Discord context to send the standby message to

        Returns:
            bool: True if transition was successful, False if bot was not in IDLE state
        """
        async with self._lock:
            if self._current_state != BotStateEnum.IDLE:
                # Only allow transition from IDLE state to prevent unexpected state changes.
                return False

            self._current_state = BotStateEnum.STANDBY
            self._standby_message = await ctx.send(self.get_message_content())
            await self._standby_message.add_reaction(
                "🎙"
            )  # Add initial reaction for control
            return True

    async def start_recording(self, user: discord.User) -> bool:
        """
        Transition from standby to recording state.

        This method:
        1. Transitions the bot from STANDBY to RECORDING state
        2. Sets the authority user to the one who started recording
        3. Updates the standby message to reflect the new state

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

            await self._update_message()  # Update UI
            return True

    async def stop_recording(self) -> bool:
        """
        Stop recording and return to standby state.

        This method:
        1. Transitions the bot from RECORDING back to STANDBY state
        2. Resets the authority user to "anyone"
        3. Updates the standby message to reflect the new state

        Returns:
            bool: True if transition was successful, False if bot was not in RECORDING state
        """
        async with self._lock:
            if self._current_state != BotStateEnum.RECORDING:
                # Only allow transition from RECORDING state.
                return False

            self._current_state = BotStateEnum.STANDBY
            self._reset_authority()  # Release specific user control

            await self._update_message()  # Update UI
            return True

    async def reset_to_idle(self) -> bool:
        """
        Reset the bot state to idle.

        This method:
        1. Transitions the bot from any state back to IDLE state
        2. Deletes the standby message if it exists
        3. Resets the authority user to "anyone"

        Returns:
            bool: True if transition was successful, False if bot was already in IDLE state
        """
        async with self._lock:
            if self._current_state == BotStateEnum.IDLE:
                # Already idle, no action needed.
                return False

            if self._standby_message:
                try:
                    await self._standby_message.delete()
                except discord.NotFound:
                    # Message might have been deleted manually, which is fine.
                    pass
                finally:
                    self._standby_message = None

            self._current_state = BotStateEnum.IDLE
            self._reset_authority()  # Reset authority
            # No UI message to update in IDLE state.
            return True

    async def _update_message(self):
        """
        Update the standby message with the current state.

        This internal method updates the standby message's content to reflect
        the current state of the bot, including the current mode and authority user.
        It's called whenever the bot's state changes to keep the UI in sync.

        The method does nothing if there is no standby message (e.g., in IDLE state).
        """
        if self._standby_message:
            await self._standby_message.edit(content=self.get_message_content())

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
        3. Updates the standby message if it exists.

        Returns:
            bool: True if transition was successful, False if bot was already in CONNECTION_ERROR state.
        """
        async with self._lock:
            if self._current_state == BotStateEnum.CONNECTION_ERROR:
                return False  # Already in the error state

            self._current_state = BotStateEnum.CONNECTION_ERROR
            self._reset_authority()  # Reset authority

            if self._standby_message:  # Update UI if a standby message exists
                await self._update_message()
            return True

    async def recover_to_standby(self) -> bool:
        """
        Attempt to recover from CONNECTION_ERROR state back to STANDBY.

        This method should be called when external connections are confirmed to be restored.
        It transitions the state to STANDBY and updates the UI if possible.

        Returns:
            bool: True if recovery to STANDBY was successful, False otherwise
                  (e.g., not in CONNECTION_ERROR state, or standby message missing).
        """
        async with self._lock:
            if self._current_state != BotStateEnum.CONNECTION_ERROR:
                return False  # Can only recover from connection error state

            if not self._standby_message:
                # If the standby message doesn't exist (e.g., error occurred before it was created,
                # or it was somehow deleted), we cannot fully recover to standby UI here.
                # This recovery path assumes the standby message context is still valid.
                return False

            self._current_state = BotStateEnum.STANDBY
            self._reset_authority()
            await (
                self._update_message()
            )  # Update the existing standby message to reflect STANDBY state
            return True
