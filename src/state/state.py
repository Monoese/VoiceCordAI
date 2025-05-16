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

from enum import Enum
from typing import Optional

import discord
from discord import Message


class BotStateEnum(Enum):
    """
    Enumeration of possible bot states.

    States:
        IDLE: Bot is not actively listening or processing voice
        STANDBY: Bot is ready to start recording when triggered
        RECORDING: Bot is actively recording audio from a user
    """
    IDLE = "idle"
    STANDBY = "standby"
    RECORDING = "recording"


class BotState:
    """
    Manages the state and permissions of the Discord bot.

    This class:
    - Tracks the current state of the bot (idle, standby, recording)
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
        self._current_state: BotStateEnum = BotStateEnum.IDLE  # Initial state is IDLE
        self._authority_user_id: Optional[str] = "anyone"  # Default authority is anyone
        self._authority_user_name: Optional[str] = "anyone"  # Default authority name
        self._standby_message: Optional[Message] = None  # No standby message initially

    @property
    def current_state(self) -> BotStateEnum:
        """
        Get the current state of the bot.

        Returns:
            BotStateEnum: The current state (IDLE, STANDBY, or RECORDING)
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

    @authority_user_id.setter
    def authority_user_id(self, value: str):
        """
        Set the ID of the user who has authority to control the bot.

        Args:
            value: The user ID to set as the authority
        """
        self._authority_user_id = value

    @property
    def standby_message(self) -> Optional[Message]:
        """
        Get the standby message that displays the bot's status and controls.

        Returns:
            Optional[Message]: The Discord message object or None if not in standby
        """
        return self._standby_message

    def get_message_content(self) -> str:
        """
        Generate the standby message content based on the current state.

        This method creates a formatted message that displays:
        - The current bot mode (IDLE, STANDBY, or RECORDING)
        - Instructions for using the bot's reaction controls
        - The current recording status
        - Which user has authority to control the bot

        Returns:
            str: Formatted message content for the standby message
        """
        return (f"**🎙 Voice Recording Bot - **{self._current_state.value}** Mode**\n\n"
                f"Here's how to control the bot:\n"
                f"---\n"
                f"### 🔄 How to Use:\n"
                f"1. **Start Recording**: React to this message with 🎙 to start recording.\n"
                f"2. **Stop Recording**: Remove your 🎙 reaction to pause recording.\n"
                f"4. **Finish Session**: Use `!-listen` to end the session and return the bot to Idle Mode.\n"
                f"---\n"
                f"### 🛠 Current Status:\n"
                f"- **Recording Status**: `{self._current_state.value}`\n"
                f"---\n"
                f"### 🧑 Authority User:\n"
                f"> `{self._authority_user_name}` can control the recording actions.")

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
        # Only allow transition from IDLE state
        if self._current_state != BotStateEnum.IDLE:
            return False

        # Transition to STANDBY state
        self._current_state = BotStateEnum.STANDBY

        # Create and send the standby message with controls
        self._standby_message = await ctx.send(self.get_message_content())
        await self._standby_message.add_reaction("🎙")
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
        # Only allow transition from STANDBY state
        if self._current_state != BotStateEnum.STANDBY:
            return False

        # Transition to RECORDING state
        self._current_state = BotStateEnum.RECORDING

        # Set the authority user
        self.authority_user_id = user.id
        self._authority_user_name = user.name

        # Update the message to reflect the new state
        await self._update_message()
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
        # Only allow transition from RECORDING state
        if self._current_state != BotStateEnum.RECORDING:
            return False

        # Transition back to STANDBY state
        self._current_state = BotStateEnum.STANDBY

        # Reset authority to "anyone"
        self.authority_user_id = "anyone"
        self._authority_user_name = "anyone"

        # Update the message to reflect the new state
        await self._update_message()
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
        # Don't do anything if already in IDLE state
        if self._current_state == BotStateEnum.IDLE:
            return False

        # Delete the standby message if it exists
        if self._standby_message:
            await self._standby_message.delete()
            self._standby_message = None

        # Transition to IDLE state
        self._current_state = BotStateEnum.IDLE
        self.authority_user_id = "anyone"
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
