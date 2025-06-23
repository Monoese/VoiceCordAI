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
from asyncio import Task
from enum import Enum
from typing import Optional

import discord
from discord import Message
from discord.ext import commands
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
    - Maintains the standby message for reaction-based controls
    - Provides methods to update the UI based on state changes
    """

    def __init__(self) -> None:
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
        self._update_queue = asyncio.Queue()
        self._ui_updater_task: Optional[Task] = None

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

    def start_updater_task(self) -> None:
        """
        Start the background UI updater task if it is not already running.
        """
        if self._ui_updater_task is None or self._ui_updater_task.done():
            self._ui_updater_task = asyncio.create_task(self._ui_updater_loop())
            logger.info("UI updater task started.")

    async def stop_updater_task(self) -> None:
        """
        Stop the background UI updater task gracefully.
        """
        if self._ui_updater_task and not self._ui_updater_task.done():
            self._ui_updater_task.cancel()
            try:
                await self._ui_updater_task
            except asyncio.CancelledError:
                pass  # This is expected.
            finally:
                self._ui_updater_task = None
                logger.info("UI updater task stopped.")

    async def _ui_updater_loop(self) -> None:
        """
        Background loop to process UI update requests from a queue.
        This method continuously waits for items on `_update_queue`. When an item
        is received, it calls `_update_message` to synchronize the UI.
        This decouples state changes from slow network I/O.
        """
        while True:
            try:
                await self._update_queue.get()
            except asyncio.CancelledError:
                logger.info("UI updater task cancelled while waiting for queue.")
                break

            try:
                await self._update_message()
            except discord.DiscordException:
                # _update_message already logged the specific error.
                # A failure to update the UI is a connection issue.
                logger.warning("UI update failed. Entering connection error state.")
                await self.enter_connection_error_state()
            except Exception as e:
                logger.critical(
                    "Unexpected critical error during UI update: %s",
                    e,
                    exc_info=True,
                )
            finally:
                self._update_queue.task_done()

    async def set_active_ai_provider_name(self, provider_name: str) -> None:
        """
        Set the name of the active AI service provider and queues a UI update.

        Args:
            provider_name: The name of the provider (e.g., "openai", "gemini").
        """
        async with self._lock:
            self._active_ai_provider_name = provider_name
            self._update_queue.put_nowait(True)

    def _reset_authority(self) -> None:
        """Atomically resets the authority user to 'anyone'."""
        self._authority_user_id = "anyone"
        self._authority_user_name = "anyone"

    def _set_authority(self, user: discord.User) -> None:
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
        main_content: str
        authority_user_note: str = "can control the recording actions."

        if self._current_state == BotStateEnum.CONNECTION_ERROR:
            main_content = (
                f"**⚠️ Voice Chat Session - CONNECTION ERROR **\n\n"
                f"---\n"
                f"### 🛠 Current State:\n"
                f"- **State**: `{self._current_state.value}`\n"
                f"- **Details**: The bot has encountered a connection issue (voice or services) and may not be fully functional.\n"
                f"- **Action**: Please try `{Config.COMMAND_PREFIX}disconnect` and then `{Config.COMMAND_PREFIX}connect` again.\n"
                f"If the issue persists, contact an administrator."
            )
            authority_user_note = "can control the recording actions (if applicable)."
        else:
            main_content = (
                f"**🎙 Voice Chat Session - **\n\n"
                f"---\n"
                f"### 🔄 How to control the bot:\n"
                f"1. **Start Recording**: React to this message with 🎙 to start recording.\n"
                f"2. **Finish Recording**: Remove your 🎙 reaction to finish recording.\n"
                f"3. **End Session**: Use `{Config.COMMAND_PREFIX}disconnect` to end the session.\n"
                f"4. **Switch AI**: Use `{Config.COMMAND_PREFIX}set <name>` (e.g., openai, gemini).\n"
                f"---\n"
                f"### 🛠 Current State:\n"
                f"- **State**: `{self._current_state.value}`"
            )

        shared_content = (
            f"---\n"
            f"### 🤖 AI Provider:\n"
            f"> Active Service: `{self.active_ai_provider_name.upper()}`\n"
            f"---\n"
            f"### 🧑 Authority User:\n"
            f"> `{self._authority_user_name}` {authority_user_note}"
        )

        return f"{main_content}\n{shared_content}"

    async def initialize_standby(self, ctx: commands.Context) -> bool:
        """
        Initialize standby state from idle state.

        This method:
        1. Transitions the bot from IDLE to STANDBY state
        2. Creates and sends the standby message with controls
        3. Adds the 🎙 reaction for users to start recording

        Args:
            ctx: The Discord command context to send the standby message to.

        Returns:
            bool: True if transition was successful, False if bot was not in IDLE state
        """
        async with self._lock:
            if self._current_state != BotStateEnum.IDLE:
                # Only allow transition from IDLE state to prevent unexpected state changes.
                return False

            self._current_state = BotStateEnum.STANDBY
            try:
                self._standby_message = await ctx.send(self.get_message_content())
                await self._standby_message.add_reaction(
                    "🎙"
                )  # Add initial reaction for control
            except discord.DiscordException as e:
                logger.error(
                    "Failed to create standby message or add reaction: %s",
                    e,
                    exc_info=True,
                )
                self._current_state = BotStateEnum.IDLE  # Revert state
                self._standby_message = None
                return False
            return True

    async def start_recording(self, user: discord.User) -> bool:
        """
        Transition from standby to recording state.

        This method:
        1. Transitions the bot from STANDBY to RECORDING state
        2. Sets the authority user to the one who started recording
        3. Queues an update for the standby message to reflect the new state

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

            self._update_queue.put_nowait(True)
            return True

    async def stop_recording(self) -> bool:
        """
        Stop recording and return to standby state.

        This method:
        1. Transitions the bot from RECORDING back to STANDBY state
        2. Resets the authority user to "anyone"
        3. Queues an update for the standby message to reflect the new state

        Returns:
            bool: True if transition was successful, False if bot was not in RECORDING state
        """
        async with self._lock:
            if self._current_state != BotStateEnum.RECORDING:
                # Only allow transition from RECORDING state.
                return False

            self._current_state = BotStateEnum.STANDBY
            self._reset_authority()  # Release specific user control

            self._update_queue.put_nowait(True)
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
        message_to_delete: Optional[Message] = None
        async with self._lock:
            if self._current_state == BotStateEnum.IDLE:
                # Already idle, no action needed.
                return False

            message_to_delete = self._standby_message
            self._standby_message = None

            self._current_state = BotStateEnum.IDLE
            self._reset_authority()  # Reset authority
            # No UI message to update in IDLE state.

        if message_to_delete:
            try:
                await message_to_delete.delete()
            except discord.NotFound:
                # Message might have been deleted manually, which is fine.
                pass
            except discord.DiscordException as e:
                # Log other errors (e.g., permissions) but proceed with state reset.
                logger.warning("Failed to delete standby message during reset: %s", e)
        return True

    async def _update_message(self) -> None:
        """
        Update the standby message with the current state.

        This internal method updates the standby message's content to reflect
        the current state of the bot, including the current mode and authority user.
        It is called by the background UI updater task to keep the UI in sync.

        The method does nothing if there is no standby message (e.g., in IDLE state).
        """
        if self._standby_message:
            try:
                await self._standby_message.edit(content=self.get_message_content())
            except discord.DiscordException as e:
                logger.error("Failed to update standby message: %s", e, exc_info=True)
                raise  # Re-raise to allow callers to handle the state transition

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
        3. Queues an update for the standby message if it exists.

        Returns:
            bool: True if transition was successful, False if bot was already in CONNECTION_ERROR state.
        """
        async with self._lock:
            if self._current_state == BotStateEnum.CONNECTION_ERROR:
                return False  # Already in the error state

            self._current_state = BotStateEnum.CONNECTION_ERROR
            self._reset_authority()  # Reset authority

            if self._standby_message:  # Update UI if a standby message exists
                self._update_queue.put_nowait(True)
            return True

    async def recover_to_standby(self) -> bool:
        """
        Attempt to recover from CONNECTION_ERROR state back to STANDBY.

        This method should be called when external connections are confirmed to be restored.
        It transitions the state to STANDBY and queues a UI update if possible.

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
            self._update_queue.put_nowait(True)
            return True
