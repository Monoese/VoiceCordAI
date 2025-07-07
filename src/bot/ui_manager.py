"""
UI Manager module for handling GuildSession UI components.
"""

import asyncio
from typing import Optional

import discord

from src.config.config import Config
from src.state.state import BotState, BotStateEnum, StateEvent
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SessionUIManager:
    """Manages the UI components of a GuildSession, primarily the standby message."""

    def __init__(self, guild_id: int, bot_state: BotState):
        self.guild_id = guild_id
        self.bot_state = bot_state
        self.standby_message: Optional[discord.Message] = None
        self._update_queue: asyncio.Queue[None] = asyncio.Queue()
        self._updater_task: Optional[asyncio.Task[None]] = None
        self.bot_state.subscribe_to_state_changes(self._on_state_change)

    async def _on_state_change(self, event: StateEvent) -> None:
        """Callback for when bot state changes, which schedules a UI update."""
        self.schedule_update()

    def get_message_content(self) -> str:
        """Generate the standby message content based on the current state."""
        current_state = self.bot_state.current_state
        main_content: str
        authority_user_note: str = "can control the recording actions."
        authority_user_name = self.bot_state.get_authority_user_name()

        if current_state == BotStateEnum.CONNECTION_ERROR:
            main_content = (
                f"**⚠️ Voice Chat Session - CONNECTION ERROR **\n\n"
                f"---\n"
                f"### 🛠 Current State:\n"
                f"- **State**: `{current_state.value}`\n"
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
                f"1. **Start Recording**: React to this message with {Config.REACTION_START_RECORDING} to start recording.\n"
                f"2. **Finish Recording**: Remove your {Config.REACTION_START_RECORDING} reaction to finish recording.\n"
                f"3. **End Session**: Use `{Config.COMMAND_PREFIX}disconnect` to end the session.\n"
                f"4. **Switch AI**: Use `{Config.COMMAND_PREFIX}set <name>` (e.g., openai, gemini).\n"
                f"---\n"
                f"### 🛠 Current State:\n"
                f"- **State**: `{current_state.value}`"
            )

        shared_content = (
            f"---\n"
            f"### 🤖 AI Provider:\n"
            f"> Active Service: `{self.bot_state.active_ai_provider_name.upper()}`\n"
            f"---\n"
            f"### 🧑 Authority User:\n"
            f"> `{authority_user_name}` {authority_user_note}"
        )

        return f"{main_content}\n{shared_content}"

    async def _update_message(self) -> None:
        """Update the standby message with the current state."""
        if self.standby_message:
            try:
                await self.standby_message.edit(content=self.get_message_content())
            except discord.DiscordException as e:
                logger.error("Failed to update standby message: %s", e, exc_info=True)
                # Re-raise to allow the updater loop to handle it.
                raise

    def schedule_update(self) -> None:
        """Schedules a UI update by putting an item in the queue."""
        self._update_queue.put_nowait(None)

    async def _updater_loop(self) -> None:
        """Background loop to process UI update requests from a queue."""
        while True:
            try:
                await self._update_queue.get()
            except asyncio.CancelledError:
                logger.info("UI updater task cancelled while waiting for queue.")
                break

            try:
                await self._update_message()
            except discord.DiscordException as e:
                logger.warning(
                    f"Failed to update standby message in background loop: {e}"
                )
            except Exception as e:
                logger.error(
                    "Unexpected error during UI update in background loop: %s",
                    e,
                    exc_info=True,
                )
            finally:
                self._update_queue.task_done()

    def start(self) -> None:
        """Start the background UI updater task if it is not already running."""
        if self._updater_task is None or self._updater_task.done():
            self._updater_task = asyncio.create_task(self._updater_loop())
            logger.info(f"UI updater task started for guild {self.guild_id}.")

    async def stop(self) -> None:
        """Stop the background UI updater task gracefully."""
        if self._updater_task and not self._updater_task.done():
            self._updater_task.cancel()
            try:
                await self._updater_task
            except asyncio.CancelledError:
                pass  # This is expected.
            finally:
                self._updater_task = None
                logger.info(f"UI updater task stopped for guild {self.guild_id}.")

    async def create(self, channel: discord.TextChannel) -> bool:
        """Creates and sends the initial standby message."""
        try:
            self.standby_message = await channel.send(self.get_message_content())
            await self.standby_message.add_reaction(Config.REACTION_START_RECORDING)
            return True
        except discord.DiscordException as e:
            logger.error(
                f"Failed to create standby message for guild {self.guild_id}: {e}",
                exc_info=True,
            )
            return False

    async def cleanup(self) -> None:
        """Cleans up UI resources, like deleting the standby message."""
        await self.stop()
        if self.standby_message:
            try:
                await self.standby_message.delete()
            except discord.NotFound:
                pass  # Message was already deleted
            except discord.DiscordException as e:
                logger.warning(
                    f"Failed to delete standby message during cleanup for guild {self.guild_id}: {e}",
                    exc_info=True,
                )
            finally:
                self.standby_message = None

    def get_message_id(self) -> Optional[int]:
        """Returns the ID of the standby message, if it exists."""
        return self.standby_message.id if self.standby_message else None
