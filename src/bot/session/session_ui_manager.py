"""
UI Manager module for handling GuildSession UI components.
"""

import asyncio
import os
from typing import Optional

import discord

from src.config.config import Config
from src.bot.state import BotState, BotStateEnum, StateEvent
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SessionUIManager:
    """Manages the UI components of a GuildSession, primarily the standby message."""

    def __init__(self, guild: discord.Guild, bot_state: BotState):
        self.guild = guild
        self.guild_id = guild.id
        self.bot_state = bot_state
        self.standby_message: Optional[discord.Message] = None
        self._update_queue: asyncio.Queue[None] = asyncio.Queue()
        self._updater_task: Optional[asyncio.Task[None]] = None
        self.bot_state.subscribe_to_state_changes(self._on_state_change)

    async def _on_state_change(self, event: StateEvent) -> None:
        """Callback for when bot state changes, which schedules a UI update."""
        self.schedule_update()

    def _get_consented_user_list(self) -> str:
        """Generates a formatted string of consented user names."""
        user_ids = self.bot_state.get_consented_user_ids()
        if not user_ids:
            return "No one has consented yet."

        names = []
        for user_id in user_ids:
            member = self.guild.get_member(user_id)
            names.append(
                member.display_name if member else f"Unknown User (ID: {user_id})"
            )
        return ", ".join(names)

    def _get_manual_control_content(self) -> str:
        """Generates the UI content for ManualControl mode."""
        state = self.bot_state.current_state
        state_info = ""

        if state == BotStateEnum.STANDBY:
            state_info = "Listening for wake word or Push-to-Talk..."
        elif state == BotStateEnum.RECORDING:
            auth_user = self.bot_state.get_authority_user_name()
            state_info = f"🔴 Recording `{auth_user}`..."

        consented_users = self._get_consented_user_list()

        # Dynamically get the wake word from the model filename
        model_filename = os.path.basename(Config.WAKE_WORD_MODEL_PATH)
        wake_word = model_filename.split("_")[0].capitalize()

        return (
            f"**🎙️ Voice Chat Session - Manual Control**\n\n"
            f"---\n"
            f"### Instructions\n"
            f"- **Push-to-Talk**: React with {Config.REACTION_TRIGGER_PTT} to start/stop recording.\n"
            f"- **Wake Word**: Say '{wake_word}' to start recording.\n"
            f"- **Give Consent**: React with {Config.REACTION_GRANT_CONSENT} to grant/revoke consent.\n"
            f"---\n"
            f"### Status\n"
            f"- **State**: `{state_info}`\n"
            f"- **Consented Users for Wake Word**: {consented_users}"
        )

    def _get_realtime_talk_content(self) -> str:
        """Generates the UI content for RealtimeTalk mode."""
        state = self.bot_state.current_state
        state_info = ""

        if state == BotStateEnum.LISTENING:
            state_info = "Actively listening..."
        elif state == BotStateEnum.SPEAKING:
            state_info = "AI is speaking..."

        consented_users = self._get_consented_user_list()
        user_list_title = (
            "Streaming Audio From"
            if consented_users != "No one has consented yet."
            else "Consented Users"
        )

        return (
            f"**🗣️ Voice Chat Session - Realtime Talk**\n\n"
            f"---\n"
            f"### Instructions\n"
            f"- **Speak Freely**: The AI will manage turn-taking automatically.\n"
            f"- **Give Consent**: React with {Config.REACTION_GRANT_CONSENT} to grant/revoke consent.\n"
            f"- **Switch Mode**: React with {Config.REACTION_MODE_MANUAL} to switch to Manual Control.\n"
            f"---\n"
            f"### Status\n"
            f"- **State**: `{state_info}`\n"
            f"- **{user_list_title}**: {consented_users}"
        )

    def get_message_content(self) -> str:
        """Generate the standby message content based on the current mode and state."""
        if self.bot_state.current_state == BotStateEnum.CONNECTION_ERROR:
            return (
                f"**⚠️ Voice Chat Session - CONNECTION ERROR **\n\n"
                f"---\n"
                f"- **State**: `{self.bot_state.current_state.value}`\n"
                f"- **Details**: The bot has encountered a connection issue.\n"
                f"- **Action**: Try `{Config.COMMAND_PREFIX}disconnect` and then `{Config.COMMAND_PREFIX}connect` again."
            )

        mode_content = self._get_manual_control_content()

        shared_content = (
            f"---\n"
            f"### 🤖 AI Provider: `{self.bot_state.active_ai_provider_name.upper()}`\n"
            f"Use `{Config.COMMAND_PREFIX}set <name>` (e.g., openai, gemini) to switch."
        )

        return f"{mode_content}\n{shared_content}"

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
            await self.standby_message.add_reaction(Config.REACTION_GRANT_CONSENT)
            await self.standby_message.add_reaction(Config.REACTION_TRIGGER_PTT)
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
