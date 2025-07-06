"""
Guild Session module for managing per-guild bot state and interactions.

This module defines the GuildSession class, which encapsulates all the logic,
state, and resources for the bot's operation within a single Discord guild. This
ensures that the bot can operate in multiple guilds simultaneously without any
state conflicts.
"""

from typing import Dict

import discord
from discord.ext import commands, tasks

from src.audio.playback import AudioPlaybackManager
from src.bot.ai_service_coordinator import AIServiceCoordinator
from src.bot.interaction_handler import InteractionHandler
from src.bot.ui_manager import SessionUIManager
from src.bot.voice_connection import VoiceConnectionManager
from src.config.config import Config
from src.state.state import BotState, BotStateEnum
from src.utils.logger import get_logger


logger = get_logger(__name__)


class GuildSession:
    """
    Manages all state and logic for the bot's interaction within a single guild.

    This class encapsulates all components required for a voice session, including
    state management, audio playback, voice connection, and AI service interaction.
    Each guild will have its own instance of this class, ensuring complete isolation.
    """

    def __init__(
        self,
        guild: discord.Guild,
        bot: commands.Bot,
        ai_service_factories: Dict[str, tuple],
    ):
        """
        Initializes a new session for a specific guild.

        Args:
            guild: The Discord guild this session belongs to.
            bot: The Discord bot instance.
            ai_service_factories: A dictionary of factories for creating AI service managers.
        """
        self.guild = guild
        self.bot = bot

        # Guild-specific instances of core components
        self.bot_state = BotState()
        self.ui_manager = SessionUIManager(guild.id, self.bot_state)
        self.audio_playback_manager = AudioPlaybackManager()
        self.voice_connection = VoiceConnectionManager(
            self.guild, self.audio_playback_manager
        )
        self.ai_coordinator = AIServiceCoordinator(
            bot_state=self.bot_state,
            audio_playback_manager=self.audio_playback_manager,
            ai_service_factories=ai_service_factories,
            guild_id=self.guild.id,
        )
        self.interaction_handler = InteractionHandler(
            guild_id=self.guild.id,
            bot=self.bot,
            bot_state=self.bot_state,
            ui_manager=self.ui_manager,
            voice_connection=self.voice_connection,
            ai_coordinator=self.ai_coordinator,
        )

    async def start_background_tasks(self) -> None:
        """Starts all persistent background tasks for the session."""
        self.ui_manager.start()
        self._connection_check_loop.start()
        logger.info(f"Background tasks started for guild {self.guild.id}.")

    async def cleanup(self) -> None:
        """
        Gracefully shuts down the session, ensuring each cleanup step is attempted.
        """
        logger.info(f"Cleaning up session for guild {self.guild.id}")
        self._connection_check_loop.cancel()
        await self.ui_manager.cleanup()
        await self.interaction_handler.cleanup()

        try:
            await self.ai_coordinator.shutdown()
        except Exception as e:
            logger.error(
                f"Error during AI provider shutdown for guild {self.guild.id}: {e}",
                exc_info=True,
            )

        try:
            if self.voice_connection.is_connected():
                await self.voice_connection.disconnect()
        except Exception as e:
            logger.error(
                f"Error during voice disconnect for guild {self.guild.id}: {e}",
                exc_info=True,
            )

        await self.bot_state.reset_to_idle()
        logger.info(f"Session for guild {self.guild.id} cleaned up successfully.")

    async def handle_reaction_add(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """
        Delegates reaction add events to the interaction handler.
        """
        await self.interaction_handler.handle_reaction_add(reaction, user)

    async def handle_reaction_remove(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """
        Delegates reaction remove events to the interaction handler.
        """
        await self.interaction_handler.handle_reaction_remove(reaction, user)

    async def connect(self, ctx: commands.Context) -> bool:
        """
        Handles the logic of the 'connect' command.
        """
        if ctx.author.voice is None:
            await ctx.send("You are not connected to a voice channel.")
            return False

        if not await self.ai_coordinator.ensure_connected(ctx):
            return False

        voice_channel = ctx.author.voice.channel
        if not await self.voice_connection.connect_to_channel(voice_channel):
            await ctx.send("Failed to connect to the voice channel.")
            if await self.bot_state.enter_connection_error_state():
                self.ui_manager.schedule_update()
            return False

        if not await self.bot_state.transition_to_standby():
            return False

        if not await self.ui_manager.create(ctx.channel):
            await self.bot_state.reset_to_idle()
            return False

        await self.start_background_tasks()
        logger.info(
            f"Connect command successful for guild {self.guild.id}. Bot is in STANDBY."
        )
        return True

    async def set_provider(self, ctx: commands.Context, provider_name: str) -> None:
        """
        Handles the logic of the 'set' command.
        """
        provider_name = provider_name.lower()
        if provider_name not in self.ai_coordinator.ai_service_factories:
            valid_providers = ", ".join(self.ai_coordinator.ai_service_factories.keys())
            await ctx.send(
                f"Invalid provider name '{provider_name}'. Valid options are: {valid_providers}."
            )
            return

        if await self.ai_coordinator.switch_provider(
            provider_name, ctx, self.voice_connection.is_connected()
        ):
            self.ui_manager.schedule_update()
            await ctx.send(f"AI provider switched to '{provider_name.upper()}'.")

            if self.bot_state.current_state == BotStateEnum.CONNECTION_ERROR:
                if await self.bot_state.recover_to_standby():
                    self.ui_manager.schedule_update()

    @tasks.loop(seconds=Config.CONNECTION_CHECK_INTERVAL)
    async def _connection_check_loop(self) -> None:
        """Periodically checks the health of critical connections."""
        if self.bot_state.current_state not in [
            BotStateEnum.STANDBY,
            BotStateEnum.RECORDING,
            BotStateEnum.CONNECTION_ERROR,
        ]:
            return

        channel_for_message = None
        if self.ui_manager.standby_message:
            channel_for_message = self.ui_manager.standby_message.channel

        if self.bot_state.current_state == BotStateEnum.CONNECTION_ERROR:
            if (
                self.voice_connection.is_connected()
                and self.ai_coordinator.is_connected()
            ):
                if await self.bot_state.recover_to_standby():
                    self.ui_manager.schedule_update()
                    return

        await self.interaction_handler.check_and_handle_connection_issues(
            channel_for_message
        )

    @_connection_check_loop.before_loop
    async def before_connection_check_loop(self) -> None:
        """Waits for the bot to be ready before starting the connection check loop."""
        await self.bot.wait_until_ready()
