"""
Guild Session module for managing per-guild bot state and interactions.

This module defines the GuildSession class, which encapsulates all the logic,
state, and resources for the bot's operation within a single Discord guild. This
ensures that the bot can operate in multiple guilds simultaneously without any
state conflicts.
"""

from typing import Dict

import discord
from discord.ext import commands

from src.audio.playback import AudioPlaybackManager
from src.bot.session.ai_service_coordinator import AIServiceCoordinator
from src.bot.session.interaction_handler import InteractionHandler
from src.bot.session.session_ui_manager import SessionUIManager
from src.bot.session.voice_connection_manager import VoiceConnectionManager
from src.bot.state import BotState
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

        self.bot_state = BotState()
        self.ui_manager = SessionUIManager(guild.id, self.bot_state)
        self.audio_playback_manager = AudioPlaybackManager(self.guild)
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
        logger.info(f"Background tasks started for guild {self.guild.id}.")

    async def cleanup(self) -> None:
        """
        Gracefully shuts down the session, ensuring each cleanup step is attempted.
        """
        logger.info(f"Cleaning up session for guild {self.guild.id}")
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

    async def _on_ai_connect(self) -> None:
        """Callback for when the AI service connects."""
        logger.info(f"AI service connected for guild {self.guild.id}.")
        # If the voice connection is also active, attempt to recover the bot's state.
        if self.voice_connection.is_connected():
            await self.bot_state.recover_to_standby()

    async def _on_ai_disconnect(self) -> None:
        """Callback for when the AI service disconnects."""
        logger.warning(f"AI service disconnected for guild {self.guild.id}.")
        await self.bot_state.enter_connection_error_state()

    async def handle_voice_connection_update(self, is_connected: bool) -> None:
        """
        Handler called by VoiceCog on voice state changes.

        This method decides whether to recover the bot to a healthy state or enter
        an error state based on the status of both the voice and AI connections.
        """
        # Query the coordinator directly to get the authoritative AI connection status.
        if is_connected and self.ai_coordinator.is_connected():
            logger.info(
                f"Voice connection active for guild {self.guild.id}, recovering if needed."
            )
            await self.bot_state.recover_to_standby()
        elif not is_connected:
            # If the voice connection is lost, always enter an error state.
            logger.warning(f"Voice connection lost for guild {self.guild.id}.")
            await self.bot_state.enter_connection_error_state()

    async def connect(self, ctx: commands.Context) -> bool:
        """
        Handles the full connection logic when a user issues the /connect command.

        This method orchestrates the entire session startup sequence:
        1. Ensures a connection to the AI service is active.
        2. Connects the bot to the user's voice channel.
        3. Transitions the bot's state to STANDBY.
        4. Creates and displays the persistent UI message.
        5. Starts all necessary background tasks for the session.

        Args:
            ctx: The command context from the user's invocation.

        Returns:
            True if the entire connection and setup process succeeds, False otherwise.
        """
        if ctx.author.voice is None:
            await ctx.send("You are not connected to a voice channel.")
            return False

        if not await self.ai_coordinator.ensure_connected(
            ctx, on_connect=self._on_ai_connect, on_disconnect=self._on_ai_disconnect
        ):
            return False

        voice_channel = ctx.author.voice.channel
        if not await self.voice_connection.connect_to_channel(voice_channel):
            await ctx.send("Failed to connect to the voice channel.")
            await self.bot_state.enter_connection_error_state()
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
            provider_name,
            ctx,
            self.voice_connection.is_connected(),
            on_connect=self._on_ai_connect,
            on_disconnect=self._on_ai_disconnect,
        ):
            await ctx.send(f"AI provider switched to '{provider_name.upper()}'.")
