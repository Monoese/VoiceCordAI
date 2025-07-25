"""
Voice Cog module for managing per-guild voice interaction sessions.

This module provides the VoiceCog class, which acts as a stateless manager for
GuildSession objects. It is responsible for receiving user commands and events,
and routing them to the appropriate session for handling. This design allows the
bot to support concurrent voice sessions across multiple guilds.
"""

import asyncio
from collections import defaultdict
from typing import Dict

import discord
from discord.ext import commands

from src.bot.session.guild_session import GuildSession
from src.bot.state import BotModeEnum
from src.exceptions import SessionError, StateTransitionError
from src.utils.logger import get_logger


logger = get_logger(__name__)


class VoiceCog(commands.Cog):
    """
    A stateless Discord Cog that manages GuildSessions for voice interactions.

    This cog acts as a dispatcher, receiving user commands and Discord events,
    and routing them to the appropriate GuildSession instance. It maintains a
    dictionary of active sessions, keyed by guild ID, ensuring that each guild's
    state is completely isolated.
    """

    def __init__(
        self,
        bot: commands.Bot,
        ai_service_factories: Dict[str, tuple],
    ):
        """
        Initializes the VoiceCog.

        Args:
            bot: The Discord bot instance.
            ai_service_factories: A dictionary of factories for creating AI service managers.
        """
        self.bot = bot
        self.ai_service_factories = ai_service_factories
        self._sessions: Dict[int, GuildSession] = {}
        self._session_locks: Dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        logger.info("VoiceCog initialized.")

    def _get_or_create_session(self, guild: discord.Guild) -> GuildSession:
        """
        Retrieves an existing session for a guild or creates a new one.

        Args:
            guild: The guild for which to get the session.

        Returns:
            The GuildSession instance for the specified guild.
        """
        if guild.id not in self._sessions:
            logger.info(f"Creating new session for guild {guild.id} ({guild.name})")
            self._sessions[guild.id] = GuildSession(
                guild=guild,
                bot=self.bot,
                ai_service_factories=self.ai_service_factories,
            )
        return self._sessions[guild.id]

    async def cog_unload(self) -> None:
        """
        Clean up all active sessions when the cog is unloaded.
        """
        logger.info(f"Unloading VoiceCog, cleaning up {len(self._sessions)} sessions.")
        cleanup_tasks = [session.cleanup() for session in self._sessions.values()]
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self._sessions.clear()
        logger.info("All active sessions cleaned up.")

    @commands.Cog.listener()
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """
        Listen for the bot's own voice connection changes to handle disconnects.

        This listener filters for events where the bot is moved, disconnected, or
        connected to a voice channel, ignoring state changes like mute or deafen.
        It then delegates the event to the appropriate GuildSession.
        """
        if member.id != self.bot.user.id or not member.guild:
            return

        if before.channel == after.channel:
            return

        session = self._sessions.get(member.guild.id)
        if not session:
            return

        is_connected = after.channel is not None
        await session.handle_voice_connection_update(is_connected)

    @commands.Cog.listener()
    async def on_reaction_add(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """
        Delegates reaction add events to the appropriate GuildSession.

        Args:
            reaction: The reaction that was added.
            user: The user who added the reaction.
        """
        if not reaction.message.guild:
            return

        session = self._sessions.get(reaction.message.guild.id)
        if session:
            try:
                await session.handle_reaction_add(reaction, user)
            except StateTransitionError as e:
                logger.critical(
                    f"Caught unrecoverable state error in guild {reaction.message.guild.id} during reaction add: {e}", 
                    exc_info=True
                )
                # Clean up the session and notify users
                await session.cleanup()
                if reaction.message.guild.id in self._sessions:
                    del self._sessions[reaction.message.guild.id]

    @commands.Cog.listener()
    async def on_reaction_remove(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """
        Delegates reaction remove events to the appropriate GuildSession.

        Args:
            reaction: The reaction that was removed.
            user: The user who removed the reaction.
        """
        if not reaction.message.guild:
            return

        session = self._sessions.get(reaction.message.guild.id)
        if session:
            try:
                await session.handle_reaction_remove(reaction, user)
            except StateTransitionError as e:
                logger.critical(
                    f"Caught unrecoverable state error in guild {reaction.message.guild.id} during reaction remove: {e}", 
                    exc_info=True
                )
                # Clean up the session and notify users
                await session.cleanup()
                if reaction.message.guild.id in self._sessions:
                    del self._sessions[reaction.message.guild.id]

    @commands.command(name="connect", aliases=["manual_connect"])
    async def connect_manual_command(self, ctx: commands.Context) -> None:
        """
        Connects the bot in ManualControl mode.

        This command is also aliased as `manual_connect`. It will fail if a
        session is already active in the guild.

        Args:
            ctx: The command context.
        """
        if not ctx.guild:
            await ctx.send("This command can only be used in a server.")
            return

        # CONCURRENCY FIX: Atomic session operations per guild
        async with self._session_locks[ctx.guild.id]:
            if ctx.guild.id in self._sessions:
                await ctx.send(
                    "I'm already in a session in this server. Use `/disconnect` to end it first."
                )
                return

            session = self._get_or_create_session(ctx.guild)
            try:
                success = await session.initialize_session(ctx, BotModeEnum.ManualControl)
                if not success:
                    logger.warning(
                        f"Manual connection process failed for guild {ctx.guild.id}. Cleaning up session."
                    )
                    if ctx.guild.id in self._sessions:
                        del self._sessions[ctx.guild.id]
            except StateTransitionError as e:
                logger.critical(
                    f"Caught unrecoverable state error in guild {ctx.guild.id} during manual connect: {e}", 
                    exc_info=True
                )
                await ctx.send("An unexpected internal error occurred. The session will now terminate.")
                await session.cleanup()
                if ctx.guild.id in self._sessions:
                    del self._sessions[ctx.guild.id]

    @commands.command(name="realtime_connect")
    async def connect_realtime_command(self, ctx: commands.Context) -> None:
        """
        Connects the bot in RealtimeTalk mode.

        This command will fail if a session is already active in the guild.

        Args:
            ctx: The command context.
        """
        if not ctx.guild:
            await ctx.send("This command can only be used in a server.")
            return

        # CONCURRENCY FIX: Atomic session operations per guild
        async with self._session_locks[ctx.guild.id]:
            if ctx.guild.id in self._sessions:
                await ctx.send(
                    "I'm already in a session in this server. Use `/disconnect` to end it first."
                )
                return

            session = self._get_or_create_session(ctx.guild)
            try:
                success = await session.initialize_session(ctx, BotModeEnum.RealtimeTalk)
                if not success:
                    logger.warning(
                        f"Realtime connection process failed for guild {ctx.guild.id}. Cleaning up session."
                    )
                    if ctx.guild.id in self._sessions:
                        del self._sessions[ctx.guild.id]
            except StateTransitionError as e:
                logger.critical(
                    f"Caught unrecoverable state error in guild {ctx.guild.id} during realtime connect: {e}", 
                    exc_info=True
                )
                await ctx.send("An unexpected internal error occurred. The session will now terminate.")
                await session.cleanup()
                if ctx.guild.id in self._sessions:
                    del self._sessions[ctx.guild.id]

    @commands.command(name="set")
    async def set_provider_command(
        self, ctx: commands.Context, provider_name: str
    ) -> None:
        """
        Delegates the 'set' command to an active GuildSession.

        This command can only be used when the bot is in an active session
        (i.e., after the 'connect' command has been used).

        Args:
            ctx: The command context.
            provider_name: The name of the AI provider to switch to.
        """
        if not ctx.guild:
            await ctx.send("This command can only be used in a server.")
            return

        session = self._sessions.get(ctx.guild.id)
        if not session:
            await ctx.send(
                "The bot is not currently in a session. Use the 'connect' command first."
            )
            return

        try:
            await session.set_provider(ctx, provider_name)
        except StateTransitionError as e:
            logger.critical(
                f"Caught unrecoverable state error in guild {ctx.guild.id} during set provider: {e}", 
                exc_info=True
            )
            await ctx.send("An unexpected internal error occurred. The session will now terminate.")
            await session.cleanup()
            if ctx.guild.id in self._sessions:
                del self._sessions[ctx.guild.id]

    @commands.command(name="disconnect")
    async def disconnect_command(self, ctx: commands.Context) -> None:
        """
        Terminates and cleans up the session for the current guild.

        This command delegates the cleanup logic to the GuildSession and ensures
        the session is removed from the cog's memory, effectively ending its
        lifecycle.

        Args:
            ctx: The command context.
        """
        if not ctx.guild:
            await ctx.send("This command can only be used in a server.")
            return

        # CONCURRENCY FIX: Atomic session operations per guild
        async with self._session_locks[ctx.guild.id]:
            session = self._sessions.get(ctx.guild.id)
            if not session:
                await ctx.send("The bot is not currently in a session in this server.")
                return

            await ctx.send("Session terminating...")
            try:
                await session.cleanup()
                await ctx.send("Session terminated successfully.")
            except StateTransitionError as e:
                logger.critical(
                    f"Caught unrecoverable state error in guild {ctx.guild.id} during disconnect: {e}", 
                    exc_info=True
                )
                await ctx.send("An unexpected internal error occurred during cleanup. The session has been forcefully removed.")
            except SessionError as e:
                logger.error(
                    f"Session error during cleanup for guild {ctx.guild.id}: {e}",
                    exc_info=True,
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error during session cleanup for guild {ctx.guild.id}: {e}",
                    exc_info=True,
                )
                await ctx.send(
                    "An error occurred during cleanup. The session has been forcefully removed."
                )
            finally:
                if ctx.guild.id in self._sessions:
                    del self._sessions[ctx.guild.id]
                logger.info(
                    f"Session for guild {ctx.guild.id} has been fully cleaned up and removed."
                )


async def setup(bot: commands.Bot) -> None:
    """
    Stub for the setup function to prevent loading this cog as a standard extension.

    This cog requires a manual instantiation with specific dependencies and should not
    be loaded via `bot.load_extension()`.

    Args:
        bot: The bot instance.
    """
    raise NotImplementedError(
        "VoiceCog requires dependencies (ai_service_factories) and cannot be loaded as a standard extension. "
        "Instantiate and add it manually in your main script."
    )
