"""
Interaction Handler module for processing user interactions within a GuildSession.
"""

from typing import TYPE_CHECKING

import discord
from discord.ext import commands

from src.config.config import Config
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.bot.session.guild_session import GuildSession
    from src.bot.session.session_ui_manager import SessionUIManager


logger = get_logger(__name__)


class InteractionHandler:
    """
    Handles user interactions like reactions for a specific GuildSession.

    This class acts as a pure dispatcher, forwarding reaction events to the
    GuildSession for stateful processing.
    """

    def __init__(
        self,
        guild_id: int,
        bot: commands.Bot,
        ui_manager: "SessionUIManager",
        guild_session: "GuildSession",
    ):
        self.guild_id = guild_id
        self.bot = bot
        self.ui_manager = ui_manager
        self.guild_session = guild_session

    async def cleanup(self) -> None:
        """No-op cleanup for the dispatcher. All resources are managed by GuildSession."""
        pass

    async def handle_reaction_add(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """
        Dispatches reaction add events based on the emoji.
        """
        message_id = self.ui_manager.get_message_id()
        if user == self.bot.user or not (
            message_id and reaction.message.id == message_id
        ):
            return

        emoji = str(reaction.emoji)

        if emoji == Config.REACTION_GRANT_CONSENT:
            await self.guild_session.handle_consent_reaction(user, added=True)
        elif emoji in (Config.REACTION_MODE_MANUAL, Config.REACTION_MODE_REALTIME):
            await self.guild_session.handle_mode_switch_reaction(user, emoji)
        elif emoji == Config.REACTION_TRIGGER_PTT:
            await self.guild_session.handle_pushtotalk_reaction(user, added=True)

    async def handle_reaction_remove(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """
        Dispatches reaction remove events based on the emoji.
        """
        message_id = self.ui_manager.get_message_id()
        if user == self.bot.user or not (
            message_id and reaction.message.id == message_id
        ):
            return

        emoji = str(reaction.emoji)

        if emoji == Config.REACTION_GRANT_CONSENT:
            await self.guild_session.handle_consent_reaction(user, added=False)
        elif emoji == Config.REACTION_TRIGGER_PTT:
            await self.guild_session.handle_pushtotalk_reaction(user, added=False)
