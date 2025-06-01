"""
Main entry point for the Discord bot application.

This module initializes all the necessary components for the bot to function:
- Audio management for voice processing
- Bot state management for tracking the bot's current state
- WebSocket communication for external service integration
- Discord bot setup with appropriate intents and command prefix

The bot is configured to use the VoiceCog for handling voice-related commands and events.
"""

import asyncio

import discord
from discord.ext import commands

from src.audio.audio import AudioManager
from src.bot.cogs.voice_cog import VoiceCog
from src.config.config import Config
from src.state.state import BotState
from src.utils.logger import get_logger
from src.websocket.event_handler import WebSocketEventHandler
from src.websocket.manager import WebSocketManager

# Configure discord.py's internal logging.
# root=False prevents it from reconfiguring the root logger we set up in src.utils.logger.
discord.utils.setup_logging(
    level=Config.LOG_CONSOLE_LEVEL, root=False
)  # Use configured console level for discord.py's loggers.

logger = get_logger(__name__)

# --- Initialize Core Application Components ---
audio_manager: AudioManager = AudioManager()
bot_state_manager: BotState = BotState()

# --- Set up WebSocket Communication Layer ---
event_handler: WebSocketEventHandler = WebSocketEventHandler(
    audio_manager=audio_manager  # EventHandler needs AudioManager to process audio events
)
websocket_manager: WebSocketManager = WebSocketManager(
    event_handler_instance=event_handler  # Manager uses EventHandler to dispatch received messages
)

# --- Configure Discord Bot ---
# Intents.all() enables all privileged intents; for production, specify only needed intents.
intents: discord.Intents = discord.Intents.all()
bot: commands.Bot = commands.Bot(command_prefix=Config.COMMAND_PREFIX, intents=intents)


async def main():
    """
    Main asynchronous function that starts the Discord bot.

    This function:
    1. Adds the VoiceCog to the bot, which handles voice commands and events
    2. Starts the bot with the Discord token from the configuration

    The function uses an async context manager to ensure proper cleanup when the bot stops.
    """
    async with bot:
        voice_cog_instance = VoiceCog(
            bot=bot,
            audio_manager=audio_manager,
            bot_state_manager=bot_state_manager,
            websocket_manager=websocket_manager,
        )
        await bot.add_cog(voice_cog_instance)
        logger.info("VoiceCog loaded and added to the bot.")

        logger.info("Starting Discord bot...")
        await bot.start(Config.DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
