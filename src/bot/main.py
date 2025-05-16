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

# Configure logger for this module
logger = get_logger(__name__)

# Initialize core components
audio_manager: AudioManager = AudioManager()
bot_state_manager: BotState = BotState()

# Set up WebSocket communication
event_handler: WebSocketEventHandler = WebSocketEventHandler(audio_manager=audio_manager)
websocket_manager: WebSocketManager = WebSocketManager(event_handler_instance=event_handler)

# Configure Discord bot with all intents for full functionality
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
        # Add the voice cog which handles all voice-related commands and events
        await bot.add_cog(VoiceCog(bot, audio_manager, bot_state_manager, websocket_manager))
        logger.info("VoiceCog loaded.")

        # Start the bot with the token from configuration
        await bot.start(Config.DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
