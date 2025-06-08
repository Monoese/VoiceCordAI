"""
Main entry point for the Discord bot application.

This module initializes all the necessary components for the bot to function:
- Audio management for voice processing
- Bot state management for tracking the bot's current state
- OpenAI Realtime API communication via OpenAIRealtimeManager
- Discord bot setup with appropriate intents and command prefix

The bot is configured to use the VoiceCog for handling voice-related commands and events.
"""

import asyncio
from typing import Dict  # Added import

import discord
from discord.ext import commands

from src.audio.audio import AudioManager
from src.bot.cogs.voice_cog import VoiceCog
from src.config.config import Config
from src.state.state import BotState
from src.utils.logger import get_logger

# Import both managers
from src.openai_adapter.manager import OpenAIRealtimeManager
from src.gemini_adapter.manager import GeminiRealtimeManager  # Added import
from src.ai_services.interface import IRealtimeAIServiceManager  # For type hinting


# Configure discord.py's internal logging.
# root=False prevents it from reconfiguring the root logger we set up in src.utils.logger.
discord.utils.setup_logging(level=Config.LOG_CONSOLE_LEVEL, root=False)

logger = get_logger(__name__)

# --- Initialize Core Application Components ---
audio_manager: AudioManager = AudioManager()
bot_state_manager: BotState = BotState()

# --- Set up AI Service Communication Layer ---
# Instantiate all available AI service managers
openai_manager = OpenAIRealtimeManager(
    audio_manager=audio_manager, service_config=Config.OPENAI_SERVICE_CONFIG
)
gemini_manager = GeminiRealtimeManager(
    audio_manager=audio_manager, service_config=Config.GEMINI_SERVICE_CONFIG
)

# Store them in a dictionary for VoiceCog
all_ai_service_managers: Dict[str, IRealtimeAIServiceManager] = {
    "openai": openai_manager,
    "gemini": gemini_manager,
}

# Validate that the default provider from Config is in our dictionary
if Config.AI_SERVICE_PROVIDER not in all_ai_service_managers:
    logger.error(
        f"Default AI_SERVICE_PROVIDER '{Config.AI_SERVICE_PROVIDER}' is not a configured manager. Available: {list(all_ai_service_managers.keys())}. Exiting."
    )
    raise SystemExit(
        f"Invalid default AI_SERVICE_PROVIDER: {Config.AI_SERVICE_PROVIDER}"
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
            ai_service_managers=all_ai_service_managers,  # Pass the dictionary of managers
        )
        await bot.add_cog(voice_cog_instance)
        logger.info("VoiceCog loaded and added to the bot.")

        logger.info("Starting Discord bot...")
        await bot.start(Config.DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
