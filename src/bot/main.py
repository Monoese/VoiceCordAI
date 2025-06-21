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
from typing import Dict

import discord
from discord.ext import commands

from src.audio.audio import AudioManager
from src.bot.cogs.voice_cog import VoiceCog
from src.config.config import Config
from src.state.state import BotState
from src.utils.logger import get_logger

# Import both managers
from src.openai_adapter.manager import OpenAIRealtimeManager
from src.gemini_adapter.manager import GeminiRealtimeManager
from src.ai_services.interface import IRealtimeAIServiceManager


# Configure discord.py's internal logging.
discord.utils.setup_logging(level=Config.LOG_CONSOLE_LEVEL, root=False)

logger = get_logger(__name__)

# --- Initialize Core Application Components ---
audio_manager: AudioManager = AudioManager()
bot_state_manager: BotState = BotState()

# --- Set up AI Service Communication Layer ---
all_ai_service_managers: Dict[str, IRealtimeAIServiceManager] = {}

if Config.OPENAI_API_KEY:
    openai_manager = OpenAIRealtimeManager(
        audio_manager=audio_manager, service_config=Config.OPENAI_SERVICE_CONFIG
    )
    all_ai_service_managers["openai"] = openai_manager
    logger.info("OpenAI Realtime Manager initialized.")
else:
    logger.info(
        "OpenAI API key not found. OpenAI Realtime Manager will not be available."
    )

if Config.GEMINI_API_KEY:
    gemini_manager = GeminiRealtimeManager(
        audio_manager=audio_manager, service_config=Config.GEMINI_SERVICE_CONFIG
    )
    all_ai_service_managers["gemini"] = gemini_manager
    logger.info("Gemini Realtime Manager initialized.")
else:
    logger.info(
        "Gemini API key not found. Gemini Realtime Manager will not be available."
    )

# Validate that at least one manager was initialized if we proceed
if not all_ai_service_managers:
    logger.error(
        "No AI service managers could be initialized. Check API key configurations. Exiting."
    )
    raise SystemExit(
        "No AI service managers available. Please configure at least one API key."
    )

# Validate that the default provider from Config is in our dictionary and was initialized
if Config.AI_SERVICE_PROVIDER not in all_ai_service_managers:
    logger.error(
        f"Default AI_SERVICE_PROVIDER '{Config.AI_SERVICE_PROVIDER}' is configured, "
        f"but its API key is missing or the manager could not be initialized. "
        f"Available managers: {list(all_ai_service_managers.keys())}. Exiting."
    )
    raise SystemExit(
        f"Default AI_SERVICE_PROVIDER '{Config.AI_SERVICE_PROVIDER}' not available. "
        f"Check API key or ensure it's a valid choice among initialized services."
    )


# --- Configure Discord Bot ---
intents = discord.Intents.default()
intents.message_content = True
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
