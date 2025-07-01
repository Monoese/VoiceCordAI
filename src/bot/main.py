"""
Main entry point for the Discord bot application.

This module initializes all the necessary components for the bot to function:
- Real-time AI service communication via a factory pattern
- Discord bot setup with appropriate intents and command prefix

The bot is configured to use the VoiceCog, which manages all voice-related commands
and events by delegating to per-guild session managers.
"""

import asyncio
from typing import Dict

import discord
from discord.ext import commands

from src.bot.cogs.voice_cog import VoiceCog
from src.config.config import Config
from src.utils.logger import get_logger

# Import managers and their configurations
from src.openai_adapter.manager import OpenAIRealtimeManager
from src.openai_adapter.config import OPENAI_SERVICE_CONFIG
from src.gemini_adapter.manager import GeminiRealtimeManager
from src.gemini_adapter.config import GEMINI_SERVICE_CONFIG


# Configure discord.py's internal logging.
discord.utils.setup_logging(level=Config.LOG_CONSOLE_LEVEL, root=False)

logger = get_logger(__name__)

# --- Set up AI Service Communication Layer ---
# Instead of instances, we create a factory registry.
ai_service_factories: Dict[str, tuple] = {
    "openai": (OpenAIRealtimeManager, OPENAI_SERVICE_CONFIG),
    "gemini": (GeminiRealtimeManager, GEMINI_SERVICE_CONFIG),
}
logger.info(
    "AI service factories registered for: %s", list(ai_service_factories.keys())
)

# Validate that the default provider from Config is a valid factory choice.
# The actual key validity will be checked on-demand when the manager is created.
if Config.AI_SERVICE_PROVIDER not in ai_service_factories:
    logger.error(
        f"Default AI_SERVICE_PROVIDER '{Config.AI_SERVICE_PROVIDER}' is not a valid choice. "
        f"Available providers: {list(ai_service_factories.keys())}. Exiting."
    )
    raise SystemExit(
        f"Default AI_SERVICE_PROVIDER '{Config.AI_SERVICE_PROVIDER}' is not a valid choice."
    )


# --- Configure Discord Bot ---
intents = discord.Intents.default()
intents.message_content = True
bot: commands.Bot = commands.Bot(command_prefix=Config.COMMAND_PREFIX, intents=intents)


async def main() -> None:
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
            ai_service_factories=ai_service_factories,  # Pass the dictionary of factories
        )
        await bot.add_cog(voice_cog_instance)
        logger.info("VoiceCog loaded and added to the bot.")

        logger.info("Starting Discord bot...")
        await bot.start(Config.DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
