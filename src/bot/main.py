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

logger = get_logger(__name__)

audio_manager: AudioManager = AudioManager()
bot_state_manager: BotState = BotState()

event_handler: WebSocketEventHandler = WebSocketEventHandler(audio_manager=audio_manager)

websocket_manager: WebSocketManager = WebSocketManager(event_handler_instance=event_handler)

intents: discord.Intents = discord.Intents.all()
bot: commands.Bot = commands.Bot(command_prefix=Config.COMMAND_PREFIX, intents=intents)


async def main():
    async with bot:
        await bot.add_cog(VoiceCog(bot, audio_manager, bot_state_manager, websocket_manager))
        logger.info("VoiceCog loaded.")

        await bot.start(Config.DISCORD_TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
