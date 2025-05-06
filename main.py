import asyncio
import base64

import discord
from discord.ext import commands
from discord.ext import voice_recv

from audio import AudioManager
from config import Config
from events import *
from logger import get_logger
from state import BotState, BotStateEnum
from websocket_manager import WebSocketManager

# Set up logger for this module
logger = get_logger(__name__)

audio_manager = AudioManager()
bot_state_manager = BotState()
websocket_manager = WebSocketManager()
voice_client = None

intents = discord.Intents.all()

bot = commands.Bot(command_prefix="!", intents=intents)


async def handle_error(event: ErrorEvent):
    logger.error(f"Error event received: {event.error}")


async def handle_session_updated(event: SessionUpdatedEvent):
    logger.info(f"Handling event: {event.type}")


async def handle_session_created(event: SessionCreatedEvent):
    logger.info(f"Handling event: {event.type}")


async def handle_response_audio_delta(event):
    logger.debug(f"Handling event: {event.type}")
    base64_audio = event.delta
    decoded_audio = base64.b64decode(base64_audio)

    audio_manager.extend_response_buffer(decoded_audio)
    logger.debug(f"Buffered audio fragment: {len(decoded_audio)} bytes")


async def handle_response_audio_done(event):
    if audio_manager.response_buffer:
        await audio_manager.enqueue_audio(audio_manager.response_buffer)
        audio_manager.clear_response_buffer()


EVENT_HANDLERS = {"error": handle_error, "session.created": handle_session_created,
                  "session.updated": handle_session_updated, "response.audio.delta": handle_response_audio_delta,
                  "response.audio.done": handle_response_audio_done, }


async def queue_session_update():
    event = SessionUpdatedEvent(event_id="event_123", type="session.update", session={"turn_detection": None})
    await websocket_manager.send_event(event)


async def process_incoming_events():
    """Process events from the WebSocketManager's incoming queue"""
    while True:
        event = await websocket_manager.get_next_event()
        logger.debug(f"Processing event in incoming queue: {event.type}")

        event_type = event.type
        handler = EVENT_HANDLERS.get(event_type)
        try:
            if handler:
                await handler(event)
            else:
                logger.warning(f"No handler found for event type: {event_type}")
        except Exception as e:
            logger.error(f"Error in handler for {event_type}: {e}")
        finally:
            websocket_manager.task_done()


async def send_audio_events(base64_audio: str):
    """Helper function to send audio-related events to the server."""

    data = {"event_id": "event_456", "type": "input_audio_buffer.append", "audio": base64_audio, }
    event = EVENT_TYPE_MAPPING["input_audio_buffer.append"].from_json(data)
    await websocket_manager.send_event(event)

    data = {"event_id": "event_789", "type": "input_audio_buffer.commit", }
    event = EVENT_TYPE_MAPPING["input_audio_buffer.commit"].from_json(data)
    await websocket_manager.send_event(event)

    data = {"event_id": "event_234", "type": "response.create"}
    event = EVENT_TYPE_MAPPING["response.create"].from_json(data)
    await websocket_manager.send_event(event)


@bot.command(name="listen")
async def start_standby(ctx):
    if await bot_state_manager.initialize_standby(ctx):
        return
    await ctx.send("Bot is already active in another state.")


@bot.command(name="-listen")
async def return_to_idle(ctx):
    if await bot_state_manager.reset_to_idle():
        return
    await ctx.send("Bot is already in idle state.")


@bot.event
async def on_reaction_add(reaction, user):
    if user == bot.user:
        return

    if (reaction.message.id == bot_state_manager.standby_message.id):
        if (reaction.emoji == "🎙" and bot_state_manager.current_state == BotStateEnum.STANDBY):

            if await bot_state_manager.start_recording(user):
                voice_client = reaction.message.guild.voice_client
                if voice_client and isinstance(voice_client, voice_recv.VoiceRecvClient):
                    sink = audio_manager.create_sink()
                    voice_client.listen(sink)
                    logger.info("Started new recording session with fresh sink")


        elif (
                reaction.emoji == "❌" and bot_state_manager.current_state == BotStateEnum.RECORDING and bot_state_manager.is_authorized(
            user)):

            if await bot_state_manager.stop_recording():
                await reaction.message.channel.send(f"{user.display_name} canceled recording. Returning to standby.")


@bot.event
async def on_reaction_remove(reaction, user):
    if user == bot.user:
        return

    if (
            reaction.message.id == bot_state_manager.standby_message.id and reaction.emoji == "🎙" and bot_state_manager.current_state == BotStateEnum.RECORDING and bot_state_manager.is_authorized(
        user)):

        voice_client = reaction.message.guild.voice_client

        if voice_client and hasattr(voice_client, "sink"):
            if voice_client.sink:
                pcm_data = voice_client.sink.audio_data
                voice_client.stop_listening()

                if pcm_data:

                    processed_audio = audio_manager.process_audio(pcm_data)
                    base64_audio = audio_manager.encode_to_base64(processed_audio)

                    await send_audio_events(base64_audio)
                else:
                    await reaction.message.channel.send("No audio data was captured.")

                await bot_state_manager.stop_recording()
            else:
                await reaction.message.channel.send("[debug] Audio sink is not available.")
        else:
            await reaction.message.channel.send("Recording was not started or no audio data was captured.")


@bot.command(name="connect")
async def join_voice_channel(ctx):
    global voice_client

    if ctx.author.voice is None:
        await ctx.send("You are not connected to a voice channel.")
        return

    voice_channel = ctx.author.voice.channel

    try:
        voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        await ctx.send(f"Connected to {voice_channel}")
        if voice_client and voice_client.is_connected():
            bot.loop.create_task(audio_manager.playback_loop(voice_client))
        else:
            await ctx.send("Bot is not connected to a voice channel.")

        """Command to initiate WebSocket connection and start event handling."""
        if websocket_manager.connection:
            await ctx.send("Already connected to the WebSocket server.")
            return

        await websocket_manager.start()
        asyncio.create_task(process_incoming_events())
        await ctx.send("Connected to WebSocket server")
        await queue_session_update()

    except discord.ClientException:
        await ctx.send("Already connected to a voice channel.")
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")


@bot.command(name="disconnect")
async def disconnect_bot(ctx):
    global voice_client

    if voice_client and voice_client.is_connected():
        await voice_client.disconnect()
        await ctx.send("Bot left voice channel and disconnected from realtime api.")

    if websocket_manager.connection:
        await websocket_manager.stop()
        await ctx.send("Disconnected from WebSocket server.")
    else:
        await ctx.send("No active WebSocket connection to disconnect.")


bot.run(Config.DISCORD_TOKEN)
