import asyncio
import base64

import discord
import websockets
from discord.ext import commands
from discord.ext import voice_recv
from websockets.asyncio.client import connect

from audio import AudioManager
from config import Config
from events import *
from state import BotState, BotStateEnum

audio_manager = AudioManager()

bot_state_manager = BotState()

ws_connection = None
voice_client = None

incoming_events = asyncio.Queue()
outgoing_events = asyncio.Queue()

intents = discord.Intents.all()

bot = commands.Bot(command_prefix="!", intents=intents)


async def handle_error(event: ErrorEvent):
    print(event.error)


async def handle_session_updated(event: SessionUpdatedEvent):
    print(f"handling {event.type}")


async def handle_session_created(event: SessionCreatedEvent):
    print(f"handling {event.type}")


async def handle_response_audio_delta(event):
    print(f"handling {event.type}")
    base64_audio = event.delta
    decoded_audio = base64.b64decode(base64_audio)

    audio_manager.extend_response_buffer(decoded_audio)
    print("Buffered audio fragment:", len(decoded_audio), "bytes")


async def handle_response_audio_done(event):
    if audio_manager.response_buffer:
        await audio_manager.enqueue_audio(audio_manager.response_buffer)
        audio_manager.clear_response_buffer()


EVENT_HANDLERS = {"error": handle_error, "session.created": handle_session_created,
                  "session.updated": handle_session_updated, "response.audio.delta": handle_response_audio_delta,
                  "response.audio.done": handle_response_audio_done, }


async def queue_session_update():
    event = SessionUpdatedEvent(event_id="event_123", type="session.update", session={"turn_detection": None})
    await outgoing_events.put(event)


async def ws_handler():
    """Establish and handle a WebSocket connection session within a 15-minute limit."""
    global ws_connection
    headers = {"Authorization": "Bearer " + Config.OPENAI_API_KEY, "OpenAI-Beta": "realtime=v1", }
    try:
        async with connect(Config.WS_SERVER_URL, additional_headers=headers) as websocket:
            ws_connection = websocket
            print("Connected to WebSocket server.")

            receive_task = asyncio.create_task(receive_events(websocket))
            print("receiving events through ws and save to queue")
            process_task = asyncio.create_task(process_incoming_event())
            send_task = asyncio.create_task(send_events(websocket))
            print("checking events in queue and send events through ws")

            await asyncio.wait([receive_task, process_task, send_task], timeout=Config.CONNECTION_TIMEOUT)

            if not receive_task.done() or not send_task.done():
                print("Connection timed out after 15 minutes. Reconnecting...")

    except websockets.ConnectionClosed:
        print("WebSocket connection closed. Reconnecting...")
    except Exception as e:
        print(f"Error in WebSocket connection: {e}. Reconnecting...")
    finally:
        print("ws connection failed to be recovered.")
        ws_connection = None
        await asyncio.sleep(1)
        asyncio.create_task(ws_handler())


async def process_incoming_event():
    while True:
        event = await incoming_events.get()
        print("Processing event in incoming queue:", event.type)

        event_type = event.type
        handler = EVENT_HANDLERS.get(event_type)
        try:
            if handler:
                await handler(event)
            else:
                print(f"No handler found for event type: {event_type}")
        except Exception as e:
            print(f"Error in handler for {event_type}: {e}")
        finally:
            incoming_events.task_done()


async def receive_events(websocket):
    """Handles receiving events from the WebSocket server and adding them to the incoming queue."""
    while True:
        message = await websocket.recv()
        data = json.loads(message)

        event = BaseEvent.from_json(data)

        if event is not None:
            print("Received event:", event.type)
            await incoming_events.put(event)
        else:
            print(f"handler for {data["type"]} not available")


async def send_events(websocket):
    """Handles sending events from the outgoing queue to the WebSocket server."""
    while True:
        event = await outgoing_events.get()
        try:
            await websocket.send(event.to_json())
            print("Sent event:", event.type)
        finally:
            outgoing_events.task_done()


async def send_audio_events(base64_audio: str):
    """Helper function to send audio-related events to the server."""

    data = {"event_id": "event_456", "type": "input_audio_buffer.append", "audio": base64_audio, }
    event = EVENT_TYPE_MAPPING["input_audio_buffer.append"].from_json(data)
    await outgoing_events.put(event)

    data = {"event_id": "event_789", "type": "input_audio_buffer.commit", }
    event = EVENT_TYPE_MAPPING["input_audio_buffer.commit"].from_json(data)
    await outgoing_events.put(event)

    data = {"event_id": "event_234", "type": "response.create"}
    event = EVENT_TYPE_MAPPING["response.create"].from_json(data)
    await outgoing_events.put(event)


@bot.command(name="listen")
async def start_standby(ctx):
    if await bot_state_manager.initialize_standby(ctx):
        return
    await ctx.send("Bot is already active in another state.")


@bot.command(name="-listen")
async def stop_listen(ctx):
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
                    print("Started new recording session with fresh sink")


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
    global voice_client, ws_connection

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
        if ws_connection:
            await ctx.send("Already connected to the WebSocket server.")
            return

        asyncio.create_task(ws_handler())
        await ctx.send("Connected to WebSocket server")
        await queue_session_update()

    except discord.ClientException:
        await ctx.send("Already connected to a voice channel.")
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")


@bot.command(name="disconnect")
async def disconnect_ws(ctx):
    global voice_client, ws_connection

    if voice_client and voice_client.is_connected():
        await voice_client.disconnect()
        await ctx.send("Bot left voice channel and disconnected from realtime api.")

    if ws_connection:
        await ws_connection.close()
        ws_connection = None

        await ctx.send("Disconnected from WebSocket server.")
    else:
        await ctx.send("No active WebSocket connection to disconnect.")


bot.run(Config.DISCORD_TOKEN)
