import json
import asyncio
import base64
import json
import os
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from typing import Dict, Any

import discord
import websockets
from discord import FFmpegPCMAudio
from discord.ext import commands
from discord.ext import voice_recv
from dotenv import load_dotenv
from pydub import AudioSegment
from websockets.asyncio.client import connect

# configure paths for this project
BASE_DIR = Path(__file__).resolve().parent

# load .env file
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# set constants
WS_SERVER_URL = ("wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01")
CONNECTION_TIMEOUT = 15 * 60
CHUNK_DURATION_MS = 500  # Duration of each chunk in milliseconds

# set variables
ws_connection = None  # Placeholder for the WebSocket connection
voice_client = None
standby_message = None
authority_user = "anyone"
bot_state = "idle"

# standby message content
standby_message_content = f"**🎙 Voice Recording Bot - **{bot_state}** Mode**\n\nHere’s how to control the bot:\n---\n### 🔄 How to Use:\n1. **Start Recording**: React to this message with 🎙 to start recording.\n2. **Stop Recording**: Remove your 🎙 reaction to pause recording.\n4. **Finish Session**: Use `!-listen` to end the session and return the bot to Idle Mode.\n---\n### 🛠 Current Status:\n- **Recording Status**: `{bot_state}`\n---\n### 🧑 Authority User:\n> `{authority_user}` can control the recording actions."


async def update_standby_message_content():
    global bot_state, authority_user, standby_message_content
    standby_message_content = f"**🎙 Voice Recording Bot - **{bot_state}** Mode**\n\nHere’s how to control the bot:\n---\n### 🔄 How to Use:\n1. **Start Recording**: React to this message with 🎙 to start recording.\n2. **Stop Recording**: Remove your 🎙 reaction to pause recording.\n4. **Finish Session**: Use `!-listen` to end the session and return the bot to Idle Mode.\n---\n### 🛠 Current Status:\n- **Recording Status**: `{bot_state}`\n---\n### 🧑 Authority User:\n> `{authority_user}` can control the recording actions."


# input audio
audio_buffer = bytearray()

# output audio
audio_queue = asyncio.Queue()

# Event queues for events
incoming_events = asyncio.Queue()
outgoing_events = asyncio.Queue()

# define intents of bot
intents = discord.Intents.all()

# initialize bot
bot = commands.Bot(command_prefix="!", intents=intents)


# audio sink class for audio recording
class MyPCM16Sink(voice_recv.AudioSink):
    def __init__(self):
        super().__init__()
        self.audio_data = bytearray()  # Buffer to store audio data

    def wants_opus(self) -> bool:
        # We want PCM, so we set this to False
        return False

    def write(self, user, data):
        # Confirm that this method is being called and receiving PCM data
        if data.pcm:
            print("Writing data for user:", user)  # Debugging: show user info
            print("Received PCM data length:", len(data.pcm))  # Debugging: check length
            self.audio_data.extend(data.pcm)  # Append PCM data to the buffer
        else:
            print("No PCM data received")  # Debugging: check for missing data

    def cleanup(self):
        # Called after recording is stopped to perform any final actions
        pass


# dataclass for events
@dataclass
class BaseEvent:
    event_id: str
    type: str

    def to_json(self) -> str:
        """Convert the event to a JSON string for sending."""
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(data: dict):
        event_type = data.get("type")
        if event_type in EVENT_TYPE_MAPPING:
            print("constructing", event_type)
            return EVENT_TYPE_MAPPING[event_type](**data)
        else:
            print(f"Unhandled event type: {event_type}")
            return None


# Define specific event types with their expected structure
@dataclass
class SessionUpdateEvent(BaseEvent):
    session: Dict[str, Any]


@dataclass
class InputAudioBufferAppendEvent(BaseEvent):
    audio: str


@dataclass
class InputAudioBufferCommitEvent(BaseEvent):
    pass


@dataclass
class SessionUpdatedEvent(BaseEvent):
    session: Dict[str, Any]


@dataclass
class SessionCreatedEvent(BaseEvent):
    session: Dict[str, Any]


@dataclass
class ConversationItemCreateEvent(BaseEvent):
    item: Dict[str, Any]


@dataclass
class ResponseCreateEvent(BaseEvent):
    pass


@dataclass
class ResponseAudioDeltaEvent(BaseEvent):
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


@dataclass
class ResponseAudioDoneEvent(BaseEvent):
    response_id: str
    item_id: str
    output_index: int
    content_index: int


@dataclass
class ErrorEvent(BaseEvent):
    error: Dict[str, Any]


# mapping between event name strings and event class
EVENT_TYPE_MAPPING = {# client events
    "session.update": SessionUpdateEvent, "conversation.item.create": ConversationItemCreateEvent,
    "input_audio_buffer.append": InputAudioBufferAppendEvent, "input_audio_buffer.commit": InputAudioBufferCommitEvent,
    "response.create": ResponseCreateEvent, # server events
    "session.created": SessionCreatedEvent, "session.updated": SessionUpdatedEvent,
    "response.audio.delta": ResponseAudioDeltaEvent, "response.audio.done": ResponseAudioDoneEvent,
    "error": ErrorEvent, }


# Event handler functions
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

    audio_buffer.extend(decoded_audio)
    print("Buffered audio fragment:", len(decoded_audio), "bytes")


async def handle_response_audio_done(event):
    global audio_buffer

    # All audio deltas have been processed, so we can now enqueue the complete audio buffer
    if audio_buffer:
        # Enqueue the audio data for playback
        await enqueue_audio(audio_buffer)

        # Clear audio_buffer for the next set of audio deltas
        audio_buffer.clear()


# mapping between event name strings and event handlers. only for incoming events.
EVENT_HANDLERS = {# server events only
    "error": handle_error, "session.created": handle_session_created, "session.updated": handle_session_updated,
    "response.audio.delta": handle_response_audio_delta, "response.audio.done": handle_response_audio_done, }


async def audio_playback_loop(voice_client):
    while True:
        # Wait until there’s audio in the queue
        audio_buffer = await audio_queue.get()

        # Use FFmpegPCMAudio to play the audio in Discord
        audio_source = FFmpegPCMAudio(audio_buffer, pipe=True)
        voice_client.play(audio_source, after=lambda e: print("Finished playing audio"))

        # Wait for the audio to finish playing
        while voice_client.is_playing():
            await asyncio.sleep(0.1)

        audio_queue.task_done()  # Mark the audio as processed


# audio codec
def process_audio(data):
    # Convert raw PCM data to an AudioSegment for easy processing
    audio_segment = AudioSegment(data=data, sample_width=2,  # 16-bit audio
        frame_rate=96000,  # Discord default
        channels=1,  # Mono
    )

    # Resample to 24kHz
    audio_segment = audio_segment.set_frame_rate(24000)
    print(f"Processed audio sample rate: {audio_segment.frame_rate}")  # Should output 24000

    # Export as raw 16-bit PCM data
    return audio_segment.raw_data


# base64 encoding
def encode_audio_to_base64(pcm_data):
    return base64.b64encode(pcm_data).decode("utf-8")


async def enqueue_audio(audio_buffer):
    # Convert raw PCM data to an AudioSegment
    audio_segment = AudioSegment(data=bytes(audio_buffer), sample_width=2,  # 16-bit PCM
        frame_rate=24000,  # Original sample rate of the audio
        channels=1,  # Mono
    )

    # Resample to 48kHz and stereo for Discord compatibility
    audio_segment = audio_segment.set_frame_rate(48000).set_channels(2)

    # Export to in-memory buffer in Opus format
    opus_buffer = BytesIO()
    audio_segment.export(opus_buffer, format="ogg", codec="libopus")
    opus_buffer.seek(0)  # Reset buffer position to the beginning

    # Enqueue the audio
    await audio_queue.put(opus_buffer)


async def queue_session_update():
    event = SessionUpdatedEvent(event_id="event_123", type="session.update", session={"turn_detection": None})
    await outgoing_events.put(event)


async def ws_handler():
    """Establish and handle a WebSocket connection session within a 15-minute limit."""
    global ws_connection
    headers = {"Authorization": "Bearer " + OPENAI_API_KEY, "OpenAI-Beta": "realtime=v1", }
    try:
        async with connect(WS_SERVER_URL, additional_headers=headers) as websocket:
            ws_connection = websocket  # Store active WebSocket connection
            print("Connected to WebSocket server.")

            # Run send and receive tasks with a timeout
            receive_task = asyncio.create_task(receive_events(websocket))
            print("receiving events through ws and save to queue")
            process_task = asyncio.create_task(process_incoming_event())
            send_task = asyncio.create_task(send_events(websocket))
            print("checking events in queue and send events through ws")

            # Wait for both tasks or timeout to expire
            await asyncio.wait([receive_task, process_task, send_task], timeout=CONNECTION_TIMEOUT)

            # Check if timeout reached
            if not receive_task.done() or not send_task.done():
                print("Connection timed out after 15 minutes. Reconnecting...")

    except websockets.ConnectionClosed:
        print("WebSocket connection closed. Reconnecting...")
    except Exception as e:
        print(f"Error in WebSocket connection: {e}. Reconnecting...")
    finally:
        print("ws connection failed to be recovered.")
        ws_connection = None
        await asyncio.sleep(1)  # Short delay before reconnection
        asyncio.create_task(ws_handler())  # Automatically reconnect after timeout


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
            incoming_events.task_done()  # Ensure task_done is always called


async def receive_events(websocket):
    """Handles receiving events from the WebSocket server and adding them to the incoming queue."""
    while True:
        message = await websocket.recv()
        data = json.loads(message)

        event = BaseEvent.from_json(data)

        if event is not None:
            print("Received event:", event.type)
            await incoming_events.put(event)  # Queue incoming event for processing
        else:
            print(f"handler for {data["type"]} not available")


async def send_events(websocket):
    """Handles sending events from the outgoing queue to the WebSocket server."""
    while True:
        event = await outgoing_events.get()  # Wait for an event to send
        try:
            await websocket.send(event.to_json())
            print("Sent event:", event.type)
        finally:
            outgoing_events.task_done()  # Ensure task_done is always called


# command for bot to join a voice channel the called is inside of
@bot.command(name="listen")
async def start_standby(ctx):
    global bot_state, standby_message

    if bot_state == "idle":
        bot_state = "standby"

        # Send standby message and add reaction for user to trigger recording
        standby_message = await ctx.send(standby_message_content)
        await standby_message.add_reaction("🎙")  # Add a specific emoji for reaction

    else:
        await ctx.send("Bot is already active in another state.")


@bot.command(name="-listen")
async def stop_listen(ctx):
    global bot_state, standby_message, authority_user

    if bot_state == "standby" or bot_state == "recording":
        bot_state = "idle"
        await standby_message.delete()
        standby_message = None
        authority_user = "anyone"

    else:
        await ctx.send("Bot is already in idle state.")


@bot.event
async def on_reaction_add(reaction, user):
    global bot_state, authority_user, standby_message, standby_message_content

    # Ignore reactions from the bot itself
    if user == bot.user:
        return

    if (bot_state == "standby" and reaction.message.id == standby_message.id and reaction.emoji == "🎙"):
        # User has reacted to start recording
        bot_state = "recording"
        authority_user = user.name

        # Start recording logic here (e.g., ctx.voice_client.listen(sink))
        voice_client = reaction.message.guild.voice_client
        print("on_reaction_add. got voice client: ", voice_client)
        if voice_client and isinstance(voice_client, voice_recv.VoiceRecvClient):
            sink = MyPCM16Sink()  # Initialize the custom sink
            voice_client.listen(sink)  # Start listening with the sink
            voice_client.sink = (sink  # Explicitly assign the sink to be accessible later
            )

            await update_standby_message_content()
            await standby_message.edit(
                content=standby_message_content)  # Make sure to set up `ctx.voice_client.sink` with your recording sink

    elif (bot_state == "recording" and reaction.message.id == standby_message.id and reaction.emoji == "❌"):
        # If user reacts with a cancel emoji, stop recording and reset to standby
        # await stop_recording()
        bot_state = "standby"
        await reaction.message.channel.send(f"{user.display_name} canceled recording. Returning to standby.")


@bot.event
async def on_reaction_remove(reaction, user):
    global bot_state, authority_user, outgoing_events, standby_message, standby_message_content

    # Ignore reactions from the bot itself
    if user == bot.user:
        return

    if (
            bot_state == "recording" and reaction.message.id == standby_message.id and reaction.emoji == "🎙" and user.name == authority_user):
        # If the recording reaction is removed, stop recording and return to standby
        # stop recording and send the event to server
        voice_client = reaction.message.guild.voice_client
        # Check if the bot is connected and has a sink
        if voice_client and hasattr(voice_client, "sink"):
            print("Sink exists on voice_client:", voice_client.sink)  # Debugging
            if voice_client.sink:  # Further check if sink is not None
                pcm_data = (voice_client.sink.audio_data)  # Access audio data from the sink
                voice_client.stop_listening()  # Stop listening

                # Check if any audio data was captured
                if pcm_data:
                    print("Audio data length:", len(pcm_data))  # Debugging: check data length
                    processed_audio = process_audio(pcm_data)  # Resample and format
                    base64_audio = encode_audio_to_base64(processed_audio)  # Encode to Base64

                    # initialize events and put them into the outgoing event queue
                    # input_audio_buffer.append
                    data = {"event_id": "event_456", "type": "input_audio_buffer.append", "audio": base64_audio, }
                    event = EVENT_TYPE_MAPPING["input_audio_buffer.append"].from_json(data)
                    await outgoing_events.put(event)

                    # input_audio_buffer.commit
                    data = {"event_id": "event_789", "type": "input_audio_buffer.commit", }

                    event = EVENT_TYPE_MAPPING["input_audio_buffer.commit"].from_json(data)
                    await outgoing_events.put(event)

                    # response.create
                    data = {"event_id": "event_234", "type": "response.create"}

                    event = EVENT_TYPE_MAPPING["response.create"].from_json(data)
                    await outgoing_events.put(event)

                else:
                    print("No audio data captured.")  # Debugging: empty data
                    await reaction.message.channel.send("No audio data was captured.")

                # reset variables
                bot_state = "standby"
                authority_user = "anyone"

                # edit standby_message in text channel
                await update_standby_message_content()
                await standby_message.edit(content=standby_message_content)

                # clear leftover audio data in sink
                voice_client.sink.cleanup()

            else:
                print("Sink was None")  # Debugging
                await reaction.message.channel.send("[debug] Audio sink is not avialable.")
        else:
            print("No voice_client or sink not set")  # Debugging
            await reaction.message.channel.send(
                "Recording was not started or no audio data was captured.")  # stopped recording and sent the event to server


@bot.command(name="connect")
async def join_voice_channel(ctx):
    global voice_client, ws_connection
    # Check if the user is in a voice channel
    if ctx.author.voice is None:
        await ctx.send("You are not connected to a voice channel.")
        return

    # get user voice channel
    voice_channel = ctx.author.voice.channel

    # connect to the voice channel and return the connection object
    try:
        # connect to voice channel
        voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        await ctx.send(f"Connected to {voice_channel}")
        if voice_client and voice_client.is_connected():
            # Start the audio playback loop in the background
            bot.loop.create_task(audio_playback_loop(voice_client))
        else:
            await ctx.send("Bot is not connected to a voice channel.")

        # connect to ws
        """Command to initiate WebSocket connection and start event handling."""
        if ws_connection:
            await ctx.send("Already connected to the WebSocket server.")
            return

        asyncio.create_task(ws_handler())
        await ctx.send("Connected to WebSocket server")
        await queue_session_update()  # send session update event to turn vad off

    except discord.ClientException:
        await ctx.send("Already connected to a voice channel.")
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")


@bot.command(name="disconnect")
async def disconnect_ws(ctx):
    global voice_client, ws_connection

    # leave voice channel
    if voice_client and voice_client.is_connected():
        await voice_client.disconnect()
        await ctx.send("Bot left voice channel and disconnected from realtime api.")

    # disconnect from realtime api ws connection
    if ws_connection:
        await ws_connection.close()
        ws_connection = None

        await ctx.send("Disconnected from WebSocket server.")
    else:
        await ctx.send("No active WebSocket connection to disconnect.")


@bot.command(name="sayhi")
async def say_hi(ctx):
    # create user message
    data = {"event_id": "evt_stPvKNm765mXF1x3F", "type": "conversation.item.create",
        "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}], }, }
    event = EVENT_TYPE_MAPPING["conversation.item.create"].from_json(data)
    await outgoing_events.put(event)

    # ask server to create response
    data = {"event_id": "evt_stPvKNm765mXF1x3F", "type": "response.create"}
    event = EVENT_TYPE_MAPPING["response.create"].from_json(data)
    await outgoing_events.put(event)


bot.run(DISCORD_TOKEN)
