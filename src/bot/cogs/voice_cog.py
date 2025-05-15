import asyncio
import uuid
from typing import Optional

import discord
from discord.ext import commands
from discord.ext import voice_recv

from src.audio.audio import AudioManager
from src.state.state import BotState, BotStateEnum
from src.utils.logger import get_logger
from src.websocket.events.events import EVENT_TYPE_MAPPING, SessionUpdatedEvent
from src.websocket.manager import WebSocketManager

logger = get_logger(__name__)


class VoiceCog(commands.Cog):
    def __init__(self, bot: commands.Bot, audio_manager: AudioManager, bot_state_manager: BotState,
                 websocket_manager: WebSocketManager):
        self.bot = bot
        self.audio_manager = audio_manager
        self.bot_state_manager = bot_state_manager
        self.websocket_manager = websocket_manager
        self.voice_client: Optional[voice_recv.VoiceRecvClient] = None
        self._playback_task: Optional[asyncio.Task] = None

    async def _queue_session_update(self) -> None:
        event = SessionUpdatedEvent(event_id=f"event_{uuid.uuid4()}", type="session.update",
                                    session={"turn_detection": None})
        await self.websocket_manager.send_event(event)

    async def _send_audio_events(self, base64_audio: str) -> None:
        """Helper function to send audio-related events to the server."""

        append_event_data = {"event_id": f"event_{uuid.uuid4()}", "type": "input_audio_buffer.append",
                             "audio": base64_audio}
        append_event = EVENT_TYPE_MAPPING["input_audio_buffer.append"](**append_event_data)
        await self.websocket_manager.send_event(append_event)

        commit_event_data = {"event_id": f"event_{uuid.uuid4()}", "type": "input_audio_buffer.commit"}
        commit_event = EVENT_TYPE_MAPPING["input_audio_buffer.commit"](**commit_event_data)
        await self.websocket_manager.send_event(commit_event)

        response_create_data = {"event_id": f"event_{uuid.uuid4()}", "type": "response.create"}
        response_create_event = EVENT_TYPE_MAPPING["response.create"](**response_create_data)
        await self.websocket_manager.send_event(response_create_event)

    @commands.command(name="listen")
    async def listen_command(self, ctx: commands.Context) -> None:
        if await self.bot_state_manager.initialize_standby(ctx):
            return
        await ctx.send("Bot is already active in another state.")

    @commands.command(name="-listen")
    async def stop_listen_command(self, ctx: commands.Context) -> None:
        if await self.bot_state_manager.reset_to_idle():
            return
        await ctx.send("Bot is already in idle state.")

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User) -> None:
        if user == self.bot.user:
            return

        if self.bot_state_manager.standby_message and reaction.message.id == self.bot_state_manager.standby_message.id:
            if reaction.emoji == "🎙" and self.bot_state_manager.current_state == BotStateEnum.STANDBY:
                if await self.bot_state_manager.start_recording(user):
                    if reaction.message.guild and reaction.message.guild.voice_client:
                        self.voice_client = reaction.message.guild.voice_client
                        if isinstance(self.voice_client, voice_recv.VoiceRecvClient):
                            sink = self.audio_manager.create_sink()
                            self.voice_client.listen(sink)
                            logger.info("Started new recording session with fresh sink using existing voice client.")
                        else:
                            logger.warning("Voice client is not a VoiceRecvClient instance.")
                    elif user.voice and user.voice.channel:
                        try:
                            logger.info(
                                f"User {user.name} is in voice channel {user.voice.channel.name}. Bot connecting.")
                            self.voice_client = await user.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
                            if isinstance(self.voice_client, voice_recv.VoiceRecvClient):
                                sink = self.audio_manager.create_sink()
                                self.voice_client.listen(sink)
                                logger.info("Connected to voice and started new recording session.")
                            else:
                                logger.error("Failed to connect as VoiceRecvClient.")
                        except Exception as e:
                            logger.error(f"Error connecting to voice channel for recording: {e}")
                            await self.bot_state_manager.stop_recording()
                            await reaction.message.channel.send("Could not join your voice channel to start recording.")
                            return
                    else:
                        logger.warning(
                            "Bot is not connected to a voice channel in this guild, and user is not in a voice channel.")
                        await self.bot_state_manager.stop_recording()
                        await reaction.message.channel.send(
                            "You need to be in a voice channel, or the bot needs to be in one, to start recording.")
                        return


            elif (
                    reaction.emoji == "❌" and self.bot_state_manager.current_state == BotStateEnum.RECORDING and self.bot_state_manager.is_authorized(
                user)):
                if await self.bot_state_manager.stop_recording():
                    if self.voice_client and self.voice_client.is_listening():
                        self.voice_client.stop_listening()
                        logger.info("Stopped listening due to cancellation.")
                    await reaction.message.channel.send(
                        f"{user.display_name} canceled recording. Returning to standby.")

    @commands.Cog.listener()
    async def on_reaction_remove(self, reaction: discord.Reaction, user: discord.User) -> None:
        if user == self.bot.user:
            return

        if (
                self.bot_state_manager.standby_message and reaction.message.id == self.bot_state_manager.standby_message.id and reaction.emoji == "🎙" and self.bot_state_manager.current_state == BotStateEnum.RECORDING and self.bot_state_manager.is_authorized(
            user)):

            if reaction.message.guild and reaction.message.guild.voice_client:
                self.voice_client = reaction.message.guild.voice_client
            else:
                logger.warning("Voice client not found in guild during reaction_remove for recording.")
                await reaction.message.channel.send("Bot is not in a voice channel or voice client is missing.")
                await self.bot_state_manager.stop_recording()
                return

            if self.voice_client and hasattr(self.voice_client, "sink") and self.voice_client.sink:
                pcm_data = bytes(self.voice_client.sink.audio_data)
                self.voice_client.stop_listening()
                logger.info("Stopped listening on reaction remove.")

                if pcm_data:
                    processed_audio = self.audio_manager.process_audio(pcm_data)
                    base64_audio = self.audio_manager.encode_to_base64(processed_audio)
                    await self._send_audio_events(base64_audio)
                else:
                    await reaction.message.channel.send("No audio data was captured.")
                await self.bot_state_manager.stop_recording()
            else:
                logger.warning("Audio sink not available or voice client missing during reaction_remove.")
                await reaction.message.channel.send("Recording was not properly started or no audio data was captured.")
                await self.bot_state_manager.stop_recording()

    @commands.command(name="connect")
    async def connect_command(self, ctx: commands.Context) -> None:
        if ctx.author.voice is None:
            await ctx.send("You are not connected to a voice channel.")
            return

        voice_channel = ctx.author.voice.channel

        if self.voice_client and self.voice_client.is_connected():
            if self.voice_client.channel != voice_channel:
                try:
                    await self.voice_client.move_to(voice_channel)
                    await ctx.send(f"Moved to {voice_channel.name}")
                except Exception as e:
                    await ctx.send(f"Error moving to voice channel: {e}")
                    return
            else:
                await ctx.send("Already connected to this voice channel.")
        else:
            try:
                self.voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
                await ctx.send(f"Connected to {voice_channel.name}")
            except discord.DiscordException as e:
                await ctx.send(f"Already connected to a voice channel or failed to connect: : {str(e)}")
                return
            except Exception as e:
                await ctx.send(f"An error occurred during connection: {str(e)}")
                return

        if self.voice_client and self.voice_client.is_connected():
            if self._playback_task is None or self._playback_task.done():
                self._playback_task = self.bot.loop.create_task(self.audio_manager.playback_loop(self.voice_client))
                logger.info("Playback loop started.")
            else:
                logger.info("Playback loop already running.")
        else:
            await ctx.send("Bot is not connected to a voice channel.")
            return

        if self.websocket_manager.connected:
            await ctx.send("Already connected to the WebSocket server.")
        else:
            try:
                await self.websocket_manager.start()
                await ctx.send("Connected to WebSocket server")
                await self._queue_session_update()
            except Exception as e:
                await ctx.send(f"Failed to connect to WebSocket server: {e}")

    @commands.command(name="disconnect")
    async def disconnect_command(self, ctx: commands.Context) -> None:
        if self.voice_client and self.voice_client.is_connected():
            if self.voice_client.is_listening():
                self.voice_client.stop_listening()

            if self._playback_task and not self._playback_task.done():
                self._playback_task.cancel()
                try:
                    await self._playback_task
                except asyncio.CancelledError:
                    logger.info("Playback loop task cancelled.")
                finally:
                    self._playback_task = None

            await self.voice_client.disconnect()
            self.voice_client = None
            await ctx.send("Bot left voice channel.")
        else:
            await ctx.send("Bot is not in a voice channel.")

        if self.websocket_manager.connected:
            try:
                await self.websocket_manager.stop()
                await ctx.send("Disconnected from WebSocket server.")
            except Exception as e:
                logger.error(f"Failed to disconnect from WebSocket server: {e}")


async def setup(bot: commands.Bot):
    # need to be defined if used as extension
    pass
