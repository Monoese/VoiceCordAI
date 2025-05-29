"""
Voice Cog module for Discord bot voice interaction functionality.

This module provides the VoiceCog class which handles all voice-related commands and events:
- Connecting to voice channels
- Recording user audio
- Processing audio through WebSocket services
- Playing back audio responses
- Managing the bot's state during voice interactions

The cog uses reaction-based controls to start and stop recording, making it user-friendly
in Discord servers.
"""

import uuid

import discord
from discord.ext import commands

from src.audio.audio import AudioManager
from src.bot.voice_connection import VoiceConnectionManager
from src.state.state import BotState, BotStateEnum
from src.utils.logger import get_logger
from src.websocket.events.events import (
    SessionUpdateEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
)
from src.websocket.manager import WebSocketManager

# Configure logger for this module
logger = get_logger(__name__)


class VoiceCog(commands.Cog):
    """
    Discord Cog that handles voice channel interactions and audio processing.

    This cog manages:
    - Voice channel connections
    - Audio recording from users
    - Sending recorded audio to external services via WebSocket
    - Playing back audio responses
    - State transitions between idle, standby, and recording states

    The cog uses reaction-based controls (🎙️, ❌) for user interaction.
    """

    def __init__(
        self,
        bot: commands.Bot,
        audio_manager: AudioManager,
        bot_state_manager: BotState,
        websocket_manager: WebSocketManager,
    ):
        """
        Initialize the VoiceCog with required dependencies.

        Args:
            bot: The Discord bot instance this cog is attached to
            audio_manager: Handles audio processing and playback
            bot_state_manager: Manages the bot's state transitions
            websocket_manager: Manages WebSocket communication with external services
        """
        self.bot = bot
        self.audio_manager = audio_manager
        self.bot_state_manager = bot_state_manager
        self.websocket_manager = websocket_manager

        # Create voice connection manager
        self.voice_connection = VoiceConnectionManager(
            bot=bot,
            audio_manager=audio_manager,
        )

    async def _queue_session_update(self) -> None:
        """
        Send a session update event to the WebSocket server.

        This method creates and sends a session.update event with turn_detection set to None,
        which configures the session for proper audio handling.
        """
        event = SessionUpdateEvent(
            event_id=f"event_{uuid.uuid4()}",
            type="session.update",
            session={"turn_detection": None},
        )
        success = await self.websocket_manager.safe_send_event(event)
        if not success:
            logger.error("Failed to send session update event")

    async def _send_audio_events(self, base64_audio: str) -> None:
        """
        Send a sequence of audio-related events to the WebSocket server.

        This method sends three events in sequence:
        1. input_audio_buffer.append - Adds the base64-encoded audio to the input buffer
        2. input_audio_buffer.commit - Commits the audio buffer for processing
        3. response.create - Requests a response based on the committed audio

        Args:
            base64_audio: Base64-encoded audio data to send to the server
        """
        append_event = InputAudioBufferAppendEvent(
            event_id=f"event_{uuid.uuid4()}",
            type="input_audio_buffer.append",
            audio=base64_audio,
        )
        if not await self.websocket_manager.safe_send_event(append_event):
            logger.error("Failed to send audio append event")
            return

        commit_event = InputAudioBufferCommitEvent(
            event_id=f"event_{uuid.uuid4()}",
            type="input_audio_buffer.commit",
        )
        if not await self.websocket_manager.safe_send_event(commit_event):
            logger.error("Failed to send audio commit event")
            return

        response_create_event = ResponseCreateEvent(
            event_id=f"event_{uuid.uuid4()}",
            type="response.create",
        )
        if not await self.websocket_manager.safe_send_event(response_create_event):
            logger.error("Failed to send response create event")
            return

    @commands.Cog.listener()
    async def on_reaction_add(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """
        Event listener for when a reaction is added to a message.

        This listener handles two main reaction scenarios:
        1. 🎙 reaction on standby message - Starts recording if in STANDBY state.
           If audio is playing, it stops playback and sends a response.cancel event.
        2. ❌ reaction on standby message - Cancels recording if in RECORDING state.

        The method ensures the bot is connected to a voice channel for recording
        and handles appropriate error scenarios.

        Args:
            reaction: The reaction that was added
            user: The user who added the reaction
        """
        # Ignore reactions from the bot itself
        if user == self.bot.user:
            return

        # Only process reactions on the standby message
        if (
            self.bot_state_manager.standby_message
            and reaction.message.id == self.bot_state_manager.standby_message.id
        ):
            # Handle 🎙 reaction to start recording
            if (
                reaction.emoji == "🎙"
                and self.bot_state_manager.current_state == BotStateEnum.STANDBY
            ):
                response_id_to_cancel = None
                guild = reaction.message.guild
                if guild and guild.voice_client and guild.voice_client.is_playing():
                    logger.info(
                        "User initiated new recording. Stopping current audio playback."
                    )
                    # Get the response_id of the stream being played by AudioManager
                    response_id_to_cancel = self.audio_manager.get_current_playing_response_id()
                    guild.voice_client.stop() # Stop discord.py audio playback

                # Send response.cancel event to the server
                cancel_event = ResponseCancelEvent(
                    event_id=f"event_{uuid.uuid4()}",
                    type="response.cancel",
                    response_id=response_id_to_cancel # This can be None if nothing was playing or ID couldn't be parsed
                )
                if response_id_to_cancel:
                    logger.info(f"Sending response.cancel event for response_id: {response_id_to_cancel}.")
                else:
                    logger.info("Sending response.cancel event for default in-progress response.")
                
                if not await self.websocket_manager.safe_send_event(cancel_event):
                    logger.error(
                        "Failed to send response.cancel event. Continuing with recording..."
                    )

                if await self.bot_state_manager.start_recording(user):
                    # Case 1: Bot is already in a voice channel in this guild
                    if reaction.message.guild and reaction.message.guild.voice_client:
                        self.voice_connection.voice_client = (
                            reaction.message.guild.voice_client
                        )
                        if not self.voice_connection.start_recording():
                            logger.warning(
                                "Failed to start recording with existing voice client."
                            )
                            await self.bot_state_manager.stop_recording()
                            return
                    # Case 2: User is in a voice channel, bot needs to connect
                    elif user.voice and user.voice.channel:
                        try:
                            logger.info(
                                f"User {user.name} is in voice channel {user.voice.channel.name}. Bot connecting."
                            )
                            if not await self.voice_connection.connect_to_channel(
                                user.voice.channel
                            ):
                                logger.error("Failed to connect to voice channel.")
                                await self.bot_state_manager.stop_recording()
                                await reaction.message.channel.send(
                                    "Could not join your voice channel to start recording."
                                )
                                return

                            if not self.voice_connection.start_recording():
                                logger.error(
                                    "Failed to start recording after connecting."
                                )
                                await self.bot_state_manager.stop_recording()
                                await reaction.message.channel.send(
                                    "Could not start recording in your voice channel."
                                )
                                return

                            logger.info(
                                "Connected to voice and started new recording session."
                            )
                        except Exception as e:
                            logger.error(
                                f"Error connecting to voice channel for recording: {e}"
                            )
                            await self.bot_state_manager.stop_recording()
                            await reaction.message.channel.send(
                                "Could not join your voice channel to start recording."
                            )
                            return
                    # Case 3: Neither bot nor user is in a voice channel
                    else:
                        logger.warning(
                            "Bot is not connected to a voice channel in this guild, and user is not in a voice channel."
                        )
                        await self.bot_state_manager.stop_recording()
                        await reaction.message.channel.send(
                            "You need to be in a voice channel, or the bot needs to be in one, to start recording."
                        )
                        return

            # Handle ❌ reaction to cancel recording
            elif (
                reaction.emoji == "❌"
                and self.bot_state_manager.current_state == BotStateEnum.RECORDING
                and self.bot_state_manager.is_authorized(user)
            ):
                if await self.bot_state_manager.stop_recording():
                    if self.voice_connection.is_recording():
                        self.voice_connection.stop_recording()
                        logger.info("Stopped listening due to cancellation.")
                    await reaction.message.channel.send(
                        f"{user.display_name} canceled recording. Returning to standby."
                    )

    @commands.Cog.listener()
    async def on_reaction_remove(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """
        Event listener for when a reaction is removed from a message.

        This listener specifically handles when a user removes the 🎙 reaction
        from the standby message while in RECORDING state. This action:
        1. Stops the recording
        2. Processes the captured audio
        3. Sends the audio to the WebSocket server
        4. Returns the bot to STANDBY state

        Args:
            reaction: The reaction that was removed
            user: The user who removed the reaction
        """
        # Ignore reactions from the bot itself
        if user == self.bot.user:
            return

        # Check if this is a 🎙 reaction removal on the standby message during recording
        if (
            self.bot_state_manager.standby_message
            and reaction.message.id == self.bot_state_manager.standby_message.id
            and reaction.emoji == "🎙"
            and self.bot_state_manager.current_state == BotStateEnum.RECORDING
            and self.bot_state_manager.is_authorized(user)
        ):
            # Verify voice connection is available and recording
            if not self.voice_connection.is_connected():
                logger.warning(
                    "Voice connection not available during reaction_remove for recording."
                )
                await reaction.message.channel.send(
                    "Bot is not in a voice channel or voice client is missing."
                )
                await self.bot_state_manager.stop_recording()
                return

            # Process recorded audio if available
            if self.voice_connection.is_recording():
                pcm_data = self.voice_connection.stop_recording()
                logger.info("Stopped listening on reaction remove.")

                if pcm_data:
                    processed_audio = await self.audio_manager.ffmpeg_to_24k_mono(
                        pcm_data
                    )
                    base64_audio = self.audio_manager.encode_to_base64(processed_audio)
                    await self._send_audio_events(base64_audio)
                else:
                    await reaction.message.channel.send("No audio data was captured.")

                await self.bot_state_manager.stop_recording()
            else:
                logger.warning("No active recording found during reaction_remove.")
                await reaction.message.channel.send(
                    "Recording was not properly started or no audio data was captured."
                )
                await self.bot_state_manager.stop_recording()

    @commands.command(name="connect")
    async def connect_command(self, ctx: commands.Context) -> None:
        """
        Command to connect the bot to a voice channel, establish WebSocket connection, and enter standby mode.

        This command:
        1. Connects to the user's current voice channel (or moves if already connected elsewhere).
        2. Ensures the audio playback loop is running for handling responses.
        3. Establishes a WebSocket connection if not already active.
        4. Transitions the bot to STANDBY state, ready for voice input.

        Args:
            ctx: The command context containing information about the invocation
        """
        # Check if the user is in a voice channel
        if ctx.author.voice is None:
            await ctx.send("You are not connected to a voice channel.")
            return

        voice_channel = ctx.author.voice.channel

        if not await self.voice_connection.connect_to_channel(voice_channel):
            await ctx.send("Failed to connect to the voice channel.")
            return

        # Connect to the WebSocket server if not already connected
        if self.websocket_manager.connected:
            await ctx.send("Already connected to the WebSocket server.")
        else:
            try:
                # Use ensure_connected to wait for the connection to be fully established
                if await self.websocket_manager.ensure_connected():
                    await self._queue_session_update()
                else:
                    await ctx.send(
                        "Failed to establish WebSocket connection within timeout"
                    )
                    return # Don't proceed to initialize standby if WebSocket connection failed
            except Exception as e:
                await ctx.send(f"Failed to connect to WebSocket server: {e}")
                return # Don't proceed to initialize standby if WebSocket connection failed

        # Initialize standby mode
        if not await self.bot_state_manager.initialize_standby(ctx):
            await ctx.send("Bot is already active in another state, but connection steps were performed if applicable.")

    @commands.command(name="disconnect")
    async def disconnect_command(self, ctx: commands.Context) -> None:
        """
        Command to disconnect the bot from voice channel, stop WebSocket connection, and return to idle state.

        This command:
        1. Resets the bot to IDLE state, stopping any active recording/listening.
        2. Disconnects from the voice channel, stopping audio playback.
        3. Stops the WebSocket connection.

        Args:
            ctx: The command context containing information about the invocation
        """
        # Reset bot state to idle first
        if not await self.bot_state_manager.reset_to_idle():
            await ctx.send("Bot was already in idle state or could not be reset.")

        if not await self.voice_connection.disconnect():
            await ctx.send("Bot is not in a voice channel.")
            # If not in a voice channel, it might still be connected to WebSocket

        # Stop WebSocket connection if connected
        if self.websocket_manager.connected:
            try:
                await self.websocket_manager.stop()
            except Exception as e:
                logger.error(f"Failed to disconnect from WebSocket server: {e}")


async def setup(bot: commands.Bot):
    raise NotImplementedError(
        "VoiceCog requires dependencies (audio_manager, bot_state_manager, websocket_manager) and cannot be loaded as a standard extension. "
        "Instantiate and add it manually in your main script."
    )
