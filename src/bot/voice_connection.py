"""
Voice Connection module for managing Discord voice channel connections.

This module provides the VoiceConnectionManager class which handles:
- Connecting to voice channels
- Disconnecting from voice channels
- Managing voice client instances
- Setting up audio recording and playback
"""

import asyncio
from typing import Optional

import discord
from discord.ext import commands, voice_recv

from src.audio.audio import AudioManager
from src.state.state import BotState
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VoiceConnectionManager:
    """
    Manages Discord voice channel connections and related functionality.

    This class handles:
    - Voice channel connections and disconnections
    - Voice client management
    - Audio recording setup
    - Playback task management
    """

    def __init__(
        self,
        bot: commands.Bot,
        audio_manager: AudioManager,
        bot_state_manager: BotState,
    ):
        """
        Initialize the VoiceConnectionManager with required dependencies.

        Args:
            bot: The Discord bot instance.
            audio_manager: Handles audio processing and playback.
            bot_state_manager: Manages the bot's state.
        """
        self.bot = bot
        self.audio_manager = audio_manager
        self.bot_state_manager = bot_state_manager
        self.voice_client: Optional[voice_recv.VoiceRecvClient] = None
        self._playback_task: Optional[asyncio.Task] = None

    async def establish_session(
        self, voice_channel: discord.VoiceChannel, ctx: commands.Context
    ) -> bool:
        """
        Handles the complete process of joining a voice channel and entering the STANDBY state.

        Args:
            voice_channel: The voice channel to connect to.
            ctx: The command context, used by BotState to send messages.

        Returns:
            bool: True if session establishment was successful, False otherwise.
        """
        logger.info(f"Attempting to establish session in voice channel: {voice_channel.name}")
        if not await self.connect_to_channel(voice_channel):
            logger.error(f"Failed to connect to voice channel {voice_channel.name} during session establishment.")
            return False

        if not await self.bot_state_manager.initialize_standby(ctx):
            logger.warning("Failed to initialize standby state. Bot might already be active.")
            # initialize_standby handles sending messages if already active.
            return False # Or True, depending on if partial success is okay. Plan implies False.

        logger.info(f"Session successfully established in voice channel: {voice_channel.name}")
        return True

    async def terminate_session(self) -> None:
        """
        Handles the complete process of disconnecting from voice and returning to the IDLE state.
        """
        logger.info("Attempting to terminate session.")
        await self.bot_state_manager.reset_to_idle()
        await self.disconnect()
        logger.info("Session terminated.")

    async def begin_recording(self, user: discord.User) -> bool:
        """
        Ensures bot is connected to the user's voice channel and starts recording.

        This method handles connecting or moving the bot if necessary, transitions
        the bot state to RECORDING, and activates the audio sink.

        Args:
            user: The Discord user initiating the recording.

        Returns:
            bool: True if recording started successfully, False otherwise.
        """
        logger.info(f"Attempting to begin recording for user: {user.name} ({user.id})")

        # 1. Ensure Voice Connection
        if user.voice and user.voice.channel:
            if not self.is_connected() or (
                self.voice_client and self.voice_client.channel != user.voice.channel
            ):
                logger.info(
                    f"Bot not connected or in a different channel. Connecting/moving to {user.voice.channel.name}."
                )
                if not await self.connect_to_channel(user.voice.channel):
                    logger.error(
                        f"Failed to connect/move to user's voice channel: {user.voice.channel.name}."
                    )
                    return False
        elif not self.is_connected(): # User not in a channel AND bot not connected
            logger.warning(
                f"User {user.name} is not in a voice channel, and bot is not connected. Cannot start recording."
            )
            return False
        # If user is not in a voice channel but bot is already connected, proceed with current channel.
        elif not user.voice or not user.voice.channel:
             logger.info(f"User {user.name} is not in a voice channel. Bot will record in its current channel: {self.voice_client.channel.name if self.voice_client else 'Unknown'}")


        # 2. Transition to Recording State
        if not await self.bot_state_manager.start_recording(user):
            logger.warning(
                f"Failed to transition bot state to RECORDING for user {user.name}. Current state: {self.bot_state_manager.current_state}"
            )
            return False
        logger.info(f"Bot state transitioned to RECORDING for user {user.name}.")

        # 3. Activate Audio Sink
        if not self.start_recording():
            logger.error("Failed to activate audio sink.")
            # 4. Rollback on Sink Failure
            logger.info("Rolling back bot state to STANDBY due to sink activation failure.")
            await self.bot_state_manager.stop_recording() # Reverts state and authority
            return False
        
        logger.info(f"Successfully started recording for user {user.name}.")
        return True

    async def finish_recording(self) -> bytes:
        """
        Stops recording, reverts the state to STANDBY, and returns the captured audio.

        Returns:
            bytes: The recorded PCM audio data.
        """
        logger.info("Attempting to finish recording.")
        pcm_data = self.stop_recording() # Stops hardware recording and gets data
        await self.bot_state_manager.stop_recording() # Transitions state back to STANDBY
        logger.info(f"Recording finished. PCM data length: {len(pcm_data)} bytes.")
        return pcm_data

    async def connect_to_channel(self, voice_channel: discord.VoiceChannel) -> bool:
        """
        Connect to a voice channel or move to it if already connected elsewhere.

        Args:
            voice_channel: The voice channel to connect to

        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            if self.voice_client and self.voice_client.is_connected():
                if self.voice_client.channel != voice_channel:
                    await self.voice_client.move_to(voice_channel)
                    logger.info(f"Moved to voice channel: {voice_channel.name}")
            else:
                self.voice_client = await voice_channel.connect(
                    cls=voice_recv.VoiceRecvClient  # Use custom client for receiving audio
                )
                logger.info(f"Connected to voice channel: {voice_channel.name}")

            # Ensure playback loop is running for this voice client
            if self.voice_client and self.voice_client.is_connected():
                if self._playback_task is None or self._playback_task.done():
                    logger.info("Starting new playback loop task.")
                    # --- Temporary Debug Prints ---
                    logger.debug(
                        f"VOICE_CONNECTION_DEBUG: Type of self.audio_manager: {type(self.audio_manager)}"
                    )
                    logger.debug(
                        f"VOICE_CONNECTION_DEBUG: Is self.audio_manager None? {self.audio_manager is None}"
                    )
                    if self.audio_manager:
                        logger.debug(
                            f"VOICE_CONNECTION_DEBUG: Attributes of self.audio_manager: {dir(self.audio_manager)}"
                        )
                        logger.debug(
                            f"VOICE_CONNECTION_DEBUG: Does it have playback_loop? {'playback_loop' in dir(self.audio_manager)}"
                        )
                    # --- End Temporary Debug Prints ---
                    self._playback_task = self.bot.loop.create_task(
                        self.audio_manager.playback_loop(self.voice_client)
                    )
                else:
                    logger.info("Playback loop task already running.")
                return True
            return False  # Should not happen if connect/move_to succeeded
        except Exception as e:
            logger.error(f"Error connecting to voice channel: {e}", exc_info=True)
            return False

    async def disconnect(self) -> bool:
        """
        Disconnect from the current voice channel and clean up resources.

        Returns:
            bool: True if disconnection was successful, False otherwise
        """
        if not self.voice_client or not self.voice_client.is_connected():
            logger.info("Not connected to a voice channel.")
            return False

        try:
            if self.voice_client.is_listening():  # Stop any active recording
                self.voice_client.stop_listening()
                logger.info("Stopped listening.")

            if (
                self._playback_task and not self._playback_task.done()
            ):  # Cancel audio playback loop
                self._playback_task.cancel()
                try:
                    await self._playback_task  # Allow task to process cancellation
                except asyncio.CancelledError:
                    logger.info("Playback loop task cancelled successfully.")
                except Exception as e_task:  # pragma: no cover
                    logger.error(f"Error during playback task cancellation: {e_task}")
                finally:
                    self._playback_task = None

            channel_name = "Unknown"
            if self.voice_client and self.voice_client.channel:
                channel_name = self.voice_client.channel.name

            await self.voice_client.disconnect()
            logger.info(f"Disconnected from voice channel: {channel_name}")
            self.voice_client = None
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from voice channel: {e}", exc_info=True)
            return False

    def start_recording(self) -> bool:
        """
        Start recording audio from the voice channel.

        Returns:
            bool: True if recording started successfully, False otherwise
        """
        if not self.voice_client or not self.voice_client.is_connected():
            logger.warning("Cannot start recording: Not connected to a voice channel.")
            return False

        try:
            if isinstance(self.voice_client, voice_recv.VoiceRecvClient):
                sink = self.audio_manager.create_sink()
                self.voice_client.listen(sink)
                logger.info("Started recording with new audio sink.")
                return True
            else:
                logger.warning("Voice client is not a VoiceRecvClient instance.")
                return False
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return False

    def stop_recording(self) -> bytes:
        """
        Stop recording audio and return the recorded data.

        Returns:
            bytes: The recorded PCM audio data, or empty bytes if no data was captured
        """
        if (
            not self.voice_client
            or not hasattr(self.voice_client, "sink")
            or not self.voice_client.sink
        ):
            logger.warning("Cannot stop recording: No active recording or sink.")
            return bytes()

        try:
            pcm_data = bytes(
                self.voice_client.sink.audio_data
            )  # Retrieve captured audio
            self.voice_client.stop_listening()
            logger.info(
                f"Stopped recording. Retrieved {len(pcm_data)} bytes of audio data."
            )
            # The sink itself will be cleaned up by discord.py when stop_listening is called
            # or when a new sink is started.
            return pcm_data
        except Exception as e:
            logger.error(f"Error stopping recording: {e}", exc_info=True)
            return bytes()

    def is_connected(self) -> bool:
        """
        Check if the bot is connected to a voice channel.

        Returns:
            bool: True if connected, False otherwise
        """
        return self.voice_client is not None and self.voice_client.is_connected()

    def is_recording(self) -> bool:
        """
        Check if the bot is currently recording audio.

        Returns:
            bool: True if recording, False otherwise
        """
        return (
            self.voice_client is not None
            and self.voice_client.is_connected()
            and self.voice_client.is_listening()
        )
