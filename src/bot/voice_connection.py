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
from discord.ext import voice_recv

from src.audio.audio import AudioManager
from src.utils.logger import get_logger

# Configure logger for this module
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

    def __init__(self, bot, audio_manager: AudioManager, ):
        """
        Initialize the VoiceConnectionManager with required dependencies.
        
        Args:
            bot: The Discord bot instance
            audio_manager: Handles audio processing and playback
            websocket_manager: Manages WebSocket communication with external services
        """
        self.bot = bot
        self.audio_manager = audio_manager
        self.voice_client: Optional[voice_recv.VoiceRecvClient] = None
        self._playback_task: Optional[asyncio.Task] = None

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
                    # Move to the specified voice channel if connected to a different one
                    await self.voice_client.move_to(voice_channel)
                    logger.info(f"Moved to voice channel: {voice_channel.name}")
            else:
                # Connect to the voice channel if not already connected
                self.voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
                logger.info(f"Connected to voice channel: {voice_channel.name}")

            # Start the audio playback loop if connected
            if self.voice_client and self.voice_client.is_connected():
                if self._playback_task is None or self._playback_task.done():
                    self._playback_task = self.bot.loop.create_task(self.audio_manager.playback_loop(self.voice_client))
                    logger.info("Playback loop started.")
                return True
            return False
        except Exception as e:
            logger.error(f"Error connecting to voice channel: {e}")
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
            # Stop listening if active
            if self.voice_client.is_listening():
                self.voice_client.stop_listening()
                logger.info("Stopped listening.")

            # Cancel playback task if running
            if self._playback_task and not self._playback_task.done():
                self._playback_task.cancel()
                try:
                    await self._playback_task
                except asyncio.CancelledError:
                    logger.info("Playback loop task cancelled.")
                finally:
                    self._playback_task = None

            # Disconnect from voice channel
            await self.voice_client.disconnect()
            self.voice_client = None
            logger.info("Disconnected from voice channel.")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from voice channel: {e}")
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
        if not self.voice_client or not hasattr(self.voice_client, "sink") or not self.voice_client.sink:
            logger.warning("Cannot stop recording: No active recording or sink.")
            return bytes()

        try:
            # Get the recorded PCM data and stop listening
            pcm_data = bytes(self.voice_client.sink.audio_data)
            self.voice_client.stop_listening()
            logger.info("Stopped recording and retrieved audio data.")
            return pcm_data
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
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
        return (self.voice_client is not None and self.voice_client.is_connected() and self.voice_client.is_listening())
