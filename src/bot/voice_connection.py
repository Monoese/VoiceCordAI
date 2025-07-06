"""
Voice Connection module for managing Discord voice channel connections.

This module provides the VoiceConnectionManager class which handles:
- Connecting to voice channels
- Disconnecting from voice channels
- Managing voice client instances
- Setting up audio recording and playback
"""

import discord
from discord.ext import voice_recv

from src.audio.playback import AudioPlaybackManager
from src.audio.recorder import create_sink
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VoiceConnectionManager:
    """
    Manages Discord voice channel connections and related functionality.

    This class is designed to be stateless regarding the voice client object.
    It retrieves the voice client directly from the guild object (`guild.voice_client`)
    to ensure it always has the authoritative and up-to-date instance from discord.py,
    preventing issues with stale references during reconnects.
    """

    def __init__(
        self,
        guild: discord.Guild,
        audio_playback_manager: AudioPlaybackManager,
    ) -> None:
        """
        Initialize the VoiceConnectionManager with required dependencies.

        Args:
            guild: The discord.Guild instance this manager is responsible for.
            audio_playback_manager: Handles audio playback for this guild.
        """
        self.guild = guild
        self.audio_playback_manager = audio_playback_manager

    def _get_voice_client(self) -> voice_recv.VoiceRecvClient | None:
        """Safely retrieves the voice client from the guild object."""
        # The return type of guild.voice_client is discord.VoiceClient | None.
        # We assume that if it's not None, it's the VoiceRecvClient we connected with.
        # A type check could be added here if other client types were used.
        vc = self.guild.voice_client
        if isinstance(vc, voice_recv.VoiceRecvClient):
            return vc
        return None

    async def connect_to_channel(self, voice_channel: discord.VoiceChannel) -> bool:
        """
        Connect to a voice channel or move to it if already connected elsewhere.

        Args:
            voice_channel: The voice channel to connect to.

        Returns:
            bool: True if connection was successful, False otherwise.
        """
        voice_client = self._get_voice_client()
        try:
            if voice_client and voice_client.is_connected():
                if voice_client.channel != voice_channel:
                    await voice_client.move_to(voice_channel)
                    logger.info(f"Moved to voice channel: {voice_channel.name}")
            else:
                # Let discord.py handle the state. It will set guild.voice_client.
                await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
                logger.info(f"Connected to voice channel: {voice_channel.name}")

            # After connecting/moving, get the authoritative client again to start playback.
            current_vc = self._get_voice_client()
            if current_vc and current_vc.is_connected():
                self.audio_playback_manager.start(current_vc)
                return True

            logger.warning(
                "Failed to get a valid voice client after connection attempt."
            )
            return False
        except Exception as e:
            logger.error(f"Error connecting to voice channel: {e}", exc_info=True)
            return False

    async def disconnect(self) -> bool:
        """
        Disconnect from the current voice channel and clean up resources.

        Returns:
            bool: True if disconnection was successful, False otherwise.
        """
        voice_client = self._get_voice_client()
        if not (voice_client and voice_client.is_connected()):
            logger.info("Not connected to a voice channel, no action needed.")
            return True  # Already disconnected, so the goal is achieved.

        channel_name = voice_client.channel.name if voice_client.channel else "Unknown"

        try:
            if voice_client.is_listening():
                voice_client.stop_listening()
                logger.info("Stopped listening.")

            await self.audio_playback_manager.stop()

            await voice_client.disconnect()
            logger.info(f"Disconnected from voice channel: {channel_name}")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from voice channel: {e}", exc_info=True)
            return False

    def start_listening(self) -> bool:
        """
        Start listening for audio from the voice channel by activating the sink.

        Returns:
            bool: True if listening started successfully, False otherwise.
        """
        voice_client = self._get_voice_client()
        if not (voice_client and voice_client.is_connected()):
            logger.warning("Cannot start listening: Not connected to a voice channel.")
            return False

        try:
            sink = create_sink()
            voice_client.listen(sink)
            logger.info("Started listening with new audio sink.")
            return True
        except Exception as e:
            logger.error(f"Error starting listening: {e}", exc_info=True)
            return False

    def stop_listening(self) -> bytes:
        """
        Stop listening for audio and return the captured data.

        Returns:
            bytes: The recorded PCM audio data, or empty bytes if no data was captured.
        """
        voice_client = self._get_voice_client()
        if not (voice_client and hasattr(voice_client, "sink") and voice_client.sink):
            logger.warning("Cannot stop listening: No active sink.")
            return bytes()

        try:
            pcm_data = bytes(voice_client.sink.audio_data)
            voice_client.stop_listening()
            logger.info(
                f"Stopped listening. Retrieved {len(pcm_data)} bytes of audio data."
            )
            return pcm_data
        except Exception as e:
            logger.error(f"Error stopping listening: {e}", exc_info=True)
            return bytes()

    def is_connected(self) -> bool:
        """
        Check if the bot is connected to a voice channel.

        Returns:
            bool: True if connected, False otherwise.
        """
        voice_client = self._get_voice_client()
        return voice_client is not None and voice_client.is_connected()

    def is_recording(self) -> bool:
        """
        Check if the bot is currently recording audio.

        Returns:
            bool: True if recording, False otherwise.
        """
        voice_client = self._get_voice_client()
        return (
            voice_client is not None
            and voice_client.is_connected()
            and voice_client.is_listening()
        )
