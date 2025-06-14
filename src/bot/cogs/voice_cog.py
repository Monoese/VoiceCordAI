"""
Voice Cog module for Discord bot voice interaction functionality.

This module provides the VoiceCog class which handles all voice-related commands and events:
- Connecting to voice channels
- Recording user audio
- Processing audio through real-time AI services
- Playing back audio responses
- Managing the bot's state during voice interactions

The cog uses reaction-based controls to start and stop recording, making it user-friendly
in Discord servers.
"""

import asyncio
from typing import Optional, Union, Dict

import discord
from discord.ext import commands, tasks

from src.audio.audio import AudioManager
from src.bot.voice_connection import VoiceConnectionManager
from src.config.config import Config
from src.state.state import BotState, BotStateEnum
from src.utils.logger import get_logger
from src.ai_services.interface import IRealtimeAIServiceManager


logger = get_logger(__name__)


class VoiceCog(commands.Cog):
    """
    Discord Cog that handles voice channel interactions and audio processing.

    This cog manages:
    - Voice channel connections
    - Audio recording from users
    - Sending recorded audio to the AI service
    - Playing back audio responses
    - State transitions between idle, standby, and recording states

    The cog uses reaction-based controls (🎙️, ❌) for user interaction.
    """

    def __init__(
        self,
        bot: commands.Bot,
        audio_manager: AudioManager,
        bot_state_manager: BotState,
        ai_service_managers: Dict[str, IRealtimeAIServiceManager],
    ):
        """
        Initialize the VoiceCog with required dependencies.

        Args:
            bot: The Discord bot instance this cog is attached to
            audio_manager: Handles audio processing and playback
            bot_state_manager: Manages the bot's state transitions
            ai_service_managers: A dictionary of available AI service managers,
                                 keyed by provider name (e.g., "openai", "gemini").
        """
        self.bot = bot
        self.audio_manager = audio_manager
        self.bot_state_manager = bot_state_manager
        self.all_ai_service_managers: Dict[str, IRealtimeAIServiceManager] = (
            ai_service_managers
        )

        # Set the active AI service manager based on config
        default_provider = Config.AI_SERVICE_PROVIDER
        if default_provider not in self.all_ai_service_managers:
            # This should ideally be caught in main.py, but as a safeguard
            logger.error(
                f"Default AI provider '{default_provider}' not found in provided managers. Falling back to first available or raising error."
            )
            raise ValueError(
                f"Default AI provider '{default_provider}' not found in available managers."
            )

        self.active_ai_service_manager: IRealtimeAIServiceManager = (
            self.all_ai_service_managers[default_provider]
        )
        logger.info(f"VoiceCog initialized with active AI service: {default_provider}")

        self.voice_connection = VoiceConnectionManager(
            bot=bot,
            audio_manager=audio_manager,
        )
        self.background_tasks = set()
        self._connection_check_loop.start()

    def cog_unload(self):
        """
        Clean up resources when the cog is unloaded, including the connection
        check loop and any running background audio processing tasks.
        """
        self._connection_check_loop.cancel()
        # Cancel any running audio processing tasks to ensure graceful shutdown
        for task in self.background_tasks:
            task.cancel()
        logger.info(
            f"Cog unloaded. Cancelled {len(self.background_tasks)} background tasks."
        )

    async def _check_and_handle_connection_issues(
        self,
        ctx_or_channel: Optional[Union[commands.Context, discord.TextChannel]] = None,
    ) -> bool:
        """
        Checks critical connections and transitions to CONNECTION_ERROR state if issues are found.

        Args:
            ctx_or_channel: Context or TextChannel to send an error message if an issue is detected.

        Returns:
            bool: True if an issue was detected and state was changed, False otherwise.
        """
        current_bot_state = self.bot_state_manager.current_state
        if current_bot_state == BotStateEnum.IDLE:
            return False

        issue_detected = False
        issue_description = ""

        if current_bot_state in [BotStateEnum.STANDBY, BotStateEnum.RECORDING]:
            if not self.voice_connection.is_connected():
                issue_detected = True
                issue_description = "Voice connection lost."
                logger.warning(
                    "Connection check: Voice connection found to be inactive while in STANDBY or RECORDING state."
                )

        if not issue_detected and not self.active_ai_service_manager.is_connected():
            if current_bot_state != BotStateEnum.CONNECTION_ERROR:
                issue_detected = True
                issue_description = "AI service connection lost."
                logger.warning(
                    "Connection check: AI service connection found to be inactive."
                )

        if issue_detected:
            logger.info(
                f"Connection issue detected: {issue_description}. Transitioning to CONNECTION_ERROR state."
            )
            state_changed = await self.bot_state_manager.enter_connection_error_state()
            if state_changed and ctx_or_channel:
                try:
                    target_channel = (
                        ctx_or_channel
                        if isinstance(ctx_or_channel, discord.TextChannel)
                        else ctx_or_channel.channel
                    )
                    await target_channel.send(
                        f"Critical error: {issue_description} Bot functionality may be limited. Current state: {self.bot_state_manager.current_state.value}"
                    )
                except Exception as e:
                    logger.error(f"Failed to send connection error message: {e}")
            return True

        return False

    async def _process_and_send_audio(
        self, pcm_data: bytes, channel: discord.TextChannel
    ) -> None:
        """
        Sends a processed audio chunk to the AI service and finalizes the turn.

        Args:
            pcm_data: Processed (e.g., resampled) PCM audio data.
            channel: The Discord text channel for sending error messages.
        """
        if not await self.active_ai_service_manager.send_audio_chunk(pcm_data):
            logger.error("Failed to send audio chunk to AI service.")
            await channel.send(
                "Sorry, I encountered an error sending your audio to the AI service. My systems might be temporarily unavailable."
            )
            await self.bot_state_manager.enter_connection_error_state()
            return

        if not await self.active_ai_service_manager.finalize_input_and_request_response():
            logger.error(
                "Failed to finalize input and request response from AI service."
            )
            await channel.send(
                "Sorry, I encountered an error requesting a response from the AI service. My systems might be temporarily unavailable."
            )
            await self.bot_state_manager.enter_connection_error_state()
            return
        logger.info("Successfully sent audio and requested response from AI service.")

    async def _process_and_dispatch_audio(
        self, pcm_data: bytes, channel: discord.TextChannel
    ):
        """
        A background task to process raw audio and send it to the AI service.
        This function is designed to be run via asyncio.create_task() to avoid
        blocking the main event loop.

        Args:
            pcm_data: Raw PCM audio data from the voice connection.
            channel: The text channel for sending status or error messages.
        """
        logger.debug(f"Starting background audio processing for {len(pcm_data)} bytes.")
        try:
            processed_pcm_data = await self.audio_manager.resample_and_convert_audio(
                pcm_data
            )
            logger.debug(
                f"Resampling complete. Processed data size: {len(processed_pcm_data)} bytes."
            )
            await self._process_and_send_audio(processed_pcm_data, channel)
        except asyncio.CancelledError:
            logger.info("Audio processing task was cancelled.")
        except RuntimeError as e:
            logger.error(f"Error processing audio with ffmpeg: {e}", exc_info=True)
            await channel.send("Error processing your audio. Please try again.")
        except Exception as e:
            logger.error(
                f"An unexpected error occurred in the audio processing task: {e}",
                exc_info=True,
            )
            await channel.send(
                "An unexpected error occurred while processing your audio."
            )

    @commands.Cog.listener()
    async def on_reaction_add(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """
        Event listener for when a reaction is added to a message.

        Handles:
        - 🎙 reaction on standby message: Starts recording if in STANDBY state.
          If audio is playing, it stops playback and requests cancellation of any ongoing AI response.
        - ❌ reaction on standby message: Cancels recording if in RECORDING state.

        Ensures the bot is connected to a voice channel for recording.

        Args:
            reaction: The reaction that was added.
            user: The user who added the reaction.
        """
        if user == self.bot.user:
            return

        if not (
            self.bot_state_manager.standby_message
            and reaction.message.id == self.bot_state_manager.standby_message.id
        ):
            return

        if self.bot_state_manager.current_state == BotStateEnum.CONNECTION_ERROR:
            logger.debug("Reaction ignored: Bot is in CONNECTION_ERROR state.")
            return

        if (
            reaction.emoji == "🎙"
            and self.bot_state_manager.current_state == BotStateEnum.STANDBY
        ):
            if await self._check_and_handle_connection_issues(reaction.message.channel):
                return

            response_id_to_cancel = None
            guild = reaction.message.guild
            if guild and guild.voice_client and guild.voice_client.is_playing():
                logger.info(
                    "User initiated new recording. Stopping current audio playback."
                )
                response_id_to_cancel = (
                    self.audio_manager.get_current_playing_response_id()
                )
                guild.voice_client.stop()

            log_msg = (
                f"Sending response.cancel for response_id: {response_id_to_cancel}."
                if response_id_to_cancel
                else "Sending response.cancel for default in-progress response."
            )
            logger.info(log_msg)

            if not await self.active_ai_service_manager.cancel_ongoing_response():
                logger.error(
                    "Failed to send cancel_ongoing_response. Continuing with recording..."
                )

            if not await self.bot_state_manager.start_recording(user):
                logger.warning(
                    f"State transition to RECORDING failed for user {user.name}."
                )
                return

            if not self.voice_connection.start_listening():
                logger.error(
                    f"Failed to start listening for user {user.name}. Rolling back state."
                )
                await self.bot_state_manager.stop_recording()
                await reaction.message.channel.send(
                    f"Sorry {user.mention}, I couldn't start recording due to a technical issue."
                )
                return

            logger.info(
                f"Successfully started recording for user {user.name} via reaction."
            )

        elif (
            reaction.emoji == "❌"
            and self.bot_state_manager.current_state == BotStateEnum.RECORDING
            and self.bot_state_manager.is_authorized(user)
        ):
            if await self.bot_state_manager.stop_recording():
                if self.voice_connection.is_recording():
                    self.voice_connection.stop_listening()
                    logger.info(
                        "Stopped listening due to cancellation. Attempting to cancel AI response."
                    )
                    if not await self.active_ai_service_manager.cancel_ongoing_response():
                        logger.error(
                            "Failed to request cancellation of ongoing response during user cancellation."
                        )
                await reaction.message.channel.send(
                    f"{user.display_name} canceled recording. Returning to standby."
                )

    @commands.Cog.listener()
    async def on_reaction_remove(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """
        Event listener for when a reaction is removed from a message.

        Handles removal of 🎙 reaction from the standby message while in RECORDING state.
        This stops recording and dispatches the audio processing to a background task,
        making the UI immediately responsive.

        Args:
            reaction: The reaction that was removed.
            user: The user who removed the reaction.
        """
        if user == self.bot.user:
            return

        if not (
            self.bot_state_manager.standby_message
            and reaction.message.id == self.bot_state_manager.standby_message.id
            and reaction.emoji == "🎙"
            and self.bot_state_manager.current_state == BotStateEnum.RECORDING
            and self.bot_state_manager.is_authorized(user)
        ):
            return

        if self.bot_state_manager.current_state == BotStateEnum.CONNECTION_ERROR:
            logger.debug("Reaction removal ignored: Bot is in CONNECTION_ERROR state.")
            return

        if await self._check_and_handle_connection_issues(reaction.message.channel):
            if self.voice_connection.is_recording():
                self.voice_connection.stop_listening()
                logger.info(
                    "Stopped listening due to connection issue detected during reaction removal."
                )
            return

        if not self.voice_connection.is_connected():
            logger.warning(
                "Voice connection not available during reaction_remove for recording."
            )
            await reaction.message.channel.send(
                "Bot is not in a voice channel or voice client is missing."
            )
            await self.bot_state_manager.stop_recording()
            return

        pcm_data = self.voice_connection.stop_listening()
        logger.info("Stopped listening on reaction remove.")

        await self.bot_state_manager.stop_recording()
        logger.info("State transitioned back to STANDBY.")

        if pcm_data:
            task = asyncio.create_task(
                self._process_and_dispatch_audio(pcm_data, reaction.message.channel)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        else:
            await reaction.message.channel.send("No audio data was captured.")

    @commands.command(name="connect")
    async def connect_command(self, ctx: commands.Context) -> None:
        """
        Command to connect the bot to a voice channel, establish AI service connection, and enter standby mode.

        This command orchestrates the entire connection sequence:
        1. Connects to the user's voice channel.
        2. Connects to the configured AI service.
        3. Transitions the bot to STANDBY state and creates the control message.

        Args:
            ctx: The command context containing information about the invocation
        """
        if ctx.author.voice is None:
            await ctx.send("You are not connected to a voice channel.")
            return

        voice_channel = ctx.author.voice.channel

        if not await self.voice_connection.connect_to_channel(voice_channel):
            await ctx.send("Failed to connect to the voice channel.")
            await self.bot_state_manager.enter_connection_error_state()
            return
        logger.info(f"Successfully connected to voice channel: {voice_channel.name}")

        if not self.active_ai_service_manager.is_connected():
            logger.info("Attempting to establish AI service connection...")
            if not await self.active_ai_service_manager.connect():
                await ctx.send("Failed to establish AI service connection.")
                await self.bot_state_manager.enter_connection_error_state()
                return
        logger.info("AI service connection is active.")

        if not await self.bot_state_manager.initialize_standby(ctx):
            logger.warning(
                "initialize_standby failed, likely because bot is already active."
            )
            return

        logger.info(
            f"Connect command successful for {ctx.author.name}. Bot is in STANDBY."
        )

    @commands.command(name="set")
    async def set_provider_command(self, ctx: commands.Context, provider_name: str):
        """
        Command to dynamically switch the AI service provider.
        Usage: /set <name> (e.g., openai, gemini)
        """
        provider_name = provider_name.lower()
        if provider_name not in self.all_ai_service_managers:
            valid_providers = ", ".join(self.all_ai_service_managers.keys())
            await ctx.send(
                f"Invalid provider name '{provider_name}'. Valid options are: {valid_providers}."
            )
            return

        if self.bot_state_manager.active_ai_provider_name == provider_name:
            await ctx.send(
                f"AI provider is already set to '{provider_name.upper()}'. No change made."
            )
            return

        logger.info(f"Attempting to switch AI provider to '{provider_name}'.")

        if self.active_ai_service_manager.is_connected():
            logger.info(
                f"Disconnecting current AI provider: {self.bot_state_manager.active_ai_provider_name}"
            )
            if self.bot_state_manager.current_state == BotStateEnum.RECORDING:
                if self.voice_connection.is_recording():
                    self.voice_connection.stop_listening()
                await self.bot_state_manager.stop_recording()
                logger.info("Stopped recording due to AI provider switch.")

            await self.active_ai_service_manager.cancel_ongoing_response()
            await self.active_ai_service_manager.disconnect()
            logger.info(
                f"Disconnected from {self.bot_state_manager.active_ai_provider_name}."
            )

        self.active_ai_service_manager = self.all_ai_service_managers[provider_name]
        await self.bot_state_manager.set_active_ai_provider_name(provider_name)
        logger.info(f"Switched active AI service manager to {provider_name}.")

        if self.bot_state_manager.current_state in [
            BotStateEnum.STANDBY,
            BotStateEnum.RECORDING,
            BotStateEnum.CONNECTION_ERROR,
        ]:
            if self.voice_connection.is_connected():
                logger.info(f"Attempting to connect new AI provider: {provider_name}")
                if await self.active_ai_service_manager.connect():
                    logger.info(
                        f"Successfully connected to new AI provider: {provider_name}."
                    )
                    if (
                        self.bot_state_manager.current_state
                        == BotStateEnum.CONNECTION_ERROR
                    ):
                        if await self.bot_state_manager.recover_to_standby():
                            logger.info(
                                "Recovered to STANDBY state after AI provider switch."
                            )
                        else:
                            logger.warning(
                                "Could not recover to STANDBY after AI provider switch, standby message might be missing."
                            )
                else:
                    logger.error(
                        f"Failed to connect to new AI provider: {provider_name}."
                    )
                    await self.bot_state_manager.enter_connection_error_state()
                    await ctx.send(
                        f"Failed to connect to '{provider_name.upper()}'. Bot is in an error state."
                    )
            else:
                logger.info(
                    "Bot is not in a voice channel, new AI provider will connect when bot joins a channel."
                )

    @commands.command(name="disconnect")
    async def disconnect_command(self, ctx: commands.Context) -> None:
        """
        Command to disconnect the bot from voice channel, stop AI service connection, and return to idle state.

        This command orchestrates the entire disconnection sequence:
        1. Resets the bot to IDLE state.
        2. Disconnects from the voice channel.
        3. Disconnects from the AI service.

        Args:
            ctx: The command context containing information about the invocation
        """
        await self.bot_state_manager.reset_to_idle()
        logger.info("Bot state has been reset to IDLE.")

        await self.voice_connection.disconnect()
        logger.info("Disconnected from voice channel.")

        if self.active_ai_service_manager.is_connected():
            try:
                logger.info("Stopping AI service connection...")
                await self.active_ai_service_manager.disconnect()
            except Exception as e:
                logger.error(f"Failed to disconnect from AI service: {e}")
                await ctx.send("Error stopping AI service connection.")

        await ctx.send("Session terminated. Bot is now idle and disconnected.")

    @tasks.loop(seconds=10.0)
    async def _connection_check_loop(self):
        """
        Periodically checks the health of critical connections (voice, AI service).

        If the bot is in STANDBY or RECORDING state, it verifies connections and
        transitions to CONNECTION_ERROR if issues are found.
        If the bot is already in CONNECTION_ERROR state, it checks if connections
        have recovered and attempts to transition back to STANDBY.
        """
        if self.bot_state_manager.current_state not in [
            BotStateEnum.STANDBY,
            BotStateEnum.RECORDING,
            BotStateEnum.CONNECTION_ERROR,
        ]:
            return

        logger.debug("Periodic connection check running...")
        channel_for_message = None
        if self.bot_state_manager.standby_message:
            channel_for_message = self.bot_state_manager.standby_message.channel

        if self.bot_state_manager.current_state == BotStateEnum.CONNECTION_ERROR:
            if (
                self.voice_connection.is_connected()
                and self.active_ai_service_manager.is_connected()
            ):
                logger.info(
                    "Connections appear to be restored while in CONNECTION_ERROR state. Attempting to recover to STANDBY."
                )
                if await self.bot_state_manager.recover_to_standby():
                    logger.info(
                        "Successfully recovered to STANDBY state from CONNECTION_ERROR."
                    )
                    return
                else:
                    logger.warning(
                        "Failed to recover to STANDBY state. Standby message might be missing or state was not CONNECTION_ERROR."
                    )

        await self._check_and_handle_connection_issues(channel_for_message)

    @_connection_check_loop.before_loop
    async def before_connection_check_loop(self):
        await self.bot.wait_until_ready()


async def setup(bot: commands.Bot):
    raise NotImplementedError(
        "VoiceCog requires dependencies (audio_manager, bot_state_manager, ai_service_manager) and cannot be loaded as a standard extension. "
        "Instantiate and add it manually in your main script."
    )
