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

from typing import Optional, Union, Dict  # Added Dict
import discord
from discord.ext import commands, tasks

from src.audio.audio import AudioManager
from src.bot.voice_connection import VoiceConnectionManager
from src.config.config import Config  # Added import
from src.state.state import BotState, BotStateEnum
from src.utils.logger import get_logger

# from src.websocket.events.events import ( # These events are no longer created directly
#     SessionUpdateEvent,
#     InputAudioBufferAppendEvent,
#     InputAudioBufferCommitEvent,
#     ResponseCreateEvent,
#     ResponseCancelEvent,
# )
# from src.websocket.manager import WebSocketManager # Replaced by OpenAIRealtimeManager
# from src.openai_adapter.manager import OpenAIRealtimeManager # Replaced by IRealtimeAIServiceManager
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
        ai_service_managers: Dict[str, IRealtimeAIServiceManager],  # Changed parameter
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
            # Fallback or raise, for now, let's assume main.py ensures it exists.
            # If you want a more robust fallback:
            # self.active_ai_service_manager = next(iter(self.all_ai_service_managers.values()))
            # logger.warning(f"Using '{next(iter(self.all_ai_service_managers.keys()))}' as fallback AI provider.")
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
        # BotState.__init__ already sets the active_ai_provider_name from Config.
        # No standby message exists here to update yet.
        self._connection_check_loop.start()  # Start the background task

    def cog_unload(self):
        self._connection_check_loop.cancel()

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
            return False  # No connections are expected to be active in IDLE state

        issue_detected = False
        issue_description = ""

        # Check voice connection if bot is supposed to be in a voice-active state
        if current_bot_state in [BotStateEnum.STANDBY, BotStateEnum.RECORDING]:
            if not self.voice_connection.is_connected():
                issue_detected = True
                issue_description = "Voice connection lost."
                logger.warning(
                    "Connection check: Voice connection found to be inactive while in STANDBY or RECORDING state."
                )

        # Check AI service connection if voice is okay (or not applicable) and bot is not already in error for it
        if not issue_detected and not self.active_ai_service_manager.is_connected():
            if current_bot_state != BotStateEnum.CONNECTION_ERROR:
                issue_detected = True
                issue_description = "AI service connection lost."  # Generic message
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
            return True  # Issue was detected and resulted in a state change attempt

        return False  # No new issue detected that required a state change

    async def _process_and_send_audio(
        self, pcm_data: bytes, channel: discord.TextChannel
    ) -> None:
        """
        Processes recorded PCM audio and sends it to the AI service.

        This method:
        1. Sends the audio chunk.
        2. Finalizes the input and requests a response from the AI service.

        Args:
            pcm_data: Raw PCM audio data captured from the user.
            channel: The Discord text channel to send error messages to.
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
        if user == self.bot.user:  # Ignore reactions from the bot itself
            return

        if not (
            self.bot_state_manager.standby_message
            and reaction.message.id == self.bot_state_manager.standby_message.id
        ):  # Only process reactions on the standby message
            return

        if self.bot_state_manager.current_state == BotStateEnum.CONNECTION_ERROR:
            logger.debug("Reaction ignored: Bot is in CONNECTION_ERROR state.")
            # Optionally, send a message or remove the reaction if desired.
            return

        # Handle 🎙 reaction to start recording
        if (
            reaction.emoji == "🎙"
            and self.bot_state_manager.current_state == BotStateEnum.STANDBY
        ):
            if await self._check_and_handle_connection_issues(reaction.message.channel):
                return  # Connection issue detected, state changed to CONNECTION_ERROR

            # Before starting a new recording, ensure any ongoing audio playback is stopped
            # and notify the AI service to cancel any in-progress response generation.
            response_id_to_cancel = None
            guild = reaction.message.guild
            if guild and guild.voice_client and guild.voice_client.is_playing():
                logger.info(
                    "User initiated new recording. Stopping current audio playback."
                )
                response_id_to_cancel = (
                    self.audio_manager.get_current_playing_response_id()
                )
                guild.voice_client.stop()  # Stop discord.py audio playback

            # Send response.cancel event to the server, even if nothing was playing (server handles default)
            log_msg = (
                f"Sending response.cancel for response_id: {response_id_to_cancel}."
                if response_id_to_cancel
                else "Sending response.cancel for default in-progress response."
            )
            logger.info(log_msg)

            # Use the new manager's method.
            if not await self.active_ai_service_manager.cancel_ongoing_response():
                logger.error(
                    "Failed to send cancel_ongoing_response. Continuing with recording..."
                )

            # --- Orchestration Logic ---
            # 1. Change state to RECORDING
            if not await self.bot_state_manager.start_recording(user):
                # This can fail if the state is not STANDBY, which we already checked.
                # This is a safeguard.
                logger.warning(
                    f"State transition to RECORDING failed for user {user.name}."
                )
                return

            # 2. Start listening for audio
            if not self.voice_connection.start_listening():
                logger.error(
                    f"Failed to start listening for user {user.name}. Rolling back state."
                )
                # Rollback: If listening fails, revert state to STANDBY
                await self.bot_state_manager.stop_recording()
                await reaction.message.channel.send(
                    f"Sorry {user.mention}, I couldn't start recording due to a technical issue."
                )
                return

            logger.info(f"Successfully started recording for user {user.name} via reaction.")

        # Handle ❌ reaction to cancel recording
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
        This stops recording, processes audio, sends it to the AI service, and returns to STANDBY.

        Args:
            reaction: The reaction that was removed.
            user: The user who removed the reaction.
        """
        if user == self.bot.user:  # Ignore reactions from the bot itself
            return

        # Check if this is the 🎙 reaction being removed from the correct message,
        # by an authorized user, while in the RECORDING state.
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
            # If the state became CONNECTION_ERROR while recording, this ensures we don't proceed.
            # The background task or next interaction would handle the state.
            return

        if await self._check_and_handle_connection_issues(reaction.message.channel):
            # If a connection issue was found, state is now CONNECTION_ERROR.
            # We might still want to stop the local recording if it was active and voice connection is okay.
            if self.voice_connection.is_recording():  # Check if it was recording
                self.voice_connection.stop_listening()
                logger.info(
                    "Stopped listening due to connection issue detected during reaction removal."
                )
            return

        if not self.voice_connection.is_connected():
            logger.warning(
                "Voice connection not available during reaction_remove for recording (and not caught by _check_and_handle_connection_issues)."
            )
            await reaction.message.channel.send(
                "Bot is not in a voice channel or voice client is missing."
            )
            await self.bot_state_manager.stop_recording()
            return

        # --- Orchestration Logic ---
        # 1. Stop listening and get audio data
        pcm_data = self.voice_connection.stop_listening()
        logger.info("Stopped listening on reaction remove.")

        # 2. Transition state back to STANDBY
        await self.bot_state_manager.stop_recording()
        logger.info("State transitioned back to STANDBY.")

        # 3. Process and send the audio if any was captured
        if pcm_data:
            logger.debug(f"Raw PCM data size from Discord: {len(pcm_data)} bytes.")
            try:
                processed_pcm_data = (
                    await self.audio_manager.resample_and_convert_audio(pcm_data)
                )
                logger.debug(
                    f"Processed PCM data size: {len(processed_pcm_data)} bytes."
                )
                await self._process_and_send_audio(
                    processed_pcm_data, reaction.message.channel
                )
            except RuntimeError as e:
                logger.error(f"Error processing audio with ffmpeg: {e}")
                await reaction.message.channel.send(
                    "Error processing your audio. Please try again."
                )
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

        # --- Orchestration Sequence ---
        # 1. Connect to voice channel
        if not await self.voice_connection.connect_to_channel(voice_channel):
            await ctx.send("Failed to connect to the voice channel.")
            await self.bot_state_manager.enter_connection_error_state()
            return
        logger.info(f"Successfully connected to voice channel: {voice_channel.name}")

        # 2. Connect to AI Service
        if not self.active_ai_service_manager.is_connected():
            logger.info("Attempting to establish AI service connection...")
            if not await self.active_ai_service_manager.connect():
                await ctx.send("Failed to establish AI service connection.")
                await self.bot_state_manager.enter_connection_error_state()
                return
        logger.info("AI service connection is active.")

        # 3. Initialize Standby State
        if not await self.bot_state_manager.initialize_standby(ctx):
            # This case handles if the bot is already active.
            # The initialize_standby method sends its own message if it fails.
            logger.warning("initialize_standby failed, likely because bot is already active.")
            return

        logger.info(f"Connect command successful for {ctx.author.name}. Bot is in STANDBY.")

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

        # 1. Handle current AI service state
        if self.active_ai_service_manager.is_connected():
            logger.info(
                f"Disconnecting current AI provider: {self.bot_state_manager.active_ai_provider_name}"
            )
            # If recording, stop it first
            if self.bot_state_manager.current_state == BotStateEnum.RECORDING:
                if self.voice_connection.is_recording():
                    self.voice_connection.stop_listening()  # Stop hardware recording
                # No need to process audio, just stop the state
                await (
                    self.bot_state_manager.stop_recording()
                )  # This resets authority and updates message (to standby)
                logger.info("Stopped recording due to AI provider switch.")

            # Cancel any ongoing response from the old service
            await self.active_ai_service_manager.cancel_ongoing_response()
            # Disconnect the old service
            await self.active_ai_service_manager.disconnect()
            logger.info(
                f"Disconnected from {self.bot_state_manager.active_ai_provider_name}."
            )

        # 2. Switch to the new provider
        self.active_ai_service_manager = self.all_ai_service_managers[provider_name]
        await self.bot_state_manager.set_active_ai_provider_name(
            provider_name
        )  # This will update the message
        logger.info(f"Switched active AI service manager to {provider_name}.")

        # 3. Connect the new AI service if the bot is in a state that requires it
        #    (e.g., STANDBY, or was RECORDING, or in voice channel)
        #    The _connection_check_loop will also handle this, but we can be proactive.
        #    We only attempt to connect if the bot is in a voice channel (implied by STANDBY or was RECORDING).
        if self.bot_state_manager.current_state in [
            BotStateEnum.STANDBY,
            BotStateEnum.RECORDING,
            BotStateEnum.CONNECTION_ERROR,
        ]:
            if (
                self.voice_connection.is_connected()
            ):  # Check if bot is in a voice channel
                logger.info(f"Attempting to connect new AI provider: {provider_name}")
                if await self.active_ai_service_manager.connect():
                    logger.info(
                        f"Successfully connected to new AI provider: {provider_name}."
                    )
                    # If previously in CONNECTION_ERROR, try to recover to STANDBY
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

        # 4. Update standby message - This is now handled by set_active_ai_provider_name

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
        # --- Orchestration Sequence ---
        # 1. Reset bot state to IDLE
        await self.bot_state_manager.reset_to_idle()
        logger.info("Bot state has been reset to IDLE.")

        # 2. Disconnect from voice channel
        await self.voice_connection.disconnect()
        logger.info("Disconnected from voice channel.")

        # 3. Disconnect from AI service
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
        if (
            self.bot_state_manager.current_state
            not in [
                BotStateEnum.STANDBY,
                BotStateEnum.RECORDING,
                BotStateEnum.CONNECTION_ERROR,  # Also check if in error, to see if it can recover.
            ]
        ):
            return

        logger.debug("Periodic connection check running...")
        channel_for_message = None
        if self.bot_state_manager.standby_message:
            channel_for_message = self.bot_state_manager.standby_message.channel

        # If in CONNECTION_ERROR state, check if connections have recovered
        if self.bot_state_manager.current_state == BotStateEnum.CONNECTION_ERROR:
            if (
                self.voice_connection.is_connected()
                and self.active_ai_service_manager.is_connected()  # Check AI service
            ):
                logger.info(
                    "Connections appear to be restored while in CONNECTION_ERROR state. Attempting to recover to STANDBY."
                )
                if await self.bot_state_manager.recover_to_standby():
                    logger.info(
                        "Successfully recovered to STANDBY state from CONNECTION_ERROR."
                    )
                    # If recovery was successful, no need to run _check_and_handle_connection_issues
                    # as the state is no longer CONNECTION_ERROR for this iteration.
                    return
                else:
                    logger.warning(
                        "Failed to recover to STANDBY state. Standby message might be missing or state was not CONNECTION_ERROR."
                    )
            # else: Connections are still not okay, remain in CONNECTION_ERROR. Loop will check again.

        # For STANDBY or RECORDING, or if recovery from CONNECTION_ERROR failed, run the standard check.
        # This will also re-trigger CONNECTION_ERROR if one of the connections dropped again immediately after a failed recovery attempt.
        await self._check_and_handle_connection_issues(channel_for_message)

    @_connection_check_loop.before_loop
    async def before_connection_check_loop(self):
        await (
            self.bot.wait_until_ready()
        )  # Wait for the bot to be ready before starting the loop


async def setup(bot: commands.Bot):
    raise NotImplementedError(
        "VoiceCog requires dependencies (audio_manager, bot_state_manager, ai_service_manager) and cannot be loaded as a standard extension. "
        "Instantiate and add it manually in your main script."
    )
