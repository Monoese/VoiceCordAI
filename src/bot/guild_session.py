"""
Guild Session module for managing per-guild bot state and interactions.

This module defines the GuildSession class, which encapsulates all the logic,
state, and resources for the bot's operation within a single Discord guild. This
ensures that the bot can operate in multiple guilds simultaneously without any
state conflicts.
"""

import asyncio
from typing import Any, Dict, Optional, Set, Union

import discord
import openai
from discord.ext import commands, tasks
from google.genai import errors as gemini_errors

from src.ai_services.interface import IRealtimeAIServiceManager
from src.audio.playback import AudioPlaybackManager
from src.audio.processor import process_recorded_audio
from src.bot.voice_connection import VoiceConnectionManager
from src.config.config import Config
from src.state.state import BotState, BotStateEnum
from src.utils.logger import get_logger


logger = get_logger(__name__)


class GuildSession:
    """
    Manages all state and logic for the bot's interaction within a single guild.

    This class encapsulates all components required for a voice session, including
    state management, audio playback, voice connection, and AI service interaction.
    Each guild will have its own instance of this class, ensuring complete isolation.
    """

    def __init__(
        self,
        guild: discord.Guild,
        bot: commands.Bot,
        ai_service_factories: Dict[str, tuple],
    ):
        """
        Initializes a new session for a specific guild.

        Args:
            guild: The Discord guild this session belongs to.
            bot: The Discord bot instance.
            ai_service_factories: A dictionary of factories for creating AI service managers.
        """
        self.guild = guild
        self.bot = bot
        self.ai_service_factories = ai_service_factories

        # Guild-specific instances of core components
        self.bot_state = BotState()
        self.audio_playback_manager = AudioPlaybackManager()
        self.voice_connection = VoiceConnectionManager(bot, self.audio_playback_manager)

        # State attributes migrated from VoiceCog
        self.active_ai_service_manager: Optional[IRealtimeAIServiceManager] = None
        self._standby_message: Optional[discord.Message] = None
        self._update_queue: asyncio.Queue[None] = asyncio.Queue()
        self._ui_updater_task: Optional[asyncio.Task[None]] = None
        self._background_tasks: Set[asyncio.Task[Any]] = set()
        self._connection_check_loop_task: Optional[asyncio.Task[None]] = None
        self._action_lock = asyncio.Lock()

    async def start_background_tasks(self) -> None:
        """Starts all persistent background tasks for the session."""
        self.start_updater_task()
        self._connection_check_loop.start()
        logger.info(f"Background tasks started for guild {self.guild.id}.")

    async def cleanup(self) -> None:
        """
        Gracefully shuts down the session, ensuring each cleanup step is attempted.

        This includes stopping background tasks, disconnecting from voice,
        and shutting down the AI service connection. Each step is wrapped in a
        try/except block to ensure that a failure in one part does not prevent
        others from executing.
        """
        logger.info(f"Cleaning up session for guild {self.guild.id}")
        self._connection_check_loop.cancel()
        await self.stop_updater_task()

        # Cancel any running audio processing tasks
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        logger.info(
            f"Cleaned up {len(self._background_tasks)} background tasks for guild {self.guild.id}."
        )

        try:
            await self._shutdown_current_ai_provider()
        except Exception as e:
            logger.error(
                f"Error during AI provider shutdown for guild {self.guild.id}: {e}",
                exc_info=True,
            )

        try:
            if self.voice_connection.is_connected():
                await self.voice_connection.disconnect()
        except Exception as e:
            logger.error(
                f"Error during voice disconnect for guild {self.guild.id}: {e}",
                exc_info=True,
            )

        try:
            if self._standby_message:
                await self._standby_message.delete()
                self._standby_message = None
        except discord.NotFound:
            pass  # Message was already deleted
        except Exception as e:
            logger.warning(
                f"Failed to delete standby message during cleanup for guild {self.guild.id}: {e}",
                exc_info=True,
            )

        await self.bot_state.reset_to_idle()
        logger.info(f"Session for guild {self.guild.id} cleaned up successfully.")

    def _get_standby_message_content(self) -> str:
        """
        Generate the standby message content based on the current state.

        Returns:
            A formatted string containing the content for the standby message.
        """
        current_state = self.bot_state.current_state
        main_content: str
        authority_user_note: str = "can control the recording actions."
        authority_user_name = self.bot_state.get_authority_user_name()

        if current_state == BotStateEnum.CONNECTION_ERROR:
            main_content = (
                f"**⚠️ Voice Chat Session - CONNECTION ERROR **\n\n"
                f"---\n"
                f"### 🛠 Current State:\n"
                f"- **State**: `{current_state.value}`\n"
                f"- **Details**: The bot has encountered a connection issue (voice or services) and may not be fully functional.\n"
                f"- **Action**: Please try `{Config.COMMAND_PREFIX}disconnect` and then `{Config.COMMAND_PREFIX}connect` again.\n"
                f"If the issue persists, contact an administrator."
            )
            authority_user_note = "can control the recording actions (if applicable)."
        else:
            main_content = (
                f"**🎙 Voice Chat Session - **\n\n"
                f"---\n"
                f"### 🔄 How to control the bot:\n"
                f"1. **Start Recording**: React to this message with {Config.REACTION_START_RECORDING} to start recording.\n"
                f"2. **Finish Recording**: Remove your {Config.REACTION_START_RECORDING} reaction to finish recording.\n"
                f"3. **End Session**: Use `{Config.COMMAND_PREFIX}disconnect` to end the session.\n"
                f"4. **Switch AI**: Use `{Config.COMMAND_PREFIX}set <name>` (e.g., openai, gemini).\n"
                f"---\n"
                f"### 🛠 Current State:\n"
                f"- **State**: `{current_state.value}`"
            )

        shared_content = (
            f"---\n"
            f"### 🤖 AI Provider:\n"
            f"> Active Service: `{self.bot_state.active_ai_provider_name.upper()}`\n"
            f"---\n"
            f"### 🧑 Authority User:\n"
            f"> `{authority_user_name}` {authority_user_note}"
        )

        return f"{main_content}\n{shared_content}"

    async def _update_standby_message(self) -> None:
        """
        Update the standby message with the current state.
        """
        if self._standby_message:
            try:
                await self._standby_message.edit(
                    content=self._get_standby_message_content()
                )
            except discord.DiscordException as e:
                logger.error("Failed to update standby message: %s", e, exc_info=True)
                # Re-raise to allow the updater loop to handle it.
                raise

    def _queue_ui_update(self) -> None:
        """Schedules a UI update by putting an item in the queue."""
        self._update_queue.put_nowait(None)

    async def _ui_updater_loop(self) -> None:
        """
        Background loop to process UI update requests from a queue.
        This method continuously waits for items on `_update_queue`. When an item
        is received, it calls `_update_standby_message` to synchronize the UI.
        This decouples state changes from slow network I/O.
        """
        while True:
            try:
                await self._update_queue.get()
            except asyncio.CancelledError:
                logger.info("UI updater task cancelled while waiting for queue.")
                break

            try:
                await self._update_standby_message()
            except discord.DiscordException as e:
                # The message might have been deleted, or we might have lost permissions.
                # This is not a critical error for the bot's state, just a UI sync failure.
                logger.warning(
                    f"Failed to update standby message in background loop: {e}"
                )
            except Exception as e:
                logger.error(
                    "Unexpected error during UI update in background loop: %s",
                    e,
                    exc_info=True,
                )
            finally:
                self._update_queue.task_done()

    def start_updater_task(self) -> None:
        """
        Start the background UI updater task if it is not already running.
        """
        if self._ui_updater_task is None or self._ui_updater_task.done():
            self._ui_updater_task = asyncio.create_task(self._ui_updater_loop())
            logger.info(f"UI updater task started for guild {self.guild.id}.")

    async def stop_updater_task(self) -> None:
        """
        Stop the background UI updater task gracefully.
        """
        if self._ui_updater_task and not self._ui_updater_task.done():
            self._ui_updater_task.cancel()
            try:
                await self._ui_updater_task
            except asyncio.CancelledError:
                pass  # This is expected.
            finally:
                self._ui_updater_task = None
                logger.info(f"UI updater task stopped for guild {self.guild.id}.")

    async def _create_and_set_manager(
        self, provider_name: str, ctx: commands.Context
    ) -> bool:
        """
        Creates and sets the AI service manager for the given provider using the factory.

        Args:
            provider_name: The name of the provider to create a manager for.
            ctx: The command context, used for sending user-facing error messages.

        Returns:
            True if the manager was created and set successfully, False otherwise.
        """
        if provider_name not in self.ai_service_factories:
            await ctx.send(f"Unknown AI provider '{provider_name}'.")
            return False

        manager_class, service_config = self.ai_service_factories[provider_name]
        try:
            # The manager's constructor may perform validation (e.g., for API keys)
            # and raise an exception on failure.
            manager_instance = manager_class(
                audio_playback_manager=self.audio_playback_manager,
                service_config=service_config,
            )
            self.active_ai_service_manager = manager_instance
            await self.bot_state.set_active_ai_provider_name(provider_name)
            logger.info(
                f"Successfully created and set AI service manager for '{provider_name}' in guild {self.guild.id}."
            )
            return True
        except (ValueError, Exception) as e:
            # Catches ValueError from config checks, and any other init errors
            logger.error(
                f"Failed to create AI service manager for '{provider_name}' in guild {self.guild.id}: {e}",
                exc_info=True,
            )
            await ctx.send(
                f"Failed to initialize AI provider '{provider_name.upper()}'. "
                f"Please check your configuration (e.g., API key). Error: {e}"
            )
            self.active_ai_service_manager = None
            return False

    async def _shutdown_current_ai_provider(self) -> None:
        """
        Shuts down the connection to the current AI provider and discards the instance.
        """
        if self.active_ai_service_manager:
            provider_name = self.bot_state.active_ai_provider_name
            logger.info(
                f"Shutting down current AI provider '{provider_name}' for guild {self.guild.id}"
            )

            if self.bot_state.current_state == BotStateEnum.RECORDING:
                if self.voice_connection.is_recording():
                    self.voice_connection.stop_listening()
                await self.bot_state.stop_recording()
                logger.info(
                    f"Stopped recording due to AI provider shutdown for guild {self.guild.id}."
                )

            if self.active_ai_service_manager.is_connected():
                await self.active_ai_service_manager.cancel_ongoing_response()
                await self.active_ai_service_manager.disconnect()
                logger.info(
                    f"Disconnected from {provider_name} for guild {self.guild.id}."
                )

            self.active_ai_service_manager = None
            logger.info(
                f"AI provider instance for {provider_name} shut down and discarded for guild {self.guild.id}."
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
        current_bot_state = self.bot_state.current_state
        if current_bot_state == BotStateEnum.IDLE:
            return False

        issue_detected = False
        issue_description = ""

        if current_bot_state in [BotStateEnum.STANDBY, BotStateEnum.RECORDING]:
            if not self.voice_connection.is_connected():
                issue_detected = True
                issue_description = "Voice connection lost."
                logger.warning(
                    f"Connection check (guild {self.guild.id}): Voice connection found to be inactive."
                )

        if not issue_detected and (
            not self.active_ai_service_manager
            or not self.active_ai_service_manager.is_connected()
        ):
            if self.active_ai_service_manager:
                if current_bot_state != BotStateEnum.CONNECTION_ERROR:
                    issue_detected = True
                    issue_description = "AI service connection lost."
                    logger.warning(
                        f"Connection check (guild {self.guild.id}): AI service connection found to be inactive."
                    )

        if issue_detected:
            logger.info(
                f"Connection issue detected for guild {self.guild.id}: {issue_description}. Transitioning to CONNECTION_ERROR."
            )
            state_changed = await self.bot_state.enter_connection_error_state()
            if state_changed:
                self._queue_ui_update()
            if state_changed and ctx_or_channel:
                try:
                    target_channel = (
                        ctx_or_channel
                        if isinstance(ctx_or_channel, discord.TextChannel)
                        else ctx_or_channel.channel
                    )
                    await target_channel.send(
                        f"Critical error: {issue_description} Bot functionality may be limited. Current state: {self.bot_state.current_state.value}"
                    )
                except discord.DiscordException as e:
                    logger.error(
                        f"Failed to send connection error message for guild {self.guild.id}: {e}"
                    )
            return True

        return False

    async def _send_audio_turn_to_service(
        self, pcm_data: bytes, channel: discord.TextChannel
    ) -> None:
        """
        Sends a processed audio chunk to the AI service and finalizes the turn.

        Args:
            pcm_data: Processed (e.g., resampled) PCM audio data.
            channel: The Discord text channel for sending error messages.
        """
        if not self.active_ai_service_manager:
            logger.error(
                f"Cannot send audio for guild {self.guild.id}: No active AI service manager."
            )
            await channel.send(
                "Sorry, the AI service is not available. Please try connecting again."
            )
            await self.bot_state.enter_connection_error_state()
            return

        if not await self.active_ai_service_manager.send_audio_chunk(pcm_data):
            logger.error(
                f"Failed to send audio chunk to AI service for guild {self.guild.id}."
            )
            await channel.send(
                "Sorry, I encountered an error sending your audio to the AI service."
            )
            await self.bot_state.enter_connection_error_state()
            return

        if not await self.active_ai_service_manager.finalize_input_and_request_response():
            logger.error(
                f"Failed to finalize input and request response from AI service for guild {self.guild.id}."
            )
            await channel.send(
                "Sorry, I encountered an error requesting a response from the AI service."
            )
            await self.bot_state.enter_connection_error_state()
            return
        logger.info(
            f"Successfully sent audio and requested response from AI service for guild {self.guild.id}."
        )

    async def _process_and_send_audio_task(
        self, pcm_data: bytes, channel: discord.TextChannel
    ) -> None:
        """
        A background task to process raw audio and send it to the AI service.

        Args:
            pcm_data: Raw PCM audio data from the voice connection.
            channel: The text channel for sending status or error messages.
        """
        if not self.active_ai_service_manager:
            logger.error(
                f"Cannot process audio for guild {self.guild.id}: No active AI service manager."
            )
            await channel.send(
                "Sorry, the AI service is not available. Please try connecting again."
            )
            return

        logger.debug(
            f"Starting background audio processing for {len(pcm_data)} bytes in guild {self.guild.id}."
        )
        try:
            target_frame_rate, target_channels = (
                self.active_ai_service_manager.processing_audio_format
            )
            logger.info(
                f"Processing audio for '{self.bot_state.active_ai_provider_name}' in guild {self.guild.id}."
            )

            processed_pcm_data = await process_recorded_audio(
                pcm_data,
                target_frame_rate=target_frame_rate,
                target_channels=target_channels,
            )
            logger.debug(
                f"Audio processing complete for guild {self.guild.id}. Processed data size: {len(processed_pcm_data)} bytes."
            )
            await self._send_audio_turn_to_service(processed_pcm_data, channel)
        except asyncio.CancelledError:
            logger.info(
                f"Audio processing task was cancelled for guild {self.guild.id}."
            )
        except RuntimeError as e:
            logger.error(
                f"Error processing audio for guild {self.guild.id}: {e}", exc_info=True
            )
            await channel.send("Error processing your audio. Please try again.")
        except (
            openai.OpenAIError,
            gemini_errors.APIError,
        ) as e:
            logger.error(
                f"AI service error in audio task for guild {self.guild.id}: {e}",
                exc_info=True,
            )
            await channel.send(
                "An unexpected error occurred while processing your audio."
            )

    async def handle_reaction_add(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """
        Handles the logic for a reaction being added to the standby message.

        Args:
            reaction: The reaction that was added.
            user: The user who added the reaction.
        """
        async with self._action_lock:
            if user == self.bot.user:
                return

            if not (
                self._standby_message
                and reaction.message.id == self._standby_message.id
            ):
                return

            if self.bot_state.current_state == BotStateEnum.CONNECTION_ERROR:
                logger.debug(
                    f"Reaction ignored in guild {self.guild.id}: Bot is in CONNECTION_ERROR state."
                )
                return

            if (
                reaction.emoji == Config.REACTION_START_RECORDING
                and self.bot_state.current_state == BotStateEnum.STANDBY
            ):
                if await self._check_and_handle_connection_issues(
                    reaction.message.channel
                ):
                    return

                if self.guild.voice_client and self.guild.voice_client.is_playing():
                    logger.info(
                        f"User initiated new recording in guild {self.guild.id}. Stopping current audio playback."
                    )
                    self.guild.voice_client.stop()

                if self.active_ai_service_manager:
                    if not await self.active_ai_service_manager.cancel_ongoing_response():
                        logger.error(
                            f"Failed to send cancel_ongoing_response for guild {self.guild.id}. Continuing..."
                        )
                else:
                    logger.warning(
                        f"Cannot cancel ongoing response for guild {self.guild.id}, no active AI manager."
                    )

                if not await self.bot_state.start_recording(user):
                    logger.warning(
                        f"State transition to RECORDING failed for user {user.name} in guild {self.guild.id}."
                    )
                    return

                self._queue_ui_update()

                if not self.voice_connection.start_listening():
                    logger.error(
                        f"Failed to start listening for user {user.name} in guild {self.guild.id}. Rolling back state."
                    )
                    await self.bot_state.stop_recording()
                    self._queue_ui_update()
                    await reaction.message.channel.send(
                        f"Sorry {user.mention}, I couldn't start recording due to a technical issue."
                    )
                    return
                logger.info(
                    f"Started recording for user {user.name} in guild {self.guild.id}."
                )

            elif (
                reaction.emoji == Config.REACTION_CANCEL_RECORDING
                and self.bot_state.current_state == BotStateEnum.RECORDING
                and self.bot_state.is_authorized(user)
            ):
                if await self.bot_state.stop_recording():
                    self._queue_ui_update()
                    if self.voice_connection.is_recording():
                        self.voice_connection.stop_listening()
                        logger.info(
                            f"Stopped listening due to cancellation in guild {self.guild.id}."
                        )
                        if (
                            self.active_ai_service_manager
                            and not await self.active_ai_service_manager.cancel_ongoing_response()
                        ):
                            logger.error(
                                f"Failed to request cancellation of ongoing response during user cancellation in guild {self.guild.id}."
                            )
                    await reaction.message.channel.send(
                        f"{user.display_name} canceled recording. Returning to standby."
                    )

    async def handle_reaction_remove(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """
        Handles the logic for a reaction being removed from the standby message.

        Args:
            reaction: The reaction that was removed.
            user: The user who removed the reaction.
        """
        async with self._action_lock:
            if user == self.bot.user:
                return

            if not (
                self._standby_message
                and reaction.message.id == self._standby_message.id
                and reaction.emoji == Config.REACTION_START_RECORDING
                and self.bot_state.current_state == BotStateEnum.RECORDING
                and self.bot_state.is_authorized(user)
            ):
                return

            if self.bot_state.current_state == BotStateEnum.CONNECTION_ERROR:
                logger.debug(
                    f"Reaction removal ignored in guild {self.guild.id}: CONNECTION_ERROR state."
                )
                return

            if await self._check_and_handle_connection_issues(reaction.message.channel):
                if self.voice_connection.is_recording():
                    self.voice_connection.stop_listening()
                return

            if not self.voice_connection.is_connected():
                await reaction.message.channel.send(
                    "Bot is not in a voice channel or voice client is missing."
                )
                await self.bot_state.stop_recording()
                return

            pcm_data = self.voice_connection.stop_listening()

            if await self.bot_state.stop_recording():
                self._queue_ui_update()

            if pcm_data:
                task = asyncio.create_task(
                    self._process_and_send_audio_task(
                        pcm_data, reaction.message.channel
                    )
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            else:
                await reaction.message.channel.send("No audio data was captured.")

    async def connect(self, ctx: commands.Context) -> bool:
        """
        Handles the logic of the 'connect' command.

        Args:
            ctx: The command context.

        Returns:
            True if the connection was successful, False otherwise.
        """
        if ctx.author.voice is None:
            await ctx.send("You are not connected to a voice channel.")
            return False

        # Ensure an AI service manager is active and connected.
        if not self.active_ai_service_manager:
            default_provider = Config.AI_SERVICE_PROVIDER
            if not await self._create_and_set_manager(default_provider, ctx):
                return False  # Creation failed, user was notified.

        if not self.active_ai_service_manager.is_connected():
            if not await self.active_ai_service_manager.connect():
                await ctx.send(
                    "Failed to connect to the AI service. Please try again later."
                )
                # Clear the manager so the next connect attempt can re-initialize.
                self.active_ai_service_manager = None
                return False

        voice_channel = ctx.author.voice.channel
        if not await self.voice_connection.connect_to_channel(voice_channel):
            await ctx.send("Failed to connect to the voice channel.")
            if await self.bot_state.enter_connection_error_state():
                self._queue_ui_update()
            return False

        if not await self.bot_state.transition_to_standby():
            return False

        try:
            self._standby_message = await ctx.send(self._get_standby_message_content())
            await self._standby_message.add_reaction(Config.REACTION_START_RECORDING)
        except discord.DiscordException as e:
            logger.error(
                f"Failed to create standby message for guild {self.guild.id}: {e}",
                exc_info=True,
            )
            await self.bot_state.reset_to_idle()
            return False

        await self.start_background_tasks()
        logger.info(
            f"Connect command successful for guild {self.guild.id}. Bot is in STANDBY."
        )
        return True

    async def set_provider(self, ctx: commands.Context, provider_name: str) -> None:
        """
        Handles the logic of the 'set' command.

        Args:
            ctx: The command context.
            provider_name: The name of the AI provider to switch to.
        """
        provider_name = provider_name.lower()
        if provider_name not in self.ai_service_factories:
            valid_providers = ", ".join(self.ai_service_factories.keys())
            await ctx.send(
                f"Invalid provider name '{provider_name}'. Valid options are: {valid_providers}."
            )
            return

        if (
            self.bot_state.active_ai_provider_name == provider_name
            and self.active_ai_service_manager
        ):
            await ctx.send(
                f"AI provider is already set to '{provider_name.upper()}'. No change made."
            )
            return

        # --- Validation Phase ---
        # 1. Create the new manager instance without making it active yet.
        manager_class, service_config = self.ai_service_factories[provider_name]
        try:
            new_manager = manager_class(
                audio_playback_manager=self.audio_playback_manager,
                service_config=service_config,
            )
        except (ValueError, Exception) as e:
            logger.error(
                f"Failed to create AI service manager for '{provider_name}' in guild {self.guild.id}: {e}",
                exc_info=True,
            )
            await ctx.send(
                f"Failed to initialize AI provider '{provider_name.upper()}'. "
                f"Please check your configuration (e.g., API key). Error: {e}"
            )
            return  # Abort, keeping the old provider active.

        # 2. If already in a voice call, try to connect the new manager before committing.
        if self.voice_connection.is_connected():
            if not await new_manager.connect():
                await ctx.send(
                    f"Failed to connect to the new AI provider '{provider_name.upper()}'. "
                    "The switch has been aborted, and the previous provider remains active."
                )
                await (
                    new_manager.disconnect()
                )  # Ensure cleanup of the failed temporary manager
                return  # Abort, keeping the old provider active.

        # --- Commit Phase ---
        # The new provider is validated and connected. Now, commit the switch.
        await self._shutdown_current_ai_provider()
        self.active_ai_service_manager = new_manager
        await self.bot_state.set_active_ai_provider_name(provider_name)
        self._queue_ui_update()
        await ctx.send(f"AI provider switched to '{provider_name.upper()}'.")

        # If we were in an error state, try to recover.
        if self.bot_state.current_state == BotStateEnum.CONNECTION_ERROR:
            # We already know voice is connected and the new AI service is connected.
            if await self.bot_state.recover_to_standby():
                self._queue_ui_update()

    @tasks.loop(seconds=Config.CONNECTION_CHECK_INTERVAL)
    async def _connection_check_loop(self) -> None:
        """Periodically checks the health of critical connections."""
        if self.bot_state.current_state not in [
            BotStateEnum.STANDBY,
            BotStateEnum.RECORDING,
            BotStateEnum.CONNECTION_ERROR,
        ]:
            return

        channel_for_message = None
        if self._standby_message:
            channel_for_message = self._standby_message.channel

        if self.bot_state.current_state == BotStateEnum.CONNECTION_ERROR:
            # Attempt recovery if a manager is active and connections are good.
            if (
                self.voice_connection.is_connected()
                and self.active_ai_service_manager
                and self.active_ai_service_manager.is_connected()
            ):
                if await self.bot_state.recover_to_standby():
                    self._queue_ui_update()
                    return

        await self._check_and_handle_connection_issues(channel_for_message)

    @_connection_check_loop.before_loop
    async def before_connection_check_loop(self) -> None:
        """Waits for the bot to be ready before starting the connection check loop."""
        await self.bot.wait_until_ready()
