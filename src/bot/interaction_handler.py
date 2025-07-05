"""
Interaction Handler module for processing user interactions within a GuildSession.
"""

import asyncio
from typing import Any, Optional, Set, Union

import discord
import openai
from discord.ext import commands
from google.genai import errors as gemini_errors

from src.audio.processor import process_recorded_audio
from src.bot.ai_service_coordinator import AIServiceCoordinator
from src.bot.ui_manager import SessionUIManager
from src.bot.voice_connection import VoiceConnectionManager
from src.config.config import Config
from src.state.state import BotState, BotStateEnum
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InteractionHandler:
    """
    Handles user interactions like reactions for a specific GuildSession.
    """

    def __init__(
        self,
        guild_id: int,
        bot: commands.Bot,
        bot_state: BotState,
        ui_manager: SessionUIManager,
        voice_connection: VoiceConnectionManager,
        ai_coordinator: AIServiceCoordinator,
    ):
        self.guild_id = guild_id
        self.bot = bot
        self.bot_state = bot_state
        self.ui_manager = ui_manager
        self.voice_connection = voice_connection
        self.ai_coordinator = ai_coordinator
        self._action_lock = asyncio.Lock()
        self._background_tasks: Set[asyncio.Task[Any]] = set()

    async def cleanup(self):
        """Cancel all background tasks."""
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        logger.info(
            f"Cleaned up {len(self._background_tasks)} background tasks for guild {self.guild_id}."
        )

    async def _handle_start_recording_reaction(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """Handles the start recording reaction."""
        if await self.check_and_handle_connection_issues(reaction.message.channel):
            return

        guild = self.bot.get_guild(self.guild_id)
        if not guild:
            return

        if guild.voice_client and guild.voice_client.is_playing():
            logger.info(
                f"User initiated new recording in guild {self.guild_id}. Stopping current audio playback."
            )
            guild.voice_client.stop()

        if not await self.ai_coordinator.cancel_ongoing_response():
            logger.error(
                f"Failed to send cancel_ongoing_response for guild {self.guild_id}. Continuing..."
            )

        if not await self.bot_state.start_recording(user):
            logger.warning(
                f"State transition to RECORDING failed for user {user.name} in guild {self.guild_id}."
            )
            return

        self.ui_manager.schedule_update()

        if not self.voice_connection.start_listening():
            logger.error(
                f"Failed to start listening for user {user.name} in guild {self.guild_id}. Rolling back state."
            )
            await self.bot_state.stop_recording()
            self.ui_manager.schedule_update()
            await reaction.message.channel.send(
                f"Sorry {user.mention}, I couldn't start recording due to a technical issue."
            )
            return
        logger.info(f"Started recording for user {user.name} in guild {self.guild_id}.")

    async def _handle_cancel_recording_reaction(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """Handles the cancel recording reaction."""
        if await self.bot_state.stop_recording():
            self.ui_manager.schedule_update()
            if self.voice_connection.is_recording():
                self.voice_connection.stop_listening()
                logger.info(
                    f"Stopped listening due to cancellation in guild {self.guild_id}."
                )
                if not await self.ai_coordinator.cancel_ongoing_response():
                    logger.error(
                        f"Failed to request cancellation of ongoing response during user cancellation in guild {self.guild_id}."
                    )
            await reaction.message.channel.send(
                f"{user.display_name} canceled recording. Returning to standby."
            )

    async def _handle_stop_recording_reaction(self, reaction: discord.Reaction) -> None:
        """Handles the stop recording reaction (reaction removal)."""
        if self.bot_state.current_state == BotStateEnum.CONNECTION_ERROR:
            logger.debug(
                f"Reaction removal ignored in guild {self.guild_id}: CONNECTION_ERROR state."
            )
            return

        if await self.check_and_handle_connection_issues(reaction.message.channel):
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
            self.ui_manager.schedule_update()

        if pcm_data:
            task = asyncio.create_task(
                self._process_and_send_audio_task(pcm_data, reaction.message.channel)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        else:
            await reaction.message.channel.send("No audio data was captured.")

    async def check_and_handle_connection_issues(
        self,
        ctx_or_channel: Optional[Union[commands.Context, discord.TextChannel]] = None,
    ) -> bool:
        """
        Checks critical connections and transitions to CONNECTION_ERROR state if issues are found.
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
                    f"Connection check (guild {self.guild_id}): Voice connection found to be inactive."
                )

        if not issue_detected and not self.ai_coordinator.is_connected():
            if current_bot_state != BotStateEnum.CONNECTION_ERROR:
                issue_detected = True
                issue_description = "AI service connection lost."
                logger.warning(
                    f"Connection check (guild {self.guild_id}): AI service connection found to be inactive."
                )

        if issue_detected:
            logger.info(
                f"Connection issue detected for guild {self.guild_id}: {issue_description}. Transitioning to CONNECTION_ERROR."
            )
            state_changed = await self.bot_state.enter_connection_error_state()
            if state_changed:
                self.ui_manager.schedule_update()
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
                        f"Failed to send connection error message for guild {self.guild_id}: {e}"
                    )
            return True

        return False

    async def _process_and_send_audio_task(
        self, pcm_data: bytes, channel: discord.TextChannel
    ) -> None:
        """
        A background task to process raw audio and send it to the AI service.
        """
        audio_format = self.ai_coordinator.get_processing_audio_format()
        if not audio_format:
            logger.error(
                f"Cannot process audio for guild {self.guild_id}: No active AI service manager."
            )
            await channel.send(
                "Sorry, the AI service is not available. Please try connecting again."
            )
            return

        logger.debug(
            f"Starting background audio processing for {len(pcm_data)} bytes in guild {self.guild_id}."
        )
        try:
            target_frame_rate, target_channels = audio_format
            logger.info(
                f"Processing audio for '{self.bot_state.active_ai_provider_name}' in guild {self.guild_id}."
            )

            processed_pcm_data = await process_recorded_audio(
                pcm_data,
                target_frame_rate=target_frame_rate,
                target_channels=target_channels,
            )
            logger.debug(
                f"Audio processing complete for guild {self.guild_id}. Processed data size: {len(processed_pcm_data)} bytes."
            )
            if not await self.ai_coordinator.send_audio_turn(processed_pcm_data):
                await channel.send(
                    "Sorry, I encountered an error sending your audio to the AI service."
                )
                await self.bot_state.enter_connection_error_state()

        except asyncio.CancelledError:
            logger.info(
                f"Audio processing task was cancelled for guild {self.guild_id}."
            )
        except RuntimeError as e:
            logger.error(
                f"Error processing audio for guild {self.guild_id}: {e}", exc_info=True
            )
            await channel.send("Error processing your audio. Please try again.")
        except (
            openai.OpenAIError,
            gemini_errors.APIError,
        ) as e:
            logger.error(
                f"AI service error in audio task for guild {self.guild_id}: {e}",
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
        """
        async with self._action_lock:
            message_id = self.ui_manager.get_message_id()
            if user == self.bot.user or not (
                message_id and reaction.message.id == message_id
            ):
                return

            if self.bot_state.current_state == BotStateEnum.CONNECTION_ERROR:
                logger.debug(
                    f"Reaction ignored in guild {self.guild_id}: Bot is in CONNECTION_ERROR state."
                )
                return

            if (
                reaction.emoji == Config.REACTION_START_RECORDING
                and self.bot_state.current_state == BotStateEnum.STANDBY
            ):
                await self._handle_start_recording_reaction(reaction, user)

            elif (
                reaction.emoji == Config.REACTION_CANCEL_RECORDING
                and self.bot_state.current_state == BotStateEnum.RECORDING
                and self.bot_state.is_authorized(user)
            ):
                await self._handle_cancel_recording_reaction(reaction, user)

    async def handle_reaction_remove(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """
        Handles the logic for a reaction being removed from the standby message.
        """
        async with self._action_lock:
            message_id = self.ui_manager.get_message_id()
            is_valid_reaction_remove = (
                user != self.bot.user
                and message_id
                and reaction.message.id == message_id
                and reaction.emoji == Config.REACTION_START_RECORDING
                and self.bot_state.current_state == BotStateEnum.RECORDING
                and self.bot_state.is_authorized(user)
            )
            if not is_valid_reaction_remove:
                return

            await self._handle_stop_recording_reaction(reaction)
