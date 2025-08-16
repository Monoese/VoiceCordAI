"""
Guild Session module for managing per-guild bot state and interactions.

This module defines the GuildSession class, which encapsulates all the logic,
state, and resources for the bot's operation within a single Discord guild. This
ensures that the bot can operate in multiple guilds simultaneously without any
state conflicts.
"""

import asyncio
from typing import Dict, Optional, Set

import discord
import openai
from discord.ext import commands
from google.genai import errors as gemini_errors

from src.audio.playback import AudioPlaybackManager
from src.audio.processor import process_recorded_audio
from src.audio.sinks import AudioSink, ManualControlSink, RealtimeMixingSink
from src.bot.session.ai_service_coordinator import AIServiceCoordinator
from src.bot.session.interaction_handler import InteractionHandler
from src.bot.session.session_ui_manager import SessionUIManager
from src.bot.session.voice_connection_manager import VoiceConnectionManager
from src.bot.state import BotState, BotModeEnum, BotStateEnum, RecordingMethod
from src.config.config import Config
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
        self._action_lock = asyncio.Lock()
        self._background_tasks: Set[asyncio.Task] = set()
        self._audio_sink: Optional["AudioSink"] = None

        self.bot_state = BotState()
        self.ui_manager = SessionUIManager(self.guild, self.bot_state)
        self.audio_playback_manager = AudioPlaybackManager(self.guild)
        self.voice_connection = VoiceConnectionManager(
            self.guild, self.audio_playback_manager
        )
        self.ai_coordinator = AIServiceCoordinator(
            bot_state=self.bot_state,
            audio_playback_manager=self.audio_playback_manager,
            ai_service_factories=ai_service_factories,
            guild_id=self.guild.id,
        )
        # InteractionHandler must be created last, as it requires a reference to the fully initialized GuildSession
        self.interaction_handler = InteractionHandler(
            guild_id=self.guild.id,
            bot=self.bot,
            ui_manager=self.ui_manager,
            guild_session=self,
        )

    async def start_background_tasks(self) -> None:
        """Starts all persistent background tasks for the session."""
        self.ui_manager.start()
        logger.info(f"Background tasks started for guild {self.guild.id}.")

    async def cleanup(self) -> None:
        """
        Gracefully shuts down the session, ensuring each cleanup step is attempted.
        """
        logger.info(f"Cleaning up session for guild {self.guild.id}")

        # Cancel all background tasks managed by this session
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        await self.ui_manager.cleanup()
        await self.interaction_handler.cleanup()  # No-op, but good practice

        # Clean up the audio sink
        if self._audio_sink:
            self._audio_sink.cleanup()
            self._audio_sink = None

        try:
            await self.ai_coordinator.shutdown()
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

        await self.bot_state.reset_to_idle()
        logger.info(f"Session for guild {self.guild.id} cleaned up successfully.")

    async def handle_reaction_add(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """
        Delegates reaction add events to the interaction handler.
        """
        await self.interaction_handler.handle_reaction_add(reaction, user)

    async def handle_reaction_remove(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """
        Delegates reaction remove events to the interaction handler.
        """
        await self.interaction_handler.handle_reaction_remove(reaction, user)

    async def _on_ai_connect(self) -> None:
        """Callback for when the AI service connects."""
        logger.info(f"AI service connected for guild {self.guild.id}.")
        # If the voice connection is also active, attempt to recover the bot's state.
        if self.voice_connection.is_connected():
            await self.bot_state.recover_to_standby()

    async def _on_ai_disconnect(self) -> None:
        """Callback for when the AI service disconnects."""
        logger.warning(f"AI service disconnected for guild {self.guild.id}.")
        await self.bot_state.enter_connection_error_state()

    async def handle_voice_connection_update(self, is_connected: bool) -> None:
        """
        Handler called by VoiceCog on voice state changes.

        This method decides whether to recover the bot to a healthy state or enter
        an error state based on the status of both the voice and AI connections.
        """
        # Query the coordinator directly to get the authoritative AI connection status.
        if is_connected and self.ai_coordinator.is_connected():
            logger.info(
                f"Voice connection active for guild {self.guild.id}, recovering if needed."
            )
            await self.bot_state.recover_to_standby()
        elif not is_connected:
            # If the voice connection is lost, always enter an error state.
            logger.warning(f"Voice connection lost for guild {self.guild.id}.")
            await self.bot_state.enter_connection_error_state()

    # --- New Handler Methods (called by InteractionHandler) ---

    async def handle_consent_reaction(self, user: discord.User, added: bool) -> None:
        async with self._action_lock:
            if added:
                logger.info(f"User {user.id} granted consent.")
                await self.bot_state.grant_consent(user.id)
                if self._audio_sink:
                    if isinstance(self._audio_sink, ManualControlSink):
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            None, self._audio_sink.add_user, user.id
                        )
                    else:
                        self._audio_sink.add_user(user.id)
            else:
                logger.info(f"User {user.id} revoked consent.")
                await self.bot_state.revoke_consent(user.id)
                if self._audio_sink:
                    self._audio_sink.remove_user(user.id)
            self.ui_manager.schedule_update()

    async def handle_mode_switch_reaction(self, user: discord.User, emoji: str) -> None:
        async with self._action_lock:
            member = self.guild.get_member(user.id)
            # Permission checks: User must be in a voice channel and have given consent.
            if not member or not member.voice or not member.voice.channel:
                return
            if user.id not in self.bot_state.get_consented_user_ids():
                return

            new_mode = (
                BotModeEnum.ManualControl
                if emoji == Config.REACTION_MODE_MANUAL
                else BotModeEnum.RealtimeTalk
            )
            await self.switch_mode(new_mode)

    async def handle_pushtotalk_reaction(self, user: discord.User, added: bool) -> None:
        async with self._action_lock:
            if self.bot_state.mode != BotModeEnum.ManualControl:
                return

            if added and self.bot_state.current_state == BotStateEnum.STANDBY:
                # This transition is immediate, no cues.
                if isinstance(self._audio_sink, ManualControlSink):
                    self._audio_sink.enable_vad(False)
                await self.bot_state.start_recording(user, RecordingMethod.PushToTalk)
                # SESSION ID SYNC: Update sink session ID after new recording starts
                if isinstance(self._audio_sink, ManualControlSink):
                    self._audio_sink.update_session_id()
            elif (
                not added
                and self.bot_state.current_state == BotStateEnum.RECORDING
                and self.bot_state.recording_method == RecordingMethod.PushToTalk
                and self.bot_state.is_authorized(user)
            ):
                if self._audio_sink:
                    sink: "ManualControlSink" = self._audio_sink  # type: ignore
                    audio_data = sink.stop_and_get_audio()
                    self._handle_finished_recording(audio_data)
                await self.bot_state.stop_recording()

    # --- New Callback Methods (passed to ManualControlSink) ---

    async def on_wake_word_detected(self, user: discord.User) -> None:
        logger.info(
            f"GuildSession: Received on_wake_word_detected event for user {user.id}"
        )
        async with self._action_lock:
            if (
                self.bot_state.mode != BotModeEnum.ManualControl
                or self.bot_state.current_state != BotStateEnum.STANDBY
            ):
                return

            # Enable VAD for the upcoming recording session.
            if isinstance(self._audio_sink, ManualControlSink):
                self._audio_sink.enable_vad(True)

            # Immediately transition to RECORDING state to capture all audio.
            await self.bot_state.start_recording(user, RecordingMethod.WakeWord)
            # SESSION ID SYNC: Update sink session ID after new recording starts
            if isinstance(self._audio_sink, ManualControlSink):
                self._audio_sink.update_session_id()

            # Commented out to test if cue playback disrupts audio reception
            # await self.audio_playback_manager.play_cue("start_recording")

    async def on_vad_speech_end(self, audio_data: bytes) -> None:
        async with self._action_lock:
            if (
                self.bot_state.current_state != BotStateEnum.RECORDING
                or self.bot_state.recording_method != RecordingMethod.WakeWord
            ):
                return

            logger.info("VAD detected end of speech. Transitioning to STANDBY.")
            # No end recording cue - proceed directly to processing

            # Start processing the audio in a background task.
            self._handle_finished_recording(audio_data)
            await self.bot_state.stop_recording()

    # --- New Core Logic Methods ---
    def _handle_finished_recording(self, audio_data: bytes) -> None:
        """
        Creates a background task to process finished recording audio.

        This helper method consolidates the common logic used by both
        push-to-talk and wake word triggered recordings.
        """
        if audio_data:
            logger.info(f"Creating audio processing task for {len(audio_data)} bytes.")
            task = asyncio.create_task(self._process_manual_audio_task(audio_data))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _process_manual_audio_task(self, audio_data: bytes) -> None:
        """Task to process a finished audio recording from ManualControl mode."""
        # Cross-session corruption fix: Capture session ID at start of processing
        session_id = self.bot_state.current_session_id

        try:
            # Get the audio format required by the active AI service
            target_format = self.ai_coordinator.get_processing_audio_format()
            if not target_format:
                logger.error(
                    f"Could not determine target audio format for guild {self.guild.id}. Aborting processing."
                )
                await self._safe_enter_error_state(session_id)
                return
            target_frame_rate, target_channels = target_format

            # Convert the raw audio to the required format
            processed_audio = await process_recorded_audio(
                raw_audio_data=audio_data,
                target_frame_rate=target_frame_rate,
                target_channels=target_channels,
            )

            if not await self.ai_coordinator.send_audio_turn(processed_audio):
                await self._safe_enter_error_state(session_id)
        except asyncio.CancelledError:
            logger.info(
                f"Manual audio processing task cancelled for guild {self.guild.id}."
            )
        except (openai.APIError, gemini_errors.APIError) as e:
            logger.error(
                f"AI service API error in manual audio task: {e}", exc_info=True
            )
            await self._safe_enter_error_state(session_id)
        except Exception as e:
            logger.error(f"Unexpected error in manual audio task: {e}", exc_info=True)
            await self._safe_enter_error_state(session_id)

    async def _safe_enter_error_state(self, original_session_id: int) -> None:
        """
        Safely enter error state only if the session hasn't changed.

        This prevents background tasks from corrupting unrelated new sessions.
        Only enters error state if we're still processing the same session
        that this background task was created for.

        Args:
            original_session_id: The session ID when this background task started
        """
        async with self._action_lock:
            if self.bot_state.current_session_id == original_session_id:
                logger.info(
                    f"Background task entering error state for session {original_session_id}"
                )
                await self.bot_state.enter_connection_error_state()
            else:
                logger.info(
                    f"Background task from session {original_session_id} failed, but current session is "
                    f"{self.bot_state.current_session_id}. Not entering error state to avoid corruption."
                )

    async def _realtime_audio_streamer_task(self) -> None:
        """Task to continuously stream audio from RealtimeMixingSink to the AI."""
        if not isinstance(self._audio_sink, RealtimeMixingSink):
            return
        try:
            while True:
                chunk = await self._audio_sink.output_queue.get()
                if not await self.ai_coordinator.send_audio_chunk(chunk):
                    await self.bot_state.enter_connection_error_state()
                    break
                self._audio_sink.output_queue.task_done()
        except asyncio.CancelledError:
            logger.info(
                f"Realtime audio streamer task cancelled for guild {self.guild.id}."
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in realtime audio streamer: {e}", exc_info=True
            )
            await self.bot_state.enter_connection_error_state()

    async def _initialize_sink(self) -> None:
        """Creates and starts the appropriate audio sink based on the current mode."""
        consented_users = self.bot_state.get_consented_user_ids()
        if self.bot_state.mode == BotModeEnum.ManualControl:
            self._audio_sink = ManualControlSink(
                bot_state=self.bot_state,
                initial_consented_users=consented_users,
                on_wake_word_detected=self.on_wake_word_detected,
                on_vad_speech_end=self.on_vad_speech_end,
            )
        elif self.bot_state.mode == BotModeEnum.RealtimeTalk:
            self._audio_sink = RealtimeMixingSink(
                bot_state=self.bot_state, initial_consented_users=consented_users
            )
            task = asyncio.create_task(self._realtime_audio_streamer_task())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        else:
            self._audio_sink = None

        if self._audio_sink:
            self.voice_connection.start_listening(self._audio_sink)
            logger.info(f"Initialized audio sink for mode: {self.bot_state.mode.value}")

    async def switch_mode(self, new_mode: BotModeEnum) -> None:
        """Switches the bot operational mode, performing a hard reset of the audio state."""
        if self.bot_state.mode == new_mode:
            return

        logger.info(f"Switching mode to {new_mode.value}")

        # Mode switch race fix: Ensure proper cleanup coordination
        async with self._action_lock:
            # Cleanup old sink with coordination
            if self._audio_sink:
                # First, stop voice connection from sending more audio to sink
                self.voice_connection.stop_listening()

                # Now cleanup the sink safely
                if hasattr(self._audio_sink, "cleanup"):
                    self._audio_sink.cleanup()
                self._audio_sink = None

            await self.bot_state.set_mode(new_mode)
            await self._initialize_sink()

            # Set default state for the new mode
            if new_mode == BotModeEnum.ManualControl:
                await self.bot_state.set_state(BotStateEnum.STANDBY)
            elif new_mode == BotModeEnum.RealtimeTalk:
                await self.bot_state.set_state(BotStateEnum.LISTENING)

    async def initialize_session(
        self, ctx: commands.Context, mode: BotModeEnum
    ) -> bool:
        """
        Handles the full connection logic when a user issues a connect command.
        """
        if ctx.author.voice is None:
            await ctx.send("You are not connected to a voice channel.")
            return False

        if not await self.ai_coordinator.ensure_connected(
            ctx, on_connect=self._on_ai_connect, on_disconnect=self._on_ai_disconnect
        ):
            return False

        voice_channel = ctx.author.voice.channel
        if not await self.voice_connection.connect_to_channel(voice_channel):
            await ctx.send("Failed to connect to the voice channel.")
            await self.bot_state.enter_connection_error_state()
            return False

        await self.bot_state.set_mode(mode)

        if not await self.ui_manager.create(ctx.channel):
            await self.bot_state.reset_to_idle()
            return False

        await self._initialize_sink()

        # Set initial state based on mode
        initial_state = (
            BotStateEnum.STANDBY
            if mode == BotModeEnum.ManualControl
            else BotStateEnum.LISTENING
        )
        await self.bot_state.set_state(initial_state)

        await self.start_background_tasks()
        logger.info(
            f"Connect command successful for guild {self.guild.id}. Bot is in {initial_state.value}."
        )
        return True

    async def set_provider(self, ctx: commands.Context, provider_name: str) -> None:
        """
        Handles the logic of the 'set' command.
        """
        provider_name = provider_name.lower()
        if provider_name not in self.ai_coordinator.ai_service_factories:
            valid_providers = ", ".join(self.ai_coordinator.ai_service_factories.keys())
            await ctx.send(
                f"Invalid provider name '{provider_name}'. Valid options are: {valid_providers}."
            )
            return

        # Provider switching race fix: Ensure atomic provider changes
        async with self._action_lock:
            if await self.ai_coordinator.switch_provider(
                provider_name,
                ctx,
                self.voice_connection.is_connected(),
                on_connect=self._on_ai_connect,
                on_disconnect=self._on_ai_disconnect,
            ):
                await ctx.send(f"AI provider switched to '{provider_name.upper()}'.")
