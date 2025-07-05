"""
AI Service Coordinator module.

This module defines the AIServiceCoordinator class, which is responsible for
managing the lifecycle and interactions with the AI service providers.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Tuple

from discord.ext import commands

from src.ai_services.interface import IRealtimeAIServiceManager
from src.audio.playback import AudioPlaybackManager
from src.config.config import Config
from src.state.state import BotState, BotStateEnum
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AIServiceCoordinator:
    """
    Manages the lifecycle and interactions with AI service providers.
    """

    def __init__(
        self,
        bot_state: BotState,
        audio_playback_manager: AudioPlaybackManager,
        ai_service_factories: Dict[str, tuple],
        guild_id: int,
    ):
        self.bot_state = bot_state
        self.audio_playback_manager = audio_playback_manager
        self.ai_service_factories = ai_service_factories
        self.guild_id = guild_id
        self.active_ai_service_manager: Optional[IRealtimeAIServiceManager] = None

    def is_connected(self) -> bool:
        """Checks if the active AI service manager is connected."""
        return (
            self.active_ai_service_manager is not None
            and self.active_ai_service_manager.is_connected()
        )

    def get_processing_audio_format(self) -> Optional[Tuple[int, int]]:
        """Gets the audio format required by the current AI service for processing."""
        if self.active_ai_service_manager:
            return self.active_ai_service_manager.processing_audio_format
        return None

    async def cancel_ongoing_response(self) -> bool:
        """Cancels any ongoing response from the AI service."""
        if self.is_connected() and self.active_ai_service_manager:
            return await self.active_ai_service_manager.cancel_ongoing_response()
        return False

    async def send_audio_turn(self, pcm_data: bytes) -> bool:
        """Sends a full audio turn (chunk + finalize) to the AI service."""
        if not self.is_connected() or not self.active_ai_service_manager:
            logger.error(
                f"Cannot send audio for guild {self.guild_id}: Not connected."
            )
            return False

        if not await self.active_ai_service_manager.send_audio_chunk(pcm_data):
            logger.error(
                f"Failed to send audio chunk to AI service for guild {self.guild_id}."
            )
            return False

        if not await self.active_ai_service_manager.finalize_input_and_request_response():
            logger.error(
                f"Failed to finalize input for AI service for guild {self.guild_id}."
            )
            return False

        logger.info(
            f"Successfully sent audio and requested response from AI service for guild {self.guild_id}."
        )
        return True

    async def shutdown(self) -> None:
        """Shuts down the connection to the current AI provider."""
        if self.active_ai_service_manager and self.is_connected():
            provider_name = self.bot_state.active_ai_provider_name
            logger.info(
                f"Shutting down AI provider '{provider_name}' for guild {self.guild_id}"
            )
            await self.cancel_ongoing_response()
            await self.active_ai_service_manager.disconnect()
            logger.info(
                f"Disconnected from {provider_name} for guild {self.guild_id}."
            )
        self.active_ai_service_manager = None

    async def ensure_connected(self, ctx: commands.Context) -> bool:
        """Ensures the AI service is initialized and connected, creating it if necessary."""
        if not self.is_connected():
            if not self.active_ai_service_manager:
                default_provider = Config.AI_SERVICE_PROVIDER
                if not await self._create_and_set_manager(default_provider, ctx):
                    return False

            if not await self.active_ai_service_manager.connect():
                await ctx.send(
                    "Failed to connect to the AI service. Please try again later."
                )
                self.active_ai_service_manager = None
                return False
        return True

    async def switch_provider(
        self, provider_name: str, ctx: commands.Context, is_voice_connected: bool
    ) -> bool:
        """Handles the logic of switching the AI provider."""
        if (
            self.bot_state.active_ai_provider_name == provider_name
            and self.is_connected()
        ):
            await ctx.send(
                f"AI provider is already set to '{provider_name.upper()}'."
            )
            return True

        new_manager = await self._validate_new_provider(
            provider_name, ctx, is_voice_connected
        )
        if not new_manager:
            return False

        await self.shutdown()
        self.active_ai_service_manager = new_manager
        await self.bot_state.set_active_ai_provider_name(provider_name)
        return True

    async def _create_and_set_manager(
        self, provider_name: str, ctx: commands.Context
    ) -> bool:
        """Creates and sets the AI service manager instance."""
        manager_class, service_config = self.ai_service_factories[provider_name]
        try:
            manager_instance = manager_class(
                audio_playback_manager=self.audio_playback_manager,
                service_config=service_config,
            )
            self.active_ai_service_manager = manager_instance
            await self.bot_state.set_active_ai_provider_name(provider_name)
            logger.info(
                f"Successfully created AI manager for '{provider_name}' in guild {self.guild_id}."
            )
            return True
        except (ValueError, Exception) as e:
            logger.error(
                f"Failed to create AI manager for '{provider_name}' in guild {self.guild_id}: {e}",
                exc_info=True,
            )
            await ctx.send(
                f"Failed to initialize AI provider '{provider_name.upper()}'. "
                f"Check configuration (e.g., API key). Error: {e}"
            )
            self.active_ai_service_manager = None
            return False

    async def _validate_new_provider(
        self, provider_name: str, ctx: commands.Context, is_voice_connected: bool
    ) -> Optional[IRealtimeAIServiceManager]:
        """Validates a new provider by attempting to create and connect it."""
        manager_class, service_config = self.ai_service_factories[provider_name]
        try:
            new_manager = manager_class(
                audio_playback_manager=self.audio_playback_manager,
                service_config=service_config,
            )
        except (ValueError, Exception) as e:
            logger.error(
                f"Failed to create AI manager for '{provider_name}' in guild {self.guild_id}: {e}",
                exc_info=True,
            )
            await ctx.send(
                f"Failed to initialize AI provider '{provider_name.upper()}'. "
                f"Check configuration. Error: {e}"
            )
            return None

        if is_voice_connected:
            if not await new_manager.connect():
                await ctx.send(
                    f"Failed to connect to '{provider_name.upper()}'. Switch aborted."
                )
                await new_manager.disconnect()
                return None
        return new_manager
