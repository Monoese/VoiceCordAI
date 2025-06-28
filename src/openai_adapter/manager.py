"""
OpenAI Realtime API Manager.

This module provides the OpenAIRealtimeManager class, which will serve as
the primary interface for the bot to interact with the OpenAI Realtime API
via the new adapter. It will coordinate connection management, event sending,
and event handling.
"""
from __future__ import annotations

import asyncio
import base64
from typing import Optional, Dict, Any

from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

from src.utils.logger import get_logger
from src.audio.audio import AudioManager
from .connection import OpenAIRealtimeConnection
from .event_mapper import OpenAIEventHandlerAdapter
from src.ai_services.interface import IRealtimeAIServiceManager

logger = get_logger(__name__)


class OpenAIRealtimeManager(IRealtimeAIServiceManager):
    """
    Manages interactions with the OpenAI Realtime API using the OpenAI Python library.

    This class handles:
    - Initializing the AsyncOpenAI client.
    - Managing the OpenAIRealtimeConnection for WebSocket communication.
    - Coordinating with OpenAIEventHandlerAdapter for processing incoming events.
    - Providing methods to send various commands/data to the OpenAI Realtime API.
    - Managing the connection lifecycle (connect, disconnect) as per the IRealtimeAIServiceManager interface.
    """

    def __init__(self, audio_manager: AudioManager, service_config: Dict[str, Any]):
        """
        Initializes the OpenAIRealtimeManager.

        Args:
            audio_manager: An instance of AudioManager, required by the event handler.
            service_config: Configuration specific to this service instance.

        Raises:
            ValueError: If the API key is missing in the service configuration.
        """
        super().__init__(audio_manager, service_config)

        api_key = self._service_config.get("api_key")
        if not api_key:
            raise ValueError("OpenAI API key is missing in the service configuration.")

        self.model_name: str = self._service_config.get("model_name")
        if not self.model_name:
            raise ValueError(
                "OpenAI model name is missing in the service configuration."
            )

        self.openai_client: AsyncOpenAI = AsyncOpenAI(api_key=api_key)

        self.event_handler_adapter: OpenAIEventHandlerAdapter = (
            OpenAIEventHandlerAdapter(
                audio_manager=self._audio_manager,
                response_audio_format=self.response_audio_format,
            )
        )
        self.connection_handler: OpenAIRealtimeConnection = OpenAIRealtimeConnection(
            client=self.openai_client, model_name=self.model_name
        )

    async def _get_active_conn(self) -> Optional[AsyncRealtimeConnection]:
        """Helper to get the active connection object if connected."""
        if self.is_connected():
            conn = self.connection_handler.get_active_connection()
            if conn:
                return conn
            else:
                logger.warning(
                    "_get_active_conn: is_connected() is True, but get_active_connection() returned None."
                )
        return None

    async def connect(self) -> bool:
        """
        Establishes a connection to the OpenAI Realtime API and initializes the session.
        """
        logger.info("OpenAIRealtimeManager: Attempting to connect...")
        if self.is_connected():
            logger.info("OpenAIRealtimeManager: Already connected.")
            return True

        await self.connection_handler.connect(self.event_handler_adapter.dispatch_event)

        timeout = self._service_config.get("connection_timeout", 30.0)
        try:
            await asyncio.wait_for(
                self.connection_handler.wait_until_connected(), timeout=timeout
            )
            logger.info(
                "OpenAIRealtimeManager: Connection successfully established with handler."
            )

            initial_session_data = self._service_config.get("initial_session_data")
            if initial_session_data:
                conn_obj = self.connection_handler.get_active_connection()
                if conn_obj:
                    try:
                        await conn_obj.session.update(session=initial_session_data)
                        logger.info(
                            f"Sent initial session.update: {initial_session_data}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error sending initial session.update: {e}",
                            exc_info=True,
                        )
                        await self.connection_handler.disconnect()
                        return False
                else:
                    logger.error(
                        "Failed to get active connection object from connection_handler for session update, though handler reported connected."
                    )
                    await self.connection_handler.disconnect()
                    return False

            return True
        except asyncio.TimeoutError:
            logger.warning(
                f"OpenAIRealtimeManager: Failed to establish connection within {timeout}s timeout."
            )
            await self.connection_handler.disconnect()
            return False

    async def disconnect(self) -> None:
        """
        Closes the connection to the OpenAI Realtime API and cleans up resources.
        """
        logger.info("OpenAIRealtimeManager: Attempting to disconnect...")
        await self.connection_handler.disconnect()
        logger.info("OpenAIRealtimeManager: Disconnected.")

    def is_connected(self) -> bool:
        """
        Checks if the manager is currently connected by delegating to the connection handler.
        """
        return self.connection_handler.is_connected()

    async def send_audio_chunk(self, audio_data: bytes) -> bool:
        """
        Sends a chunk of raw audio data to the OpenAI service.
        The data is base64 encoded in a background thread to avoid blocking the event loop.
        """
        conn = await self._get_active_conn()
        if not conn:
            logger.error("Cannot send audio chunk: Not connected.")
            return False
        try:
            loop = asyncio.get_running_loop()
            audio_b64_bytes = await loop.run_in_executor(
                None, base64.b64encode, audio_data
            )
            audio_b64 = audio_b64_bytes.decode("utf-8")
            await conn.input_audio_buffer.append(audio=audio_b64)
            logger.debug("Sent input_audio_buffer.append")
            return True
        except Exception as e:
            logger.error(f"Error sending input_audio_buffer.append: {e}", exc_info=True)
            return False


    async def finalize_input_and_request_response(self) -> bool:
        """
        Signals that all audio for the current turn has been sent and requests a response.
        For OpenAI, this commits the audio buffer and then creates a response.
        """
        conn = await self._get_active_conn()
        if not conn:
            logger.error("Cannot finalize input and request response: Not connected.")
            return False
        try:
            await conn.input_audio_buffer.commit()
            logger.info("Sent input_audio_buffer.commit")

            response_data = self._service_config.get("response_creation_data")
            await conn.response.create(response=response_data or {})
            logger.info(f"Sent response.create (data: {response_data or {}})")
            return True
        except Exception as e:
            logger.error(
                f"Error in finalize_input_and_request_response: {e}", exc_info=True
            )
            return False

    async def cancel_ongoing_response(self) -> bool:
        """
        Attempts to cancel any AI response currently being generated or streamed.
        """
        conn = await self._get_active_conn()
        if not conn:
            logger.error("Cannot cancel response: Not connected.")
            return False

        response_id_to_cancel = self._audio_manager.get_current_playing_response_id()

        payload: Dict[str, Any] = {"type": "response.cancel"}
        if response_id_to_cancel:
            payload["response_id"] = response_id_to_cancel

        try:
            await conn.send(payload)
            logger.info(f"Sent response.cancel (payload: {payload})")
            return True
        except Exception as e:
            logger.error(f"Error sending response.cancel: {e}", exc_info=True)
            return False
