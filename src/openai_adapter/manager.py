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

from src.config.config import Config
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
        """
        super().__init__(audio_manager, service_config)

        self.openai_client: AsyncOpenAI = AsyncOpenAI(
            api_key=self._service_config.get("api_key", Config.OPENAI_API_KEY)
        )

        self.event_handler_adapter: OpenAIEventHandlerAdapter = (
            OpenAIEventHandlerAdapter(
                audio_manager=self._audio_manager,
                response_audio_format=self.response_audio_format,
            )
        )
        self.connection_handler: OpenAIRealtimeConnection = OpenAIRealtimeConnection(
            client=self.openai_client
        )

    async def _get_active_conn(self) -> Optional[AsyncRealtimeConnection]:
        """Helper to get the active connection object if connected."""
        if not self._is_connected_flag:
            logger.debug("_get_active_conn: Not connected based on internal flag.")
            return None

        if self.connection_handler.is_connected():
            conn = self.connection_handler.get_active_connection()
            if conn:
                return conn
            else:
                logger.warning(
                    "_get_active_conn: _is_connected_flag is True, but connection_handler.get_active_connection returned None."
                )
        else:
            logger.warning(
                "_get_active_conn: _is_connected_flag is True, but connection_handler.is_connected is False. State inconsistency."
            )
        return None

    async def connect(self) -> bool:
        """
        Establishes a connection to the OpenAI Realtime API and initializes the session.
        """
        logger.info("OpenAIRealtimeManager: Attempting to connect...")
        if self._is_connected_flag:
            logger.info("OpenAIRealtimeManager: Already connected.")
            return True

        await self.connection_handler.connect(self.event_handler_adapter.dispatch_event)

        timeout = self._service_config.get("connection_timeout", 30.0)
        wait_interval = 0.1
        max_attempts = int(timeout / wait_interval)

        for attempt in range(max_attempts):
            if self.connection_handler.is_connected():
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
                            self._is_connected_flag = False
                            return False
                    else:
                        logger.error(
                            "Failed to get active connection object from connection_handler for session update, though handler reported connected."
                        )
                        await self.connection_handler.disconnect()
                        self._is_connected_flag = False
                        return False

                self._is_connected_flag = True
                return True
            if attempt < max_attempts - 1:
                await asyncio.sleep(wait_interval)

        logger.warning(
            f"OpenAIRealtimeManager: Failed to establish connection within {timeout}s timeout."
        )
        await self.connection_handler.disconnect()
        self._is_connected_flag = False
        return False

    async def disconnect(self) -> None:
        """
        Closes the connection to the OpenAI Realtime API and cleans up resources.
        """
        logger.info("OpenAIRealtimeManager: Attempting to disconnect...")
        await self.connection_handler.disconnect()
        self._is_connected_flag = False
        logger.info("OpenAIRealtimeManager: Disconnected.")

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

    async def send_text_message(self, text: str, finalize_turn: bool) -> bool:
        """
        Sends a text message. OpenAI's real-time API might not directly support
        this in the same way as audio. This is a placeholder or needs specific mapping.
        Using conversation.item.create for now if text is provided.
        The `finalize_turn` parameter is not directly used by OpenAI's `conversation.item.create`
        in the same way as Gemini's `turn_complete`.
        """
        conn = await self._get_active_conn()
        if not conn:
            logger.error("Cannot send text message: Not connected.")
            return False

        if not text:
            logger.warning("send_text_message called with empty text.")
            return False

        item_data = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }
        try:
            await conn.conversation.item.create(item=item_data)
            logger.info(
                f"Sent conversation.item.create for text message: {text} (finalize_turn: {finalize_turn})"
            )
            if finalize_turn:
                return await self.finalize_input_and_request_response()
            return True
        except Exception as e:
            logger.error(
                f"Error sending conversation.item.create for text: {e}", exc_info=True
            )
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
        except AttributeError:
            logger.error(
                "The 'send' method is not available on the connection object. "
                "This method for cancelling responses might need updating based on library capabilities."
            )
            return False
        except Exception as e:
            logger.error(f"Error sending response.cancel: {e}", exc_info=True)
            return False
