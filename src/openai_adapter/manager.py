"""
OpenAI Realtime API Manager.

This module provides the OpenAIRealtimeManager class, which will serve as
the primary interface for the bot to interact with the OpenAI Realtime API
via the new adapter. It will coordinate connection management, event sending,
and event handling.
"""

import asyncio
import base64  # Added import
from typing import Optional, Dict, Any

from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import (
    AsyncRealtimeConnection as OpenAIConnectionObject,
)

from src.config.config import Config
from src.utils.logger import get_logger
from src.audio.audio import AudioManager
from .connection import OpenAIRealtimeConnection
from .event_mapper import OpenAIEventHandlerAdapter
from src.ai_services.interface import IRealtimeAIServiceManager  # Added import

logger = get_logger(__name__)


class OpenAIRealtimeManager(
    IRealtimeAIServiceManager
):  # Inherit from IRealtimeAIServiceManager
    """
    Manages interactions with the OpenAI Realtime API using the OpenAI Python library.

    This class handles:
    - Initializing the AsyncOpenAI client.
    - Managing the OpenAIRealtimeConnection for WebSocket communication.
    - Coordinating with OpenAIEventHandlerAdapter for processing incoming events.
    - Providing methods to send various commands/data to the OpenAI Realtime API.
    - Managing the connection lifecycle (connect, disconnect) as per the IRealtimeAIServiceManager interface.
    """

    def __init__(
        self, audio_manager: AudioManager, service_config: Dict[str, Any]
    ):  # Added service_config
        """
        Initializes the OpenAIRealtimeManager.

        Args:
            audio_manager: An instance of AudioManager, required by the event handler.
            service_config: Configuration specific to this service instance.
        """
        super().__init__(audio_manager, service_config)  # Call super().__init__
        # self.audio_manager is now set by super()
        # self._service_config is now set by super()
        # self._is_connected_flag is now set by super()

        # Initialize the OpenAI client. It uses OPENAI_API_KEY from env by default.
        # Config.OPENAI_API_KEY ensures it's loaded.
        # service_config could override API key if needed, e.g., self._service_config.get("api_key", Config.OPENAI_API_KEY)
        self.openai_client: AsyncOpenAI = AsyncOpenAI(
            api_key=self._service_config.get("api_key", Config.OPENAI_API_KEY)
        )

        self.event_handler_adapter: OpenAIEventHandlerAdapter = (
            OpenAIEventHandlerAdapter(audio_manager=self._audio_manager)
        )
        self.connection_handler: OpenAIRealtimeConnection = OpenAIRealtimeConnection(
            client=self.openai_client
        )
        # self._main_task: Optional[asyncio.Task] = None # This was not used actively

    async def _get_active_conn(self) -> Optional[OpenAIConnectionObject]:
        """Helper to get the active connection object if connected."""
        # Use the flag from the parent class for the primary check
        if not self._is_connected_flag:
            logger.debug("_get_active_conn: Not connected based on internal flag.")
            return None

        # Double check with connection_handler for sanity, though _is_connected_flag should be authoritative
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

    # --- Interface Methods Implementation ---

    async def connect(self) -> bool:
        """
        Establishes a connection to the OpenAI Realtime API and initializes the session.
        """
        logger.info("OpenAIRealtimeManager: Attempting to connect...")
        if self._is_connected_flag:
            logger.info("OpenAIRealtimeManager: Already connected.")
            return True

        await self.connection_handler.connect(self.event_handler_adapter.dispatch_event)

        # Wait for the connection to be established
        timeout = self._service_config.get("connection_timeout", 30.0)
        wait_interval = 0.1
        max_attempts = int(timeout / wait_interval)

        for attempt in range(max_attempts):
            if (
                self.connection_handler.is_connected()
            ):  # Check underlying connection status
                logger.info(
                    "OpenAIRealtimeManager: Connection successfully established with handler."
                )

                # Send initial session update if configured
                initial_session_data = self._service_config.get("initial_session_data")
                if initial_session_data:
                    # Directly use connection_handler to get the connection object at this stage
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
                            # Clean up using the connection_handler directly and ensure our flag is false
                            await self.connection_handler.disconnect()
                            self._is_connected_flag = False
                            return False
                    else:
                        # This case should ideally not be hit if connection_handler.is_connected() was true
                        logger.error(
                            "Failed to get active connection object from connection_handler for session update, though handler reported connected."
                        )
                        await self.connection_handler.disconnect()
                        self._is_connected_flag = False
                        return False

                self._is_connected_flag = True  # Set our authoritative flag
                return True
            if attempt < max_attempts - 1:
                await asyncio.sleep(wait_interval)

        logger.warning(
            f"OpenAIRealtimeManager: Failed to establish connection within {timeout}s timeout."
        )
        await (
            self.connection_handler.disconnect()
        )  # Ensure cleanup if connection attempt failed
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

    # is_connected() is inherited from IRealtimeAIServiceManager and uses self._is_connected_flag

    async def send_audio_chunk(self, audio_data: bytes) -> bool:
        """
        Sends a chunk of raw audio data to the OpenAI service.
        The data is base64 encoded before sending.
        """
        conn = await self._get_active_conn()
        if not conn:
            logger.error("Cannot send audio chunk: Not connected.")
            return False
        try:
            audio_b64 = base64.b64encode(audio_data).decode("utf-8")
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

        # Example mapping to conversation.item.create
        # This assumes the text is a user message.
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
            # If finalize_turn is True, we might immediately want to request a response.
            # This depends on the desired interaction flow.
            if finalize_turn:
                return (
                    await self.finalize_input_and_request_response()
                )  # This will call create_response
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
            # Commit audio buffer
            await conn.input_audio_buffer.commit()
            logger.info("Sent input_audio_buffer.commit")

            # Create response
            # response_data can be specified in service_config or passed if needed
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

        # The OpenAI library example `push_to_talk_app.py` uses `connection.send` for this.
        # `connection.send` takes a dictionary.
        # The response_id might be tracked by AudioManager or VoiceCog if needed.
        # For now, sending a general cancel.
        response_id_to_cancel = (
            self._audio_manager.get_current_playing_response_id()
        )  # Use self._audio_manager

        payload: Dict[str, Any] = {"type": "response.cancel"}
        if response_id_to_cancel:
            # The OpenAI API expects the response_id of the response being generated,
            # not the stream_id used by AudioManager.
            # This needs careful mapping if response_id is available.
            # Assuming for now that if audio_manager has a response_id, it's the one to cancel.
            payload["response_id"] = response_id_to_cancel
            # This assumes get_current_playing_response_id() returns the actual OpenAI response_id

        try:
            await conn.send(payload)
            logger.info(f"Sent response.cancel (payload: {payload})")
            return True
        except AttributeError:  # Should not happen with official library
            logger.error(
                "The 'send' method is not available on the connection object. "
                "This method for cancelling responses might need updating based on library capabilities."
            )
            return False
        except Exception as e:
            logger.error(f"Error sending response.cancel: {e}", exc_info=True)
            return False

    # --- Old methods to be removed or adapted ---
    # send_session_update, original send_audio_chunk, commit_audio_buffer,
    # send_conversation_item, create_response, original cancel_response
    # start, stop, ensure_connected, original is_connected

    # The following methods are effectively replaced by the interface methods.
    # If any specific internal logic from them is needed, it's merged into the interface methods.

    # Original send_session_update - logic moved to connect() for initial setup
    # Original send_audio_chunk - logic moved to new send_audio_chunk(self, audio_data: bytes)
    # Original commit_audio_buffer - logic moved to finalize_input_and_request_response
    # Original send_conversation_item - can be used by send_text_message or kept as helper
    # Original create_response - logic moved to finalize_input_and_request_response
    # Original cancel_response - logic moved to cancel_ongoing_response
    # Original start - logic moved to connect()
    # Original stop - logic moved to disconnect()
    # Original ensure_connected - this specific pattern is not directly part of the interface.
    # VoiceCog will call connect() if not is_connected().
    # Original is_connected - replaced by parent's implementation using _is_connected_flag
