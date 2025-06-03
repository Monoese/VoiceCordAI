"""
OpenAI Realtime API Manager.

This module provides the OpenAIRealtimeManager class, which will serve as
the primary interface for the bot to interact with the OpenAI Realtime API
via the new adapter. It will coordinate connection management, event sending,
and event handling.
"""

import asyncio
from typing import Optional, Dict, Any

from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import (
    AsyncRealtimeConnection as OpenAIConnectionObject,
)

from src.config.config import Config
from src.utils.logger import get_logger
from src.audio.audio import AudioManager  # Required for OpenAIEventHandlerAdapter
from .connection import OpenAIRealtimeConnection
from .event_mapper import OpenAIEventHandlerAdapter

logger = get_logger(__name__)


class OpenAIRealtimeManager:
    """
    Manages interactions with the OpenAI Realtime API using the OpenAI Python library.

    This class handles:
    - Initializing the AsyncOpenAI client.
    - Managing the OpenAIRealtimeConnection for WebSocket communication.
    - Coordinating with OpenAIEventHandlerAdapter for processing incoming events.
    - Providing methods to send various commands/data to the OpenAI Realtime API.
    - Managing the lifecycle (start, stop) of the connection.
    """

    def __init__(self, audio_manager: AudioManager):
        """
        Initializes the OpenAIRealtimeManager.

        Args:
            audio_manager: An instance of AudioManager, required by the event handler.
        """
        self.audio_manager: AudioManager = audio_manager

        # Initialize the OpenAI client. It uses OPENAI_API_KEY from env by default.
        # Config.OPENAI_API_KEY ensures it's loaded.
        self.openai_client: AsyncOpenAI = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

        self.event_handler_adapter: OpenAIEventHandlerAdapter = (
            OpenAIEventHandlerAdapter(audio_manager=self.audio_manager)
        )
        self.connection_handler: OpenAIRealtimeConnection = OpenAIRealtimeConnection(
            client=self.openai_client
        )
        self._main_task: Optional[asyncio.Task] = None

    async def _get_active_conn(self) -> Optional[OpenAIConnectionObject]:
        """Helper to get the active connection object if connected."""
        if self.connection_handler.is_connected():
            conn = self.connection_handler.get_active_connection()
            if conn:
                return conn
            else:
                logger.warning(
                    "_get_active_conn: is_connected is True, but get_active_connection returned None."
                )
        logger.debug("_get_active_conn: Not connected or no active connection object.")
        return None

    # --- Methods for Sending Events to OpenAI Realtime API ---

    async def send_session_update(self, session_data: Dict[str, Any]) -> bool:
        """
        Updates the session configuration.
        Example: `session_data={"turn_detection": {"type": "server_vad"}}`
                 `session_data={"modalities": ["text", "audio"]}`
        """
        conn = await self._get_active_conn()
        if not conn:
            logger.error("Cannot send session update: Not connected.")
            return False
        try:
            await conn.session.update(session=session_data)
            logger.info(f"Sent session.update: {session_data}")
            return True
        except Exception as e:
            logger.error(f"Error sending session.update: {e}", exc_info=True)
            return False

    async def send_audio_chunk(self, audio_b64: str) -> bool:
        """Appends a base64 encoded audio chunk to the input audio buffer."""
        conn = await self._get_active_conn()
        if not conn:
            logger.error("Cannot send audio chunk: Not connected.")
            return False
        try:
            await conn.input_audio_buffer.append(audio=audio_b64)
            logger.debug("Sent input_audio_buffer.append")
            return True
        except Exception as e:
            logger.error(f"Error sending input_audio_buffer.append: {e}", exc_info=True)
            return False

    async def commit_audio_buffer(self) -> bool:
        """Commits the currently buffered input audio for processing."""
        conn = await self._get_active_conn()
        if not conn:
            logger.error("Cannot commit audio buffer: Not connected.")
            return False
        try:
            await conn.input_audio_buffer.commit()
            logger.info("Sent input_audio_buffer.commit")
            return True
        except Exception as e:
            logger.error(f"Error sending input_audio_buffer.commit: {e}", exc_info=True)
            return False

    async def send_conversation_item(self, item_data: Dict[str, Any]) -> bool:
        """
        Creates a new item in the conversation.
        Example: `item_data={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}`
        """
        conn = await self._get_active_conn()
        if not conn:
            logger.error("Cannot send conversation item: Not connected.")
            return False
        try:
            await conn.conversation.item.create(item=item_data)
            logger.info(f"Sent conversation.item.create: {item_data}")
            return True
        except Exception as e:
            logger.error(f"Error sending conversation.item.create: {e}", exc_info=True)
            return False

    async def create_response(
        self, response_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Requests the creation of a new response from the server.
        `response_data` can specify modalities, e.g., `{"modalities": ["text", "audio"]}`.
        If None, defaults are used by the API.
        """
        conn = await self._get_active_conn()
        if not conn:
            logger.error("Cannot create response: Not connected.")
            return False
        try:
            await conn.response.create(response=response_data or {})
            logger.info(f"Sent response.create (data: {response_data or {}})")
            return True
        except Exception as e:
            logger.error(f"Error sending response.create: {e}", exc_info=True)
            return False

    async def cancel_response(self, response_id: Optional[str] = None) -> bool:
        """
        Sends a request to cancel an in-progress response.
        If `response_id` is None, the server attempts to cancel the default/current response.
        """
        conn = await self._get_active_conn()
        if not conn:
            logger.error("Cannot cancel response: Not connected.")
            return False

        # The OpenAI library example `push_to_talk_app.py` uses `connection.send` for this.
        # `connection.send` takes a dictionary.
        payload: Dict[str, Any] = {"type": "response.cancel"}
        if response_id:
            payload["response_id"] = response_id

        try:
            # Assuming `conn.send` is available and is the correct method as per library examples
            # for messages not covered by specific helper methods.
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

    # --- Lifecycle Management ---

    async def start(self) -> None:
        """
        Starts the connection to OpenAI Realtime API and the event processing loop.
        This method is idempotent; it won't start a new connection if one is already active or starting.
        The actual connection and event loop are managed by `OpenAIRealtimeConnection`,
        which creates its own background task.
        """
        logger.info("OpenAIRealtimeManager: Attempting to start connection...")
        if self.connection_handler.is_connected():
            logger.info("OpenAIRealtimeManager: Connection is already active.")
            return

        # OpenAIRealtimeConnection.connect is designed to be idempotent and manages its own task.
        # We pass the event handler's dispatch method as the callback.
        await self.connection_handler.connect(self.event_handler_adapter.dispatch_event)
        # The _main_task in this manager isn't strictly necessary if connection_handler manages its own task
        # and provides sufficient status. For now, we assume connection_handler.connect() is non-blocking
        # in terms of starting its internal task.
        logger.info(
            "OpenAIRealtimeManager: Connection process initiated via connection_handler."
        )

    async def stop(self) -> None:
        """
        Stops the connection to OpenAI Realtime API and cleans up resources.
        """
        logger.info("OpenAIRealtimeManager: Attempting to stop connection...")
        await self.connection_handler.disconnect()
        # The OpenAIRealtimeConnection.disconnect() method handles waiting for its internal task.
        logger.info("OpenAIRealtimeManager: Connection stopped via connection_handler.")

    async def ensure_connected(self, timeout: float = 30.0) -> bool:
        """
        Ensures the connection to OpenAI Realtime API is active.
        If not connected, it attempts to start the connection and waits for it to establish.

        Args:
            timeout: Maximum time in seconds to wait for the connection to establish.

        Returns:
            True if the connection is active or successfully established, False otherwise.
        """
        if self.is_connected():
            logger.debug("ensure_connected: Already connected.")
            return True

        logger.info(
            "ensure_connected: Not connected. Attempting to start and connect..."
        )
        await self.start()  # Initiates the connection process

        # Wait for the connection to be established
        # This loop polls the is_connected status.
        # A more sophisticated approach might involve an asyncio.Event signaled by OpenAIRealtimeConnection.
        wait_interval = 0.1  # seconds
        max_attempts = int(timeout / wait_interval)

        for attempt in range(max_attempts):
            if self.is_connected():
                logger.info("ensure_connected: Connection successfully established.")
                return True
            if attempt < max_attempts - 1:  # Don't sleep on the last attempt
                await asyncio.sleep(wait_interval)
            else:  # Log before final check
                logger.debug(
                    f"ensure_connected: Waited {timeout}s, final connection check."
                )

        if self.is_connected():  # Final check after loop
            logger.info(
                "ensure_connected: Connection successfully established (checked after loop)."
            )
            return True

        logger.warning(
            f"ensure_connected: Failed to establish connection within {timeout}s timeout."
        )
        return False

    def is_connected(self) -> bool:
        """Checks if the underlying connection handler reports being connected."""
        return self.connection_handler.is_connected()

    # TODO: Add health metrics similar to WebSocketManager if needed
    # async def get_health_metrics(self) -> dict:
    #     ...
