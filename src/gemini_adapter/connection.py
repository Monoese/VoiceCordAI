"""
Gemini Realtime Connection Management.

This module provides the GeminiRealtimeConnection class, responsible for:
- Establishing and maintaining the connection to Google's Gemini Live API.
- Handling the asynchronous event stream from the API.
- Translating the API's turn-based responses into a stream of synthetic events.
"""

import uuid
from typing import Optional

from google import genai
from google.genai import types

from src.ai_services.base_connection import BaseConnectionHandler
from src.utils.logger import get_logger
from .event_handler import TurnStartEvent, TurnMessageEvent, TurnEndEvent

logger = get_logger(__name__)


class GeminiRealtimeConnection(BaseConnectionHandler):
    """
    Manages a connection to the Gemini Live API by implementing
    the provider-specific logic on top of BaseConnectionHandler.
    """

    def __init__(
        self,
        gemini_client: genai.Client,
        model_name: str,
        live_connect_config_params: dict,
    ) -> None:
        """Initializes the GeminiRealtimeConnection.

        Args:
            gemini_client: An instance of genai.Client.
            model_name: The name of the Gemini model to use.
            live_connect_config_params: Configuration for the live connection.
        """
        super().__init__()
        self.gemini_client: genai.Client = gemini_client
        self.model_name: str = model_name
        self.live_connect_config_params: dict = live_connect_config_params
        self._session_object: Optional[genai.live.AsyncSession] = None

    async def _connection_logic(self) -> None:
        """
        Establishes and processes the Gemini Live API connection.
        """
        live_connect_config_obj = types.LiveConnectConfig(
            **self.live_connect_config_params
        )
        logger.info(
            f"Attempting to connect to Gemini Live API with model: {self.model_name}..."
        )

        try:
            async with self.gemini_client.aio.live.connect(
                model=self.model_name, config=live_connect_config_obj
            ) as session:
                self._session_object = session
                self._connected_event.set()
                logger.info(
                    "Successfully connected to Gemini Live API. Resetting retry delay."
                )
                self._retry_delay = 1.0  # Reset delay on successful connection

                while self.is_connected() and not self._shutdown_signal.is_set():
                    try:
                        # This is a blocking call that waits for the server to initiate a turn.
                        turn_iterator = self._session_object.receive()
                        # Gemini does not provide a turn ID, so we generate one for internal tracking.
                        turn_id = str(uuid.uuid4())
                        await self._event_callback(TurnStartEvent(turn_id=turn_id))

                        async for message in turn_iterator:
                            if self._shutdown_signal.is_set():
                                break
                            await self._event_callback(
                                TurnMessageEvent(message=message)
                            )

                        await self._event_callback(TurnEndEvent(turn_id=turn_id))
                        if self._shutdown_signal.is_set():
                            break
                    except Exception as e:
                        logger.error(
                            f"GeminiRealtimeConnection: Error in receive loop: {e}",
                            exc_info=True,
                        )
                        break  # Break inner loop to trigger reconnection
        finally:
            self._session_object = None
            self._connected_event.clear()

    def get_active_session(
        self,
    ) -> Optional[genai.live.AsyncSession]:
        """
        Returns the active Gemini AsyncSession object if connected.

        Returns:
            The active `AsyncSession` object, or `None` if not connected.
        """
        return self._session_object
