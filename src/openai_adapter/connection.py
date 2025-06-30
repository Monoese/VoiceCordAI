"""
OpenAI Realtime Connection Management.

This module provides the OpenAIRealtimeConnection class, responsible for:
- Establishing and maintaining the connection to OpenAI's Realtime API using the openai library.
- Handling the asynchronous event stream from the API.
- Providing an interface to send commands/data to the API.
"""

from typing import Optional

from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import (
    AsyncRealtimeConnection as OpenAIConnectionObject,
)

from src.ai_services.base_connection import BaseConnectionHandler
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIRealtimeConnection(BaseConnectionHandler):
    """
    Manages a connection to the OpenAI Realtime API by implementing
    the provider-specific logic on top of BaseConnectionHandler.
    """

    def __init__(self, client: AsyncOpenAI, model_name: str) -> None:
        """
        Initializes the OpenAIRealtimeConnection.

        Args:
            client: An instance of AsyncOpenAI client.
            model_name: The name of the OpenAI model to use for the connection.

        Returns:
            None.
        """
        super().__init__()
        self.client: AsyncOpenAI = client
        self.model_name: str = model_name
        self._connection_object: Optional[OpenAIConnectionObject] = None

    async def _connection_logic(self) -> None:
        """
        Establishes and processes the OpenAI Realtime API connection.
        """
        logger.info(
            f"Attempting to connect to OpenAI Realtime API with model: {self.model_name}..."
        )
        try:
            # The `async with` block manages the WebSocket connection lifecycle.
            async with self.client.beta.realtime.connect(model=self.model_name) as conn:
                self._connection_object = conn
                self._connected_event.set()
                logger.info(
                    "Successfully connected to OpenAI Realtime API. Resetting retry delay."
                )
                self._retry_delay = 1.0  # Reset delay on successful connection

                async for event in conn:
                    if self._shutdown_signal.is_set():
                        logger.info(
                            "Shutdown signal detected during event iteration. Breaking loop."
                        )
                        break
                    await self._event_callback(event)
        finally:
            self._connection_object = None
            self._connected_event.clear()

    def get_active_connection(self) -> Optional[OpenAIConnectionObject]:
        """
        Returns the active OpenAI library connection object if connected.

        This object can be used by the manager to send events to the API.

        Returns:
            The AsyncRealtimeConnection object from the OpenAI library, or None if not connected.
        """
        return self._connection_object
