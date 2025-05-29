"""
WebSocket Manager module for handling WebSocket events and coordinating with the connection layer.

This module provides the WebSocketManager class which:
- Coordinates with WebSocketConnection for low-level connection management
- Sends events to the WebSocket server
- Receives and processes events from the WebSocket server
- Serializes events to JSON for transmission
- Deserializes JSON messages to event objects

The manager provides a clean interface for other components to send events
without worrying about the underlying connection details, while delegating
the actual connection management to the WebSocketConnection class.
"""

import json

from src.config.config import Config
from src.utils.logger import get_logger
from src.websocket.connection import WebSocketConnection
from src.websocket.connection_state import ConnectionState
from src.websocket.events.events import BaseEvent

log = get_logger(__name__)


class WebSocketManager:
    """
    Manages WebSocket events and coordinates with the connection layer.

    This class handles:
    - Coordinating with WebSocketConnection for low-level connection management
    - Sending events to the WebSocket server
    - Receiving and processing events from the WebSocket server
    - Serializing events to JSON for transmission
    - Deserializing JSON messages to event objects

    The manager provides a simple interface for other components to send events
    without having to worry about connection state or reconnection logic.
    """

    def __init__(self, event_handler_instance) -> None:
        """
        Initialize the WebSocketManager with an event handler.

        Args:
            event_handler_instance: The event handler to use for processing events
        """
        self._event_handler = event_handler_instance

        url: str = Config.WS_SERVER_URL
        headers = {
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }

        self._connection = WebSocketConnection(
            url=url, headers=headers, message_handler=self._handle_message
        )

    async def _handle_message(self, message: str) -> None:
        """
        Handle a message received from the WebSocket connection.

        Parses the JSON message into an event object, dispatches it to the
        event handler, and logs any errors during processing.

        Args:
            message: The raw message received from the WebSocket
        """
        try:
            event_dict = json.loads(message)
            event = BaseEvent.from_json(event_dict)
            if event:
                await self._event_handler.dispatch_event(event)
            else:
                # This case is handled by BaseEvent.from_json logging a warning,
                # but an additional debug log here can be useful for dropped messages.
                log.debug(f"Dropping unknown event type: {event_dict.get('type')}")
        except json.JSONDecodeError as e:
            log.error(f"Failed to decode JSON message: {e}")
        except Exception as exc:
            log.exception(f"Failed to process message: {exc}")

    async def start(self) -> None:
        """
        Start the WebSocket connection (idempotent).

        Delegates to the WebSocketConnection instance to start the connection.
        """
        await self._connection.start()

    async def stop(self) -> None:
        """
        Signal the WebSocket connection to stop and wait for a clean shutdown.

        Delegates to the WebSocketConnection instance to stop the connection.
        """
        await self._connection.stop()

    async def close(self) -> None:
        """Alias for stop() method."""
        await self.stop()

    async def ensure_connected(self, timeout: float = 30.0) -> bool:
        """
        Ensure that the WebSocket connection is established.

        Starts the connection if not already started, then waits for it to
        reach the CONNECTED state.

        Args:
            timeout: Maximum time to wait for connection in seconds

        Returns:
            bool: True if connected successfully, False otherwise
        """
        if not self._connection.connected:
            await self.start()

        if self._connection.connected:  # Check again in case start() was instant
            return True

        log.info("Waiting for WebSocket connection to be established...")
        result = await self._connection.wait_for_state(
            ConnectionState.CONNECTED, timeout
        )

        if result:
            log.info("WebSocket connection established successfully")
        else:
            log.warning(
                f"Failed to establish WebSocket connection within {timeout}s timeout"
            )

        return result

    async def send_event(self, event: BaseEvent) -> None:
        """
        Send an event to the server.

        This method serializes the event to JSON and sends it over the WebSocket connection.
        Note: This method does not check if the connection is established.
        Use safe_send_event() if you need to ensure the connection is established.

        Args:
            event: The event to send to the WebSocket server
        """
        try:
            json_data = event.to_json()
            await self._connection.send(json_data)
            log.debug(f"Sent event: {event.type}")
        except Exception as e:
            log.error(f"Error sending event {event.type}: {e}")
            raise  # Re-raise to allow caller to handle the error

    async def safe_send_event(self, event: BaseEvent, timeout: float = 30.0) -> bool:
        """
        Ensure connection is established and then send an event.

        Ensures the WebSocket connection is established before sending the event.

        Args:
            event: The event to send to the WebSocket server
            timeout: Maximum time to wait for connection in seconds

        Returns:
            bool: True if the event was sent successfully, False otherwise
        """
        if not await self.ensure_connected(timeout):
            log.error(f"Failed to send event {event.type}: WebSocket not connected")
            return False

        try:
            await self.send_event(event)
            log.debug(f"Successfully sent event {event.type}")
            return True
        except Exception as e:
            log.error(f"Error sending event {event.type}: {e}")
            return False

    @property
    def connected(self) -> bool:
        """
        Check if the WebSocket connection is currently established.

        Returns:
            bool: True if the connection is in CONNECTED state, False otherwise
        """
        return self._connection.connected

    @property
    def state(self) -> ConnectionState:
        """
        Get the current connection state.

        Returns:
            ConnectionState: The current state of the WebSocket connection
        """
        return self._connection.state

    @property
    def reconnect_attempts(self) -> int:
        """
        Get the number of reconnection attempts made.

        This counter resets when a connection is successfully established.

        Returns:
            int: The number of reconnection attempts
        """
        return self._connection.reconnect_attempts

    async def get_health_metrics(self) -> dict:
        """
        Get health metrics for the WebSocket connection.

        This method returns a dictionary with various metrics about the connection
        health, which can be useful for monitoring and debugging.

        Returns:
            dict: A dictionary containing health metrics
        """
        connection_metrics = await self._connection.get_health_metrics()
        manager_metrics = {"manager_type": "WebSocketManager"}
        return {**connection_metrics, **manager_metrics}

    async def wait_for_state(
        self, target_state: ConnectionState, timeout: float = None
    ) -> bool:
        """
        Wait for the connection to reach a specific state.

        This method is useful for components that need to wait for the WebSocket
        connection to reach a certain state before proceeding.

        Args:
            target_state: The state to wait for
            timeout: Maximum time to wait in seconds (None for no timeout)

        Returns:
            bool: True if the target state was reached, False if timed out
        """
        return await self._connection.wait_for_state(target_state, timeout)
