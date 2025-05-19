"""
WebSocket Manager module for handling WebSocket connections to external services.

This module provides the WebSocketManager class which:
- Establishes and maintains WebSocket connections
- Handles reconnection with exponential backoff on connection failures
- Sends events to the WebSocket server
- Receives and processes events from the WebSocket server
- Manages background tasks for sending and receiving data
- Implements a state machine to track connection status

The manager uses asyncio for asynchronous operation and provides a clean
interface for other components to send events without worrying about
the underlying connection details.
"""

import asyncio
import json
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed

from src.config.config import Config
from src.websocket.connection_state import ConnectionState
from src.websocket.events.events import BaseEvent
from src.utils.logger import get_logger

# Configure logger for this module
log = get_logger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections to external services.

    This class handles:
    - Establishing and maintaining WebSocket connections
    - Reconnecting automatically with exponential backoff on failures
    - Sending events to the WebSocket server via a queue system
    - Receiving events from the WebSocket server and dispatching them
    - Managing background tasks for connection, sending, and receiving
    - Tracking connection state using a state machine

    The manager provides a simple interface for other components to send events
    without having to worry about connection state or reconnection logic.
    """
    def __init__(self, event_handler_instance) -> None:
        self._url: str = Config.WS_SERVER_URL
        self._headers = {"Authorization": f"Bearer {Config.OPENAI_API_KEY}", "OpenAI-Beta": "realtime=v1", }
        self._event_handler = event_handler_instance

        self._outgoing_event_queue: asyncio.Queue[BaseEvent] = asyncio.Queue()

        self._websocket_connection: Optional[websockets.ClientConnection] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None
        self._connection_task: Optional[asyncio.Task] = None

        self._connection_active = asyncio.Event()
        self._connection_active.clear()

        # Initialize state machine
        self._state = ConnectionState.DISCONNECTED
        self._state_lock = asyncio.Lock()  # For thread-safe state transitions
        self._reconnect_attempts = 0

    async def _set_state(self, new_state: ConnectionState) -> None:
        """
        Set the connection state with proper logging and synchronization.

        This method ensures thread-safe state transitions and logs the state change
        for debugging purposes.

        Args:
            new_state: The new connection state to transition to
        """
        async with self._state_lock:
            old_state = self._state
            if old_state != new_state:
                log.info(f"WebSocket state transition: {old_state.name} → {new_state.name}")
                self._state = new_state

                # Reset reconnect attempts when successfully connected
                if new_state == ConnectionState.CONNECTED:
                    self._reconnect_attempts = 0
                # Increment reconnect attempts when reconnecting
                elif new_state == ConnectionState.RECONNECTING:
                    self._reconnect_attempts += 1

    async def start(self) -> None:
        """
        Start the background-reconnecting task (idempotent).

        Transitions the state to CONNECTING and starts the connection process.
        """
        if self._connection_task and not self._connection_task.done():
            return

        await self._set_state(ConnectionState.CONNECTING)
        self._connection_active.set()
        self._connection_task = asyncio.create_task(self._maintain_connection())

    async def stop(self) -> None:
        """
        Signal all loops to finish and wait for a clean shutdown.

        Transitions the state to DISCONNECTING and then to DISCONNECTED
        after all resources are cleaned up.
        """
        await self._set_state(ConnectionState.DISCONNECTING)
        self._connection_active.clear()

        if self._websocket_connection:
            await self._websocket_connection.close()

        for task in (self._receive_task, self._send_task, self._connection_task):
            if task and not task.done():
                task.cancel()

        await asyncio.gather(*(t for t in (self._receive_task, self._send_task, self._connection_task) if t),
            return_exceptions=True, )

        self._websocket_connection = None
        self._receive_task = None
        self._send_task = None
        self._connection_task = None

        await self._set_state(ConnectionState.DISCONNECTED)

    async def close(self) -> None:
        """Alias for stop() method."""
        await self.stop()

    async def ensure_connected(self, timeout: float = 30.0) -> bool:
        """
        Ensure that the WebSocket connection is established.

        This convenience method:
        1. Starts the connection if it's not already started
        2. Waits for the connection to reach the CONNECTED state
        3. Returns True if connected successfully, False otherwise

        Args:
            timeout: Maximum time to wait for connection in seconds

        Returns:
            bool: True if connected successfully, False otherwise
        """
        # Start the connection if it's not already started
        if not self._connection_task or self._connection_task.done():
            await self.start()

        # If we're already connected, return immediately
        if self._state == ConnectionState.CONNECTED:
            return True

        # Wait for the connection to be established
        log.info("Waiting for WebSocket connection to be established...")
        result = await self.wait_for_state(ConnectionState.CONNECTED, timeout)

        if result:
            log.info("WebSocket connection established successfully")
        else:
            log.warning(f"Failed to establish WebSocket connection within {timeout}s timeout")

        return result

    async def send_event(self, event: BaseEvent) -> None:
        """
        Queue an event to be sent to the server.

        Note: This method does not check if the connection is established.
        Use safe_send_event() if you need to ensure the connection is established.

        Args:
            event: The event to send to the WebSocket server
        """
        await self._outgoing_event_queue.put(event)

    async def safe_send_event(self, event: BaseEvent, timeout: float = 30.0) -> bool:
        """
        Ensure connection is established and then send an event.

        This convenience method:
        1. Ensures the WebSocket connection is established
        2. Sends the event if connected
        3. Returns True if the event was queued successfully, False otherwise

        Args:
            event: The event to send to the WebSocket server
            timeout: Maximum time to wait for connection in seconds

        Returns:
            bool: True if the event was queued successfully, False otherwise
        """
        # Ensure the connection is established
        if not await self.ensure_connected(timeout):
            log.error(f"Failed to send event {event.type}: WebSocket not connected")
            return False

        # Send the event
        try:
            await self.send_event(event)
            log.debug(f"Successfully queued event {event.type} for sending")
            return True
        except Exception as e:
            log.error(f"Error queuing event {event.type}: {e}")
            return False

    @property
    def connected(self) -> bool:
        """
        Check if the WebSocket connection is currently established.

        Returns:
            bool: True if the connection is in CONNECTED state, False otherwise
        """
        return self._state == ConnectionState.CONNECTED

    @property
    def connection(self):
        """
        Get the current WebSocket connection object.

        Returns:
            The WebSocket connection object or None if not connected
        """
        return self._websocket_connection

    @property
    def state(self) -> ConnectionState:
        """
        Get the current connection state.

        Returns:
            ConnectionState: The current state of the WebSocket connection
        """
        return self._state

    @property
    def reconnect_attempts(self) -> int:
        """
        Get the number of reconnection attempts made.

        This counter resets when a connection is successfully established.

        Returns:
            int: The number of reconnection attempts
        """
        return self._reconnect_attempts

    async def get_health_metrics(self) -> dict:
        """
        Get health metrics for the WebSocket connection.

        This method returns a dictionary with various metrics about the connection
        health, which can be useful for monitoring and debugging.

        Returns:
            dict: A dictionary containing health metrics
        """
        metrics = {
            "state": self._state.name,
            "connected": self.connected,
            "reconnect_attempts": self._reconnect_attempts,
            "outgoing_queue_size": self._outgoing_event_queue.qsize(),
            "outgoing_queue_empty": self._outgoing_event_queue.empty(),
            "has_active_connection": self._websocket_connection is not None,
            "has_receive_task": self._receive_task is not None and not self._receive_task.done(),
            "has_send_task": self._send_task is not None and not self._send_task.done(),
            "has_connection_task": self._connection_task is not None and not self._connection_task.done(),
        }

        # Add WebSocket-specific metrics if connected
        if self._websocket_connection is not None:
            try:
                metrics.update({
                    "ws_open": not self._websocket_connection.closed,
                    "ws_closing": self._websocket_connection.closing,
                })
            except Exception:
                # Ignore errors when trying to access WebSocket properties
                pass

        return metrics

    async def wait_for_state(self, target_state: ConnectionState, timeout: float = None) -> bool:
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
        if self._state == target_state:
            return True

        # Create a future that will be set when the state changes to the target state
        future = asyncio.Future()

        # Define a callback to check state changes
        async def state_change_callback(old_state: ConnectionState, new_state: ConnectionState):
            if new_state == target_state and not future.done():
                future.set_result(True)

        # Store the original _set_state method
        original_set_state = self._set_state

        # Override _set_state to check for the target state
        async def wrapped_set_state(new_state: ConnectionState):
            old_state = self._state
            await original_set_state(new_state)
            await state_change_callback(old_state, new_state)

        # Replace the method temporarily
        self._set_state = wrapped_set_state

        try:
            # Wait for the future to be set or timeout
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            log.warning(f"Timeout waiting for state {target_state.name}")
            return False
        finally:
            # Restore the original method
            self._set_state = original_set_state

    async def _maintain_connection(self) -> None:
        """
        Maintain a WebSocket connection with exponential back‑off until ``stop`` is called.

        This method manages the connection lifecycle and state transitions:
        - CONNECTING: Initial connection attempt
        - CONNECTED: Successfully connected
        - RECONNECTING: Connection lost, attempting to reconnect
        - ERROR: Connection error occurred

        The method implements exponential backoff for reconnection attempts.
        """
        backoff = 1
        while self._connection_active.is_set():
            try:
                # If we're reconnecting after a failure, update the state
                if self._state == ConnectionState.ERROR:
                    await self._set_state(ConnectionState.RECONNECTING)

                # Establish a single connection (this will set state to CONNECTED if successful)
                await self._establish_single_connection()

                # Reset backoff on successful connection
                backoff = 1
            except asyncio.CancelledError:
                # Don't change state on cancellation as it's handled by stop()
                raise
            except Exception as exc:
                # Set state to ERROR on connection failure
                await self._set_state(ConnectionState.ERROR)
                log.error(f"WebSocket error: {exc} – reconnecting in {backoff}s (attempt {self._reconnect_attempts})")

                # Wait before reconnecting with exponential backoff
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _establish_single_connection(self) -> None:
        """
        Establish a single WebSocket connection; returns when it closes.

        This method:
        1. Establishes a WebSocket connection
        2. Transitions to CONNECTED state when successful
        3. Sets up receive and send tasks
        4. Waits for tasks to complete or fail
        5. Transitions back to appropriate state when connection closes
        """
        try:
            # Maintain CONNECTING state from _connect_forever or start()
            async with websockets.connect(self._url, additional_headers=self._headers) as ws:
                self._websocket_connection = ws

                # Transition to CONNECTED state
                await self._set_state(ConnectionState.CONNECTED)
                log.info(f"Connected to WebSocket server → {self._url}")

                # Start receive and send tasks
                self._receive_task = asyncio.create_task(self._receive_loop())
                self._send_task = asyncio.create_task(self._send_loop())

                # Wait for either task to complete or fail
                done, pending = await asyncio.wait(
                    (self._receive_task, self._send_task),
                    return_when=asyncio.FIRST_EXCEPTION,
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()

                # Check for exceptions in completed tasks
                for task in done:
                    try:
                        task.result()
                    except Exception as e:
                        log.error(f"Task failed with error: {e}")
                        # We'll transition to ERROR state in _connect_forever
        finally:
            # Clear the WebSocket reference
            self._websocket_connection = None

            # If we're not already in DISCONNECTING or ERROR state (handled by other methods),
            # transition to DISCONNECTED if the connection was closed normally
            if self._state not in (ConnectionState.DISCONNECTING, ConnectionState.ERROR):
                await self._set_state(ConnectionState.DISCONNECTED)

    async def _receive_loop(self) -> None:
        """
        Background task: convert raw JSON → ``BaseEvent`` → queue.

        This method processes incoming WebSocket messages and dispatches events
        to the event handler. It handles errors gracefully to prevent the connection
        from being terminated unnecessarily.
        """
        assert self._websocket_connection is not None
        try:
            async for message in self._websocket_connection:
                try:
                    event_dict = json.loads(message)
                    event = BaseEvent.from_json(event_dict)
                    if event:
                        await self._event_handler.dispatch_event(event)
                    else:
                        log.debug(f"Dropping unknown event type: {event_dict.get('type')}")
                except json.JSONDecodeError as e:
                    log.error(f"Failed to decode JSON message: {e}")
                except Exception as exc:
                    log.exception(f"Failed to process message: {exc}")
        except ConnectionClosed as e:
            # Normal closure or connection error
            if e.code == 1000:  # Normal closure
                log.info(f"WebSocket connection closed normally: {e}")
            else:
                log.warning(f"WebSocket connection closed with code {e.code}: {e}")
                # The state transition will be handled by _connect_once and _connect_forever
        except Exception as e:
            log.error(f"Unexpected error in receive loop: {e}")
            # The state transition will be handled by _connect_once and _connect_forever

    async def _send_loop(self) -> None:
        """
        Background task: pop events from queue and transmit.

        This method sends events from the outgoing queue to the WebSocket server.
        It handles connection errors gracefully and ensures the queue is properly
        managed even when errors occur.
        """
        assert self._websocket_connection is not None
        try:
            while True:
                event: BaseEvent = await self._outgoing_event_queue.get()
                try:
                    await self._websocket_connection.send(event.to_json())
                    log.debug(f"Sent event: {event.type}")
                except ConnectionClosed as e:
                    log.warning(f"Connection closed while sending: {e}")
                    # Put the event back in the queue if it's important
                    # This could be enhanced with priority or persistence
                    await self._outgoing_event_queue.put(event)
                    # Break the loop to allow reconnection
                    break
                except TypeError as e:
                    log.warning(f"Message doesn't have a supported type: {e}")
                except Exception as e:
                    log.error(f"Error sending event {event.type}: {e}")
                finally:
                    self._outgoing_event_queue.task_done()
        except Exception as e:
            log.error(f"Unexpected error in send loop: {e}")
            # The state transition will be handled by _connect_once and _connect_forever
