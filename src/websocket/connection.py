"""
WebSocket Connection module for handling low-level WebSocket connections.

This module provides the WebSocketConnection class which:
- Establishes and maintains WebSocket connections
- Handles reconnection with exponential backoff on connection failures
- Manages the connection state machine
- Provides methods for sending and receiving data
- Handles connection errors and cleanup

The connection class focuses on the low-level connection management,
while higher-level event handling is delegated to the WebSocketManager.
"""

import asyncio
from typing import Optional, Callable, Awaitable, Dict

import websockets
from websockets.exceptions import ConnectionClosed

from src.utils.logger import get_logger
from src.websocket.connection_state import ConnectionState

log = get_logger(__name__)


class WebSocketConnection:
    """
    Handles low-level WebSocket connection management.

    This class handles:
    - Establishing and maintaining WebSocket connections
    - Reconnecting automatically with exponential backoff on failures
    - Managing the connection state machine
    - Providing methods for sending and receiving data
    - Handling connection errors and cleanup

    The connection class is responsible for the network layer,
    while event processing is handled by the WebSocketManager.
    """

    def __init__(
        self,
        url: str,
        headers: Dict[str, str],
        message_handler: Callable[[str], Awaitable[None]],
    ) -> None:
        """
        Initialize the WebSocketConnection with connection parameters.

        Args:
            url: The WebSocket server URL to connect to
            headers: HTTP headers to use for the connection
            message_handler: Callback function to handle received messages
        """
        self._url = url
        self._headers = headers
        self._message_handler = message_handler

        self._websocket: Optional[websockets.ClientConnection] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None
        self._connection_task: Optional[asyncio.Task] = None

        self._connection_active = (
            asyncio.Event()
        )  # Controls the _maintain_connection loop
        self._connection_active.clear()

        self._state = ConnectionState.DISCONNECTED
        self._state_lock = asyncio.Lock()  # Protects access to _state
        self._reconnect_attempts = 0
        self._state_condition = asyncio.Condition(
            lock=self._state_lock
        )  # For waiting on state changes

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
                log.info(
                    f"WebSocket state transition: {old_state.name} → {new_state.name}"
                )
                self._state = new_state

                # Reset reconnect attempts when successfully connected
                if new_state == ConnectionState.CONNECTED:
                    self._reconnect_attempts = 0  # Reset on successful connection
                elif new_state == ConnectionState.RECONNECTING:
                    self._reconnect_attempts += 1  # Increment on attempt to reconnect

                self._state_condition.notify_all()  # Notify waiters of state change

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

        if self._websocket:
            await self._websocket.close()

        for task in (self._receive_task, self._send_task, self._connection_task):
            if task and not task.done():
                task.cancel()

        await asyncio.gather(
            *(
                t
                for t in (self._receive_task, self._send_task, self._connection_task)
                if t
            ),
            return_exceptions=True,
        )

        self._websocket = None
        self._receive_task = None
        self._send_task = None
        self._connection_task = None

        await self._set_state(ConnectionState.DISCONNECTED)

    async def close(self) -> None:
        """Alias for stop() method."""
        await self.stop()

    async def send(self, message: str) -> None:
        """
        Send a message over the WebSocket connection.

        Args:
            message: The message to send (should be a JSON string)

        Raises:
            ConnectionClosed: If the connection is closed
            RuntimeError: If the connection is not established
        """
        if not self._websocket:
            raise RuntimeError("WebSocket connection not established")

        await self._websocket.send(message)

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
        async with self._state_condition:  # Acquires the lock for condition variable
            if self._state == target_state:
                return True

            try:
                # self._state_condition.wait_for releases the lock while waiting
                # and reacquires it before returning.
                await asyncio.wait_for(
                    self._state_condition.wait_for(lambda: self._state == target_state),
                    timeout=timeout,
                )
                return True  # State successfully reached target_state
            except asyncio.TimeoutError:
                log.warning(f"Timeout waiting for state {target_state.name}")
                return self._state == target_state  # Final check after timeout
            # asyncio.CancelledError will propagate if the waiting task is cancelled.

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
                if self._state == ConnectionState.ERROR:  # If previous attempt failed
                    await self._set_state(ConnectionState.RECONNECTING)

                # This call blocks until the connection is established and then subsequently closed or fails.
                await self._establish_single_connection()

                backoff = 1  # Reset backoff after a successful connection period
            except asyncio.CancelledError:
                # Cancellation is handled by stop(), so just re-raise.
                raise
            except Exception as exc:
                await self._set_state(ConnectionState.ERROR)  # Mark state as error
                log.error(
                    f"WebSocket error: {exc} – reconnecting in {backoff}s (attempt {self._reconnect_attempts})"
                )
                await asyncio.sleep(backoff)  # Exponential backoff
                backoff = min(backoff * 2, 60)  # Cap backoff at 60 seconds

    async def _establish_single_connection(self) -> None:
        """
        Establish a single WebSocket connection; returns when it closes.

        This method:
        1. Establishes a WebSocket connection
        2. Transitions to CONNECTED state when successful
        3. Sets up receive task
        4. Waits for task to complete or fail
        5. Transitions back to appropriate state when connection closes
        """
        try:
            # This context manager handles the WebSocket handshake and connection.
            async with websockets.connect(
                self._url, additional_headers=self._headers
            ) as ws:
                self._websocket = ws
                await self._set_state(ConnectionState.CONNECTED)
                log.info(f"Connected to WebSocket server → {self._url}")

                self._receive_task = asyncio.create_task(self._receive_loop())

                # This will block until _receive_loop finishes (due to connection close or error)
                # or is cancelled.
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    # Expected during shutdown via stop()
                    raise
                except Exception as e:
                    log.error(f"Receive task failed with error: {e}")
                    # The outer _maintain_connection loop will handle setting ERROR state.
        finally:
            self._websocket = None  # Clear WebSocket reference

            # If the connection closed and we are not in a deliberate DISCONNECTING state
            # or an ERROR state (which _maintain_connection will handle),
            # then transition to DISCONNECTED. This covers normal closures.
            if self._state not in (
                ConnectionState.DISCONNECTING,  # stop() is handling this
                ConnectionState.ERROR,  # _maintain_connection will handle this
            ):
                await self._set_state(ConnectionState.DISCONNECTED)

    async def _receive_loop(self) -> None:
        """
        Background task: receive messages and pass them to the message handler.

        This method processes incoming WebSocket messages and passes them to
        the message handler callback. It handles errors gracefully to prevent
        the connection from being terminated unnecessarily.
        """
        assert self._websocket is not None
        try:
            async for message in self._websocket:
                try:
                    await self._message_handler(message)
                except Exception as exc:
                    log.exception(f"Failed to process message: {exc}")
        except ConnectionClosed as e:
            # Normal closure or connection error
            if e.code == 1000:  # Standard normal closure
                log.info(f"WebSocket connection closed normally: {e}")
            else:  # Other closure codes indicate potential issues
                log.warning(f"WebSocket connection closed with code {e.code}: {e}")
            # State transitions due to closure are handled by _establish_single_connection's finally block
            # or by _maintain_connection if it was an unexpected closure.
        except Exception as e:
            log.error(f"Unexpected error in receive loop: {e}")
            # Similar to ConnectionClosed, state transitions are handled by callers.

    @property
    def connected(self) -> bool:
        """
        Check if the WebSocket connection is currently established.

        Returns:
            bool: True if the connection is in CONNECTED state, False otherwise
        """
        return self._state == ConnectionState.CONNECTED

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
            "connected": self.connected,  # True if state is CONNECTED
            "reconnect_attempts": self._reconnect_attempts,
            "has_active_connection_object": self._websocket is not None,
            "is_receive_task_active": self._receive_task is not None
            and not self._receive_task.done(),
            "is_connection_maintenance_task_active": self._connection_task is not None
            and not self._connection_task.done(),
        }

        if (
            self._websocket is not None
        ):  # Add more specific WebSocket state if available
            try:
                metrics.update(
                    {
                        "websocket_is_open": not self._websocket.closed,
                        "websocket_is_closing": self._websocket.closing,
                    }
                )
            except Exception:  # pragma: no cover
                # Websocket object might be in an unstable state; avoid crashing health check.
                log.debug(
                    "Could not retrieve detailed websocket state for health metrics."
                )
                pass

        return metrics
