"""
Defines a base class for real-time AI service connection handlers.

This module provides BaseConnectionHandler, which encapsulates common logic
for managing WebSocket-like connections, including state management,
automatic reconnection with exponential backoff, and graceful shutdown.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Callable, Awaitable, Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseConnectionHandler(ABC):
    """
    Abstract base class for managing a resilient real-time connection.

    This class handles the generic logic for:
    - Establishing and tearing down a connection.
    - An event loop that automatically attempts to reconnect on failure.
    - Exponential backoff for reconnection attempts.
    - State management (connected, shutdown signals).
    """

    def __init__(self) -> None:
        """Initializes the BaseConnectionHandler."""
        self._event_loop_task: Optional[asyncio.Task] = None
        self._connected_event = asyncio.Event()
        self._shutdown_signal = asyncio.Event()
        self._is_attempting_connection: bool = False
        self._event_callback: Optional[Callable[[Any], Awaitable[None]]] = None
        self._on_connect_callback: Optional[Callable[[], Awaitable[None]]] = None
        self._on_disconnect_callback: Optional[Callable[[], Awaitable[None]]] = None

        self._retry_delay = 1.0
        self._max_retry_delay = 30.0

    @abstractmethod
    async def _connection_logic(self) -> None:
        """
        Provider-specific connection logic.

        This method should be implemented by subclasses to handle the
        actual connection establishment and event processing loop.
        It is executed within the main retry loop of `_run_event_loop`.
        Implementations MUST:
        - Set `self._connected_event` on success.
        - Clear `self._connected_event` in a `finally` block.
        - Reset `self._retry_delay = 1.0` on successful connection.
        """
        pass

    async def connect(
        self,
        event_callback: Callable[[Any], Awaitable[None]],
        on_connect: Callable[[], Awaitable[None]],
        on_disconnect: Callable[[], Awaitable[None]],
    ) -> None:
        """
        Establishes the connection and starts the event processing loop.

        Args:
            event_callback: An async callable to be invoked with each event.
            on_connect: An async callable to be invoked on successful connection.
            on_disconnect: An async callable to be invoked on disconnection.

        Returns:
            None.
        """
        class_name = self.__class__.__name__
        if self._is_attempting_connection:
            logger.info(f"{class_name}: Connection attempt already in progress.")
            return
        if self._event_loop_task and not self._event_loop_task.done():
            logger.info(f"{class_name}: Connection task is already active.")
            return

        self._is_attempting_connection = True
        self._shutdown_signal.clear()
        self._event_callback = event_callback
        self._on_connect_callback = on_connect
        self._on_disconnect_callback = on_disconnect

        # This wrapper ensures that `_is_attempting_connection` is reset
        # even if the `_run_event_loop` task is cancelled or fails.
        async def _connect_task_wrapper():
            try:
                await self._run_event_loop()
            finally:
                self._is_attempting_connection = False

        self._event_loop_task = asyncio.create_task(_connect_task_wrapper())
        logger.info(f"{class_name}: Connection task initiated.")

    async def _run_event_loop(self) -> None:
        """
        Manages the connection and event stream with a retry loop.
        """
        class_name = self.__class__.__name__

        while not self._shutdown_signal.is_set():
            try:
                # Subclasses must call self._on_connect_callback after connection.
                await self._connection_logic()
                logger.info(
                    f"{class_name}: Event stream ended. Connection likely closed by server."
                )
            except Exception as e:
                logger.error(
                    f"{class_name} connection error: {e}. Will attempt to reconnect.",
                    exc_info=True,
                )
            finally:
                if self.is_connected() and self._on_disconnect_callback:
                    asyncio.create_task(self._on_disconnect_callback())
                self._connected_event.clear()

            if self._shutdown_signal.is_set():
                logger.info(
                    f"{class_name}: Shutdown signal detected. Halting reconnection attempts."
                )
                break

            logger.info(
                f"Waiting {self._retry_delay:.1f} seconds before next connection attempt."
            )
            try:
                await asyncio.wait_for(
                    self._shutdown_signal.wait(), timeout=self._retry_delay
                )
                logger.info(
                    f"{class_name}: Shutdown signal received while waiting to retry. Exiting loop."
                )
                break
            except asyncio.TimeoutError:
                # This is expected; the sleep duration has passed.
                pass

            # Increase delay for the next attempt (exponential backoff).
            self._retry_delay = min(self._retry_delay * 2, self._max_retry_delay)

        logger.info(f"{class_name} event loop has terminated.")

    async def disconnect(self) -> None:
        """
        Signals the connection to shut down gracefully.

        Returns:
            None.
        """
        class_name = self.__class__.__name__
        logger.info(f"Attempting to disconnect from {class_name}...")
        self._shutdown_signal.set()

        if self._event_loop_task and not self._event_loop_task.done():
            logger.info("Waiting for event loop task to complete shutdown...")
            try:
                await asyncio.wait_for(self._event_loop_task, timeout=5.0)
                logger.info("Event loop task completed.")
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout waiting for event loop task to complete. Cancelling."
                )
                self._event_loop_task.cancel()
                try:
                    await self._event_loop_task
                except asyncio.CancelledError:
                    logger.info("Event loop task was cancelled.")
            except Exception as e:
                logger.error(
                    f"Error during event loop task shutdown: {e}", exc_info=True
                )

        self._event_loop_task = None
        self._connected_event.clear()
        self._is_attempting_connection = False
        logger.info(f"Disconnected from {class_name}.")

    def is_connected(self) -> bool:
        """Checks if the connection is currently active.

        Returns:
            True if connected, False otherwise.
        """
        return self._connected_event.is_set()

    async def wait_until_connected(self) -> None:
        """Blocks until the connection is established.

        Returns:
            None.
        """
        await self._connected_event.wait()
