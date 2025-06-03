"""
OpenAI Realtime Connection Management.

This module provides the OpenAIRealtimeConnection class, responsible for:
- Establishing and maintaining the connection to OpenAI's Realtime API using the openai library.
- Handling the asynchronous event stream from the API.
- Providing an interface to send commands/data to the API.
"""

import asyncio
from typing import Callable, Awaitable, Optional

from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import (
    AsyncRealtimeConnection as OpenAIConnectionObject,
)

# from openai.types.beta.realtime import RealtimeEvent # IDE has trouble finding this
from typing import Any as RealtimeEvent  # Using Any as a fallback

from src.config.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIRealtimeConnection:
    """
    Manages a connection to the OpenAI Realtime API.

    This class handles the lifecycle of the connection, including connecting,
    processing incoming events, and disconnecting.
    """

    def __init__(self, client: AsyncOpenAI):
        """
        Initializes the OpenAIRealtimeConnection.

        Args:
            client: An instance of AsyncOpenAI client.
        """
        self.client: AsyncOpenAI = client
        self.model_name: str = Config.OPENAI_REALTIME_MODEL_NAME

        self._connection_object: Optional[OpenAIConnectionObject] = None
        self._event_loop_task: Optional[asyncio.Task] = None
        self._is_attempting_connection: bool = False
        self._shutdown_signal = asyncio.Event()

    async def connect(
        self, event_callback: Callable[[RealtimeEvent], Awaitable[None]]
    ) -> None:
        """
        Establishes the connection and starts the event processing loop.

        If a connection attempt is already in progress or active, this method returns early.

        Args:
            event_callback: An asynchronous callable that will be invoked with each event received
                            from the OpenAI Realtime API.
        """
        if self._is_attempting_connection:
            logger.info("Connection attempt already in progress.")
            return
        if self._event_loop_task and not self._event_loop_task.done():
            logger.info("Connection task is already active.")
            return

        self._is_attempting_connection = True
        self._shutdown_signal.clear()

        async def _connect_task_wrapper():
            try:
                await self._run_event_loop(event_callback)
            finally:
                # Ensure this flag is reset regardless of how _run_event_loop exits
                self._is_attempting_connection = False

        self._event_loop_task = asyncio.create_task(_connect_task_wrapper())
        logger.info("OpenAI Realtime connection task initiated.")

    async def _run_event_loop(
        self, event_callback: Callable[[RealtimeEvent], Awaitable[None]]
    ) -> None:
        """
        Internal method to manage the connection context and event stream.
        """
        try:
            logger.info(
                f"Attempting to connect to OpenAI Realtime API with model: {self.model_name}..."
            )
            # The `async with` block manages the WebSocket connection lifecycle.
            async with self.client.beta.realtime.connect(model=self.model_name) as conn:
                self._connection_object = conn
                # _is_attempting_connection is fully reset by the wrapper's finally block.
                logger.info("Successfully connected to OpenAI Realtime API.")

                try:
                    async for event in conn:
                        if self._shutdown_signal.is_set():
                            logger.info(
                                "Shutdown signal detected during event iteration. Breaking loop."
                            )
                            break
                        await event_callback(event)
                except Exception as e:
                    logger.error(
                        f"Error during event iteration or callback: {e}", exc_info=True
                    )
                finally:
                    logger.info("Exiting event processing loop.")

            logger.info(
                "Exited OpenAI Realtime API 'async with' block (connection closed)."
            )

        except Exception as e:
            # This catches errors during the client.beta.realtime.connect() call itself
            # or other unexpected errors in the connection management.
            logger.error(
                f"Failed to connect or unexpected error in OpenAI Realtime API connection: {e}",
                exc_info=True,
            )
        finally:
            logger.info("OpenAI Realtime event loop processing finished.")
            self._connection_object = (
                None  # Ensure connection object is cleared on any exit
            )
            # _is_attempting_connection is reset by the calling wrapper's finally block.

    async def disconnect(self) -> None:
        """
        Signals the connection to shut down gracefully and waits for the event loop to terminate.
        """
        logger.info("Attempting to disconnect from OpenAI Realtime API...")
        self._shutdown_signal.set()

        if self._event_loop_task and not self._event_loop_task.done():
            logger.info("Waiting for event loop task to complete shutdown...")
            try:
                # Wait for the task to finish, which should happen once the loop breaks
                await asyncio.wait_for(self._event_loop_task, timeout=5.0)
                logger.info("Event loop task completed.")
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout waiting for event loop task to complete. Cancelling."
                )
                self._event_loop_task.cancel()
                try:
                    await self._event_loop_task  # Await the cancellation
                except asyncio.CancelledError:
                    logger.info("Event loop task was cancelled.")
            except Exception as e:  # Catch other potential errors during await
                logger.error(
                    f"Error during event loop task shutdown: {e}", exc_info=True
                )

        self._connection_object = None
        self._event_loop_task = None
        # _is_attempting_connection should be false if task finished or was cancelled.
        # It's primarily managed by connect() and its wrapper.
        logger.info("Disconnected from OpenAI Realtime API.")

    def is_connected(self) -> bool:
        """
        Checks if the connection to OpenAI Realtime API is currently active.

        Returns:
            True if the connection object exists (i.e., within an active 'async with' block),
            False otherwise.
        """
        return self._connection_object is not None

    def get_active_connection(self) -> Optional[OpenAIConnectionObject]:
        """
        Returns the active OpenAI library connection object if connected.

        This object can be used by the manager to send events to the API.

        Returns:
            The AsyncRealtimeConnection object from the OpenAI library, or None if not connected.
        """
        return self._connection_object
