"""
Gemini Realtime Connection Management.

This module provides the GeminiRealtimeConnection class, responsible for:
- Establishing and maintaining the connection to Google's Gemini Live API.
- Handling the asynchronous event stream from the API.
- Translating the API's turn-based responses into a stream of synthetic events.
"""

import asyncio
import uuid
from typing import Optional, Callable, Awaitable, Any

from google import genai
from google.genai import types

from src.utils.logger import get_logger
from .event_handler import TurnStartEvent, TurnMessageEvent, TurnEndEvent

logger = get_logger(__name__)


class GeminiRealtimeConnection:
    """
    Manages a connection to the Gemini Live API.

    This class acts as a transport layer, handling the lifecycle of the connection,
    including connecting, disconnecting, and processing the incoming event stream.
    It translates the API's turn-based structure into a series of synthetic events
    and features a self-healing mechanism with exponential backoff to automatically
    reconnect after transient network failures.
    """

    def __init__(
        self,
        gemini_client: genai.Client,
        model_name: str,
        live_connect_config_params: dict,
    ):
        self.gemini_client: genai.Client = gemini_client
        self.model_name: str = model_name
        self.live_connect_config_params: dict = live_connect_config_params

        self._session_object: Optional[genai.live.AsyncSession] = None
        self._event_loop_task: Optional[asyncio.Task] = None
        self._connected_event = asyncio.Event()
        self._shutdown_signal = asyncio.Event()
        self._is_attempting_connection: bool = (
            False  # To prevent multiple concurrent connect calls
        )

    async def connect(self, event_callback: Callable[[Any], Awaitable[None]]) -> None:
        """
        Establishes the connection and starts the event processing loop.
        If a connection attempt is already in progress or active, this method returns early.

        Args:
            event_callback: An asynchronous callable that will be invoked with each
                            synthetic event generated from the API stream.
        """
        if self._is_attempting_connection:
            logger.info(
                "GeminiRealtimeConnection: Connection attempt already in progress."
            )
            return
        if self._event_loop_task and not self._event_loop_task.done():
            logger.info("GeminiRealtimeConnection: Connection task is already active.")
            return

        self._is_attempting_connection = True
        self._shutdown_signal.clear()

        async def _connect_task_wrapper():
            try:
                await self._run_event_loop(event_callback)
            finally:
                self._is_attempting_connection = False

        self._event_loop_task = asyncio.create_task(_connect_task_wrapper())
        logger.info("GeminiRealtimeConnection: Connection task initiated.")

    async def _run_event_loop(
        self, event_callback: Callable[[Any], Awaitable[None]]
    ) -> None:
        """
        Internal method to manage the connection and event stream with a retry loop.

        This loop continuously attempts to connect and process events. If the connection
        is lost, it waits for a period (exponential backoff) before retrying, making
        the connection resilient to transient network issues.
        """
        retry_delay = 1.0
        max_retry_delay = 30.0

        while not self._shutdown_signal.is_set():
            try:
                live_connect_config_obj = types.LiveConnectConfig(
                    **self.live_connect_config_params
                )
                logger.info(
                    f"Attempting to connect to Gemini Live API with model: {self.model_name}..."
                )

                async with self.gemini_client.aio.live.connect(
                    model=self.model_name, config=live_connect_config_obj
                ) as session:
                    self._session_object = session
                    self._connected_event.set()
                    logger.info(
                        "Successfully connected to Gemini Live API. Resetting retry delay."
                    )
                    retry_delay = 1.0  # Reset delay on successful connection

                    while self.is_connected() and not self._shutdown_signal.is_set():
                        try:
                            logger.debug(
                                "GeminiRealtimeConnection: Waiting for next turn from session.receive()..."
                            )
                            turn_iterator = self._session_object.receive()
                            turn_id = str(uuid.uuid4())

                            # Emit a start event for the new turn
                            await event_callback(TurnStartEvent(turn_id=turn_id))

                            # Emit message events for each message in the turn
                            async for message in turn_iterator:
                                if self._shutdown_signal.is_set():
                                    break
                                await event_callback(TurnMessageEvent(message=message))

                            # Emit an end event after the turn is complete
                            await event_callback(TurnEndEvent(turn_id=turn_id))

                            if self._shutdown_signal.is_set():
                                break

                        except Exception as e:
                            logger.error(
                                f"GeminiRealtimeConnection: Error in receive loop: {e}",
                                exc_info=True,
                            )
                            break  # Break inner loop to trigger reconnection

            except Exception as e:
                logger.error(
                    f"Gemini connection error: {e}. Will attempt to reconnect.",
                    exc_info=True,
                )

            finally:
                if self._session_object:
                    try:
                        await self._session_object.close()
                    except Exception as e_close:
                        logger.error(
                            f"Error closing session in finally block: {e_close}",
                            exc_info=True,
                        )
                self._session_object = None
                self._connected_event.clear()

            if self._shutdown_signal.is_set():
                logger.info("Shutdown signal detected. Halting reconnection attempts.")
                break

            logger.info(
                f"Waiting {retry_delay:.1f} seconds before next connection attempt."
            )
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)

        logger.info("Gemini Realtime event loop has terminated.")

    async def disconnect(self) -> None:
        """
        Signals the connection to shut down gracefully and waits for the event loop to terminate.
        """
        logger.info("Attempting to disconnect from Gemini Live API...")
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

        self._session_object = None
        self._event_loop_task = None
        logger.info("Disconnected from Gemini Live API.")

    def is_connected(self) -> bool:
        """
        Checks if the connection to Gemini Live API is currently active.
        """
        return self._connected_event.is_set()

    async def wait_until_connected(self):
        """Blocks until the connection is established."""
        await self._connected_event.wait()

    def get_active_session(
        self,
    ) -> Optional[genai.live.AsyncSession]:
        """
        Returns the active Gemini AsyncSession object if connected.
        """
        return self._session_object
