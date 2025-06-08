"""
Gemini Realtime Connection Management.

This module provides the GeminiRealtimeConnection class, responsible for:
- Establishing and maintaining the connection to Google's Gemini Live API.
- Handling the asynchronous event stream from the API.
- Coordinating with the event handler adapter to process messages.
"""

import asyncio
import uuid
from typing import Optional, TYPE_CHECKING

from google import genai
from google.genai import types

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from .event_handler import GeminiEventHandlerAdapter
    from src.audio.audio import AudioManager

logger = get_logger(__name__)


class GeminiRealtimeConnection:
    """
    Manages a connection to the Gemini Live API.
    """

    def __init__(
        self,
        gemini_client: genai.Client,
        model_name: str,
        live_connect_config_params: dict,
        event_handler_adapter: "GeminiEventHandlerAdapter",
        audio_manager: "AudioManager",  # Passed through to event handler logic for stream IDs
    ):
        self.gemini_client: genai.Client = gemini_client
        self.model_name: str = model_name
        self.live_connect_config_params: dict = live_connect_config_params
        self.event_handler_adapter: "GeminiEventHandlerAdapter" = event_handler_adapter
        self._audio_manager: "AudioManager" = (
            audio_manager  # For managing stream IDs per turn
        )

        self._session_object: Optional[genai.live.AsyncSession] = None
        self._event_loop_task: Optional[asyncio.Task] = None
        self._shutdown_signal = asyncio.Event()
        self._is_attempting_connection: bool = (
            False  # To prevent multiple concurrent connect calls
        )

    async def connect(self) -> None:
        """
        Establishes the connection and starts the event processing loop.
        If a connection attempt is already in progress or active, this method returns early.
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
                await self._run_event_loop()
            finally:
                self._is_attempting_connection = False

        self._event_loop_task = asyncio.create_task(_connect_task_wrapper())
        logger.info("GeminiRealtimeConnection: Connection task initiated.")

    async def _run_event_loop(self) -> None:
        """
        Internal method to manage the connection context and event stream.
        """
        try:
            live_connect_config_obj = types.LiveConnectConfig(
                **self.live_connect_config_params
            )
            logger.info(
                f"GeminiRealtimeConnection: _run_event_loop connecting with config: {self.live_connect_config_params}"
            )

            async with self.gemini_client.aio.live.connect(
                model=self.model_name, config=live_connect_config_obj
            ) as session:
                self._session_object = session
                logger.info(
                    "GeminiRealtimeConnection: Successfully connected to Gemini Live API and session established."
                )

                while self.is_connected() and not self._shutdown_signal.is_set():
                    try:
                        logger.debug(
                            "GeminiRealtimeConnection: Waiting for next turn from session.receive()..."
                        )
                        turn_iterator = self._session_object.receive()

                        current_turn_stream_id: Optional[str] = (
                            None  # Holds the stream ID for the current turn
                        )

                        async for message in turn_iterator:
                            if self._shutdown_signal.is_set():
                                logger.info(
                                    "GeminiRealtimeConnection: Shutdown signal received during message processing. Breaking inner loop."
                                )
                                break

                            # Check for actual audio data in the message
                            if message.server_content and message.data:
                                if (
                                    current_turn_stream_id is None
                                ):  # First audio chunk for this turn
                                    current_turn_stream_id = str(uuid.uuid4())
                                    logger.info(
                                        f"GeminiRealtimeConnection: Starting new audio stream for turn: {current_turn_stream_id} upon first audio data."
                                    )
                                    await self._audio_manager.start_new_audio_stream(
                                        current_turn_stream_id
                                    )

                            await (
                                self.event_handler_adapter.process_live_server_message(
                                    message
                                )
                            )

                        logger.info(
                            "GeminiRealtimeConnection: Finished processing turn."
                        )
                        if (
                            current_turn_stream_id
                        ):  # Only end stream if one was actually started for this turn
                            logger.info(
                                f"GeminiRealtimeConnection: Ending audio stream: {current_turn_stream_id}"
                            )
                            await self._audio_manager.end_audio_stream()
                        # Else, if no audio was received in this turn, no stream was started, so no need to end one.

                        if self._shutdown_signal.is_set():
                            logger.info(
                                "GeminiRealtimeConnection: Shutdown signal received after turn processing. Breaking outer loop."
                            )
                            break

                    except asyncio.CancelledError:
                        logger.info(
                            "GeminiRealtimeConnection: _run_event_loop's receive operation cancelled."
                        )
                        break
                    except Exception as e:
                        logger.error(
                            f"GeminiRealtimeConnection: Error in receive loop: {e}",
                            exc_info=True,
                        )
                        break

        except asyncio.CancelledError:
            logger.info("GeminiRealtimeConnection: _run_event_loop task was cancelled.")
        except Exception as e:
            logger.error(
                f"GeminiRealtimeConnection: Fatal error in _run_event_loop (e.g., connection failed): {e}",
                exc_info=True,
            )
        finally:
            logger.info("GeminiRealtimeConnection: _run_event_loop is shutting down.")
            if self._session_object:
                try:
                    await self._session_object.close()
                    logger.info(
                        "GeminiRealtimeConnection: Session closed from _run_event_loop finally block."
                    )
                except Exception as e_close:
                    logger.error(
                        f"GeminiRealtimeConnection: Error closing session in _run_event_loop finally: {e_close}",
                        exc_info=True,
                    )
            self._session_object = None
            logger.info(
                "GeminiRealtimeConnection: _run_event_loop finished and connection status updated."
            )

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
        return self._session_object is not None

    def get_active_session(
        self,
    ) -> Optional[genai.live.AsyncSession]:  # Changed genai_types to types
        """
        Returns the active Gemini AsyncSession object if connected.
        """
        return self._session_object
