"""
WebSocket Event Handler module for processing events from WebSocket connections.

This module provides the WebSocketEventHandler class which:
- Registers handlers for different types of WebSocket events
- Dispatches incoming events to the appropriate handler
- Processes audio data received from the WebSocket server
- Handles error events and session management events
- Coordinates with the AudioManager for audio processing

The event handler acts as a bridge between the WebSocket communication layer
and the audio processing system, ensuring events are properly processed and
audio data is correctly handled.
"""

import base64
from typing import Callable, Awaitable, Dict

from src.audio.audio import AudioManager
from src.websocket.events.events import (
    BaseEvent,
    ErrorEvent,
    SessionUpdatedEvent,
    SessionCreatedEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
)
from src.utils.logger import get_logger

# Configure logger for this module
logger = get_logger(__name__)


class WebSocketEventHandler:
    """
    Handles and processes events received from WebSocket connections.

    This class:
    - Maps event types to their corresponding handler methods
    - Dispatches incoming events to the appropriate handler
    - Processes audio data from response events
    - Coordinates with the AudioManager to handle audio playback
    - Logs event processing for debugging and monitoring

    The handler maintains a dictionary of event types to handler functions,
    allowing for a clean separation of concerns and easy extension with
    new event types.
    """

    def __init__(self, audio_manager: AudioManager):
        """
        Initialize the WebSocketEventHandler with an AudioManager.

        This constructor sets up the event handler with:
        - A reference to the AudioManager for audio processing
        - A mapping of event types to their handler methods

        Args:
            audio_manager: The AudioManager instance to use for audio processing
        """
        self.audio_manager = audio_manager

        # Map event types to their handler methods
        self.EVENT_HANDLERS: Dict[str, Callable[[BaseEvent], Awaitable[None]]] = {
            "error": self._handle_error,
            "session.created": self._handle_session_created,
            "session.updated": self._handle_session_updated,
            "response.audio.delta": self._handle_response_audio_delta,
            "response.audio.done": self._handle_response_audio_done,
        }

    async def _handle_error(self, event: ErrorEvent) -> None:
        logger.error(f"Error event received: {event.error}")

    async def _handle_session_updated(self, event: SessionUpdatedEvent) -> None:
        logger.info(f"Handling event: {event.type}")

    async def _handle_session_created(self, event: SessionCreatedEvent) -> None:
        logger.info(f"Handling event: {event.type}")

    async def _handle_response_audio_delta(
        self, event: ResponseAudioDeltaEvent
    ) -> None:
        logger.info(f"Handling event: {event.type}")
        base64_audio = event.delta
        decoded_audio = base64.b64decode(base64_audio)

        self.audio_manager.extend_response_buffer(decoded_audio)
        logger.debug(f"Buffered audio fragment: {len(decoded_audio)} bytes")

    async def _handle_response_audio_done(self, event: ResponseAudioDoneEvent) -> None:
        logger.info(f"Handling event: {event.type}")
        if self.audio_manager.response_buffer:
            await self.audio_manager.enqueue_audio(self.audio_manager.response_buffer)
            self.audio_manager.clear_response_buffer()

    async def dispatch_event(self, event: BaseEvent) -> None:
        """
        Dispatch an incoming event to its appropriate handler method.

        This method:
        1. Looks up the handler for the event type in EVENT_HANDLERS
        2. Calls the handler with the event if one is found
        3. Logs warnings for unknown event types
        4. Catches and logs any exceptions that occur during handling

        Args:
            event: The BaseEvent object to dispatch to a handler
        """
        logger.debug(f"Dispatching event: {event.type}")
        event_type = event.type

        # Look up the handler for this event type
        handler = self.EVENT_HANDLERS.get(event_type)

        try:
            if handler:
                # Call the appropriate handler with the event
                await handler(event)
            else:
                logger.warning(f"No handler found for event type: {event_type}")
        except Exception as e:
            # Log any errors that occur during event handling
            logger.error(f"Error in handler for {event_type}: {e}")
