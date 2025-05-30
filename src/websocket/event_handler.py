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
from typing import Callable, Awaitable, Dict, Optional

from src.audio.audio import AudioManager
from src.websocket.events.events import (
    BaseEvent,
    ErrorEvent,
    SessionUpdatedEvent,
    SessionCreatedEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseCancelledEvent,
)
from src.utils.logger import get_logger

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
        - An ID for the currently active response audio stream

        Args:
            audio_manager: The AudioManager instance to use for audio processing
        """
        self.audio_manager = audio_manager
        # Tracks the composite ID ("response_id-item_id") of the current audio stream being received.
        self._active_response_stream_id: Optional[str] = None

        self.EVENT_HANDLERS: Dict[str, Callable[[BaseEvent], Awaitable[None]]] = {
            "error": self._handle_error,
            "session.created": self._handle_session_created,
            "session.updated": self._handle_session_updated,
            "response.audio.delta": self._handle_response_audio_delta,
            "response.audio.done": self._handle_response_audio_done,
            "response.cancelled": self._handle_response_cancelled,
        }

    async def _handle_error(self, event: ErrorEvent) -> None:
        """Handles error events from the WebSocket."""
        logger.error(f"Error event received: {event.error}")

    async def _handle_session_updated(self, event: SessionUpdatedEvent) -> None:
        """Handles session updated events from the WebSocket."""
        # TODO: Implement logic for session.updated event.
        logger.info(f"Placeholder handler for event: {event.type} - {event}")

    async def _handle_session_created(self, event: SessionCreatedEvent) -> None:
        """Handles session created events from the WebSocket."""
        # TODO: Implement logic for session.created event.
        logger.info(f"Placeholder handler for event: {event.type} - {event}")

    async def _handle_response_audio_delta(
        self, event: ResponseAudioDeltaEvent
    ) -> None:
        """Handles incoming audio chunks (deltas) for a response."""
        # A stream is uniquely identified by the combination of response_id and item_id.
        current_event_stream_id = f"{event.response_id}-{event.item_id}"

        if self._active_response_stream_id != current_event_stream_id:
            # This delta belongs to a new audio stream (e.g., new item_id or new response_id).
            self._active_response_stream_id = current_event_stream_id
            logger.info(
                f"EventHandler: Starting new audio stream '{self._active_response_stream_id}' for {event.type}."
            )
            await self.audio_manager.start_new_audio_stream(
                self._active_response_stream_id
            )

        base64_audio = event.delta
        decoded_audio = base64.b64decode(base64_audio)

        await self.audio_manager.add_audio_chunk(decoded_audio)

    async def _handle_response_audio_done(self, event: ResponseAudioDoneEvent) -> None:
        """Handles the completion of an audio stream for a response."""
        event_stream_id = f"{event.response_id}-{event.item_id}"
        logger.info(
            f"EventHandler: Received audio done for stream '{event_stream_id}' (Current active: '{self._active_response_stream_id}')"
        )

        if self._active_response_stream_id == event_stream_id:
            # This 'done' message corresponds to the stream we are currently processing.
            await self.audio_manager.end_audio_stream()
            self._active_response_stream_id = (
                None  # Reset, ready for the next audio stream
            )
        else:
            # This can happen if 'done' arrives late or for a stream already superseded.
            logger.warning(
                f"EventHandler: Received audio done for stream '{event_stream_id}', "
                f"but current active stream is '{self._active_response_stream_id}'. Ignoring."
            )

    async def _handle_response_cancelled(self, event: ResponseCancelledEvent) -> None:
        """Handles response.cancelled events from the WebSocket server."""
        logger.info(
            f"EventHandler: Received response.cancelled event (ID: {event.event_id}). "
            f"Server acknowledged cancellation for response_id: {event.cancelled_response_id}."
        )

        if event.cancelled_response_id and self._active_response_stream_id:
            # The _active_response_stream_id is a composite "response_id-item_id".
            # We need to check if the `response_id` part matches the `cancelled_response_id`.
            active_response_id_part = self._active_response_stream_id.split("-", 1)[0]
            if active_response_id_part == event.cancelled_response_id:
                logger.info(
                    f"Clearing active response stream ID '{self._active_response_stream_id}' "
                    f"as its response_id part matches the cancelled_response_id '{event.cancelled_response_id}'."
                )
                self._active_response_stream_id = None
                # Note: The AudioManager's playback for this stream should have already been
                # stopped by VoiceCog calling voice_client.stop(), which then cleans up
                # in the playback_loop. No explicit call to audio_manager.end_audio_stream()
                # is made here to avoid potential race conditions or redundant signals,
                # as the primary mechanism for stopping playback is via voice_client.stop().

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

        handler = self.EVENT_HANDLERS.get(event_type)

        try:
            if handler:
                await handler(event)
            else:
                logger.warning(f"No handler found for event type: {event_type}")
        except Exception as e:
            logger.error(f"Error in handler for {event_type}: {e}", exc_info=True)
