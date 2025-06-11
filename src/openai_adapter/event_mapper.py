"""
OpenAI Event Mapper and Handler.

This module provides the OpenAIEventHandlerAdapter class, responsible for:
- Receiving RealtimeEvent objects from the OpenAI Realtime API (via the openai library).
- Dispatching these events to appropriate handler methods based on event.type.
- Adapting the logic from the existing WebSocketEventHandler to work with OpenAI's event structures.
"""

import base64
from typing import Callable, Awaitable, Dict, Optional

# Using Any for OpenAI event attributes for now, will refine if specific types are confirmed
# from openai.types.beta.realtime import RealtimeEvent, Error # IDE has trouble finding these
from openai.types.beta.realtime import Session
from openai.types.beta.realtime.error_event import Error  # Error type for error details
from typing import Any as RealtimeEvent  # Using Any as a fallback

from src.audio.audio import AudioManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIEventHandlerAdapter:
    """
    Handles and processes events received from the OpenAI Realtime API.

    This class adapts the event handling logic from the previous WebSocketEventHandler
    to work with the event types provided by the OpenAI Python library.
    """

    def __init__(self, audio_manager: AudioManager):
        """
        Initializes the OpenAIEventHandlerAdapter.

        Args:
            audio_manager: The AudioManager instance to use for audio processing.
        """
        self.audio_manager: AudioManager = audio_manager
        self._active_response_stream_id: Optional[str] = None

        self.EVENT_HANDLERS: Dict[str, Callable[[RealtimeEvent], Awaitable[None]]] = {
            "error": self._handle_error,
            "session.created": self._handle_session_created,
            "session.updated": self._handle_session_updated,
            "response.audio.delta": self._handle_response_audio_delta,
            "response.audio.done": self._handle_response_audio_done,
            "response.cancelled": self._handle_response_cancelled,
            # TODO: Add handlers for other event types as needed, e.g.:
            # "response.text.delta", "response.text.done",
            # "response.audio_transcript.delta", "response.audio_transcript.done",
            # "response.done"
        }

    async def dispatch_event(self, event: RealtimeEvent) -> None:
        """
        Dispatches an incoming OpenAI RealtimeEvent to its appropriate handler method.

        Args:
            event: The RealtimeEvent object received from the OpenAI library.
        """
        logger.debug(f"Dispatching OpenAI event: {event.type}")

        handler = self.EVENT_HANDLERS.get(event.type)

        try:
            if handler:
                await handler(event)
            else:
                # Use model_dump_json for Pydantic models if RealtimeEvent is one
                event_data_str = (
                    event.model_dump_json(indent=2)
                    if hasattr(event, "model_dump_json")
                    else str(event)
                )
                logger.warning(
                    f"No handler found for OpenAI event type: {event.type}. Event data: {event_data_str}"
                )
        except Exception as e:
            logger.error(
                f"Error in handler for OpenAI event type {event.type}: {e}",
                exc_info=True,
            )

    async def _handle_error(self, event: RealtimeEvent) -> None:
        """Handles 'error' events from the OpenAI Realtime API."""
        # When event.type == "error", event.error contains the error details object
        error_details_obj: Optional[Error] = getattr(event, "error", None)
        if error_details_obj and isinstance(error_details_obj, Error):
            logger.error(
                f"OpenAI Realtime API Error: Type='{error_details_obj.type}', Code='{error_details_obj.code}', Message='{error_details_obj.message}', EventID='{error_details_obj.event_id}'"
            )
        else:
            event_data_str = (
                event.model_dump_json(indent=2)
                if hasattr(event, "model_dump_json")
                else str(event)
            )
            logger.error(
                f"OpenAI Realtime API Error event received with unexpected structure: {event_data_str}"
            )

    async def _handle_session_created(self, event: RealtimeEvent) -> None:
        """Handles 'session.created' events."""
        session_details: Optional[Session] = getattr(event, "session", None)
        if session_details and isinstance(session_details, Session):
            logger.info(
                f"OpenAI Session Created: ID='{session_details.id}'"
            )
        else:
            event_data_str = (
                event.model_dump_json(indent=2)
                if hasattr(event, "model_dump_json")
                else str(event)
            )
            logger.warning(
                f"Session.created event with unexpected structure: {event_data_str}"
            )

    async def _handle_session_updated(self, event: RealtimeEvent) -> None:
        """Handles 'session.updated' events."""
        session_details: Optional[Session] = getattr(event, "session", None)
        if session_details and isinstance(session_details, Session):
            logger.info(
                f"OpenAI Session Updated: ID='{session_details.id}'"
            )
        else:
            event_data_str = (
                event.model_dump_json(indent=2)
                if hasattr(event, "model_dump_json")
                else str(event)
            )
            logger.warning(
                f"Session.updated event with unexpected structure: {event_data_str}"
            )

    async def _handle_response_audio_delta(self, event: RealtimeEvent) -> None:
        """Handles 'response.audio.delta' events containing audio chunks."""
        # Based on push_to_talk_app.py, event.delta is the base64 audio string.
        # item_id and response_id are also expected.
        base64_audio_chunk: Optional[str] = getattr(event, "delta", None)
        item_id: Optional[str] = getattr(event, "item_id", None)
        response_id: Optional[str] = getattr(event, "response_id", None)

        if not (base64_audio_chunk and item_id is not None and response_id is not None):
            event_data_str = (
                event.model_dump_json(indent=2)
                if hasattr(event, "model_dump_json")
                else str(event)
            )
            logger.warning(
                f"Response.audio.delta event with missing fields (delta, item_id, or response_id): {event_data_str}"
            )
            return

        current_event_stream_id = f"{response_id}-{item_id}"

        if self._active_response_stream_id != current_event_stream_id:
            self._active_response_stream_id = current_event_stream_id
            logger.info(
                f"OpenAIEventHandlerAdapter: Starting new audio stream '{self._active_response_stream_id}' for {event.type}."
            )
            await self.audio_manager.start_new_audio_stream(
                self._active_response_stream_id
            )

        try:
            decoded_audio = base64.b64decode(base64_audio_chunk)
            await self.audio_manager.add_audio_chunk(decoded_audio)
        except Exception as e:
            logger.error(
                f"Error decoding or adding audio chunk for stream '{self._active_response_stream_id}': {e}",
                exc_info=True,
            )

    async def _handle_response_audio_done(self, event: RealtimeEvent) -> None:
        """Handles 'response.audio.done' events."""
        item_id: Optional[str] = getattr(event, "item_id", None)
        response_id: Optional[str] = getattr(event, "response_id", None)

        if item_id is None or response_id is None:
            event_data_str = (
                event.model_dump_json(indent=2)
                if hasattr(event, "model_dump_json")
                else str(event)
            )
            logger.warning(
                f"Response.audio.done event with missing fields (item_id or response_id): {event_data_str}"
            )
            return

        event_stream_id = f"{response_id}-{item_id}"
        logger.info(
            f"OpenAIEventHandlerAdapter: Received audio done for stream '{event_stream_id}' (Current active: '{self._active_response_stream_id}')"
        )

        if self._active_response_stream_id == event_stream_id:
            await self.audio_manager.end_audio_stream()
            self._active_response_stream_id = None
        else:
            logger.warning(
                f"OpenAIEventHandlerAdapter: Received audio done for stream '{event_stream_id}', "
                f"but current active stream is '{self._active_response_stream_id}'. Ignoring."
            )

    async def _handle_response_cancelled(self, event: RealtimeEvent) -> None:
        """
        Handles 'response.cancelled' events from the server.
        The client sends `{"type": "response.cancel"}`.
        The server responds with a 'response.cancelled' event including 'response_id' and 'item_id'.
        """
        event_data_str = (
            event.model_dump_json(indent=2)
            if hasattr(event, "model_dump_json")
            else str(event)
        )
        logger.info(
            f"OpenAIEventHandlerAdapter: Received response.cancelled event: {event_data_str}"
        )

        # Attempt to get a response_id that was cancelled.
        # Server-sent 'response.cancelled' event includes 'response_id' (ID of the cancelled response) and 'item_id'.
        server_cancelled_response_id: Optional[str] = getattr(
            event, "response_id", None
        )
        # item_id is also available: item_id: Optional[str] = getattr(event, 'item_id', None)

        if server_cancelled_response_id and self._active_response_stream_id:
            active_response_id_part = self._active_response_stream_id.split("-", 1)[0]
            if active_response_id_part == server_cancelled_response_id:
                logger.info(
                    f"Clearing active response stream ID '{self._active_response_stream_id}' "
                    f"as its response_id part matches the server_cancelled_response_id '{server_cancelled_response_id}'."
                )
                self._active_response_stream_id = None
        elif self._active_response_stream_id:
            # If a general cancellation event comes without a specific ID,
            # or if the ID didn't match, we might assume it's for the current active stream.
            logger.info(
                f"Received a response.cancelled event. Assuming it's for the current active stream "
                f"'{self._active_response_stream_id}' and clearing it, as no specific matching ID was found in the event."
            )
            self._active_response_stream_id = None
        else:
            logger.info(
                "Received response.cancelled event, but no active stream ID was set."
            )
