import base64
from typing import Callable, Awaitable, Dict

from src.audio.audio import AudioManager
from src.websocket.events.events import BaseEvent, ErrorEvent, SessionUpdatedEvent, SessionCreatedEvent, \
    ResponseAudioDeltaEvent, ResponseAudioDoneEvent
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WebSocketEventHandler:
    def __init__(self, audio_manager: AudioManager):
        self.audio_manager = audio_manager
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

    async def _handle_response_audio_delta(self, event: ResponseAudioDeltaEvent) -> None:
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
        """Dispatch the event to the appropriate handler."""
        logger.debug(f"Dispatching event: {event.type}")
        event_type = event.type
        handler = self.EVENT_HANDLERS.get(event_type)
        try:
            if handler:
                await handler(event)
            else:
                logger.warning(f"No handler found for event type: {event_type}")
        except Exception as e:
            logger.error(f"Error in handler for {event_type}: {e}")
