"""
Gemini Event Handler Adapter.

This module provides the GeminiEventHandlerAdapter class, responsible for
processing synthetic events generated from the Gemini Live API stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Awaitable, Dict, TYPE_CHECKING, Union, Optional

from google.genai import types

from src.audio.audio import AudioManager
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.ai_services.interface import IRealtimeAIServiceManager

logger = get_logger(__name__)


# --- Synthetic Event Definitions ---
@dataclass
class TurnStartEvent:
    """Event fired when a new response turn begins."""

    turn_id: str


@dataclass
class TurnMessageEvent:
    """Event fired for each message within a turn."""

    message: types.LiveServerMessage


@dataclass
class TurnEndEvent:
    """Event fired when a response turn ends."""

    turn_id: str


# Union type for all possible events this handler can dispatch
GeminiRealtimeEvent = Union[TurnStartEvent, TurnMessageEvent, TurnEndEvent]


class GeminiEventHandlerAdapter:
    """
    Handles and processes events synthesized from the Gemini Live API stream.

    This class is stateful and manages the audio stream lifecycle based on
    turn start/end events. It uses a dispatcher pattern to route events to
    the appropriate handler methods. It employs a "just-in-time" approach
    to starting audio streams, only initializing them upon receiving the
    first audio chunk of a turn.
    """

    def __init__(
        self, audio_manager: AudioManager, manager: "IRealtimeAIServiceManager"
    ):
        """
        Initializes the GeminiEventHandlerAdapter.

        Args:
            audio_manager: The AudioManager instance to use for audio processing.
            manager: The parent AI service manager, used to access service-specific
                     configurations like audio formats.
        """
        self.audio_manager: AudioManager = audio_manager
        self.manager: "IRealtimeAIServiceManager" = manager
        self._active_turn_id: Optional[str] = None
        self._stream_started_for_turn: bool = False

        self.EVENT_HANDLERS: Dict[
            type, Callable[[GeminiRealtimeEvent], Awaitable[None]]
        ] = {
            TurnStartEvent: self._handle_turn_start,
            TurnMessageEvent: self._handle_turn_message,
            TurnEndEvent: self._handle_turn_end,
        }

    async def dispatch_event(self, event: GeminiRealtimeEvent) -> None:
        """Dispatches a synthetic Gemini event to its appropriate handler."""
        handler = self.EVENT_HANDLERS.get(type(event))
        if handler:
            try:
                await handler(event)
            except Exception as e:
                logger.error(
                    f"Error in handler for Gemini event type {type(event).__name__}: {e}",
                    exc_info=True,
                )
        else:
            logger.warning(
                f"No handler found for Gemini event type: {type(event).__name__}"
            )

    async def _handle_turn_start(self, event: TurnStartEvent) -> None:
        """
        Handles the start of a new response turn by setting the active turn ID
        and resetting the stream state. This prepares the handler to initiate
        an audio stream upon receiving the first audio chunk.
        """
        logger.info(f"GeminiEventHandler: New turn started with ID: {event.turn_id}")
        self._active_turn_id = event.turn_id
        self._stream_started_for_turn = False

    async def _handle_turn_message(self, event: TurnMessageEvent) -> None:
        """
        Handles a message within a turn. If the message contains the first audio
        chunk for the current turn, it first initializes the audio stream in the
        AudioManager before adding the chunk.
        """
        message = event.message
        if message.server_content:
            audio_data = message.data
            text_data = message.text

            if audio_data:
                # If this is the first audio chunk for this turn, start the stream.
                if not self._stream_started_for_turn:
                    if self._active_turn_id:
                        logger.info(
                            f"GeminiEventHandler: First audio chunk received for turn '{self._active_turn_id}'. Starting new audio stream."
                        )
                        await self.audio_manager.start_new_audio_stream(
                            self._active_turn_id, self.manager.response_audio_format
                        )
                        self._stream_started_for_turn = True
                    else:
                        logger.warning(
                            "Received audio data but no active turn ID is set. Ignoring chunk."
                        )
                        return

                logger.debug(f"Received audio data chunk, size: {len(audio_data)} bytes.")
                try:
                    await self.audio_manager.add_audio_chunk(audio_data)
                except Exception as e:
                    logger.error(
                        f"Error adding audio chunk to AudioManager: {e}", exc_info=True
                    )

            if text_data:
                logger.info(f"Gemini text response: {text_data}")

        elif message.go_away:
            logger.warning(f"Received 'go_away' message from Gemini: {message.go_away}")
        elif message.usage_metadata:
            logger.debug(
                f"Received 'usage_metadata' from Gemini: {message.usage_metadata}"
            )

    async def _handle_turn_end(self, event: TurnEndEvent) -> None:
        """
        Handles the end of a response turn by signaling the end of the audio
        stream and clearing the active turn state.
        """
        if event.turn_id == self._active_turn_id:
            logger.info(f"GeminiEventHandler: Ending audio stream for turn: {event.turn_id}")
            # Only end the stream if it was actually started.
            if self._stream_started_for_turn:
                await self.audio_manager.end_audio_stream()
            self._active_turn_id = None
            self._stream_started_for_turn = False
        else:
            logger.warning(
                f"Received TurnEndEvent for '{event.turn_id}', but active turn is '{self._active_turn_id}'. Ignoring."
            )
