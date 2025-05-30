"""
WebSocket Events module for defining and handling event data structures.

This module provides:
- A base event class (BaseEvent) that all events inherit from
- Specific event classes for different types of WebSocket events
- A registration system to map event types to their respective classes
- Serialization and deserialization of events to/from JSON
- A central registry of all event types for easy lookup

The event system uses Python dataclasses for clean, type-hinted event definitions
and provides a consistent interface for all event types.
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Type

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global registry mapping event type strings (e.g., "session.created")
# to their corresponding event dataclass (e.g., SessionCreatedEvent).
EVENT_TYPE_MAPPING: Dict[str, Type["BaseEvent"]] = {}


def register_event(event_type: str):
    """
    Decorator to register an event class with a specific event type.

    This decorator adds the decorated class to the EVENT_TYPE_MAPPING dictionary,
    allowing events of this type to be automatically instantiated when received
    from the WebSocket server.

    Args:
        event_type: The string identifier for this event type

    Returns:
        A decorator function that registers the class and returns it unchanged
    """

    def wrapper(cls):
        EVENT_TYPE_MAPPING[event_type] = cls
        return cls

    return wrapper


@dataclass
class BaseEvent:
    """
    Base class for all WebSocket events.

    This class defines the common attributes for all events: `event_id` and `type`.
    It also provides `to_json` for serialization and `from_json` for deserialization.
    """

    event_id: str  # Unique identifier for this specific event instance.
    type: str  # String identifying the kind of event (e.g., "session.created").

    def to_json(self) -> str:
        """
        Convert the event to a JSON string for sending over WebSocket.

        This method serializes the event object to a JSON string by:
        1. Converting the dataclass to a dictionary using asdict()
        2. Serializing the dictionary to a JSON string

        Returns:
            str: JSON string representation of the event
        """
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(data: dict) -> Optional["BaseEvent"]:
        """
        Create an event object from JSON data received from WebSocket.

        This static method:
        1. Extracts the event type from the data
        2. Looks up the corresponding event class in EVENT_TYPE_MAPPING
        3. Instantiates the appropriate event class with the data

        Args:
            data: Dictionary containing the event data from JSON

        Returns:
            BaseEvent: An instance of the appropriate event subclass, or None if the event type is unknown or data is malformed.
        """
        event_type = data.get("type")
        if event_type in EVENT_TYPE_MAPPING:
            event_class = EVENT_TYPE_MAPPING[event_type]
            logger.debug(
                f"Constructing event of type: {event_type} with class {event_class}"
            )
            try:
                return event_class(**data)
            except TypeError as e:
                logger.error(
                    f"Failed to instantiate event {event_type} with data {data}. "
                    f"Missing or mismatched fields: {e}"
                )
                return None
        else:
            logger.warning(f"Unhandled event type: {event_type}, data: {data}")
            return None


@register_event("session.update")
@dataclass
class SessionUpdateEvent(BaseEvent):
    """Event sent by the client to update the session state."""

    session: Dict[str, Any]  # Dictionary containing the session data to update.


@register_event("input_audio_buffer.append")
@dataclass
class InputAudioBufferAppendEvent(BaseEvent):
    """Event to append audio data to the input buffer for the session."""

    audio: str  # Base64 encoded audio data chunk.


@register_event("input_audio_buffer.commit")
@dataclass
class InputAudioBufferCommitEvent(BaseEvent):
    """Event to commit the currently buffered input audio for processing."""

    pass


@register_event("session.updated")
@dataclass
class SessionUpdatedEvent(BaseEvent):
    """Event received from the server when the session state has been updated."""

    session: Dict[str, Any]  # Dictionary containing the full updated session data.


@register_event("session.created")
@dataclass
class SessionCreatedEvent(BaseEvent):
    """Event received from the server when a new session has been successfully created."""

    session: Dict[
        str, Any
    ]  # Dictionary containing the data for the newly created session.


@register_event("conversation.item.create")
@dataclass
class ConversationItemCreateEvent(BaseEvent):
    """Event sent by the client to create a new item in the conversation."""

    item: Dict[str, Any]  # Dictionary representing the conversation item to be created.


@register_event("response.create")
@dataclass
class ResponseCreateEvent(BaseEvent):
    """Event sent by the client to request the creation of a new response from the server."""

    pass


@register_event("response.audio.delta")
@dataclass
class ResponseAudioDeltaEvent(BaseEvent):
    """Event received from the server containing a chunk of response audio."""

    response_id: str  # Identifier for the response this audio chunk belongs to.
    item_id: str  # Identifier for the conversation item this response is for.
    output_index: int  # Index of the output stream (e.g., for multiple audio outputs).
    content_index: int  # Index of the content block within the output.
    delta: str  # Base64 encoded audio data chunk.


@register_event("response.audio.done")
@dataclass
class ResponseAudioDoneEvent(BaseEvent):
    """Event received from the server indicating that all audio for a response has been sent."""

    response_id: str  # Identifier for the response that is now complete.
    item_id: str  # Identifier for the conversation item this response was for.
    output_index: int  # Index of the output stream.
    content_index: int  # Index of the content block.


@register_event("error")
@dataclass
class ErrorEvent(BaseEvent):
    """Event received from the server when an error occurs."""

    error: Dict[str, Any]  # Dictionary containing details about the error.


@register_event("response.cancel")
@dataclass
class ResponseCancelEvent(BaseEvent):
    """Event sent by the client to cancel an in-progress response."""

    response_id: Optional[str] = (
        None  # Specific response ID to cancel; if None, server cancels default.
    )


@register_event("response.cancelled")
@dataclass
class ResponseCancelledEvent(BaseEvent):
    """Event received from the server confirming a response cancellation."""

    cancelled_response_id: Optional[str] = (
        None  # The ID of the response that was actually cancelled by the server.
    )
