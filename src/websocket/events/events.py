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
from typing import Dict, Any

from src.utils.logger import get_logger

# Configure logger for this module
logger = get_logger(__name__)

# Registry mapping event type strings to their respective event classes
EVENT_TYPE_MAPPING: Dict[str, "BaseEvent"] = {}


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

    This class defines the common attributes that all events must have:
    - event_id: A unique identifier for the event
    - type: The type of the event, used for routing and handling

    It also provides methods for serialization and deserialization of events
    to and from JSON format for WebSocket transmission.
    """

    event_id: str  # Unique identifier for the event
    type: str  # Type of the event, used for routing

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
    def from_json(data: dict):
        """
        Create an event object from JSON data received from WebSocket.

        This static method:
        1. Extracts the event type from the data
        2. Looks up the corresponding event class in EVENT_TYPE_MAPPING
        3. Instantiates the appropriate event class with the data

        Args:
            data: Dictionary containing the event data from JSON

        Returns:
            BaseEvent: An instance of the appropriate event subclass, or None if the event type is unknown
        """
        event_type = data.get("type")
        if event_type in EVENT_TYPE_MAPPING:
            logger.debug(f"Constructing event of type: {event_type}")
            return EVENT_TYPE_MAPPING[event_type](**data)
        else:
            logger.warning(f"Unhandled event type: {event_type}")
            return None


@register_event("session.update")
@dataclass
class SessionUpdateEvent(BaseEvent):
    session: Dict[str, Any]


@register_event("input_audio_buffer.append")
@dataclass
class InputAudioBufferAppendEvent(BaseEvent):
    audio: str


@register_event("input_audio_buffer.commit")
@dataclass
class InputAudioBufferCommitEvent(BaseEvent):
    pass


@register_event("session.updated")
@dataclass
class SessionUpdatedEvent(BaseEvent):
    session: Dict[str, Any]


@register_event("session.created")
@dataclass
class SessionCreatedEvent(BaseEvent):
    session: Dict[str, Any]


@register_event("conversation.item.create")
@dataclass
class ConversationItemCreateEvent(BaseEvent):
    item: Dict[str, Any]


@register_event("response.create")
@dataclass
class ResponseCreateEvent(BaseEvent):
    pass


@register_event("response.audio.delta")
@dataclass
class ResponseAudioDeltaEvent(BaseEvent):
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


@register_event("response.audio.done")
@dataclass
class ResponseAudioDoneEvent(BaseEvent):
    response_id: str
    item_id: str
    output_index: int
    content_index: int


@register_event("error")
@dataclass
class ErrorEvent(BaseEvent):
    error: Dict[str, Any]
