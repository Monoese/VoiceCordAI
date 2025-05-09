import json
from dataclasses import dataclass, asdict
from typing import Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

EVENT_TYPE_MAPPING: Dict[str, "BaseEvent"] = {}


def register_event(event_type: str):
    def wrapper(cls):
        EVENT_TYPE_MAPPING[event_type] = cls
        return cls

    return wrapper


@dataclass
class BaseEvent:
    event_id: str
    type: str

    def to_json(self) -> str:
        """Convert the event to a JSON string for sending."""
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(data: dict):
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
