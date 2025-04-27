import json
from dataclasses import dataclass, asdict
from typing import Dict, Any


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
            print("constructing", event_type)
            return EVENT_TYPE_MAPPING[event_type](**data)
        else:
            print(f"Unhandled event type: {event_type}")
            return None


@dataclass
class SessionUpdateEvent(BaseEvent):
    session: Dict[str, Any]


@dataclass
class InputAudioBufferAppendEvent(BaseEvent):
    audio: str


@dataclass
class InputAudioBufferCommitEvent(BaseEvent):
    pass


@dataclass
class SessionUpdatedEvent(BaseEvent):
    session: Dict[str, Any]


@dataclass
class SessionCreatedEvent(BaseEvent):
    session: Dict[str, Any]


@dataclass
class ConversationItemCreateEvent(BaseEvent):
    item: Dict[str, Any]


@dataclass
class ResponseCreateEvent(BaseEvent):
    pass


@dataclass
class ResponseAudioDeltaEvent(BaseEvent):
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


@dataclass
class ResponseAudioDoneEvent(BaseEvent):
    response_id: str
    item_id: str
    output_index: int
    content_index: int


@dataclass
class ErrorEvent(BaseEvent):
    error: Dict[str, Any]


EVENT_TYPE_MAPPING = {"session.update": SessionUpdateEvent, "conversation.item.create": ConversationItemCreateEvent,
    "input_audio_buffer.append": InputAudioBufferAppendEvent, "input_audio_buffer.commit": InputAudioBufferCommitEvent,
    "response.create": ResponseCreateEvent, "session.created": SessionCreatedEvent,
    "session.updated": SessionUpdatedEvent, "response.audio.delta": ResponseAudioDeltaEvent,
    "response.audio.done": ResponseAudioDoneEvent, "error": ErrorEvent, }
