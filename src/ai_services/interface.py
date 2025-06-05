"""
Defines the IRealtimeAIServiceManager abstract base class (ABC).

This interface specifies the contract for real-time AI service managers,
allowing the bot to interact with different AI services (like OpenAI, Gemini)
through a common set of methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

# Assuming AudioManager is in src.audio.audio
# Adjust the import path if AudioManager is located elsewhere.
from src.audio.audio import AudioManager


class IRealtimeAIServiceManager(ABC):
    """
    Abstract Base Class for a real-time AI service manager.

    This interface defines the common methods that VoiceCog will use to interact
    with an underlying AI service for voice chat functionalities, including
    connection management, sending audio/text, and controlling responses.
    """

    def __init__(self, audio_manager: AudioManager, service_config: Dict[str, Any]):
        """
        Initializes the AI service manager.

        Args:
            audio_manager: An instance of AudioManager for handling audio playback.
                           Concrete implementations will use this to play audio received
                           from the AI service.
            service_config: A dictionary containing service-specific configurations,
                            such as API keys, model names, endpoint URLs,
                            initial session parameters, or other settings required by
                            the specific AI service implementation.
        """
        self._audio_manager: AudioManager = audio_manager
        self._service_config: Dict[str, Any] = service_config
        self._is_connected_flag: bool = (
            False  # Internal state managed by implementations
        )

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establishes a connection to the real-time AI service and initializes the session.

        Implementations should handle all necessary setup, including authentication
        and sending any initial configuration or session parameters derived from
        `self._service_config`. Upon successful connection and session initialization,
        `self._is_connected_flag` should be set to True.

        Returns:
            True if the connection and session setup were successful, False otherwise.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Closes the connection to the AI service and cleans up any associated resources.

        Implementations should ensure that all tasks related to the connection are
        properly terminated and `self._is_connected_flag` is set to False.
        """
        pass

    def is_connected(self) -> bool:
        """
        Checks if the manager is currently connected to the AI service.

        This method relies on the `_is_connected_flag` which should be accurately
        maintained by the `connect` and `disconnect` methods of the concrete
        implementation.

        Returns:
            True if connected (i.e., `_is_connected_flag` is True), False otherwise.
        """
        return self._is_connected_flag

    @abstractmethod
    async def send_audio_chunk(self, audio_data: bytes) -> bool:
        """
        Sends a chunk of raw audio data to the AI service.

        Implementations are responsible for any necessary encoding (e.g., to base64
        for some services) or formatting (e.g., wrapping in a specific JSON structure
        or Blob type) required by the underlying AI service API.

        Args:
            audio_data: Raw bytes of the audio data (e.g., PCM).

        Returns:
            True if the audio chunk was successfully sent or queued for sending,
            False otherwise (e.g., if not connected, or an error occurred during sending).
        """
        pass

    @abstractmethod
    async def send_text_message(self, text: str, finalize_turn: bool) -> bool:
        """
        Sends a text message to the AI service.

        This can be used for user text input or bot-generated text prompts.
        The `finalize_turn` parameter indicates if this message should conclude
        the current speaker's turn and prompt the AI for a response.

        Args:
            text: The text content to send.
            finalize_turn: If True, signals that this text message concludes the
                           current turn, and the AI should process and respond.
                           (e.g., for Gemini, this maps to `turn_complete=True`).
                           If False, the message is sent as part of an ongoing turn.

        Returns:
            True if the text message was successfully sent, False otherwise.
        """
        pass

    @abstractmethod
    async def finalize_input_and_request_response(self) -> bool:
        """
        Signals to the AI service that all input for the current turn has been provided
        and a response is now expected from the AI.

        This is particularly relevant if the turn involved sending a stream of audio chunks
        and did not conclude with a `send_text_message(..., finalize_turn=True)` call.
        Implementations will map this to the specific mechanism of their AI service:
        - For OpenAI: This would typically involve operations like 'commit_audio_buffer'
          and then 'create_response'.
        - For Gemini (if primarily audio was sent):
            - If automatic Voice Activity Detection (VAD) is used by the service, this might
              involve sending an `audio_stream_end=True` signal.
            - If manual VAD is used, this might involve sending an `activity_end=True` signal.
        - If the last action was `send_text_message` with `finalize_turn=True`, this method
          might be a no-op or simply ensure the state is consistent.

        Returns:
            True if the signal was successfully sent and a response is anticipated,
            False otherwise.
        """
        pass

    @abstractmethod
    async def cancel_ongoing_response(self) -> bool:
        """
        Attempts to cancel any AI response that is currently being generated or streamed.

        Implementations should send the appropriate cancellation command to the AI service.
        This method should also coordinate with the `AudioManager` (via `self._audio_manager`)
        to stop any local playback of audio from the AI service if a cancellation is initiated.
        For services where a direct client-side cancellation API is not available or limited,
        this method should perform a best-effort cancellation (e.g., stop local playback,
        cease sending further data if applicable).

        Returns:
            True if cancellation was successfully requested (or best-effort initiated),
            False otherwise.
        """
        pass

    # Note on internal event handling by implementations:
    # Concrete implementations of this interface are responsible for:
    # 1. Receiving events/data from their respective AI services (e.g., audio chunks for playback,
    #    transcribed text, errors, stream lifecycle events).
    # 2. Using the `self._audio_manager` instance (provided in __init__) to handle audio playback.
    #    This involves calling methods like:
    #    - `self._audio_manager.start_new_audio_stream(stream_id)`
    #    - `self._audio_manager.add_audio_chunk(audio_chunk_bytes)`
    #    - `self._audio_manager.end_audio_stream()`
    #    The `stream_id` would be generated or obtained by the manager to uniquely identify
    #    an AI response audio stream being played.
    #
    # 3. Handling text responses or transcriptions from the AI. This might involve logging,
    #    or if `VoiceCog` needs to act on these directly, a callback mechanism could be
    #    added to the interface (e.g., `register_text_handler(callback)`), though this
    #    is not part of the current interface design for simplicity.
