"""
Defines the IRealtimeAIServiceManager abstract base class (ABC).

This interface specifies the contract for real-time AI service managers,
allowing the bot to interact with different AI services (like OpenAI, Gemini)
through a common set of methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Callable, Awaitable
from src.audio.playback import AudioPlaybackManager


class IRealtimeAIServiceManager(ABC):
    """
    Abstract Base Class for a real-time AI service manager.

    This interface defines the common methods that VoiceCog will use to interact
    with an underlying AI service for voice chat functionalities, including
    connection management, sending audio/text, and controlling responses.

    It also defines properties for retrieving service-specific audio formats.
    """

    def __init__(
        self,
        audio_playback_manager: AudioPlaybackManager,
        service_config: Dict[str, Any],
    ) -> None:
        """
        Initializes the AI service manager.

        Args:
            audio_playback_manager: An instance of AudioPlaybackManager for handling audio playback.
                           Concrete implementations will use this to play audio received
                           from the AI service.
            service_config: A dictionary containing service-specific configurations,
                            such as API keys, model names, endpoint URLs,
                            initial session parameters, or other settings required by
                            the specific AI service implementation. The following keys
                            are required:
                            - `processing_audio_frame_rate` and `processing_audio_channels`
                            - `response_audio_frame_rate` and `response_audio_channels`
        """
        self._audio_playback_manager: AudioPlaybackManager = audio_playback_manager
        self._service_config: Dict[str, Any] = service_config

        # --- Audio Format Configuration ---
        # Directly access configuration keys. A KeyError will be raised if a key is
        # missing, which indicates a configuration error that should be fixed.
        self._processing_audio_format: Tuple[int, int] = (
            service_config["processing_audio_frame_rate"],
            service_config["processing_audio_channels"],
        )

        self._response_audio_format: Tuple[int, int] = (
            service_config["response_audio_frame_rate"],
            service_config["response_audio_channels"],
        )

    @property
    def processing_audio_format(self) -> Tuple[int, int]:
        """
        Returns the audio format required by the AI service for processing input audio.

        Returns:
            A tuple containing (frame_rate: int, channels: int).
        """
        return self._processing_audio_format

    @property
    def response_audio_format(self) -> Tuple[int, int]:
        """
        Returns the audio format of the response audio from the AI service.

        This is used to correctly configure the playback system (e.g., FFmpeg)
        to handle the audio stream received from the service.

        Returns:
            A tuple containing (frame_rate: int, channels: int).
        """
        return self._response_audio_format

    @abstractmethod
    async def connect(
        self,
        on_connect: Callable[[], Awaitable[None]],
        on_disconnect: Callable[[], Awaitable[None]],
    ) -> bool:
        """
        Establishes a connection to the real-time AI service and initializes the session.

        Implementations should handle all necessary setup, including authentication
        and sending any initial configuration or session parameters derived from
        `self._service_config`.

        Args:
            on_connect: An async callback to be invoked upon successful connection.
            on_disconnect: An async callback to be invoked upon disconnection.

        Returns:
            True if the connection and session setup were successful, False otherwise.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Closes the connection to the AI service and cleans up any associated resources.

        Implementations should ensure that all tasks related to the connection are
        properly terminated.

        Returns:
            None.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Checks if the manager is currently connected to the AI service.

        Implementations should provide the logic to accurately report the
        connection status of the underlying service.

        Returns:
            True if connected, False otherwise.
        """
        pass

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
    async def finalize_input_and_request_response(self) -> bool:
        """
        Signals to the AI service that all input for the current turn has been provided
        and a response is now expected from the AI.

        This is particularly relevant for turns involving a stream of audio chunks.
        Implementations will map this to the specific mechanism of their AI service:
        - For OpenAI: This would typically involve operations like 'commit_audio_buffer'
          and then 'create_response'.
        - For Gemini (if primarily audio was sent):
            - If automatic Voice Activity Detection (VAD) is used by the service, this might
              involve sending an `audio_stream_end=True` signal.
            - If manual VAD is used, this might involve sending an `activity_end=True` signal.

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
    # 2. Using the `self._audio_playback_manager` instance (provided in __init__) to handle audio playback.
    #    This involves calling methods like:
    #    - `self._audio_playback_manager.start_new_audio_stream(stream_id, self.response_audio_format)`
    #    - `self._audio_playback_manager.add_audio_chunk(audio_chunk_bytes)`
    #    - `self._audio_playback_manager.end_audio_stream()`
    #    The `stream_id` would be generated or obtained by the manager to uniquely identify
    #    an AI response audio stream being played.
    #
    # 3. Handling text responses or transcriptions from the AI. This might involve logging,
    #    or if `VoiceCog` needs to act on these directly, a callback mechanism could be
    #    added to the interface (e.g., `register_text_handler(callback)`), though this
    #    is not part of the current interface design for simplicity.
