"""
Gemini Realtime API Manager.

This module provides the GeminiRealtimeManager class, which serves as
the primary interface for the bot to interact with the Google Gemini Live API.
It coordinates connection management, event sending, and event handling.
"""

from typing import Optional, Dict, Any, Callable, Awaitable

from google import genai
from google.genai import types

from src.ai_services.base_manager import BaseRealtimeManager
from src.audio.playback import AudioPlaybackManager
from src.utils.logger import get_logger

from .event_handler import GeminiEventHandlerAdapter
from .connection import GeminiRealtimeConnection

logger = get_logger(__name__)


class GeminiRealtimeManager(BaseRealtimeManager):
    """
    Manages interactions with the Google Gemini Live API.

    This class handles:
    - Initializing the google.generativeai.Client.
    - Managing the Live API session for communication.
    - Coordinating with GeminiEventHandlerAdapter for processing incoming events.
    - Providing methods to send various commands/data to the Gemini Live API.
    - Managing the connection lifecycle (connect, disconnect) as per the IRealtimeAIServiceManager interface.
    """

    def __init__(
        self,
        audio_playback_manager: AudioPlaybackManager,
        service_config: Dict[str, Any],
    ):
        """
        Initializes the GeminiRealtimeManager.

        Args:
            audio_playback_manager: An instance of AudioPlaybackManager, required by the event handler.
            service_config: Configuration specific to this service instance.

        Raises:
            ValueError: If the API key or model name is missing in the configuration.
        """
        super().__init__(audio_playback_manager, service_config)

        self._live_connect_config_params: Dict[str, Any] = self._service_config.get(
            "live_connect_config", {}
        )

        # Let genai.Client raise its own exception on an invalid key
        self._gemini_client: genai.Client = genai.Client(api_key=self._api_key)
        logger.info("Gemini client initialized with API key.")

        self._event_handler_adapter: GeminiEventHandlerAdapter = (
            GeminiEventHandlerAdapter(
                self._audio_playback_manager, self.response_audio_format
            )
        )

        self._connection_handler: GeminiRealtimeConnection = GeminiRealtimeConnection(
            gemini_client=self._gemini_client,
            model_name=self._model_name,
            live_connect_config_params=self._live_connect_config_params,
        )

    @property
    def _event_callback(self) -> Callable[[Any], Awaitable[None]]:
        return self._event_handler_adapter.dispatch_event

    async def _post_connect_hook(self) -> bool:
        # Gemini does not require a post-connection setup step.
        return True

    async def _get_active_session(
        self,
    ) -> Optional[genai.live.AsyncSession]:
        """Helper to get the active session object from the connection handler.

        Returns:
            The active `AsyncSession` object if connected, otherwise `None`.
        """
        if self.is_connected():
            session = self._connection_handler.get_active_session()
            if session:
                return session
            else:
                logger.warning(
                    "_get_active_session: is_connected() is True, but get_active_session() returned None."
                )
        return None

    async def send_audio_chunk(self, audio_data: bytes) -> bool:
        """
        Sends a chunk of raw audio data to the Gemini Live API.
        The audio is wrapped in a types.Blob and sent via the `media` parameter.
        """
        session = await self._get_active_session()
        if not session:
            logger.error(
                "GeminiRealtimeManager: Cannot send audio chunk, not connected or session not available."
            )
            return False

        try:
            # Get the required audio frame rate from the service-specific configuration.
            frame_rate, _ = self.processing_audio_format
            mime_type = f"audio/pcm;rate={frame_rate}"

            # Wrap the audio data in a types.Blob and use the 'media' parameter.
            audio_blob = types.Blob(data=audio_data, mime_type=mime_type)

            await session.send_realtime_input(media=audio_blob)
            logger.debug(
                f"Sent audio chunk ({len(audio_data)} bytes) as Blob with MIME type '{mime_type}' to Gemini."
            )
            return True
        except Exception as e:
            logger.error(
                f"GeminiRealtimeManager: Error sending audio chunk: {e}", exc_info=True
            )
            return False

    async def finalize_input_and_request_response(self) -> bool:
        """
        Signals to Gemini that all audio for the current turn has been provided
        and a response is now expected. Uses `send_realtime_input(audio_stream_end=True)`.
        """
        session = await self._get_active_session()
        if not session:
            logger.error(
                "GeminiRealtimeManager: Cannot finalize input, not connected or session not available."
            )
            return False

        try:
            await session.send_realtime_input(audio_stream_end=True)
            logger.info(
                "GeminiRealtimeManager: Sent audio_stream_end=True to finalize input."
            )
            return True
        except Exception as e:
            logger.error(
                f"GeminiRealtimeManager: Error sending audio_stream_end: {e}",
                exc_info=True,
            )
            return False

    async def cancel_ongoing_response(self) -> bool:
        """
        Attempts to cancel any AI response currently being generated or streamed.
        For Gemini, there's no explicit server-side cancellation API mentioned in doc.md.
        This will perform a client-side cancellation by stopping local audio playback.
        """
        if not self.is_connected():  # Check general connection status
            logger.warning(
                "GeminiRealtimeManager: Cannot cancel response, not connected."
            )
            # Still attempt local audio manager cleanup if it was playing.

        logger.info(
            "GeminiRealtimeManager: Attempting to cancel ongoing response (client-side playback stop)."
        )

        # Get the current playing stream ID from AudioManager (if any)
        # This ID was generated client-side per turn for Gemini.
        current_playing_response_id = (
            self._audio_playback_manager.get_current_playing_response_id()
        )

        if current_playing_response_id:
            try:
                logger.info(
                    f"GeminiRealtimeManager: Requesting AudioPlaybackManager to end stream: {current_playing_response_id}"
                )
                await (
                    self._audio_playback_manager.end_audio_stream()
                )  # This stops playback and clears queue for the stream
                # Note: For Gemini, the _event_loop also calls end_audio_stream when a turn naturally ends.
                # Calling it here ensures immediate stop if user interrupts.
                # The _event_loop's subsequent call for the same stream_id should be harmless
                # if AudioPlaybackManager handles repeated calls to end_audio_stream gracefully.
            except Exception as e:
                logger.error(
                    f"GeminiRealtimeManager: Error during client-side cancellation: {e}",
                    exc_info=True,
                )
                return False
        else:
            logger.info(
                "GeminiRealtimeManager: No active audio stream reported by AudioPlaybackManager to cancel."
            )

        # Since there's no server-side cancellation signal to send for Gemini based on current docs,
        # we consider the client-side action (stopping playback) as success.
        return True
