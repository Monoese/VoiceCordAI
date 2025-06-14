"""
Gemini Realtime API Manager.

This module provides the GeminiRealtimeManager class, which serves as
the primary interface for the bot to interact with the Google Gemini Live API.
It coordinates connection management, event sending, and event handling.
"""

import asyncio
from typing import Optional, Dict, Any

from google import genai
from google.genai import types

from src.ai_services.interface import IRealtimeAIServiceManager
from src.audio.audio import AudioManager
from src.config.config import Config
from src.utils.logger import get_logger

from .event_handler import GeminiEventHandlerAdapter
from .connection import GeminiRealtimeConnection

logger = get_logger(__name__)


class GeminiRealtimeManager(IRealtimeAIServiceManager):
    """
    Manages interactions with the Google Gemini Live API.

    This class handles:
    - Initializing the google.generativeai.Client.
    - Managing the Live API session for communication.
    - Coordinating with GeminiEventHandlerAdapter for processing incoming events.
    - Providing methods to send various commands/data to the Gemini Live API.
    - Managing the connection lifecycle (connect, disconnect) as per the IRealtimeAIServiceManager interface.
    """

    def __init__(self, audio_manager: AudioManager, service_config: Dict[str, Any]):
        """
        Initializes the GeminiRealtimeManager.

        Args:
            audio_manager: An instance of AudioManager, required by the event handler.
            service_config: Configuration specific to this service instance.
                            Expected keys:
                            - "api_key": Gemini API key.
                            - "model_name": Name of the Gemini model to use.
                            - "live_connect_config": Dict for genai.types.LiveConnectConfig parameters.
        """
        super().__init__(audio_manager, service_config)

        self.api_key: Optional[str] = self._service_config.get("api_key")
        self.model_name: Optional[str] = self._service_config.get("model_name")
        self.live_connect_config_params: Dict[str, Any] = self._service_config.get(
            "live_connect_config", {}
        )

        if not self.api_key:
            logger.warning(
                "Gemini API key not found in service_config. Falling back to Config.GEMINI_API_KEY (if defined)."
            )
            # This assumes GEMINI_API_KEY will be added to Config class later
            self.api_key = getattr(Config, "GEMINI_API_KEY", None)

        if not self.api_key:
            # If still no API key, log an error. Connection will likely fail.
            logger.error(
                "Gemini API key is not configured. GeminiRealtimeManager will not be able to connect."
            )
            # Initialization can proceed, but connect() will fail or not be attempted.

        self.gemini_client: Optional[genai.Client] = None
        if self.api_key:
            try:
                # Initialize client directly with API key
                self.gemini_client = genai.Client(api_key=self.api_key)
                logger.info("Gemini client initialized with API key.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}", exc_info=True)
                # self.gemini_client remains None
        else:
            logger.warning(
                "Gemini API key not available. GeminiRealtimeManager will not be able to connect."
            )

        self.event_handler_adapter: GeminiEventHandlerAdapter = (
            GeminiEventHandlerAdapter(self._audio_manager)
        )

        # Initialize connection_handler if gemini_client and model_name are available
        if self.gemini_client and self.model_name:
            self.connection_handler: GeminiRealtimeConnection = GeminiRealtimeConnection(
                gemini_client=self.gemini_client,
                model_name=self.model_name,
                live_connect_config_params=self.live_connect_config_params,
                event_handler_adapter=self.event_handler_adapter,
                audio_manager=self._audio_manager,  # Pass audio_manager for stream ID logic
            )
        else:
            logger.error(
                "GeminiRealtimeManager: Cannot initialize ConnectionHandler due to missing client or model_name."
            )
            # This state will prevent connection attempts.
            self.connection_handler = None  # type: ignore # Explicitly None if not initializable

    async def _get_active_session(
        self,
    ) -> Optional[genai.live.AsyncSession]:
        """Helper to get the active session object from the connection handler."""
        if not self._is_connected_flag or not self.connection_handler:
            logger.debug(
                "_get_active_session: Not connected or connection_handler not initialized."
            )
            return None

        if self.connection_handler.is_connected():
            session = self.connection_handler.get_active_session()
            if session:
                return session
            else:
                logger.warning(
                    "_get_active_session: _is_connected_flag is True, but connection_handler.get_active_session returned None."
                )
        else:
            logger.warning(
                "_get_active_session: _is_connected_flag is True, but connection_handler.is_connected is False. State inconsistency."
            )
        return None

    async def connect(self) -> bool:
        """
        Establishes a connection to the Gemini Live API via the connection handler.
        """
        logger.info("GeminiRealtimeManager: Attempting to connect...")
        if self._is_connected_flag:
            logger.info("GeminiRealtimeManager: Already connected.")
            return True

        if not self.connection_handler:
            logger.error(
                "GeminiRealtimeManager: ConnectionHandler not initialized. Cannot connect."
            )
            return False

        if (
            not self.gemini_client
        ):  # Redundant check if connection_handler init implies it, but safe
            logger.error(
                "GeminiRealtimeManager: Cannot connect, Gemini client not initialized."
            )
            return False

        if not self.model_name:
            logger.error(
                "GeminiRealtimeManager: Cannot connect, model_name not configured."
            )
            return False

        await self.connection_handler.connect()

        timeout = self._service_config.get("connection_timeout", 30.0)
        wait_interval = 0.1
        max_attempts = int(timeout / wait_interval)

        for attempt in range(max_attempts):
            if self.connection_handler.is_connected():
                logger.info(
                    "GeminiRealtimeManager: Connection successfully established with handler."
                )
                self._is_connected_flag = True
                return True
            if attempt < max_attempts - 1:
                await asyncio.sleep(wait_interval)

        logger.warning(
            f"GeminiRealtimeManager: Failed to establish connection within {timeout}s timeout."
        )
        await self.connection_handler.disconnect()  # Ensure cleanup
        self._is_connected_flag = False
        return False

    async def disconnect(self) -> None:
        """
        Closes the connection to the Gemini Live API via the connection handler.
        """
        logger.info("GeminiRealtimeManager: Attempting to disconnect...")
        if self.connection_handler:
            await self.connection_handler.disconnect()
        self._is_connected_flag = False
        logger.info("GeminiRealtimeManager: Disconnected.")

    async def send_audio_chunk(self, audio_data: bytes) -> bool:
        """
        Sends a chunk of raw audio data to the Gemini Live API.
        Uses `send_realtime_input(audio=audio_data)`.
        """
        session = await self._get_active_session()
        if not session:
            logger.error(
                "GeminiRealtimeManager: Cannot send audio chunk, not connected or session not available."
            )
            return False

        try:
            # Wrap the audio data in a types.Blob and use the 'media' parameter
            # The MIME type should reflect the format of `audio_data`
            # which is PCM, 1 channel (mono), 24000 Hz from our processing.
            mime_type = f"audio/pcm;rate={Config.PROCESSING_AUDIO_FRAME_RATE}"
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

    async def send_text_message(self, text: str, finalize_turn: bool) -> bool:
        """
        Sends a text message to the Gemini Live API.
        This is a placeholder for now, as the primary focus is audio.
        If `finalize_turn` is True, it could signal activity_end or audio_stream_end.
        """
        session = await self._get_active_session()
        if (
            not session
        ):  # Check if session is None, implying not connected or session not available
            logger.error(
                "GeminiRealtimeManager: Cannot send text message, not connected or session not available."
            )
            return False

        logger.debug(
            f"send_text_message called with text: '{text}', finalize_turn: {finalize_turn}. This is a placeholder for Gemini audio focus."
        )

        # The `doc.md` suggests `send_client_content` for non-realtime turn-based content,
        # or `send_realtime_input(text=...)` for realtime text.
        # Since this interface method is generic, and our focus is audio,
        # we'll keep it minimal. If `finalize_turn` is true, we can mimic
        # ending the audio stream as if text was the final part of a spoken turn.
        if finalize_turn:
            try:
                # Option 2: If text is just a signal to finalize, and no actual text content is expected by Gemini
                # in this audio-centric flow, just send audio_stream_end.
                # This aligns with the idea that text might be a secondary signal.
                # For now, let's assume text itself isn't sent, but finalize_turn is honored.
                if text:  # Log if text was provided but not sent
                    logger.info(
                        f"Text content '{text}' provided to send_text_message but not sent to Gemini in this audio-focused placeholder."
                    )

                await session.send_realtime_input(
                    audio_stream_end=True
                )  # Or activity_end=True
                logger.info(
                    "GeminiRealtimeManager: Sent audio_stream_end=True due to send_text_message with finalize_turn=True."
                )
                return True
            except Exception as e:
                logger.error(
                    f"GeminiRealtimeManager: Error in send_text_message (finalizing turn): {e}",
                    exc_info=True,
                )
                return False
        else:
            # If not finalizing turn, and we are not sending text, this is a no-op.
            if text:
                logger.info(
                    f"Text content '{text}' provided to send_text_message (finalize_turn=False) but not sent to Gemini in this audio-focused placeholder."
                )
            return True  # Successfully did nothing as per placeholder nature.

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
            self._audio_manager.get_current_playing_response_id()
        )

        if current_playing_response_id:
            logger.info(
                f"GeminiRealtimeManager: Requesting AudioManager to end stream: {current_playing_response_id}"
            )
            await (
                self._audio_manager.end_audio_stream()
            )  # This stops playback and clears queue for the stream
            # Note: For Gemini, the _event_loop also calls end_audio_stream when a turn naturally ends.
            # Calling it here ensures immediate stop if user interrupts.
            # The _event_loop's subsequent call for the same stream_id should be harmless
            # if AudioManager handles repeated calls to end_audio_stream gracefully.
        else:
            logger.info(
                "GeminiRealtimeManager: No active audio stream reported by AudioManager to cancel."
            )

        # Since there's no server-side cancellation signal to send for Gemini based on current docs,
        # we consider the client-side action (stopping playback) as success.
        return True
