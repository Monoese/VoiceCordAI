"""
Gemini Event Handler Adapter.

This module provides the GeminiEventHandlerAdapter class, responsible for
processing messages received from the Gemini Live API.
"""

from google.genai import types

from src.audio.audio import AudioManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiEventHandlerAdapter:
    """
    Handles and processes messages received from the Gemini Live API.
    """

    def __init__(self, audio_manager: AudioManager):
        """
        Initializes the GeminiEventHandlerAdapter.

        Args:
            audio_manager: The AudioManager instance to use for audio processing.
        """
        self.audio_manager: AudioManager = audio_manager
        # For Gemini, the stream_id for audio playback is generated per "turn"
        # in the GeminiRealtimeConnection's event loop.

    async def process_live_server_message(
        self, message: types.LiveServerMessage
    ) -> None:
        """
        Processes a single LiveServerMessage from a Gemini API turn.

        This method is called by the manager's event loop for each message.
        The stream_id context (start_new_audio_stream/end_audio_stream) is managed
        by the calling event loop in GeminiRealtimeManager per "turn".

        Args:
            message: The LiveServerMessage object received from the Gemini API.
        """
        if message.server_content:
            # The `data` property concatenates all inline data parts (audio).
            # The `text` property concatenates all text parts.
            audio_data = message.data
            text_data = message.text

            if audio_data:
                logger.debug(
                    f"Received audio data chunk, size: {len(audio_data)} bytes."
                )
                try:
                    await self.audio_manager.add_audio_chunk(audio_data)
                except Exception as e:
                    logger.error(
                        f"Error adding audio chunk to AudioManager: {e}", exc_info=True
                    )

            if text_data:
                # For now, just log text. Future enhancements could involve callbacks.
                logger.info(f"Gemini text response: {text_data}")

        elif message.go_away:
            logger.warning(f"Received 'go_away' message from Gemini: {message.go_away}")
            # The manager's event loop might need to handle this by attempting to reconnect or shutting down.

        elif message.tool_call:
            logger.info(f"Received 'tool_call' from Gemini: {message.tool_call}")
            # Not handled in this audio-focused MVP.

        elif message.tool_call_cancellation:
            logger.info(
                f"Received 'tool_call_cancellation' from Gemini: {message.tool_call_cancellation}"
            )
            # Not handled in this audio-focused MVP.

        elif message.usage_metadata:
            logger.debug(
                f"Received 'usage_metadata' from Gemini: {message.usage_metadata}"
            )

        elif message.setup_complete:
            logger.info(
                f"Received 'setup_complete' from Gemini: {message.setup_complete}"
            )

        elif message.session_resumption_update:
            logger.info(
                f"Received 'session_resumption_update' from Gemini: {message.session_resumption_update}"
            )

        # Check for other potential fields if the API evolves or if there are error fields not explicitly listed.
        # The LiveServerMessage structure provided earlier didn't have a direct 'error' field,
        # but errors might come as part of 'go_away' or other mechanisms.
        # If an error occurs that terminates the stream, the manager's event loop should catch it.
