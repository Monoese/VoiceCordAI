"""
Unified Audio Processing Framework

This module provides a comprehensive audio processing interface that consolidates
all audio format conversion logic throughout the VoiceCordAI codebase. It supports
both real-time stateful processing (using audioop) for streaming use cases and
high-quality batch processing (using pydub) for one-shot conversions.

The framework automatically selects the optimal processing strategy based on
the use case, while also allowing manual strategy selection when needed.
"""

import asyncio
import base64
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ByteString, Dict, Optional, Tuple, Union

import audioop
from pydub import AudioSegment

from src.config.config import Config
from src.exceptions import AudioProcessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AudioFormat:
    """
    Immutable audio format specification.

    Defines the key parameters needed for audio processing:
    sample rate, channel count, and sample width.
    """

    sample_rate: int  # Samples per second (Hz)
    channels: int  # Number of audio channels (1=mono, 2=stereo)
    sample_width: int = 2  # Sample width in bytes (2 = 16-bit)

    def __str__(self) -> str:
        return f"{self.sample_rate}Hz, {self.channels}ch, {self.sample_width * 8}bit"

    @property
    def is_mono(self) -> bool:
        return self.channels == 1

    @property
    def is_stereo(self) -> bool:
        return self.channels == 2


# Standard audio format definitions used throughout the application
DISCORD_FORMAT = AudioFormat(
    sample_rate=Config.DISCORD_AUDIO_FRAME_RATE,
    channels=Config.DISCORD_AUDIO_CHANNELS,
    sample_width=Config.SAMPLE_WIDTH,
)

WAKE_WORD_FORMAT = AudioFormat(
    sample_rate=Config.WAKE_WORD_SAMPLE_RATE,
    channels=1,
    sample_width=Config.SAMPLE_WIDTH,
)

VAD_FORMAT = AudioFormat(
    sample_rate=Config.VAD_SAMPLE_RATE, channels=1, sample_width=Config.SAMPLE_WIDTH
)

OPENAI_FORMAT = AudioFormat(
    sample_rate=24000,  # OpenAI Realtime API requirement
    channels=1,
    sample_width=Config.SAMPLE_WIDTH,
)

GEMINI_INPUT_FORMAT = AudioFormat(
    sample_rate=16000,  # Gemini input requirement
    channels=1,
    sample_width=Config.SAMPLE_WIDTH,
)

GEMINI_OUTPUT_FORMAT = AudioFormat(
    sample_rate=24000,  # Gemini output format
    channels=1,
    sample_width=Config.SAMPLE_WIDTH,
)


class ProcessingStrategy(Enum):
    """
    Audio processing strategy selection.

    REALTIME: Fast, stateful processing for streaming audio (uses audioop)
    QUALITY: High-quality, stateless processing for batch operations (uses pydub)
    AUTO: Automatically select optimal strategy based on context
    """

    REALTIME = "realtime"
    QUALITY = "quality"
    AUTO = "auto"


class BaseAudioProcessor(ABC):
    """
    Abstract base class for audio processing implementations.

    Defines the interface that all audio processors must implement,
    supporting both synchronous and asynchronous operations.
    """

    @abstractmethod
    async def convert_async(
        self,
        audio_data: bytes,
        source_format: AudioFormat,
        target_format: AudioFormat,
        **kwargs,
    ) -> bytes:
        """
        Asynchronously convert audio from source format to target format.

        Args:
            audio_data: Raw PCM audio data to convert
            source_format: Current audio format
            target_format: Desired audio format
            **kwargs: Implementation-specific parameters

        Returns:
            bytes: Converted PCM audio data

        Raises:
            AudioProcessingError: If conversion fails
        """
        pass

    @abstractmethod
    def convert_sync(
        self,
        audio_data: bytes,
        source_format: AudioFormat,
        target_format: AudioFormat,
        **kwargs,
    ) -> bytes:
        """
        Synchronously convert audio from source format to target format.

        Same as convert_async but for synchronous contexts.
        """
        pass

    @abstractmethod
    def reset_state(self, state_key: str) -> None:
        """
        Reset any internal state for the given context.

        Args:
            state_key: Identifier for the processing context to reset
        """
        pass


class RealtimeAudioProcessor(BaseAudioProcessor):
    """
    Real-time audio processor using audioop for stateful, streaming conversion.

    This processor is optimized for low-latency streaming scenarios where
    small chunks of audio need to be processed continuously with minimal
    overhead. It maintains state between chunks to ensure smooth conversion
    without artifacts at chunk boundaries.

    Best for: Wake word detection, VAD processing, real-time streaming
    """

    def __init__(self):
        """Initialize the real-time processor with empty state tracking."""
        self._resample_states: Dict[str, Optional[Any]] = {}
        self._stats = {"chunks_processed": 0, "total_bytes": 0}

    async def convert_async(
        self,
        audio_data: bytes,
        source_format: AudioFormat,
        target_format: AudioFormat,
        state_key: Optional[str] = None,
        **kwargs,
    ) -> bytes:
        """
        Asynchronously convert audio using audioop with state preservation.

        Args:
            audio_data: Raw PCM audio data from source format
            source_format: Current audio format specification
            target_format: Desired audio format specification
            state_key: Unique key for maintaining conversion state across chunks

        Returns:
            bytes: Converted PCM audio data in target format
        """
        # For real-time processing, run in executor to avoid blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.convert_sync, audio_data, source_format, target_format, state_key
        )

    def convert_sync(
        self,
        audio_data: bytes,
        source_format: AudioFormat,
        target_format: AudioFormat,
        state_key: Optional[str] = None,
        **kwargs,
    ) -> bytes:
        """
        Synchronously convert audio using audioop with state preservation.

        This method handles the core conversion logic including:
        1. Channel reduction (stereo to mono if needed)
        2. Sample rate conversion with anti-aliasing
        3. State preservation for smooth chunk-to-chunk transitions
        """
        if not audio_data:
            return b""

        try:
            # Handle formats that are already correct
            if source_format == target_format:
                logger.debug(f"No conversion needed: {source_format}")
                return audio_data

            processed_audio = audio_data

            # Step 1: Convert to mono if needed
            if source_format.channels > target_format.channels:
                if source_format.is_stereo and target_format.is_mono:
                    processed_audio = audioop.tomono(
                        processed_audio,
                        source_format.sample_width,
                        1,
                        1,  # Equal weight for both channels
                    )
                    logger.debug(
                        f"Converted stereo to mono: {len(audio_data)} -> {len(processed_audio)} bytes"
                    )
                else:
                    raise AudioProcessingError(
                        f"Unsupported channel conversion: {source_format.channels} -> {target_format.channels}"
                    )

            # Step 2: Resample if sample rates differ
            if source_format.sample_rate != target_format.sample_rate:
                current_state = None
                if state_key:
                    current_state = self._resample_states.get(state_key)

                resampled_audio, new_state = audioop.ratecv(
                    processed_audio,
                    target_format.sample_width,
                    target_format.channels,
                    source_format.sample_rate,
                    target_format.sample_rate,
                    current_state,
                )

                # Update state for next chunk
                if state_key:
                    self._resample_states[state_key] = new_state

                processed_audio = resampled_audio
                logger.debug(
                    f"Resampled {source_format.sample_rate}Hz -> {target_format.sample_rate}Hz: "
                    f"{len(audio_data)} -> {len(processed_audio)} bytes [state_key: {state_key}]"
                )

            # Update processing statistics
            self._stats["chunks_processed"] += 1
            self._stats["total_bytes"] += len(audio_data)

            return processed_audio

        except audioop.error as e:
            logger.error(f"audioop conversion failed: {e}")
            raise AudioProcessingError(f"Real-time audio conversion failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in real-time audio processing: {e}")
            raise AudioProcessingError(f"Real-time audio processing error: {e}") from e

    def reset_state(self, state_key: str) -> None:
        """
        Reset resampling state for a specific context.

        Args:
            state_key: The state key to reset
        """
        if state_key in self._resample_states:
            del self._resample_states[state_key]
            logger.debug(f"Reset real-time processing state for: {state_key}")

    def clear_all_states(self) -> None:
        """Clear all resampling states."""
        state_count = len(self._resample_states)
        self._resample_states.clear()
        logger.debug(
            f"Cleared all real-time processing states ({state_count} contexts)"
        )



class QualityAudioProcessor(BaseAudioProcessor):
    """
    High-quality audio processor using pydub for stateless, batch conversion.

    This processor prioritizes audio quality over speed and is designed for
    one-shot conversions of complete audio buffers. It provides more sophisticated
    resampling algorithms and format support than the real-time processor.

    Best for: AI service preparation, audio cue loading, post-processing
    """

    def __init__(self):
        """Initialize the quality processor."""
        self._stats = {"conversions_processed": 0, "total_bytes": 0}

    async def convert_async(
        self,
        audio_data: bytes,
        source_format: AudioFormat,
        target_format: AudioFormat,
        **kwargs,
    ) -> bytes:
        """
        Asynchronously convert audio using pydub for high quality.

        Args:
            audio_data: Raw PCM audio data from source format
            source_format: Current audio format specification
            target_format: Desired audio format specification

        Returns:
            bytes: High-quality converted PCM audio data
        """
        # Run pydub operations in executor to avoid blocking event loop
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.convert_sync, audio_data, source_format, target_format
        )

    def convert_sync(
        self,
        audio_data: bytes,
        source_format: AudioFormat,
        target_format: AudioFormat,
        **kwargs,
    ) -> bytes:
        """
        Synchronously convert audio using pydub for high quality.

        This method provides comprehensive format conversion with:
        1. Automatic format detection and loading
        2. High-quality resampling algorithms
        3. Flexible channel and bit depth conversion
        """
        if not audio_data:
            return b""

        try:
            # Handle formats that are already correct
            if source_format == target_format:
                logger.debug(f"No conversion needed: {source_format}")
                return audio_data

            # Create AudioSegment from raw PCM data
            audio_segment = AudioSegment(
                data=audio_data,
                sample_width=source_format.sample_width,
                frame_rate=source_format.sample_rate,
                channels=source_format.channels,
            )

            # Apply format conversions
            processed_segment = audio_segment

            # Convert sample rate if needed
            if source_format.sample_rate != target_format.sample_rate:
                processed_segment = processed_segment.set_frame_rate(
                    target_format.sample_rate
                )
                logger.debug(
                    f"Resampled {source_format.sample_rate}Hz -> {target_format.sample_rate}Hz "
                    f"using high-quality algorithm"
                )

            # Convert channels if needed
            if source_format.channels != target_format.channels:
                processed_segment = processed_segment.set_channels(
                    target_format.channels
                )
                logger.debug(
                    f"Converted {source_format.channels}ch -> {target_format.channels}ch"
                )

            # Export to raw PCM bytes
            buffer = io.BytesIO()
            processed_segment.export(buffer, format="raw")
            result = buffer.getvalue()

            # Update processing statistics
            self._stats["conversions_processed"] += 1
            self._stats["total_bytes"] += len(audio_data)

            logger.debug(
                f"Quality audio conversion: {len(audio_data)} -> {len(result)} bytes "
                f"({source_format} -> {target_format})"
            )

            return result

        except Exception as e:
            logger.error(f"pydub conversion failed: {e}")
            raise AudioProcessingError(f"Quality audio conversion failed: {e}") from e

    def reset_state(self, state_key: str) -> None:
        """
        Quality processor is stateless, so this is a no-op.

        Args:
            state_key: Ignored for stateless processor
        """
        pass  # Stateless processor - nothing to reset



class UnifiedAudioProcessor:
    """
    Unified facade for all audio processing operations.

    This class provides a single interface for audio processing that automatically
    selects the optimal processing strategy based on the context. It also allows
    manual strategy selection when specific behavior is required.

    Strategy Selection Logic:
    - REALTIME: For streaming with state_key provided
    - QUALITY: For batch processing of large buffers
    - AUTO: Automatically choose based on context and data size
    """

    def __init__(self):
        """Initialize the unified processor with both processing backends."""
        self._realtime_processor = RealtimeAudioProcessor()
        self._quality_processor = QualityAudioProcessor()
        self._stats = {"auto_selections": {"realtime": 0, "quality": 0}}

    async def convert(
        self,
        source_format: AudioFormat,
        target_format: AudioFormat,
        audio_data: bytes,
        strategy: ProcessingStrategy = ProcessingStrategy.AUTO,
        state_key: Optional[str] = None,
        **kwargs,
    ) -> bytes:
        """
        Convert audio using the unified processing interface.

        Args:
            source_format: Current audio format
            target_format: Desired audio format
            audio_data: Raw PCM audio data to convert
            strategy: Processing strategy (AUTO, REALTIME, or QUALITY)
            state_key: Optional state key for stateful processing

        Returns:
            bytes: Converted PCM audio data
        """
        if not audio_data:
            return b""

        # Select processing strategy
        selected_strategy = self._select_strategy(strategy, audio_data, state_key)

        # Route to appropriate processor
        if selected_strategy == ProcessingStrategy.REALTIME:
            return await self._realtime_processor.convert_async(
                audio_data, source_format, target_format, state_key=state_key, **kwargs
            )
        else:  # QUALITY
            return await self._quality_processor.convert_async(
                audio_data, source_format, target_format, **kwargs
            )

    def convert_sync(
        self,
        source_format: AudioFormat,
        target_format: AudioFormat,
        audio_data: bytes,
        strategy: ProcessingStrategy = ProcessingStrategy.AUTO,
        state_key: Optional[str] = None,
        **kwargs,
    ) -> bytes:
        """
        Synchronously convert audio using the unified processing interface.

        This method is specifically designed for use in synchronous contexts like
        Discord's audio thread where async/await is not available. It provides
        the same strategy selection and routing as the async method.

        Args:
            source_format: Current audio format
            target_format: Desired audio format
            audio_data: Raw PCM audio data to convert
            strategy: Processing strategy (AUTO, REALTIME, or QUALITY)
            state_key: Optional state key for stateful processing

        Returns:
            bytes: Converted PCM audio data
        """
        if not audio_data:
            return b""

        # Select processing strategy
        selected_strategy = self._select_strategy(strategy, audio_data, state_key)

        # Route to appropriate processor using synchronous methods
        if selected_strategy == ProcessingStrategy.REALTIME:
            return self._realtime_processor.convert_sync(
                audio_data, source_format, target_format, state_key=state_key, **kwargs
            )
        else:  # QUALITY
            return self._quality_processor.convert_sync(
                audio_data, source_format, target_format, **kwargs
            )

    def _select_strategy(
        self,
        requested_strategy: ProcessingStrategy,
        audio_data: bytes,
        state_key: Optional[str],
    ) -> ProcessingStrategy:
        """
        Select the optimal processing strategy based on context.

        Args:
            requested_strategy: User-requested strategy
            audio_data: Audio data being processed (for size analysis)
            state_key: State key presence indicates streaming context

        Returns:
            ProcessingStrategy: Selected strategy (REALTIME or QUALITY)
        """
        if requested_strategy in (
            ProcessingStrategy.REALTIME,
            ProcessingStrategy.QUALITY,
        ):
            return requested_strategy

        # AUTO selection logic
        data_size = len(audio_data)

        # Use REALTIME if:
        # 1. State key provided (indicates streaming context)
        # 2. Small chunk size (likely streaming)
        if state_key is not None or data_size < 100_000:  # 100KB threshold
            self._stats["auto_selections"]["realtime"] += 1
            return ProcessingStrategy.REALTIME

        # Use QUALITY for large buffers (likely batch processing)
        self._stats["auto_selections"]["quality"] += 1
        return ProcessingStrategy.QUALITY

    def reset_state(self, state_key: str) -> None:
        """Reset state in the real-time processor."""
        self._realtime_processor.reset_state(state_key)

    def clear_all_states(self) -> None:
        """Clear all states in the real-time processor."""
        self._realtime_processor.clear_all_states()



# Utility functions for common operations
async def load_audio_file(file_path: Union[Path, str]) -> Tuple[bytes, AudioFormat]:
    """
    Load an audio file and return raw PCM data with format information.

    Args:
        file_path: Path to the audio file (supports MP3, WAV, etc.)

    Returns:
        Tuple[bytes, AudioFormat]: Raw PCM data and detected format

    Raises:
        AudioProcessingError: If file loading fails
    """
    try:
        loop = asyncio.get_running_loop()

        def _load_file():
            audio_segment = AudioSegment.from_file(str(file_path))
            buffer = io.BytesIO()
            audio_segment.export(buffer, format="raw")

            format_info = AudioFormat(
                sample_rate=audio_segment.frame_rate,
                channels=audio_segment.channels,
                sample_width=audio_segment.sample_width,
            )

            return buffer.getvalue(), format_info

        pcm_data, format_info = await loop.run_in_executor(None, _load_file)
        logger.debug(
            f"Loaded audio file {file_path}: {format_info}, {len(pcm_data)} bytes"
        )
        return pcm_data, format_info

    except Exception as e:
        logger.error(f"Failed to load audio file {file_path}: {e}")
        raise AudioProcessingError(f"Audio file loading failed: {e}") from e


def encode_pcm_to_base64(pcm_data: ByteString) -> str:
    """
    Encode PCM audio data to base64 string for transmission.

    This is commonly used for WebSocket transmission to AI services
    where binary data needs to be represented as text.

    Args:
        pcm_data: Raw PCM audio data to encode

    Returns:
        str: Base64-encoded string representation
    """
    return base64.b64encode(pcm_data).decode("utf-8")


def decode_base64_to_pcm(base64_data: str) -> bytes:
    """
    Decode base64 string back to PCM audio data.

    Args:
        base64_data: Base64-encoded audio data

    Returns:
        bytes: Raw PCM audio data
    """
    return base64.b64decode(base64_data)


# Async Base64 utilities for AI services
async def encode_pcm_to_base64_async(pcm_data: bytes) -> str:
    """
    Asynchronously encode PCM audio data to base64 string.
    
    Args:
        pcm_data: Raw PCM audio data to encode
        
    Returns:
        str: Base64 encoded audio data
    """
    loop = asyncio.get_running_loop()
    b64_bytes = await loop.run_in_executor(None, base64.b64encode, pcm_data)
    return b64_bytes.decode("utf-8")


async def decode_base64_to_pcm_async(base64_data: str) -> bytes:
    """
    Asynchronously decode base64 string to PCM audio data.
    
    Args:
        base64_data: Base64 encoded audio data
        
    Returns:
        bytes: Raw PCM audio data
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, base64.b64decode, base64_data)
